import pandas as pd
import streamlit as st
import numpy as np
import pickle
import dill
import requests
import plotly.express as px

from dashboard_functions import *

@st.cache(allow_output_mutation=True)
def read_df(path):
    return pd.read_csv(path,
                verbose=False,
                encoding='ISO-8859-1',
                )
def read_pickle(path):
    with open(path,"rb") as f:
        object = pickle.load(f)
    return object

st.set_page_config(page_title= 'Credit Score App', layout="wide", initial_sidebar_state='expanded')
                
st.title("Evaluate your client's credit capacity.")
placeholder = st.empty()
placeholder_bis = st.empty()
return_button = st.empty()
#----------------------------------------------------------------------------------#
#                                 LOADING DATA                                     #
#----------------------------------------------------------------------------------#

df = read_df('data/dataset_sample.csv')

df = df.replace([np.inf, -np.inf], np.nan)

preprocessor = read_pickle('ressource/pipeline')

clf = read_pickle('ressource/classifier')

#----------------------------------------------------------------------------------#
#                                   SIDEBAR                                        #
#----------------------------------------------------------------------------------#
page = None
st.sidebar.markdown(
    """
    <style>
    [data-baseweb="select"] {
        margin-top: -100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

all_clients_id = df['SK_ID_CURR'].unique()
def selectbox_with_default(text, values, default='Client ID:', sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.sidebar.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

client_id = selectbox_with_default('', all_clients_id)

if client_id == 'Client ID:':
    with placeholder.container():
        st.write("This WebApps is a decision making helping tool.\n\
        A supervised binary classifier algorithm has been trained in order to predict the client's probability to make a default payment or not.")
        st.warning("Please select a client !")

else:
    data_client= df[df["SK_ID_CURR"]==client_id]
    client_index = data_client.index[0]

    placeholder.write(f"You've selected client #{client_id}.")

    gender = data_client.loc[client_index, "CODE_GENDER"]
    if gender == 1:
        gender = 'Male'
    else:
        gender = 'Female'

    family_status = data_client.loc[client_index, "NAME_FAMILY_STATUS"]
    loan_type = data_client.loc[client_index, "NAME_CONTRACT_TYPE"]
    education = data_client.loc[client_index, "NAME_EDUCATION_TYPE"]
    credit = data_client.loc[client_index, "AMT_CREDIT"]
    annuity = data_client.loc[client_index, "AMT_ANNUITY"]
    fam_members = data_client.loc[client_index, "CNT_FAM_MEMBERS"]
    childs = data_client.loc[client_index, "CNT_CHILDREN"]
    income_per_person = data_client.loc[client_index, "INCOME_PER_PERSON"]
    payment_rate = data_client.loc[client_index, "PAYMENT_RATE"]
    income_type = data_client.loc[client_index, "NAME_INCOME_TYPE"]
    occupation_type = data_client.loc[client_index, "OCCUPATION_TYPE"]
    work = income_type

    days_birth = data_client.loc[client_index, "DAYS_BIRTH"]
    age = -int(round(days_birth/365))
    
    days_employed = data_client.loc[client_index, "DAYS_EMPLOYED"]
    try: 
        years_work = -int(round(days_employed/365))
        if years_work < 1: 
            years_work = 'Less than a year'
        elif years_work == 1:
            years_work = str(years_work) + ' year'
        else: 
            years_work = str(years_work) + ' years'
    except:
        years_work = 'no information'
    
    st.sidebar.subheader('General informations:')
    
    st.sidebar.write('**Gender:** %s' %gender)
    st.sidebar.write('**Age:** %s' %age)
    st.sidebar.write('**Education level:** %s' %education)

    st.sidebar.write('**Marital status:** %s' %family_status)
    st.sidebar.write('**Family members :** %s (including %s children) '%(int(round(fam_members)), int(round(childs))))

    st.sidebar.write('**Work:** %s' %work)
    st.sidebar.write('**Work experiences:** %s ' %years_work)
    st.sidebar.write('**Income:** {:,} $'.format(round(income_per_person)))

    st.sidebar.write('')
    st.sidebar.subheader('Credit informations:')
    st.sidebar.write('**Credit amount:** {:,} $'.format(round(credit)))
    st.sidebar.write('**Annuity amount:** {:,} $'.format(round(annuity)))
    st.sidebar.write('**Payment rate:** {:.2%}'.format(payment_rate))
    page_selection = st.sidebar.empty()
    run_button = st.sidebar.empty()
    
    
    page = page_selection.radio('', ['Check credit score', 'Client more informations'])


if page == 'Check credit score':
    check = run_button.button('Run')
    if check:
        run_button.empty()
    #----------------------------------------------------------------------------------#
    #                                 PREPROCESSING                                    #
    #----------------------------------------------------------------------------------#

        X = data_client.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = data_client['TARGET']

        X = preprocessor.transform(X)

    #----------------------------------------------------------------------------------#
    #                           PREDICT, WITH API                                      #
    #----------------------------------------------------------------------------------#
        with st.spinner(f"Model working..."):
            url_api = 'https://credit-score-fastapi.herokuapp.com'
            data_json = {"id" : int(client_id)}
            headers = {"Content-Type": "application/json"}
            response = requests.request(method="POST", headers=headers, url=url_api, json=data_json).json()
            prob = response["Credit score"]
    #----------------------------------------------------------------------------------#
    #                               DISPLAY RESULTS                                    #
    #----------------------------------------------------------------------------------#
        with placeholder.container():
            left_column_recom, right_column_recom = st.columns(2)

            with left_column_recom:
                gauge_place = st.empty()
                for value in np.arange(0, round(prob*100, 1) +.1, step=.1):
                    fig_gauge = plot_gauge(value)
                    gauge_place.plotly_chart(fig_gauge, use_container_width=True)

            with right_column_recom: 
                # Display clients data and prediction
                st.write(f"Client #{client_id} has **{prob*100:.2f} % of risk** to make default.")
                st.markdown('##')

                if prob*100 < 30:
                    st.success(f"We recommand to **accept** client's application to loan.")
                elif (prob*100 >= 30) & (prob*100 <= 50):
                    st.warning(f"Client's chances to make default are between 30 and 50% . We recommand \
                                to **analyse closely** the data to make your decision.")
                else:
                    st.error(f"We recommand to **reject** client's application to loan.")
                
                st.markdown('##')
                st.markdown('##')
                st.caption(f'''Below 30% of default risk, we recommand to accept client application.\
                            Above 50% of default risk, we recommand to reject client application. \
                            Between 30 and 50%, your expertise will be your best advice in your decision making.\
                            You can use the "client more informations" page to help in the evaluation.''')

            # Display shap explainer of client's prediction            
            with st.spinner('Analysing...'):
                
                with open('ressource/feats', 'rb') as f:
                    feats = dill.load(f)
                SHAP_explainer = read_pickle('ressource/shap_explainer')

                shap_vals = SHAP_explainer.shap_values(X[0])

                shap_explained, most_important_features = format_shap_values(shap_vals[1], feats)
                explained_chart = plot_important_features(shap_explained, most_important_features)

            st.subheader('Prediction explanation')
            st.bokeh_chart(explained_chart, use_container_width=True)
            st.write('Red color indicates features that are pushing the prediction higher, and red color indicates just the opposite.')
        if return_button.button('Return'):
            scoring = run_button.button('Check credit score')

if page == 'Client more informations':
    description = pd.read_csv('data/HomeCredit_columns_description.csv',
                                    verbose=False,
                                    encoding='ISO-8859-1',
                                    )
    info_type = placeholder.radio('Select the type of information you want:', ['Current application', 
                                                                               'Previous application',
                                                                               'Credit Card balance',
                                                                               'Installment payments', 
                                                                               'POS CASH balance'])
    with placeholder_bis.container():
        if info_type == 'Current application': 
            data = read_df('data/application_sample.csv')
            data["TARGET"] = data["TARGET"].astype(str)
            st.write('Select any information about the client:')
            st.markdown('##')
            st.markdown('##')
            selected_features = st.multiselect('', data[data['SK_ID_CURR'] == client_id].dropna(axis=1).select_dtypes('float').columns)
            st.write('Select one second information about the client:')
            st.markdown('##')
            st.markdown('##')
            def selectbox_with_default_2(text, values, default="Client's variable: ", sidebar=False):
                func = st.selectbox if sidebar else st.selectbox
                return func(text, np.insert(np.array(values, object), 0, default))
            selected_features_2 = selectbox_with_default_2('', data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)
            graph_place = st.empty()
            if selected_features and selected_features_2 == "Client's variable: ":
                for features in selected_features:
                    st.write(features, ': ',description.loc[description['Row'] == features, 'Description'].values[0])
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            data_client_value = data.loc[data['SK_ID_CURR'] == client_id, features].values
                            data_client_target = data.loc[data['SK_ID_CURR'] == client_id, 'TARGET'].values

                            # Generate distribution data
                            hist, edges = np.histogram(data.loc[:, features].dropna(), bins=20)
                            hist_source_df = pd.DataFrame({"edges_left": edges[:-1], "edges_right": edges[1:], "hist":hist})
                            max_histogram = hist_source_df["hist"].max()
                            client_line = pd.DataFrame({"x": [data_client_value, data_client_value],
                                                        "y": [0, max_histogram]})
                            hist_source = ColumnDataSource(data=hist_source_df)
                            plot = plot_feature_distrib(features,
                                                        client_line,
                                                        hist_source,
                                                        data_client_value,
                                                        max_histogram)
                            st.bokeh_chart(plot, use_container_width=True)
                        
                        with col2:
                            fig = px.box(data, x='TARGET', y=features, points="outliers", color=data['TARGET'], height=580)
                            fig.update_traces(quartilemethod="inclusive")
                            fig.add_trace(go.Scatter(x=data_client_target,
                                                    y=data_client_value,
                                                    mode='markers',
                                                    marker=dict(size=10),
                                                    showlegend=False,
                                                    name='client'))
                            
                            st.plotly_chart(fig, use_container_width=True) 
                    st.write('---')
            elif selected_features and selected_features_2 != "Client's variable: ": 
                with graph_place.container():
                    for features in selected_features:
                        if selected_features_2 in data.select_dtypes('float').columns.to_list():
                            data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                            data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                            fig = px.scatter(data, x=features, y=selected_features_2, color='TARGET', height=580, opacity=.3)
                            fig.add_trace(go.Scattergl(x=data_client_value_1,
                                                    y=data_client_value_2,
                                                    mode='markers',
                                                    marker=dict(size=10, color = 'red'),
                                                    name='client'))
                            fig.update_layout(legend=dict(
                                                yanchor="top",
                                                y=1,
                                                xanchor="left",
                                                x=1
                                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            st.write('---')
                        else:
                            data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                            data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                            fig = px.box(data, x=selected_features_2, y=features, points="outliers", color=selected_features_2, height=580)
                            fig.update_traces(quartilemethod="inclusive")
                            fig.add_trace(go.Scatter(x=data_client_value_2,
                                                     y=data_client_value_1,
                                                     mode='markers',
                                                     marker=dict(size=10),
                                                     showlegend=False,
                                                     name='client'))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.write('---')

    with placeholder_bis.container():
        if info_type == 'Previous application': 
            data = read_df('data/previous_application_sample.csv')
            if (data['SK_ID_CURR'] == client_id).mean() > 0: 
                st.markdown('##')
                st.write('Select information about the client: ')
                st.markdown('##')
                st.markdown('##')                        
                selected_features = st.multiselect('', np.concatenate((['All'], data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)))
                if selected_features:
                    if 'All' in selected_features:
                        selected_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
                        
                    st.dataframe(data.loc[data['SK_ID_CURR'] == client_id, selected_features])
                    st.write('Feature description: ')
                    
                    for features in selected_features:
                        try:
                            st.write(features, ': ', description.loc[description['Row'] == features, 'Description'].values[0])
                        except:
                            st.write('')
            else:
                st.write('No information on previous application.')

    with placeholder_bis.container():
        if info_type == 'Credit Card balance': 
            data = read_df('data/credit_card_balance_sample.csv')
            if (data['SK_ID_CURR'] == client_id).mean() > 0: 
                st.markdown('##')
                st.write('Select any information about the client: ')
                st.markdown('##')
                st.markdown('##')                        
                selected_features = st.multiselect('', np.concatenate((['All'], data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)))
                if selected_features:
                    if  'All' in selected_features:
                        selected_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
                        
                    st.dataframe(data.loc[data['SK_ID_CURR'] == client_id, selected_features])
                    st.write('Feature description: ')
                    
                    for features in selected_features:
                        try:
                            st.write(features, ': ', description.loc[description['Row'] == features, 'Description'].values[0])
                        except:
                            st.write('')
            else:
                st.write('No information on credit card balance.')
    
    with placeholder_bis.container():
        if info_type == 'Installment payments': 
            data = read_df('data/installments_payments_sample.csv')
            if (data['SK_ID_CURR'] == client_id).mean() > 0: 
                st.markdown('##')
                st.write('Select any information about the client: ')
                st.markdown('##')
                st.markdown('##')                        
                selected_features = st.multiselect('', np.concatenate((['All'], data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)))
                if selected_features:
                    if  'All' in selected_features:
                        selected_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
                        
                    st.dataframe(data.loc[data['SK_ID_CURR'] == client_id, selected_features])
                    st.write('Feature description: ')
                    
                    for features in selected_features:
                        try:
                            st.write(features, ': ', description.loc[description['Row'] == features, 'Description'].values[0])
                        except:
                            st.write('')
            else:
                st.write('No information on installment payments.')

    with placeholder_bis.container():
        if info_type == 'POS CASH balance': 
            data = read_df('data/POS_CASH_balance_sample.csv')
            if (data['SK_ID_CURR'] == client_id).mean() > 0: 
                st.markdown('##')
                st.write('Select any information about the client: ')
                st.markdown('##')
                st.markdown('##')                        
                selected_features = st.multiselect('', np.concatenate((['All'], data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)))
                if selected_features:
                    if  'All' in selected_features:
                        selected_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
                        
                    st.dataframe(data.loc[data['SK_ID_CURR'] == client_id, selected_features])
                    st.write('Feature description: ')
                    
                    for features in selected_features:
                        try:
                            st.write(features, ': ', description.loc[description['Row'] == features, 'Description'].values[0])
                        except:
                            st.write('')
            else:
                st.write('No information on POS CASH balance.')

#----------------------------------------------------------------------------------#
#                                    BOTTOM                                        #
#----------------------------------------------------------------------------------#
st.write("---")

col_about, col_FAQ, col_doc, col_contact = st.columns(4)

with col_about:
    st.write("About us")

with col_FAQ:
    st.write("FAQ")

with col_doc:
    st.write("Technical documentation")

with col_contact:
    st.write("Contact")