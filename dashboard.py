import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go
import dill

from dashboard_functions import *
st.set_page_config(page_title= 'Credit Score App', layout="wide", initial_sidebar_state='expanded')
                
st.title("Evaluate your client's credit capacity.")

placeholder = st.empty()
return_button = st.empty()
with placeholder.container():
#----------------------------------------------------------------------------------#
#                                 LOADING DATA                                     #
#----------------------------------------------------------------------------------#

    df = pd.read_csv('data/dataset_sample.csv',
                    engine='pyarrow',
                    verbose=False,
                    encoding='ISO-8859-1',
                    )
    df = df.replace([np.inf, -np.inf], np.nan)

    with open('ressource/pipeline',"rb") as f:
        preprocessor = pickle.load(f)

    with open('ressource/classifier',"rb") as f:
        clf = pickle.load(f)

#----------------------------------------------------------------------------------#
#                                   SIDEBAR                                        #
#----------------------------------------------------------------------------------#
scoring = None
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
    placeholder.warning("Please select a client !")

else:
    data_client= df[df["SK_ID_CURR"]==client_id]
    client_index = data_client.index[0]

    with placeholder.container():
        st.write(f"You've selected client #{client_id}.")
        show_all = st.checkbox(f"Show client's all data.")
        if show_all:
                st.write(data_client.drop(['SK_ID_CURR', 'TARGET'], axis=1).to_html(index=False), unsafe_allow_html=True)
            

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
    credit_button = st.sidebar.empty()

    scoring = credit_button.button('Check credit score')

    if scoring:
        placeholder.empty()
        credit_button.empty()
#----------------------------------------------------------------------------------#
#                                 PREPROCESSING                                    #
#----------------------------------------------------------------------------------#

        X = data_client.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = data_client['TARGET']

        X = preprocessor.transform(X)

#----------------------------------------------------------------------------------#
#                           PREDICT, EXPLAIN FEATURES                              #
#----------------------------------------------------------------------------------#
        with st.spinner(f"Model working..."):
            prob = clf.predict_proba(X)[:, 1]
#----------------------------------------------------------------------------------#
#                               DISPLAY RESULTS                                    #
#----------------------------------------------------------------------------------#
        with placeholder.container():
            left_column_recom, right_column_recom = st.columns(2)

            with left_column_recom:
                gauge_place = st.empty()
                for value in np.arange(0, round(prob[0]*100, 1) +.1, step=.1):
                    fig_gauge = plot_gauge(value)
                    gauge_place.plotly_chart(fig_gauge)

            with right_column_recom: 
                # Display clients data and prediction
                st.write(f"Client #{client_id} has **{prob[0]*100:.2f} % of risk** to make default.")

                if prob[0]*100 < 30:
                    st.write(f"We recommand to **accept** client's application to loan.")
                elif (prob[0]*100 >= 30) & (prob[0]*100 <= 50):
                    st.write(f"Client's chances to make default are between 30 and 50% . We recommand \
                                to **analyse closely** the data to make your decision.")
                else:
                    st.write(f"We recommand to **reject** client's application to loan.")
                
                st.caption(f"Below 30% of default risk, we recommand to accept client application.\
                            Above 50% of default risk, we recommand to reject client application. \
                            Between 30 and 50%, your expertise will be your best advice in your decision making.")

            
                # Display explainer HTML object
           # if st.checkbox("Explain Results"):
                
            with st.spinner('Calculating...'):
                
                with open('ressource/feats', 'rb') as f:
                    feats = dill.load(f)
                with open('ressource/lime_explainer', 'rb') as f:
                    LIME_explainer = dill.load(f)

                exp = LIME_explainer.explain_instance(X[0],
                                                        clf.predict_proba,
                                                        num_features=10,
                                                        top_labels=1)
            html_lime = exp.as_html()

            st.subheader('Lime Explanation')
            components.html(html_lime, width=1100, height=350, scrolling=True)

        if return_button.button('Return'):
            scoring = credit_button.button('Check credit score')
            






    '''st.write('More info')
    data_type = st.radio('Select the type of information you want:', ['application'])

    if data_type == 'application': 
        data = pd.read_csv('data/application_sample.csv',
                        engine='pyarrow',
                        verbose=False,
                        encoding='ISO-8859-1',
                        )
        st.markdown('##')
        st.markdown('##')
        selected_data = st.multiselect("", data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)
        import plotly.figure_factory as ff   
        if selected_data:
            for col in selected_data:
                st.write(data.loc[data['SK_ID_CURR'] == client_id, col].values)
                fig = ff.create_distplot([data[col].dropna().to_list()], group_labels=[col])
                fig.add_vline(x=float(data.loc[data['SK_ID_CURR'] == client_id, col].values), line_dash = 'dash', line_color = 'firebrick')
                st.plotly_chart(fig)'''
                    
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