import os
import pandas as pd
import streamlit as st
import numpy as np
import warnings

import plotly.express as px
import plotly.graph_objects as go

from utils import *

# Suppress warnings for clean interface
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*TreeExplainer shap values output.*')

@st.cache_data
def _read_df_cached(path):
    return read_df(path)

# Feature name mapping for user-friendly display
@st.cache_data
def get_friendly_feature_names():
    """Map technical feature names to user-friendly labels"""
    return {
        # Personal Information
        'CNT_CHILDREN': 'Number of Children',
        'CNT_FAM_MEMBERS': 'Family Members',
        'DAYS_BIRTH': 'Age (days)',
        'DAYS_EMPLOYED': 'Employment Duration (days)',
        'DAYS_REGISTRATION': 'Registration Duration (days)',
        'DAYS_ID_PUBLISH': 'ID Publication Date (days)',
        'DAYS_LAST_PHONE_CHANGE': 'Phone Change Date (days)',
        'OWN_CAR_AGE': 'Car Age (years)',
        
        # Financial Information
        'AMT_INCOME_TOTAL': 'Total Income',
        'AMT_CREDIT': 'Credit Amount',
        'AMT_ANNUITY': 'Loan Annuity',
        'AMT_GOODS_PRICE': 'Goods Price',
        'REGION_POPULATION_RELATIVE': 'Regional Population (relative)',
        
        # External Scores
        'EXT_SOURCE_1': 'External Score 1',
        'EXT_SOURCE_2': 'External Score 2',
        'EXT_SOURCE_3': 'External Score 3',
        
        # Building Information
        'APARTMENTS_AVG': 'Apartments (avg)',
        'BASEMENTAREA_AVG': 'Basement Area (avg)',
        'YEARS_BEGINEXPLUATATION_AVG': 'Building Age (avg)',
        'YEARS_BUILD_AVG': 'Construction Year (avg)',
        'COMMONAREA_AVG': 'Common Area (avg)',
        'ELEVATORS_AVG': 'Elevators (avg)',
        'ENTRANCES_AVG': 'Entrances (avg)',
        'FLOORSMAX_AVG': 'Max Floors (avg)',
        'FLOORSMIN_AVG': 'Min Floors (avg)',
        'LANDAREA_AVG': 'Land Area (avg)',
        'LIVINGAPARTMENTS_AVG': 'Living Apartments (avg)',
        'LIVINGAREA_AVG': 'Living Area (avg)',
        'NONLIVINGAPARTMENTS_AVG': 'Non-living Apartments (avg)',
        'NONLIVINGAREA_AVG': 'Non-living Area (avg)',
        'TOTALAREA_MODE': 'Total Area (mode)',
        
        # Building Information (Mode)
        'APARTMENTS_MODE': 'Apartments (mode)',
        'BASEMENTAREA_MODE': 'Basement Area (mode)',
        'YEARS_BEGINEXPLUATATION_MODE': 'Building Age (mode)',
        'YEARS_BUILD_MODE': 'Construction Year (mode)',
        'COMMONAREA_MODE': 'Common Area (mode)',
        'ELEVATORS_MODE': 'Elevators (mode)',
        'ENTRANCES_MODE': 'Entrances (mode)',
        'FLOORSMAX_MODE': 'Max Floors (mode)',
        'FLOORSMIN_MODE': 'Min Floors (mode)',
        'LANDAREA_MODE': 'Land Area (mode)',
        'LIVINGAPARTMENTS_MODE': 'Living Apartments (mode)',
        'LIVINGAREA_MODE': 'Living Area (mode)',
        'NONLIVINGAPARTMENTS_MODE': 'Non-living Apartments (mode)',
        'NONLIVINGAREA_MODE': 'Non-living Area (mode)',
        
        # Building Information (Median)
        'APARTMENTS_MEDI': 'Apartments (median)',
        'BASEMENTAREA_MEDI': 'Basement Area (median)',
        'YEARS_BEGINEXPLUATATION_MEDI': 'Building Age (median)',
        'YEARS_BUILD_MEDI': 'Construction Year (median)',
        'COMMONAREA_MEDI': 'Common Area (median)',
        'ELEVATORS_MEDI': 'Elevators (median)',
        'ENTRANCES_MEDI': 'Entrances (median)',
        'FLOORSMAX_MEDI': 'Max Floors (median)',
        'FLOORSMIN_MEDI': 'Min Floors (median)',
        'LANDAREA_MEDI': 'Land Area (median)',
        'LIVINGAPARTMENTS_MEDI': 'Living Apartments (median)',
        'LIVINGAREA_MEDI': 'Living Area (median)',
        'NONLIVINGAPARTMENTS_MEDI': 'Non-living Apartments (median)',
        'NONLIVINGAREA_MEDI': 'Non-living Area (median)',
        
        # Social Circle
        'OBS_30_CNT_SOCIAL_CIRCLE': 'Social Circle Observations (30 days)',
        'DEF_30_CNT_SOCIAL_CIRCLE': 'Social Circle Defaults (30 days)',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'Social Circle Observations (60 days)',
        'DEF_60_CNT_SOCIAL_CIRCLE': 'Social Circle Defaults (60 days)',
        
        # Credit Bureau
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'Credit Bureau Requests (last hour)',
        'AMT_REQ_CREDIT_BUREAU_DAY': 'Credit Bureau Requests (last day)',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'Credit Bureau Requests (last week)',
        'AMT_REQ_CREDIT_BUREAU_MON': 'Credit Bureau Requests (last month)',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'Credit Bureau Requests (last quarter)',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'Credit Bureau Requests (last year)',
        
        # Regional Ratings
        'REGION_RATING_CLIENT': 'Regional Client Rating',
        'REGION_RATING_CLIENT_W_CITY': 'Regional City Rating',
        
        # Flags
        'REG_REGION_NOT_LIVE_REGION': 'Registration Region ≠ Living Region',
        'REG_REGION_NOT_WORK_REGION': 'Registration Region ≠ Work Region',
        'LIVE_REGION_NOT_WORK_REGION': 'Living Region ≠ Work Region',
        'REG_CITY_NOT_LIVE_CITY': 'Registration City ≠ Living City',
        'REG_CITY_NOT_WORK_CITY': 'Registration City ≠ Work City',
        'LIVE_CITY_NOT_WORK_CITY': 'Living City ≠ Work City',
        
        # Documents
        'FLAG_DOCUMENT_2': 'Document 2 Provided',
        'FLAG_DOCUMENT_3': 'Document 3 Provided',
        'FLAG_DOCUMENT_4': 'Document 4 Provided',
        'FLAG_DOCUMENT_5': 'Document 5 Provided',
        'FLAG_DOCUMENT_6': 'Document 6 Provided',
        'FLAG_DOCUMENT_7': 'Document 7 Provided',
        'FLAG_DOCUMENT_8': 'Document 8 Provided',
        'FLAG_DOCUMENT_9': 'Document 9 Provided',
        'FLAG_DOCUMENT_10': 'Document 10 Provided',
        'FLAG_DOCUMENT_11': 'Document 11 Provided',
        'FLAG_DOCUMENT_12': 'Document 12 Provided',
        'FLAG_DOCUMENT_13': 'Document 13 Provided',
        'FLAG_DOCUMENT_14': 'Document 14 Provided',
        'FLAG_DOCUMENT_15': 'Document 15 Provided',
        'FLAG_DOCUMENT_16': 'Document 16 Provided',
        'FLAG_DOCUMENT_17': 'Document 17 Provided',
        'FLAG_DOCUMENT_18': 'Document 18 Provided',
        'FLAG_DOCUMENT_19': 'Document 19 Provided',
        'FLAG_DOCUMENT_20': 'Document 20 Provided',
        'FLAG_DOCUMENT_21': 'Document 21 Provided',
        
        # Contact Flags
        'FLAG_MOBIL': 'Mobile Phone Provided',
        'FLAG_EMP_PHONE': 'Work Phone Provided',
        'FLAG_WORK_PHONE': 'Work Phone Available',
        'FLAG_CONT_MOBILE': 'Mobile Contact Available',
        'FLAG_PHONE': 'Home Phone Provided',
        'FLAG_EMAIL': 'Email Provided',
        
        # Application Details
        'HOUR_APPR_PROCESS_START': 'Application Hour',
        'WEEKDAY_APPR_PROCESS_START': 'Application Weekday',
    }

def format_feature_name(feature_name):
    """Convert technical feature name to user-friendly label"""
    friendly_names = get_friendly_feature_names()
    return friendly_names.get(feature_name, feature_name)

# Page configuration with custom theme
st.set_page_config(
    page_title='💳 Credit Risk Dashboard',
    layout="wide",
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Fix sidebar metrics visibility - keep white background but make text dark */
    section[data-testid="stSidebar"] .stMetric {
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stMetric label {
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricDelta"] {
        color: #666666 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Credit Risk Assessment Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Credit scoring and risk analysis platform</p>', unsafe_allow_html=True)

placeholder = st.empty()
placeholder_bis = st.empty()
return_button = st.empty()
#----------------------------------------------------------------------------------#
#                                 LOADING DATA                                     #
#----------------------------------------------------------------------------------#

df = _read_df_cached('data/dataset_sample.csv')

df = df.replace([np.inf, -np.inf], np.nan)

# Load ML models
with st.spinner('🤖 Loading models...'):
    import joblib
    pipeline = joblib.load('ressource/pipeline.joblib')
    preprocessor = pipeline[:-1]  # All steps except classifier
    clf = pipeline.named_steps['classifier']  # Extract classifier

#----------------------------------------------------------------------------------#
#                              SIDEBAR                                             #
#----------------------------------------------------------------------------------#
st.sidebar.markdown("## Client Selection")
all_clients_id = df['SK_ID_CURR'].unique()

# Client selector with search
client_id = st.sidebar.selectbox(
    "",
    options=[''] + list(all_clients_id),
    format_func=lambda x: "Choose a client..." if x == '' else f"Client #{int(x)}"
)

if client_id == '':
    with placeholder.container():
        # Welcome message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
                ### Welcome to the Credit Risk Dashboard
                
                This platform helps you evaluate credit applications using advanced machine learning.
                
                **Features:**
                - Real-time credit risk assessment
                - Interactive data visualization
                - Models predictions
                - Detailed client analytics
                
                **Get started:** Select a client from the sidebar to begin the analysis.
            """)
            
    st.stop()

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
    
    # Display client info in beautiful sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Personal Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("👤 Gender", gender)
        st.metric("🎂 Age", f"{age} yrs")
    with col2:
        st.metric("👨‍👩‍👧‍👦 Family", int(round(fam_members)))
        st.metric("👶 Children", int(round(childs)))
    
    with st.sidebar.expander("📚 More Details", expanded=False):
        st.write(f"**Education:** {education}")
        st.write(f"**Marital Status:** {family_status}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 Professional Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("💵 Income", f"${round(income_per_person):,}")
    with col2:
        st.metric("📅 Experience", years_work if isinstance(years_work, str) else f"{years_work} yrs")
    
    with st.sidebar.expander("💼 Work Details", expanded=False):
        st.write(f"**Occupation:** {work}")
        st.write(f"**Type:** {occupation_type if occupation_type else 'N/A'}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💳 Credit Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("💰 Credit", f"${round(credit):,}")
    with col2:
        st.metric("📊 Annuity", f"${round(annuity):,}")
    
    st.sidebar.metric("📈 Payment Rate", f"{payment_rate:.1%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    page = st.sidebar.radio(
        "Select Analysis Type",
        ['Check credit score', 'Client more informations'],
        label_visibility="collapsed"
    )
    
    if page == 'Check credit score':
        run_button = st.sidebar.button('Run Credit Analysis', type="primary", use_container_width=True)
    else:
        run_button = False


if page == 'Check credit score':
    if run_button:
        #----------------------------------------------------------------------------------#
        #                                 PREPROCESSING                                    #
        #----------------------------------------------------------------------------------#

        X = data_client.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = data_client['TARGET']

        # Do not transform X here — the helper will call the API first.
        # If local fallback is used, helper will call preprocessor.transform.

        #----------------------------------------------------------------------------------#
        #                           PREDICT, WITH API                                      #
        #----------------------------------------------------------------------------------#
        with st.status("Model Analysis in Progress...", expanded=True) as status:
            st.write("Connecting to API...")
            # API is the default source of truth; make it configurable via env var
            url_api = os.getenv('CREDIT_SCORE_API_URL', 'https://credit-score-api-572900860091.europe-west1.run.app')
            st.write("Processing credit data...")
            # Use helper which tries API first, then falls back to local classifier
            prob = predict_with_api_or_local(client_id,
                                            X,
                                            api_url=url_api,
                                            classifier=clf,
                                            preprocessor=preprocessor)
            st.write("Analysis complete!")
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        #----------------------------------------------------------------------------------#
        #                           RESULTS DISPLAY                                        #
        #----------------------------------------------------------------------------------#
        with placeholder.container():
            # Header for results
            st.markdown("## 📊 Credit Risk Analysis Results")
            st.markdown("---")
            
            risk_score = prob * 100
            
            # Determine risk level and styling
            if risk_score < 30:
                risk_level = "Low Risk"
                risk_icon = "✅"
                risk_color = "#28a745"
                decision = "APPROVED"
                decision_icon = "✅"
                recommendation = "We recommend ACCEPTING this client's loan application."
            elif risk_score <= 50:
                risk_level = "Medium Risk"
                risk_icon = "⚠️"
                risk_color = "#ffc107"
                decision = "REVIEW NEEDED"
                decision_icon = "⚠️"
                recommendation = "Client's default risk is moderate. We recommend reviewing additional factors before making a decision."
            else:
                risk_level = "High Risk"
                risk_icon = "❌"
                risk_color = "#dc3545"
                decision = "REJECTED"
                decision_icon = "❌"
                recommendation = "We recommend  REJECTING this client's loan application due to high default risk."
            
            # Main results in columns with fixed layout
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                # Container for gauge with fixed height
                with st.container():
                    # Display gauge directly - Plotly handles smooth animation internally
                    fig_gauge = plot_gauge(risk_score)
                    st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Centered container with proper vertical alignment
                st.markdown(f"""
                    <div style='display: flex; align-items: center; height: 400px;'>
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); width: 100%; display: flex; 
                             flex-direction: column; justify-content: center;'>
                            <h1 style='color: white; margin: 0; font-size: 3.5rem;'>{risk_icon}</h1>
                            <h2 style='color: white; margin: 0.5rem 0;'>{risk_score:.1f}%</h2>
                            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1rem;'>{risk_level}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Centered container with proper vertical alignment
                st.markdown(f"""
                    <div style='display: flex; align-items: center; height: 400px;'>
                        <div style='padding: 2rem; border-left: 5px solid {risk_color}; 
                             background-color: rgba(128,128,128,0.05); border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                             width: 100%; display: flex; flex-direction: column; justify-content: center;'>
                            <h2 style='color: {risk_color}; margin: 0 0 1rem 0;'>{decision_icon} {decision}</h2>
                            <p style='margin: 0; font-size: 1.1rem;'>{recommendation}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Risk thresholds explanation
            st.info("""
                **Decision Guidelines:**
                - **< 30%**: Low risk → Loan approval recommended
                - **30-50%**: Medium risk → Additional review suggested
                - **> 50%**: High risk → Loan rejection recommended
                
                💡 *Use the "Client more information" tab for detailed analysis to support your decision.*
            """)
            
            # SHAP Analysis Section
            st.markdown("## Model Explanation")
            st.markdown("Understanding which factors influenced the prediction")
            
            with st.spinner('Analyzing feature importance...'):
                    
                feats = read_pickle('ressource/feats')
                mapping = {f"Column_{i}": name for i, name in enumerate(df.columns)}
                SHAP_explainer = read_pickle('ressource/shap_explainer')

                # SHAP explainer expects preprocessed input; transform X for explanation only
                X_trans = pipeline[:-1].transform(X)
                
                # Get SHAP values
                X_sample = np.array(X_trans)[0:1]
                shap_vals = SHAP_explainer.shap_values(X_sample)
                
                # TreeExplainer returns a list for binary classification [class0, class1]
                if isinstance(shap_vals, list):
                    shap_vals_class1 = shap_vals[1][0]  # Class 1, first sample
                else:
                    shap_vals_class1 = shap_vals[0]  # First sample

                shap_explained, most_important_features = format_shap_values(shap_vals_class1, feats)
                
                # Replace technical names with friendly names in SHAP results
                shap_explained["features"] = shap_explained["features"].map(mapping).fillna(shap_explained["features"])

                most_important_features = [format_feature_name(f) for f in most_important_features]
                
                explained_chart = plot_important_features(shap_explained, most_important_features)

            # SHAP Visualization
            st.markdown("### 📊 Feature Importance Analysis")
            st.plotly_chart(explained_chart, use_container_width=True)
            st.caption('🔴 Red bars indicate features increasing default risk | 🟢 Green bars indicate features decreasing default risk')

if page == 'Client more informations':
    description = pd.read_csv('data/HomeCredit_columns_description.csv',
                                    encoding='ISO-8859-1',
                                    )
    
    # Display client application data analysis
    st.markdown("### 📊 Client Application Data Analysis")
    st.info("💡 Explore and compare the client's application features with the overall population")
    
    st.divider()
    
    data = read_df('data/application_sample.csv')
    data["TARGET"] = data["TARGET"].astype(str)

    # Features selection interface
    st.markdown("### 📊 Features analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary features** (select multiple)")
        # Get available features and create friendly mapping
        available_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).select_dtypes('float').columns
        feature_options = {format_feature_name(f): f for f in available_features}
        
        selected_friendly = st.multiselect(
            'Choose features to analyze:',
            options=list(feature_options.keys()),
            help="Select one or more numeric features to visualize"
        )
        # Convert back to technical names for processing
        selected_features = [feature_options[f] for f in selected_friendly]
    
    with col2:
        st.markdown("**Secondary feature** (optional)")
        all_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
        all_feature_options = {format_feature_name(f): f for f in all_features}
        
        selected_friendly_2 = st.selectbox(
            'Compare with another feature:',
            ['Choose a variable...'] + list(all_feature_options.keys()),
            help="Optional: Select a second feature for comparison"
        )
        # Convert back to technical name
        if selected_friendly_2 != 'Choose a variable...':
            selected_features_2 = all_feature_options[selected_friendly_2]
        else:
            selected_features_2 = 'Choose a variable...'
    st.divider()
    
    graph_place = st.empty()
    if selected_features and selected_features_2 == "Choose a variable...":
        for features in selected_features:
            # Get friendly name for display
            friendly_name = format_feature_name(features)
            
            
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
                    # Pass dict instead of ColumnDataSource for Plotly
                    hist_source = hist_source_df.to_dict('list')
                    plot = plot_feature_distrib(friendly_name,
                                                client_line,
                                                hist_source,
                                                data_client_value,
                                                max_histogram)
                    st.plotly_chart(plot, use_container_width=True)
                
                with col2:
                    # Create box plot
                    fig = px.box(data, x='TARGET', y=features, points="outliers", color='TARGET', height=580)
                    fig.update_traces(quartilemethod="inclusive")
                    fig.add_trace(go.Scatter(x=data_client_target,
                                            y=data_client_value,
                                            mode='markers',
                                            marker=dict(size=10),
                                            showlegend=False,
                                            name='client'))
                    # Update axis labels with friendly names
                    fig.update_layout(
                        yaxis_title=friendly_name,
                        xaxis_title='Default status (0=No, 1=Yes)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True) 

                    # Feature description card
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                     padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                    <h4 style='color: white; margin: 0;'>{friendly_name}</h4>
                    <p style='color: white; margin: 5px 0 0 0; opacity: 0.9; font-size: 0.85em;'>
                        Technical name: {features}<br>
                        {description.loc[description['Row'] == features, 'Description'].values[0] if len(description.loc[description['Row'] == features, 'Description'].values) > 0 else ''}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.divider()
    elif selected_features and selected_features_2 != "Choose a variable...": 
        with graph_place.container():
            for features in selected_features:
                # Get friendly names
                friendly_name_1 = format_feature_name(features)
                friendly_name_2 = format_feature_name(selected_features_2)
                
                if selected_features_2 in data.select_dtypes('float').columns.to_list():
                    data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                    data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                    
                    # Create scatter plot with technical names, then update labels
                    fig = px.scatter(data, x=features, y=selected_features_2, color='TARGET', height=580, opacity=.3)
                    fig.add_trace(go.Scattergl(x=data_client_value_1,
                                            y=data_client_value_2,
                                            mode='markers',
                                            marker=dict(size=10, color = 'red'),
                                            name='client'))
                    fig.update_layout(
                        legend=dict(yanchor="top", y=1, xanchor="left", x=1),
                        xaxis_title=friendly_name_1,
                        yaxis_title=friendly_name_2
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                else:
                    data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                    data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                    
                    # Create box plot with technical names, then update labels
                    fig = px.box(data, x=selected_features_2, y=features, points="outliers", color=selected_features_2, height=580)
                    fig.update_traces(quartilemethod="inclusive")
                    fig.add_trace(go.Scatter(x=data_client_value_2,
                                             y=data_client_value_1,
                                             mode='markers',
                                             marker=dict(size=10),
                                             showlegend=False,
                                             name='client'))
                    fig.update_layout(
                        xaxis_title=friendly_name_2,
                        yaxis_title=friendly_name_1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()

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