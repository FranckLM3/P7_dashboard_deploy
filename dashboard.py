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

def custom_metric(label, value):
    """Affiche une métrique personnalisée avec encadré blanc et texte gris"""
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.2rem;">{label}</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #555;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def custom_plotly_chart(fig, title=None, use_container_width=True):
    """Affiche un graphique Plotly directement sans encadré"""
    st.plotly_chart(fig, use_container_width=use_container_width)

@st.cache_data
def _read_df_cached(path):
    return read_df(path)

# Feature name mapping for user-friendly display
@st.cache_data
def get_friendly_feature_names():
    """Map technical feature names to user-friendly labels"""
    return {
        # Informations personnelles
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

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Check for mobile device and block access if needed
mobile_block_html = """
<div id="mobile-detector" style="display: none;">
    <div id="mobile-warning" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        font-family: Arial, sans-serif;
        z-index: 99999;
    ">
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 3rem;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 400px;
        ">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">📱</h1>
            <h2 style="margin-bottom: 1rem; font-size: 1.5rem;">Dashboard Non Compatible Mobile</h2>
            <p style="margin-bottom: 2rem; font-size: 1rem; line-height: 1.6;">
                Ce dashboard de scoring crédit est optimisé pour les ordinateurs de bureau et tablettes.
                Pour une expérience optimale, veuillez accéder au site depuis :
            </p>
            <ul style="text-align: left; margin-bottom: 2rem; font-size: 0.9rem; list-style-type: none; padding: 0;">
                <li style="margin-bottom: 0.5rem;">💻 Un ordinateur de bureau</li>
                <li style="margin-bottom: 0.5rem;">💻 Un ordinateur portable</li>
                <li style="margin-bottom: 0.5rem;">📱 Une tablette en mode paysage</li>
            </ul>
            <p style="font-size: 0.8rem; opacity: 0.8;">
                Écran minimum requis : 768px de largeur
            </p>
        </div>
    </div>
</div>

<script>
function checkAndShowMobileWarning() {
    const isMobile = window.innerWidth <= 768 || 
                    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    const mobileWarning = document.getElementById('mobile-warning');
    const stApp = document.querySelector('[data-testid="stApp"]');
    const mainContent = document.querySelector('.main');
    
    if (isMobile) {
        console.log('Mobile détecté, affichage de l\'avertissement');
        
        // Hide Streamlit content
        if (stApp) stApp.style.display = 'none';
        if (mainContent) mainContent.style.display = 'none';
        
        // Show mobile warning
        if (mobileWarning) {
            mobileWarning.style.display = 'flex';
        }
    } else {
        console.log('Desktop détecté');
        
        // Show Streamlit content
        if (stApp) stApp.style.display = 'block';
        if (mainContent) mainContent.style.display = 'block';
        
        // Hide mobile warning
        if (mobileWarning) {
            mobileWarning.style.display = 'none';
        }
    }
}

// Test immédiat
console.log('Test détection mobile - largeur:', window.innerWidth);
checkAndShowMobileWarning();

// Test sur resize
window.addEventListener('resize', function() {
    console.log('Resize détecté - nouvelle largeur:', window.innerWidth);
    checkAndShowMobileWarning();
});

// Tests répétés pour s'assurer que ça marche
setTimeout(checkAndShowMobileWarning, 100);
setTimeout(checkAndShowMobileWarning, 500);
setTimeout(checkAndShowMobileWarning, 1000);
setTimeout(checkAndShowMobileWarning, 2000);
</script>
"""

st.markdown(mobile_block_html, unsafe_allow_html=True)

# Custom CSS - minimal styling
st.markdown("""
    <style>
    /* Clean default styling */
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Style cohérent pour les graphiques */
    .stPlotlyChart {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        padding: 0 !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
        overflow: hidden !important;
    }
    
    /* Style pour les containers de graphiques */
    .stPlotlyChart > div {
        background: white !important;
        border-radius: 8px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Contenir les graphiques dans leurs encadrés */
    .stPlotlyChart .plotly {
        width: 100% !important;
        height: auto !important;
    }
    
    /* Style pour les onglets plus grands */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px !important;
        padding: 16px 32px !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px 12px 0 0 !important;
        border: 2px solid #e9ecef !important;
        border-bottom: none !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(108, 117, 125, 0.15) !important;
        border-color: #6c757d !important;
        transform: translateY(-2px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #495057 !important;
        color: white !important;
        border-color: #495057 !important;
        box-shadow: 0 4px 12px rgba(73, 80, 87, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px !important;
    }
    
    /* Desktop optimized (mobile is blocked) */
    @media (min-width: 769px) {
        .main-header { font-size: 3rem; }
    }
    
    /* Ensure desktop experience */
    @media (max-width: 768px) {
        body { display: none !important; }
    }
    </style>
""", unsafe_allow_html=True)

placeholder = st.empty()
placeholder_bis = st.empty()
return_button = st.empty()

df = _read_df_cached('data/dataset_sample.csv')
df = df.replace([np.inf, -np.inf], np.nan)

# Load ML models
with st.spinner('⚙️ Chargement des modèles...'):
    import joblib
    pipeline = joblib.load('ressource/pipeline.joblib')
    preprocessor = pipeline[:-1]  # All steps except classifier
    clf = pipeline.named_steps['classifier']  # Extract classifier

st.sidebar.markdown("*Choisissez un client pour commencer l'analyse*")

all_clients_id = df['SK_ID_CURR'].unique()

# Initialize session state for client selection
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = ''

client_id = st.sidebar.selectbox(
    "Sélectionnez l'identifiant d'un client",
    options=[''] + list(all_clients_id),
    format_func=lambda x: "🔍 Choisissez un client..." if x == '' else f"Client #{int(x)}",
    label_visibility="collapsed",
    help="Sélectionnez un client dans la liste pour voir son profil de risque",
    index=0 if st.session_state.selected_client == '' else list(all_clients_id).index(st.session_state.selected_client) + 1 if st.session_state.selected_client in all_clients_id else 0
)

# Update session state when user manually selects a client
if client_id != st.session_state.selected_client:
    st.session_state.selected_client = client_id

# Analysis section
if client_id != '':
    
    run_button = False
else:
    page = 'Évaluation Risque Crédit'
    run_button = False

if client_id == '':
    page = 'Évaluation Risque Crédit'
    run_button = False

if client_id == '':
    with placeholder.container():
        # Header principal uniquement sur la page d'accueil
        st.markdown('<h1 class="main-header">Dashboard Scoring Crédit</h1>', unsafe_allow_html=True)
        
        # Enhanced welcome page with clear instructions
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
            <h2 style="text-align: center; margin-bottom: 1rem;">Dashboard de Scoring Crédit - "Prêt à dépenser"</h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Outil d'aide à la décision pour les chargés de relation client
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear instructions
        st.markdown("### Comment utiliser cet outil ?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                text-align: center;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.8rem;">📋</div>
                <h4 style="color: #333; margin-bottom: 0.8rem;">Étape 1 : Choisir un client</h4>
                <ul style="color: #666; font-size: 0.9rem; text-align: left; list-style: none; padding: 0;">
                    <li>• Sélectionnez un client dans la liste déroulante</li>
                    <li>• Plus de 300 000 clients disponibles</li>
                    <li>• Informations affichées automatiquement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                text-align: center;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.8rem;">👤</div>
                <h4 style="color: #333; margin-bottom: 0.8rem;">Étape 2 : Consulter le profil</h4>
                <ul style="color: #666; font-size: 0.9rem; text-align: left; list-style: none; padding: 0;">
                    <li>• Informations personnelles et professionnelles</li>
                    <li>• Données affichées dans la barre latérale</li>
                    <li>• Navigation automatique vers l'analyse</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                text-align: center;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.8rem;">📊</div>
                <h4 style="color: #333; margin-bottom: 0.8rem;">Étape 3 : Explorer les détails</h4>
                <ul style="color: #666; font-size: 0.9rem; text-align: left; list-style: none; padding: 0;">
                    <li>• Score de risque avec explication SHAP</li>
                    <li>• Onglet "Analyse Détaillée" pour comparaisons</li>
                    <li>• Visualisations interactives</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # Process explanation
        st.markdown("---")
        st.markdown("### Processus détaillé d'utilisation")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #ddd;
                border-left: 4px solid #667eea;
                border-radius: 12px;
                padding: 2rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.8rem; margin-right: 0.8rem;">📈</div>
                    <h4 style="color: #333; margin: 0;">Analyse du risque crédit</h4>
                </div>
                <ol style="color: #666; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                    <li><strong style="color: #333;">Score de probabilité :</strong> Calcul du risque de défaut (0-100%)</li>
                    <li><strong style="color: #333;">Recommandation :</strong> Faible/Modéré/Élevé avec seuils clairs</li>
                    <li><strong style="color: #333;">Explication SHAP :</strong> Facteurs qui influencent le score</li>
                    <li><strong style="color: #333;">Aide à la décision :</strong> Comprendre le "pourquoi" du score</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #ddd;
                border-left: 4px solid #28a745;
                border-radius: 12px;
                padding: 2rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.8rem; margin-right: 0.8rem;">🔍</div>
                    <h4 style="color: #333; margin: 0;">Exploration comparative</h4>
                </div>
                <ol style="color: #666; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                    <li><strong style="color: #333;">Sélection variables :</strong> Choisir les caractéristiques à analyser</li>
                    <li><strong style="color: #333;">Distribution :</strong> Position du client vs population totale</li>
                    <li><strong style="color: #333;">Comparaison :</strong> Profil par rapport aux défaillants/non-défaillants</li>
                    <li><strong style="color: #333;">Compréhension :</strong> Contextualiser le score obtenu</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

else:
    data_client= df[df["SK_ID_CURR"]==client_id]
    client_index = data_client.index[0]
    
    # Load description data for feature explanations (always available)
    description = pd.read_csv('data/HomeCredit_columns_description.csv',
                                encoding='ISO-8859-1',
                                )

    gender = data_client.loc[client_index, "CODE_GENDER"]
    if gender == 1:
        gender = 'Homme'
    else:
        gender = 'Femme'

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
            years_work = "Moins d'un an"
        elif years_work == 1:
            years_work = str(years_work) + ' an'
        else: 
            years_work = str(years_work) + ' ans'
    except:
        years_work = 'no information'
    
    # Display client info in sidebar
    st.sidebar.markdown("### 👤 Informations personnelles")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_metric("👤 Sexe", gender)
        custom_metric("🎂 Âge", f"{age} ans")
    with col2:
        custom_metric("👨‍👩‍👧‍👦 Famille", int(round(fam_members)))
        custom_metric("👶 Enfants", int(round(childs)))
    
    with st.sidebar.expander("📚 Plus de Détails", expanded=False):
        st.write(f"**Éducation :** {education}")
        st.write(f"**Statut Marital :** {family_status}")
    
    st.sidebar.markdown("### 💼 Informations Professionnelles")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_metric("💵 Revenus", f"${round(income_per_person):,}")
    with col2:
        custom_metric("📅 Expérience", years_work if isinstance(years_work, str) else f"{years_work} ans")
    
    with st.sidebar.expander("💼 Détails Travail", expanded=False):
        st.write(f"**Profession :** {work}")
        st.write(f"**Type :** {occupation_type if occupation_type else 'N/A'}")
    
    st.sidebar.markdown("### 💰 Informations Crédit")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_metric("💰 Crédit", f"${round(credit):,}")
    with col2:
        custom_metric("📊 Annuité", f"${round(annuity):,}")
    
    with st.sidebar:
        custom_metric("📈 Taux de Paiement", f"{payment_rate:.1%}")
    
    # Système d'onglets pour l'analyse
    tab1, tab2 = st.tabs(["🎯 Risque de crédit", "📊 Analyse détaillée"])
    
    with tab1:
        # Analyse automatique
        
        # Analyse automatique sans bouton
        
        X = data_client.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = data_client['TARGET']

        # Do not transform X here — the helper will call the API first.
        # If local fallback is used, helper will call preprocessor.transform.

        url_api = os.getenv('CREDIT_SCORE_API_URL', 'https://credit-score-api-572900860091.europe-west1.run.app')
        prob = predict_with_api_or_local(client_id,
                                        X,
                                        api_url=url_api,
                                        classifier=clf,
                                        preprocessor=preprocessor)
        
        #----------------------------------------------------------------------------------#
        #                           RESULTS DISPLAY                                        #
        #----------------------------------------------------------------------------------#
        with placeholder.container():
            pass
        
        # Calculate risk score
        risk_score = prob * 100
        
        # Determine risk level and styling
        if risk_score < 30:
            risk_level = "Risque Faible"
            risk_icon = "✅"
            risk_color = "#28a745"
            decision = "RISQUE FAIBLE"
            decision_icon = "✅"
            recommendation = "Le modèle évalue ce profil comme présentant un risque faible de défaut de paiement."
        elif risk_score <= 50:
            risk_level = "Risque Modéré"
            risk_icon = "⚠️"
            risk_color = "#ffc107"
            decision = "RISQUE MODÉRÉ"
            decision_icon = "⚠️"
            recommendation = "Le modèle détecte un risque modéré. Une analyse plus approfondie serait recommandée."
        else:
            risk_level = "Risque Élevé"
            risk_icon = "❌"
            risk_color = "#dc3545"
            decision = "RISQUE ÉLEVÉ"
            decision_icon = "❌"
            recommendation = "Le modèle identifie un risque élevé de défaut de paiement pour ce profil."
        
        # Main results: two columns layout
        col1, col2 = st.columns([1, 2])

        with col1:        
            # Jauge de risque en bas
            with st.container():
                fig_gauge = plot_gauge(risk_score)
                st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("")
            st.markdown("")
            # Décision et recommandation centrée par rapport à la gauge
            st.markdown(f"""
                <div style='display: flex; align-items: center; justify-content: center; min-height: 350px;'>
                    <div style='padding: 1.2rem; border-left: 5px solid {risk_color}; 
                         background-color: rgba(128,128,128,0.05); border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);'>
                        <h2 style='color: {risk_color}; margin: 0 0 0.8rem 0;'>{decision_icon} {decision}</h2>
                        <p style='margin: 0; font-size: 1.1rem; line-height: 1.6;'>{recommendation}</p>
                        <div style='margin-top: 1.2rem; padding: 0.8rem; background: rgba(255,255,255,0.7); border-radius: 8px;'>
                            <h4 style='color: #333; margin-bottom: 0.8rem;'>Interprétation des scores :</h4>
                            <ul style='color: #666; font-size: 0.9rem; margin: 0;'>
                                <li><strong>< 30%</strong> : Risque faible de défaut</li>
                                <li><strong>30-50%</strong> : Risque modéré</li>
                                <li><strong>> 50%</strong> : Risque élevé</li>
                            </ul>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        # SHAP Analysis Section in Expander
        with st.expander("Explicabilité du Modèle (SHAP) - Facteurs d'influence", expanded=False):
            st.markdown("""
            L'analyse SHAP révèle 
            quelles caractéristiques du profil influencent le plus la prédiction :
            """)
            
            with st.spinner('Analyse des facteurs d\'influence...'):
                feats = read_pickle('ressource/feats')
                mapping = {f"Column_{i}": name for i, name in enumerate(df.columns)}
                # Load SHAP explainer via robust utils fallback
                SHAP_explainer = load_shap_explainer('ressource/shap_explainer', clf)

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
                
                # Replace technical names with friendly names
                shap_explained["features"] = shap_explained["features"].map(mapping).fillna(shap_explained["features"])

                most_important_features = [format_feature_name(f) for f in most_important_features]
                
                explained_chart = plot_important_features(shap_explained, most_important_features)

            # SHAP Visualization
            custom_plotly_chart(explained_chart, "Analyse SHAP - Facteurs d'Influence")
            
            # Explanation colors
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div style='padding: 1rem; background-color: #f8d7da; border-radius: 8px; border-left: 4px solid #dc3545;'>
                    <strong style='color: #721c24;'>🔴 Barres rouges</strong><br>
                    <small>Facteurs qui <strong>augmentent</strong> le risque de défaut</small>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style='padding: 1rem; background-color: #d4edda; border-radius: 8px; border-left: 4px solid #28a745;'>
                    <strong style='color: #155724;'>🟢 Barres vertes</strong><br>
                    <small>Facteurs qui <strong>diminuent</strong> le risque de défaut</small>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.success("""
            💡 **Comment interpréter ces résultats ?**
            - Plus la barre est longue, plus le facteur influence la prédiction
            - Les facteurs sont classés par ordre d'importance décroissante
            - Cette analyse démontre la transparence du modèle statistique
            """)

    with tab2:
        # Display client application data analysis
        st.info("💡 Explorez et comparez les caractéristiques de ce client avec l'ensemble de la population")
            
        data = read_df('data/application_sample.csv')
        data["TARGET"] = data["TARGET"].astype(str)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Variables principales** (sélection multiple)")
            # Get available features and create friendly mapping
            available_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).select_dtypes('float').columns
            feature_options = {format_feature_name(f): f for f in available_features}
            
            selected_friendly = st.multiselect(
                'Choisissez les variables à analyser :',
                options=list(feature_options.keys()),
                help="Sélectionnez une ou plusieurs variables numériques à visualiser"
            )
            # Convert back to technical names for processing
            selected_features = [feature_options[f] for f in selected_friendly]
        
        with col2:
            st.markdown("**Variable secondaire** (optionnel)")
            all_features = data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns
            all_feature_options = {format_feature_name(f): f for f in all_features}
            
            selected_friendly_2 = st.selectbox(
                'Comparer avec une autre variable :',
                ['Choisissez une variable...'] + list(all_feature_options.keys()),
                help="Optionnel : Sélectionnez une seconde variable pour comparaison"
            )
            # Convert back to technical name
            if selected_friendly_2 != 'Choisissez une variable...':
                selected_features_2 = all_feature_options[selected_friendly_2]
            else:
                selected_features_2 = 'Choisissez une variable...'
        st.divider()
        
        graph_place = st.empty()
        if selected_features and selected_features_2 == "Choisissez une variable...":
            for features in selected_features:
                # Get friendly name for display
                friendly_name = format_feature_name(features)
                
                
                with st.container():
                    data_client_value = data.loc[data['SK_ID_CURR'] == client_id, features].values
                    data_client_target = data.loc[data['SK_ID_CURR'] == client_id, 'TARGET'].values

                    # Generate distribution data
                    hist, edges = np.histogram(data.loc[:, features].dropna(), bins=20)
                    hist_source_df = pd.DataFrame({"edges_left": edges[:-1], "edges_right": edges[1:], "hist":hist})
                    max_histogram = hist_source_df["hist"].max()
                    client_line = pd.DataFrame({"x": [data_client_value, data_client_value],
                                                "y": [0, max_histogram]})
                    hist_source = hist_source_df.to_dict('list')

                    # Display in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        plot = plot_feature_distrib(friendly_name, client_line, hist_source, data_client_value, max_histogram)
                        custom_plotly_chart(plot, f"Distribution - {friendly_name}")
                    with col2:
                        fig = px.box(data, x='TARGET', y=features, points="outliers", color='TARGET', height=580)
                        fig.update_traces(quartilemethod="inclusive")
                        fig.add_trace(go.Scatter(x=data_client_target,
                                                y=data_client_value,
                                                mode='markers',
                                                marker=dict(size=10),
                                                showlegend=False,
                                                name='client'))
                        fig.update_layout(
                            yaxis_title=friendly_name,
                            xaxis_title='Default status (0=No, 1=Yes)'
                        )
                        custom_plotly_chart(fig, f"Comparaison par Statut - {friendly_name}")

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
                        
                        # Create scatter plot
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
                        
                        custom_plotly_chart(fig, f"Corrélation : {friendly_name_1} vs {friendly_name_2}")
                        st.divider()
                    else:
                        data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                        data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                        
                        # Create box plot
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
                        
                        custom_plotly_chart(fig, f"Analyse par Catégorie : {friendly_name_1} par {friendly_name_2}")
                        st.divider()