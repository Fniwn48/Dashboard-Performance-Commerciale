import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from io import BytesIO
import random

# Configuration de la page
st.set_page_config(
    page_title="Analyse Conversions Devis → Commandes",
    page_icon="📊",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    /* Réduction générale des tailles */
    .main > div {
        padding-top: 1rem;
    }
    
    .metric-card {
        background-color: #fafafa;
        padding: 12px 8px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 13px;
    }
    
    .alert-box {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-weight: 500;
        font-size: 11px;
        line-height: 1.4;
    }
    
    .alert-critical {
        background-color: #fef2f2;
        border-left: 3px solid #dc2626;
        color: #7f1d1d;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border-left: 3px solid #d97706;
        color: #78350f;
    }
    
    .alert-success {
        background-color: #f0fdf4;
        border-left: 3px solid #059669;
        color: #14532d;
    }
    
    .kpi-section {
        background: #f9fafb;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
    .kpi-title {
        color: #374151;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 12px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-container {
        background: white;
        padding: 10px 8px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        text-align: center;
        height: 100%;
        border: 1px solid #f3f4f6;
    }
    
    .metric-value {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin: 4px 0;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 11px;
        font-weight: 500;
        line-height: 1.3;
    }
    
    /* Style pour les DataFrames */
    .dataframe {
        font-size: 11px !important;
    }
    
    div[data-testid="stDataFrame"] > div {
        background-color: white;
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    /* Masquer l'index */
    thead tr th:first-child {
        display: none !important;
    }
    tbody tr td:first-child {
        display: none !important;
    }
    
    /* Réduire la taille des headers */
    h3 {
        font-size: 18px !important;
        margin-bottom: 12px !important;
        margin-top: 20px !important;
    }
    
    h4 {
        font-size: 16px !important;
        margin-bottom: 10px !important;
    }
    
    h5 {
        font-size: 14px !important;
        margin-bottom: 8px !important;
        color: #374151;
    }
    
    h6 {
        font-size: 12px !important;
        margin-bottom: 6px !important;
        color: #4b5563;
    }
    
    /* Ajuster les espaces */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Petits textes */
    small {
        font-size: 10px !important;
        color: #9ca3af;
    }
    
    /* Ajuster les graphiques Plotly */
    .js-plotly-plot {
        margin-bottom: 12px !important;
    }
    
    /* Sidebar plus compacte */
    .css-1d391kg {
        padding: 1rem 1rem !important;
    }
    
    /* Boutons plus petits */
    .stButton > button {
        font-size: 13px !important;
        padding: 0.4rem 1rem !important;
    }
    
    /* Select boxes plus petites */
    .stSelectbox label {
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def parse_date_column(df, date_column='Created On Date - Line Local'):
    """Parse la colonne date et ajoute les colonnes temporelles - VERSION CORRIGÉE"""
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df[date_column], format='%m/%d/%Y', errors='coerce')
    
    # Si la conversion échoue, essayer d'autres approches
    if df['Date'].isna().sum() > len(df) * 0.5:
        df['Date'] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=False)
    
    # Supprimer les lignes où la date n'a pas pu être parsée
    df = df[df['Date'].notna()].copy()
    
    df['Année'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Nom_Mois'] = df['Date'].dt.strftime('%B')
    
    # ✅ CORRECTION PRINCIPALE : Calcul de l'année fiscale IDENTIQUE au code Python
    # L'année fiscale N commence en août N-1 et se termine en juillet N
    df['Année_Fiscale'] = df.apply(
        lambda row: int(row['Année'] + 1) if row['Mois'] >= 8 else int(row['Année']), 
        axis=1
    )
    
    # ✅ CORRECTION : Mois fiscal IDENTIQUE au code Python
    # Août = 1, Septembre = 2, ..., Juillet = 12
    df['Mois_Fiscal'] = df.apply(
        lambda row: ((int(row['Mois']) - 8) % 12) + 1 if pd.notna(row['Mois']) else pd.NA, 
        axis=1
    )
    
    return df

def _detect_net_value_column(df: pd.DataFrame, fiscal_year: int) -> pd.DataFrame:
    """Détecte et nettoie la colonne Net Value - IDENTIQUE AU CODE PYTHON"""
    import re
    
    # Chercher les colonnes contenant "net value"
    value_cols = [c for c in df.columns if "net value" in str(c).lower()]
    if not value_cols:
        raise KeyError("❌ Colonne 'Net Value' introuvable (ex: '2024  Net Value').")
    
    # Fonction pour extraire l'année de la colonne
    def extract_year(c):
        m = re.match(r"\s*(\d{4})\s+.*net value", str(c).lower())
        return int(m.group(1)) if m else -1
    
    # Prendre la colonne avec l'année la plus récente
    value_col = max(value_cols, key=extract_year)
    
    # Renommer la colonne
    df = df.rename(columns={value_col: "Net Value"})
    
    # ✅ CORRECTION : Nettoyer exactement comme dans le code Python
    df["Net Value"] = pd.to_numeric(
        df["Net Value"].astype(str).str.replace(r"[^\d\-,\.]", "", regex=True),
        errors="coerce"
    )
    
    return df

def aggregate_orders_by_doc(orders: pd.DataFrame, IC_VALUE="IC-Inbound Call") -> pd.DataFrame:
    """Agrège les commandes par document - IDENTIQUE AU CODE PYTHON"""
    
    # Flag IC au niveau commande (si au moins une ligne IC)
    ic_flag = (orders.assign(_is_ic=(orders["Purchase Order Type"] == IC_VALUE))
                     .groupby("Sales Document #")["_is_ic"].any()
                     .rename("HasIC").reset_index())

    # ✅ AGRÉGATION IDENTIQUE AU CODE PYTHON
    agg = (orders.sort_values(["Sales Document #","Date"])
                  .groupby("Sales Document #", as_index=False)
                  .agg(Date=("Date","min"),
                       **{"SoldTo #":("SoldTo #","first")},
                       **{"SoldTo Name":("SoldTo Name","first")},  # Ajouter SoldTo Name
                        **{"SoldTo Managed Group":("SoldTo Managed Group","first")},  
                       **{"Net Value":("Net Value","sum")},
                       **{"Created By Line":("Created By Line","first")},
                       **{"Purchase Order Type":("Purchase Order Type","first")},  # Ajouter Purchase Order Type
                       FY=("Année_Fiscale","max"),
                       FiscalMonth=("Mois_Fiscal","min"),
                       Month=("Mois","min"),
                       MonthName=("Nom_Mois","first")))
    
    # Merger avec le flag IC
    agg = agg.merge(ic_flag, on="Sales Document #", how="left").fillna({"HasIC":False})
    return agg

def build_real_orders_for_user_python_style(orders: pd.DataFrame, attrib: pd.DataFrame, user_sap: str) -> pd.DataFrame:
    """Construit les commandes réelles EXACTEMENT comme le code Python"""
    
    # ✅ ÉTAPE 1 : Détecter et nettoyer la colonne Net Value
    orders_cleaned = _detect_net_value_column(orders.copy(), orders['Année_Fiscale'].iloc[0])
    
    # ✅ ÉTAPE 2 : Agrégation par document
    aggs = aggregate_orders_by_doc(orders_cleaned)

    # ✅ ÉTAPE 3 : Attribution (INNER JOIN)
    attrib_unique = attrib[["Order Document #","Created By Header"]].drop_duplicates("Order Document #")
    step1 = aggs.merge(attrib_unique, left_on="Sales Document #", right_on="Order Document #", how="inner")
    step1["Commercial Réel"] = step1["Created By Header"]

    # ✅ ÉTAPE 4 : IC non présentes dans attribution
    deja = set(step1["Sales Document #"])
    step2 = aggs[(aggs["HasIC"] == True) & (~aggs["Sales Document #"].isin(deja))].copy()
    step2["Commercial Réel"] = step2["Created By Line"]

    # ✅ ÉTAPE 5 : Union
    real = pd.concat([step1, step2], ignore_index=True)

    # ✅ ÉTAPE 6 : Filtre sur le commercial demandé
    real_user = real[real["Commercial Réel"] == user_sap].copy()

    # ✅ ÉTAPE 7 : Renommer les colonnes pour correspondre aux attentes Streamlit
    column_mapping = {
        'FY': 'Année_Fiscale',
        'FiscalMonth': 'Mois_Fiscal',
        'MonthName': 'Nom_Mois'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in real_user.columns:
            real_user = real_user.rename(columns={old_col: new_col})

    # ✅ ÉTAPE 8 : Colonnes finales
    keep = ["Sales Document #","Date","SoldTo #","SoldTo Name","Net Value","Commercial Réel","Année_Fiscale","Mois_Fiscal","Nom_Mois","Purchase Order Type"]
    available_cols = [col for col in keep if col in real_user.columns]
    return real_user[available_cols]



@st.cache_data
def load_and_process_file(uploaded_file):
    """Charge et traite un fichier Excel"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes (enlever les espaces en début/fin)
        df.columns = df.columns.str.strip()
        
        # Vérifier que la colonne Created On Date - Line Local existe
        if 'Created On Date - Line Local' not in df.columns:
            st.error(f"La colonne 'Created On Date - Line Local' n'a pas été trouvée dans le fichier. Colonnes disponibles : {', '.join(df.columns)}")
            return None
        
        df = parse_date_column(df)        
    
        # Vérifier qu'il reste des données après le filtrage
        if len(df) == 0:
            st.error("Aucune donnée valide après le traitement des dates. Vérifiez le format des dates (MM/DD/YYYY).")
            return None
            
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None
    
@st.cache_data
def load_objectifs_file(uploaded_file):
    """Charge le fichier des objectifs"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes (enlever les espaces en début/fin)
        df.columns = df.columns.str.strip()
        
        # Créer un mapping des mois français vers les mois fiscaux
        mois_mapping = {
            'Août': 1, 'Septembre': 2, 'Octobre': 3, 'Novembre': 4, 'Décembre': 5,
            'Janvier': 6, 'Février': 7, 'Mars': 8, 'Avril': 9, 'Mai': 10, 'Juin': 11, 'Juillet': 12
        }
        
        # Si on a une colonne Mois en français, la convertir
        if 'Mois' in df.columns:
            # Nettoyer la colonne Mois (enlever espaces et accents)
            df['Mois'] = df['Mois'].astype(str).str.strip()
            df['Mois_Fiscal'] = df['Mois'].map(mois_mapping)
            
            # Vérifier s'il y a des mois non mappés
            mois_non_mappes = df[df['Mois_Fiscal'].isna()]['Mois'].unique()
            if len(mois_non_mappes) > 0:
                st.error(f"Mois non reconnus dans le fichier objectifs : {list(mois_non_mappes)}")
        else:
            st.error("Colonne 'Mois' non trouvée dans le fichier objectifs")
            return None
        
        # Nettoyer les colonnes d'objectifs (enlever les symboles € et convertir en numérique)
        for col in df.columns:
            if 'Objectifs' in col or 'Objectif' in col:
                if df[col].dtype == 'object':
                    # Enlever les espaces, € et convertir les virgules en points
                    df[col] = df[col].astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier objectifs : {str(e)}")
        return None
    
@st.cache_data
def load_commerciaux_mapping(uploaded_file):
    """Charge le fichier de mapping des commerciaux"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes (enlever les espaces en début/fin)
        df.columns = df.columns.str.strip()
        
        # Vérifier les colonnes nécessaires
        required_cols = ['Prénom', 'Nom', 'User Sap']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes : {', '.join(missing_cols)}")
            return None
        
        # Créer une colonne nom complet
        df['Nom_Complet'] = df['Prénom'].astype(str) + ' ' + df['Nom'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier commerciaux : {str(e)}")
        return None

@st.cache_data
def load_commerciaux_saisie_mapping(uploaded_file):
    """Charge le fichier de mapping des commerciaux de saisie"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes (enlever les espaces en début/fin)
        df.columns = df.columns.str.strip()
        
        # Vérifier les colonnes nécessaires
        required_cols = ['Prénom', 'Nom', 'User Sap']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes : {', '.join(missing_cols)}")
            return None
        
        # Créer une colonne nom complet
        df['Nom_Complet'] = df['Prénom'].astype(str) + ' ' + df['Nom'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier commerciaux de saisie : {str(e)}")
        return None

@st.cache_data
def load_objectifs_personnalises(uploaded_file):
    """Charge le fichier des objectifs personnalisés"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes (enlever les espaces en début/fin)
        df.columns = df.columns.str.strip()
        
        # Créer un mapping des mois français vers les mois fiscaux
        mois_mapping = {
            'Août': 1, 'Septembre': 2, 'Octobre': 3, 'Novembre': 4, 'Décembre': 5,
            'Janvier': 6, 'Février': 7, 'Mars': 8, 'Avril': 9, 'Mai': 10, 'Juin': 11, 'Juillet': 12
        }
        
        # Si on a une colonne Mois en français, la convertir
        if 'Mois' in df.columns:
            # Nettoyer la colonne Mois (enlever espaces et accents)
            df['Mois'] = df['Mois'].astype(str).str.strip()
            df['Mois_Fiscal'] = df['Mois'].map(mois_mapping)
            
            # Vérifier s'il y a des mois non mappés
            mois_non_mappes = df[df['Mois_Fiscal'].isna()]['Mois'].unique()
            if len(mois_non_mappes) > 0:
                st.error(f"Mois non reconnus dans le fichier objectifs personnalisés : {list(mois_non_mappes)}")
        else:
            st.error("Colonne 'Mois' non trouvée dans le fichier objectifs personnalisés")
            return None
        
        # Nettoyer les colonnes d'objectifs (enlever les symboles € et convertir en numérique)
        for col in df.columns:
            if 'Objectifs' in col or 'Objectif' in col:
                if df[col].dtype == 'object':
                    # Enlever les espaces, € et convertir les virgules en points
                    df[col] = df[col].astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier objectifs personnalisés : {str(e)}")
        return None
    

def get_objectif_priorite(commercial, mois_fiscal, fiscal_year, df_objectifs_personnalises, df_objectifs_standards):
    """Obtient l'objectif avec priorité : personnalisé d'abord, sinon standard"""
    
    # Chercher d'abord dans les objectifs personnalisés
    if df_objectifs_personnalises is not None:
        cle_personnalisee = f'Objectifs {commercial} {fiscal_year}'
        
        # Vérifier si la colonne existe
        if cle_personnalisee in df_objectifs_personnalises.columns:
            objectif_row = df_objectifs_personnalises[df_objectifs_personnalises['Mois_Fiscal'] == mois_fiscal]
            if len(objectif_row) > 0:
                objectif_value = objectif_row[cle_personnalisee].iloc[0]
                if pd.notna(objectif_value) and objectif_value > 0:
                    return float(objectif_value)
    
    # Sinon utiliser les objectifs standards
    if df_objectifs_standards is not None:
        objectif_col = get_objectif_column(df_objectifs_standards, fiscal_year)
        if objectif_col and objectif_col in df_objectifs_standards.columns:
            objectif_row = df_objectifs_standards[df_objectifs_standards['Mois_Fiscal'] == mois_fiscal]
            if len(objectif_row) > 0:
                objectif_value = objectif_row[objectif_col].iloc[0]
                if pd.notna(objectif_value) and objectif_value > 0:
                    return float(objectif_value)
    
    return 0

def get_commercial_name(user_sap, df_mapping):
    """Obtient le nom complet d'un commercial à partir de son User SAP - VERSION CORRIGÉE"""
    if df_mapping is None:
        return user_sap
    
    # Convertir en string pour éviter les problèmes de type
    user_sap_str = str(user_sap)
    df_mapping_copy = df_mapping.copy()
    df_mapping_copy['User Sap'] = df_mapping_copy['User Sap'].astype(str)
    
    match = df_mapping_copy[df_mapping_copy['User Sap'] == user_sap_str]
    if len(match) > 0:
        return match.iloc[0]['Nom_Complet']
    return user_sap

def categorize_commercials(df_devis, df_commandes, df_mapping_commerciaux, df_mapping_saisie):
    """Catégorise les commerciaux entre commerciaux et commerciaux de saisie - VERSION CORRIGÉE"""
    # Obtenir tous les commerciaux présents dans les données (CONVERSION EN STRING)
    commercials_devis = df_devis['Created By Line'].dropna().astype(str).unique()
    commercials_commandes = df_commandes['Created By Line'].dropna().astype(str).unique()
    all_commercials = list(set(commercials_devis) | set(commercials_commandes))
    
    # Séparer selon les fichiers de mapping
    commerciaux_list = []
    commerciaux_saisie_list = []
    
    # Commerciaux avec objectifs (CONVERSION EN STRING)
    if df_mapping_commerciaux is not None:
        commerciaux_avec_objectifs = df_mapping_commerciaux['User Sap'].astype(str).unique()
        for commercial in all_commercials:
            if str(commercial) in commerciaux_avec_objectifs:  # CONVERSION EXPLICITE
                commerciaux_list.append(str(commercial))
    
    # ✅ CORRECTION : Commerciaux de saisie (SEULEMENT ceux des commandes)
    if df_mapping_saisie is not None:
        commerciaux_saisie_mapping = df_mapping_saisie['User Sap'].astype(str).unique()
        for commercial in commercials_commandes:  # ← CHANGEMENT ICI : seulement commercials_commandes
            if str(commercial) in commerciaux_saisie_mapping:  # CONVERSION EXPLICITE
                commerciaux_saisie_list.append(str(commercial))
    
    return commerciaux_list, commerciaux_saisie_list

def get_net_value_column(df, fiscal_year):
    """Détermine la colonne de valeur nette à utiliser selon l'année fiscale"""
    possible_cols = [f'{fiscal_year} Net Value', f'{fiscal_year}  Net Value', f'Net Value {fiscal_year}']
    
    for col in possible_cols:
        if col in df.columns:
            return col
    
    # Si aucune colonne spécifique trouvée, chercher une colonne générique
    for col in df.columns:
        if 'net value' in col.lower():
            return col
    
    return None

def get_objectif_column(df_objectifs, fiscal_year):
    """Détermine la colonne d'objectifs à utiliser selon l'année fiscale"""
    possible_cols = [f'Objectifs {fiscal_year}', f'Objectif {fiscal_year}']
    
    for col in possible_cols:
        if col in df_objectifs.columns:
            return col
    
    # Si aucune colonne spécifique trouvée, chercher une colonne avec "Objectif" et l'année
    for col in df_objectifs.columns:
        if 'Objectif' in col and str(fiscal_year) in col:
            return col
    
    # Dernière tentative : chercher n'importe quelle colonne avec "Objectif"
    for col in df_objectifs.columns:
        if 'Objectif' in col:
            return col
    
    return None

def filter_data(df, fiscal_year=None, month=None, period=None, commercial=None):
    """Filtre les données selon les critères sélectionnés"""
    filtered_df = df.copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        filtered_df = filtered_df[filtered_df['Created By Line'] == commercial]
    
    return filtered_df

def filter_data_by_commercial_list(df, commercials_list, fiscal_year=None, month=None, period=None, commercial=None):
    """Filtre les données selon la liste des commerciaux et autres critères - VERSION CORRIGÉE"""
    # Créer une copie et convertir les types
    df_copy = df.copy()
    df_copy['Created By Line'] = df_copy['Created By Line'].astype(str)
    
    # Convertir la liste des commerciaux en string
    commercials_list_str = [str(x) for x in commercials_list]
    
    # Filtrer par la liste des commerciaux autorisés
    filtered_df = df_copy[df_copy['Created By Line'].isin(commercials_list_str)].copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        filtered_df = filtered_df[filtered_df['Created By Line'] == str(commercial)]
    
    return filtered_df


def calculate_kpis(df_devis, df_commandes, df_objectifs=None, df_mapping=None, fiscal_year=None, period_months=None, df_objectifs_personnalises=None):
    """Calcule les KPIs principaux avec objectifs"""
    kpis = {}
    
    # Déterminer la colonne de valeur à utiliser
    net_value_col_devis = get_net_value_column(df_devis, fiscal_year)
    net_value_col_commandes = get_net_value_column(df_commandes, fiscal_year)
    
    # KPIs de base
    kpis['nb_devis'] = df_devis['Sales Document #'].nunique()
    kpis['nb_commandes'] = df_commandes['Sales Document #'].nunique()
    kpis['valeur_devis'] = df_devis[net_value_col_devis].sum() if net_value_col_devis else 0
    kpis['valeur_commandes'] = df_commandes[net_value_col_commandes].sum() if net_value_col_commandes else 0
    
    # Nombre de commerciaux uniques - PRENDRE TOUS CEUX DU MAPPING
    if df_mapping is not None:
        # Prendre TOUS les commerciaux du fichier de mapping
        all_commercials = df_mapping['User Sap'].astype(str).unique().tolist()
        nb_commerciaux = len(all_commercials)
    else:
        # Fallback si pas de mapping : prendre ceux des données
        commercials_devis = df_devis['Created By Line'].dropna().astype(str).unique()
        commercials_commandes = df_commandes['Created By Line'].dropna().astype(str).unique()
        all_commercials = list(set(commercials_devis) | set(commercials_commandes))
        nb_commerciaux = len(all_commercials)
    
    kpis['nb_commerciaux'] = nb_commerciaux

    # Ajuster le nombre si on filtre sur un commercial spécifique
    commercials_in_data = list(set(
        list(df_devis['Created By Line'].dropna().astype(str).unique()) + 
        list(df_commandes['Created By Line'].dropna().astype(str).unique())
    ))

    if len(commercials_in_data) == 1:
        nb_commerciaux = 1
        kpis['nb_commerciaux'] = 1

    # Moyennes par commercial
    if nb_commerciaux > 0:
        kpis['moy_nb_devis_par_commercial'] = kpis['nb_devis'] / nb_commerciaux
        kpis['moy_nb_commandes_par_commercial'] = kpis['nb_commandes'] / nb_commerciaux
        kpis['moy_valeur_devis_par_commercial'] = kpis['valeur_devis'] / nb_commerciaux
        kpis['moy_valeur_commandes_par_commercial'] = kpis['valeur_commandes'] / nb_commerciaux
    else:
        kpis['moy_nb_devis_par_commercial'] = 0
        kpis['moy_nb_commandes_par_commercial'] = 0
        kpis['moy_valeur_devis_par_commercial'] = 0
        kpis['moy_valeur_commandes_par_commercial'] = 0
    
    # Taux de conversion clients
    kpis['nb_clients_devis'] = df_devis['SoldTo #'].nunique()
    kpis['nb_clients_commandes'] = df_commandes['SoldTo #'].nunique()
    kpis['taux_conversion_clients'] = (kpis['nb_clients_commandes'] / kpis['nb_clients_devis'] * 100) if kpis['nb_clients_devis'] > 0 else 0
    

    # Objectif global - CALCUL FINAL avec priorité personnalisée
    if df_objectifs is not None and fiscal_year and period_months:
        objectif_total_global = 0
        
        # Déterminer quels commerciaux considérer selon les données filtrées
        commercials_in_data = list(set(
            list(df_devis['Created By Line'].dropna().astype(str).unique()) + 
            list(df_commandes['Created By Line'].dropna().astype(str).unique())
        ))
        
        # Si les données ne concernent qu'un seul commercial, calculer seulement pour lui
        if len(commercials_in_data) == 1:
            commercials_to_calculate = commercials_in_data
        else:
            commercials_to_calculate = all_commercials
        
        # Calculer l'objectif pour chaque commercial individuellement
        for commercial in commercials_to_calculate:
            objectif_commercial_periode = 0
            
            # Parcourir tous les mois de la période pour ce commercial
            for month_fiscal in period_months:
                # Utiliser la fonction de priorité pour obtenir l'objectif
                objectif_mois = get_objectif_priorite(commercial, month_fiscal, fiscal_year, df_objectifs_personnalises, df_objectifs)
                objectif_commercial_periode += objectif_mois
            
            objectif_total_global += objectif_commercial_periode
        
        kpis['objectif_total'] = objectif_total_global
        kpis['pct_objectif_atteint'] = (kpis['valeur_commandes'] / kpis['objectif_total'] * 100) if kpis['objectif_total'] > 0 else 0
    else:
        kpis['objectif_total'] = 0
        kpis['pct_objectif_atteint'] = 0
            
    return kpis

def calculate_kpis_saisie_only(df_commandes, df_mapping_saisie, fiscal_year=None, period_months=None):
    """Calcule les KPIs UNIQUEMENT pour les commandes (commerciaux de saisie) - VERSION CORRIGÉE"""
    kpis = {}
    
    # Déterminer la colonne de valeur à utiliser
    net_value_col_commandes = get_net_value_column(df_commandes, fiscal_year)
    
    # KPIs de base (SEULEMENT commandes)
    kpis['nb_devis'] = 0  # Pas de devis pour commerciaux de saisie
    kpis['nb_commandes'] = df_commandes['Sales Document #'].nunique()
    kpis['valeur_devis'] = 0  # Pas de devis pour commerciaux de saisie
    kpis['valeur_commandes'] = df_commandes[net_value_col_commandes].sum() if net_value_col_commandes else 0
    
    # ✅ CORRECTION : Compter seulement les commerciaux qui ont des données dans df_commandes
    # ET qui sont dans le mapping de saisie
    if df_mapping_saisie is not None:
        # Commerciaux qui ont des commandes dans les données filtrées
        commercials_with_data = df_commandes['Created By Line'].dropna().astype(str).unique()
        # Commerciaux autorisés selon le mapping
        commercials_in_mapping = df_mapping_saisie['User Sap'].astype(str).unique()
        # ✅ INTERSECTION : seulement ceux qui sont dans les deux
        valid_commercials = [c for c in commercials_with_data if c in commercials_in_mapping]
        nb_commerciaux = len(valid_commercials)
    else:
        # Si pas de mapping, compter directement les commerciaux dans les données
        nb_commerciaux = df_commandes['Created By Line'].dropna().astype(str).nunique()
    
    # ✅ CORRECTION IMPORTANTE : Si aucun commercial trouvé, mettre 1 pour éviter division par zéro
    if nb_commerciaux == 0:
        nb_commerciaux = 1
    
    kpis['nb_commerciaux'] = nb_commerciaux
    
    # Moyennes par commercial (SEULEMENT commandes)
    kpis['moy_nb_devis_par_commercial'] = 0  # Pas de devis
    kpis['moy_nb_commandes_par_commercial'] = kpis['nb_commandes'] / nb_commerciaux
    kpis['moy_valeur_devis_par_commercial'] = 0  # Pas de devis
    kpis['moy_valeur_commandes_par_commercial'] = kpis['valeur_commandes'] / nb_commerciaux
    
    # Clients (SEULEMENT commandes)
    kpis['nb_clients_devis'] = 0  # Pas de devis
    kpis['nb_clients_commandes'] = df_commandes['SoldTo #'].nunique()
    kpis['taux_conversion_clients'] = 0  # Pas de conversion possible sans devis
    
    # Pas d'objectifs pour les commerciaux de saisie
    kpis['objectif_total'] = 0
    kpis['pct_objectif_atteint'] = 0
    
    return kpis

def calculate_commercial_performance_saisie(df_commandes, df_mapping_saisie, fiscal_year):
    """Calcule les performances pour commerciaux de saisie (SEULEMENT commandes) - VERSION CORRIGÉE"""
    
    # ✅ CORRECTION : Prendre seulement les commerciaux qui ont des données dans df_commandes
    # ET qui sont dans le mapping de saisie
    if df_mapping_saisie is not None:
        # Commerciaux qui ont des commandes dans les données filtrées
        commercials_with_data = df_commandes['Created By Line'].dropna().astype(str).unique()
        # Commerciaux autorisés selon le mapping
        commercials_in_mapping = df_mapping_saisie['User Sap'].astype(str).unique()
        # ✅ INTERSECTION : seulement ceux qui sont dans les deux
        all_commercials_saisie = [c for c in commercials_with_data if c in commercials_in_mapping]
    else:
        all_commercials_saisie = df_commandes['Created By Line'].dropna().astype(str).unique()
    
    # Déterminer la colonne de valeur
    net_value_col_commandes = get_net_value_column(df_commandes, fiscal_year)
    
    stats = []
    
    for commercial in all_commercials_saisie:
        commandes_com = df_commandes[df_commandes['Created By Line'] == commercial]
        
        nom_complet = get_commercial_name(commercial, df_mapping_saisie)
        
        valeur_commandes = commandes_com[net_value_col_commandes].sum() if net_value_col_commandes else 0
        
        stat = {
            'Commercial': nom_complet,
            'Nb Commandes': commandes_com['Sales Document #'].nunique(),
            'Valeur Commandes': valeur_commandes,
            'Nb Clients Commandes': commandes_com['SoldTo #'].nunique()
        }
        
        stats.append(stat)
    
    df_stats = pd.DataFrame(stats)
    
    return df_stats, []  # Pas d'alertes pour commerciaux de saisie

def calculate_order_type_stats(df_devis, df_commandes):
    """Calcule les statistiques par type de commande (Purchase Order Type) - Documents uniques seulement"""
    stats = {
        'devis': {},
        'commandes': {}
    }
    
    # Vérifier si la colonne existe
    if 'Purchase Order Type' in df_devis.columns:
        # Pour éviter la duplication, prendre seulement une ligne par Sales Document #
        df_devis_unique = df_devis.drop_duplicates(subset=['Sales Document #', 'Purchase Order Type'])
        
        # Stats pour les devis (documents uniques)
        devis_types = df_devis_unique.groupby('Purchase Order Type')['Sales Document #'].nunique()
        total_devis = df_devis['Sales Document #'].nunique()
        
        for order_type, count in devis_types.items():
            stats['devis'][order_type] = {
                'count': count,
                'percentage': (count / total_devis * 100) if total_devis > 0 else 0
            }
    
    if 'Purchase Order Type' in df_commandes.columns:
        # Pour éviter la duplication, prendre seulement une ligne par Sales Document #
        df_commandes_unique = df_commandes.drop_duplicates(subset=['Sales Document #', 'Purchase Order Type'])
        
        # Stats pour les commandes (documents uniques)
        commandes_types = df_commandes_unique.groupby('Purchase Order Type')['Sales Document #'].nunique()
        total_commandes = df_commandes['Sales Document #'].nunique()
        
        for order_type, count in commandes_types.items():
            stats['commandes'][order_type] = {
                'count': count,
                'percentage': (count / total_commandes * 100) if total_commandes > 0 else 0
            }
    
    return stats

def calculate_commercial_performance(df_devis, df_commandes, df_objectifs, df_mapping, fiscal_year, period_months, df_objectifs_personnalises=None, df_real_orders=None, vue_type=None):
    """Calcule les performances détaillées par commercial avec alertes"""
    # PRENDRE TOUS les commerciaux du fichier de mapping (même sans données)
    if df_mapping is not None:
        all_commercials = df_mapping['User Sap'].astype(str).unique().tolist()
    else:
        # Fallback si pas de mapping
        commercials_devis = df_devis['Created By Line'].dropna().astype(str).unique()
        commercials_commandes = df_commandes['Created By Line'].dropna().astype(str).unique()
        all_commercials = list(set(commercials_devis) | set(commercials_commandes))
    
    # Déterminer les colonnes de valeur
    net_value_col_devis = get_net_value_column(df_devis, fiscal_year)
    net_value_col_commandes = get_net_value_column(df_commandes, fiscal_year)
    
    stats = []
    alerts = []
    
    for commercial in all_commercials:
        devis_com = df_devis[df_devis['Created By Line'] == commercial]
        commandes_com = df_commandes[df_commandes['Created By Line'] == commercial]
        
        nom_complet = get_commercial_name(commercial, df_mapping)
        
        valeur_devis = devis_com[net_value_col_devis].sum() if net_value_col_devis else 0
        valeur_commandes = commandes_com[net_value_col_commandes].sum() if net_value_col_commandes else 0
        
        if vue_type == "Commercial de saisie":
            # Pour Commercial de saisie : seulement les commandes
            stat = {
                'Commercial': nom_complet,
                'Nb Commandes': commandes_com['Sales Document #'].nunique(),
                'Valeur Commandes': valeur_commandes,
                'Nb Clients Commandes': commandes_com['SoldTo #'].nunique()
            }
        else:
            # Pour Vue globale et Commercial : toutes les données
            stat = {
                'Commercial': nom_complet,
                'Nb Devis': devis_com['Sales Document #'].nunique(),
                'Nb Commandes': commandes_com['Sales Document #'].nunique(),
                'Valeur Devis': valeur_devis,
                'Valeur Commandes': valeur_commandes,
                'Nb Clients Devis': devis_com['SoldTo #'].nunique(),
                'Nb Clients Commandes': commandes_com['SoldTo #'].nunique(),
                '% Conversion Clients': (commandes_com['SoldTo #'].nunique() / devis_com['SoldTo #'].nunique() * 100) if devis_com['SoldTo #'].nunique() > 0 else 0
            }
            
            # Ajouter les KPIs réels si les données existent
            if df_real_orders is not None and not df_real_orders.empty:
                commandes_reelles_com = df_real_orders[df_real_orders['Commercial Réel'] == commercial]
                stat['Nb Commandes Réelles'] = commandes_reelles_com['Sales Document #'].nunique()
                stat['Valeur Commandes Réelles'] = commandes_reelles_com['Net Value'].sum()
                stat['Nb Clients Réels'] = commandes_reelles_com['SoldTo #'].nunique()
            else:
                stat['Nb Commandes Réelles'] = 0
                stat['Valeur Commandes Réelles'] = 0
                stat['Nb Clients Réels'] = 0

        # Calculer l'objectif pour ce commercial avec priorité personnalisée
        if df_objectifs is not None and period_months and vue_type != "Commercial de saisie":

            objectif_commercial_periode = 0
            for month_fiscal in period_months:
                # Utiliser la fonction de priorité pour obtenir l'objectif
                objectif_mois = get_objectif_priorite(commercial, month_fiscal, fiscal_year, df_objectifs_personnalises, df_objectifs)
                objectif_commercial_periode += objectif_mois
            
            # ✅ SORTIR DE LA BOUCLE
            stat['Objectif'] = objectif_commercial_periode
            stat['% Objectif Atteint'] = (valeur_commandes / objectif_commercial_periode * 100) if objectif_commercial_periode > 0 else 0

            # Objectif atteint réel
            if df_real_orders is not None and not df_real_orders.empty:
                stat['% Objectif Réel Atteint'] = (stat['Valeur Commandes Réelles'] / objectif_commercial_periode * 100) if objectif_commercial_periode > 0 else 0
            else:
                stat['% Objectif Réel Atteint'] = 0
            
            # Créer l'alerte pour ce commercial
            alert = {
                'commercial': nom_complet,
                'pct_atteint': stat['% Objectif Réel Atteint'] if vue_type == "Commercial" else stat['% Objectif Atteint'],
                'valeur_commandes': stat['Valeur Commandes Réelles'] if vue_type == "Commercial" and df_real_orders is not None and not df_real_orders.empty else valeur_commandes,
                'objectif': objectif_commercial_periode
            }
            # Classification selon performance - UTILISER LES DONNÉES RÉELLES
            pct_pour_classement = stat['% Objectif Réel Atteint'] if vue_type == "Commercial" else stat['% Objectif Atteint']

            if pct_pour_classement < 60:
                alert['niveau'] = 'critical'
                alert['emoji'] = '🔴'
                alert['status'] = 'En difficulté'
            elif pct_pour_classement <= 90:
                alert['niveau'] = 'warning'
                alert['emoji'] = '🟡'
                alert['status'] = 'À surveiller'
            else:
                alert['niveau'] = 'success'
                alert['emoji'] = '🟢'
                alert['status'] = 'Performant'
                        
            alerts.append(alert)
        else:
            if vue_type != "Commercial de saisie":
                stat['Objectif'] = 0
                stat['% Objectif Atteint'] = 0
                stat['% Objectif Réel Atteint'] = 0

            
            alert = {
                'commercial': nom_complet,
                'pct_atteint': 0,
                'valeur_commandes': valeur_commandes,
                'objectif': 0,
                'niveau': 'critical',
                'emoji': '🔴',
                'status': 'En difficulté'
            }
            alerts.append(alert)
        
        stats.append(stat)
    
    df_stats = pd.DataFrame(stats)
    
    # Trier les alertes par performance décroissante (les meilleurs en premier)
    alerts.sort(key=lambda x: x['pct_atteint'], reverse=True)
    
    return df_stats, alerts

def create_order_type_comparison_charts(df_devis_current, df_commandes_current, df_devis_prev, df_commandes_prev, fiscal_year):
    """Crée les graphiques de comparaison des types de commandes avec tableaux détaillés"""
    
    # Calculer les statistiques pour l'année actuelle et précédente
    stats_current = calculate_order_type_stats(df_devis_current, df_commandes_current)
    stats_prev = calculate_order_type_stats(df_devis_prev, df_commandes_prev)
    
    # Obtenir les types actifs pour DEVIS uniquement
    active_devis_types = set()
    if 'Purchase Order Type' in df_devis_current.columns and len(df_devis_current) > 0:
        for ptype, group in df_devis_current.groupby('Purchase Order Type'):
            if pd.notna(ptype) and group['Sales Document #'].nunique() > 0:
                active_devis_types.add(ptype)
    
    if 'Purchase Order Type' in df_devis_prev.columns and len(df_devis_prev) > 0:
        for ptype, group in df_devis_prev.groupby('Purchase Order Type'):
            if pd.notna(ptype) and group['Sales Document #'].nunique() > 0:
                active_devis_types.add(ptype)
    
    # Obtenir les types actifs pour COMMANDES uniquement
    active_commandes_types = set()
    if 'Purchase Order Type' in df_commandes_current.columns and len(df_commandes_current) > 0:
        for ptype, group in df_commandes_current.groupby('Purchase Order Type'):
            if pd.notna(ptype) and ptype != 'nan' and str(ptype).strip() != '' and group['Sales Document #'].nunique() > 0:
                active_commandes_types.add(ptype)
    
    if 'Purchase Order Type' in df_commandes_prev.columns and len(df_commandes_prev) > 0:
        for ptype, group in df_commandes_prev.groupby('Purchase Order Type'):
            if pd.notna(ptype) and ptype != 'nan' and str(ptype).strip() != '' and group['Sales Document #'].nunique() > 0:
                active_commandes_types.add(ptype)
    
    if not active_devis_types and not active_commandes_types:
        fig_empty = go.Figure()
        fig_empty.add_annotation(
            text="Aucun type de commande actif trouvé",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig_empty, fig_empty, None, None
    
    devis_types = sorted(list(active_devis_types))
    commandes_types = sorted(list(active_commandes_types))
    
    # Calculer les valeurs et statistiques détaillées pour les tableaux
    def get_detailed_stats(df, fiscal_year, doc_type):
        """Calcule les statistiques détaillées par type de commande - Documents uniques seulement"""
        if 'Purchase Order Type' not in df.columns:
            return pd.DataFrame()
        
        net_value_col = get_net_value_column(df, fiscal_year)
        if not net_value_col:
            return pd.DataFrame()
        
        # Éviter la duplication : une commande = un type
        df_unique = df.drop_duplicates(subset=['Sales Document #', 'Purchase Order Type'])
        
        # Calculer les stats par type avec agrégation correcte
        stats_by_type = []
        for order_type in df_unique['Purchase Order Type'].unique():
            df_type = df[df['Purchase Order Type'] == order_type]
            nb_docs = df_type['Sales Document #'].nunique()
            valeur = df_type[net_value_col].sum()
            
            # Inclure tous les types, même ceux qui n'existent que dans une source
            stats_by_type.append({
                'Type': order_type,
                f'Nb {doc_type}': nb_docs,
                'Nb Clients': df_type['SoldTo #'].nunique(),
                'Valeur (€)': valeur
            })
        
        result_df = pd.DataFrame(stats_by_type)
        if not result_df.empty:
            # Calculer les pourcentages
            total_docs = result_df[f'Nb {doc_type}'].sum()
            total_valeur = result_df['Valeur (€)'].sum()
            result_df[f'% {doc_type}'] = (result_df[f'Nb {doc_type}'] / total_docs * 100).round(1) if total_docs > 0 else 0
            result_df['% Valeur'] = (result_df['Valeur (€)'] / total_valeur * 100).round(1) if total_valeur > 0 else 0
            return result_df.sort_values('Valeur (€)', ascending=False)
        return result_df
    
    # Données pour les graphiques SÉPARÉS (chaque graphique ses propres types)
    # Devis avec seulement les types de devis
    devis_current = [stats_current['devis'].get(t, {}).get('count', 0) for t in devis_types]
    devis_prev = [stats_prev['devis'].get(t, {}).get('count', 0) for t in devis_types]
    
    # Commandes avec seulement les types de commandes
    commandes_current = [stats_current['commandes'].get(t, {}).get('count', 0) for t in commandes_types]
    commandes_prev = [stats_prev['commandes'].get(t, {}).get('count', 0) for t in commandes_types]
    
    # Calculer les pourcentages pour chaque graphique séparément
    total_devis_current = sum(devis_current) if sum(devis_current) > 0 else 1
    total_devis_prev = sum(devis_prev) if sum(devis_prev) > 0 else 1
    total_commandes_current = sum(commandes_current) if sum(commandes_current) > 0 else 1
    total_commandes_prev = sum(commandes_prev) if sum(commandes_prev) > 0 else 1
    
    devis_current_pct = [(v/total_devis_current*100) for v in devis_current]
    devis_prev_pct = [(v/total_devis_prev*100) for v in devis_prev]
    commandes_current_pct = [(v/total_commandes_current*100) for v in commandes_current]
    commandes_prev_pct = [(v/total_commandes_prev*100) for v in commandes_prev]
    
    # Graphique 1 : Comparaison des devis (seulement types de devis)
    fig_devis = go.Figure()
    
    fig_devis.add_trace(go.Bar(
        name=f'{fiscal_year}',
        x=devis_types,
        y=devis_current,
        marker_color='#3b82f6',
        text=[f'{v}<br>({p:.1f}%)' if v > 0 else '' for v, p in zip(devis_current, devis_current_pct)],
        textposition='outside',
        textfont=dict(size=10, color='#1f2937')
    ))
    
    fig_devis.add_trace(go.Bar(
        name=f'{fiscal_year-1}',
        x=devis_types,
        y=devis_prev,
        marker_color='#93c5fd',
        text=[f'{v}<br>({p:.1f}%)' if v > 0 else '' for v, p in zip(devis_prev, devis_prev_pct)],
        textposition='outside',
        textfont=dict(size=10, color='#1f2937')
    ))
    
    fig_devis.update_layout(
        title=f'📋 Comparaison des Devis par Type: {fiscal_year} vs {fiscal_year-1}',
        xaxis_title='Type de commande',
        yaxis_title='Nombre de devis',
        barmode='group',
        height=500,
        showlegend=True,
        font=dict(size=11),
        title_font_size=15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45)
    )
    fig_devis.update_xaxes(gridcolor='#f3f4f6')
    fig_devis.update_yaxes(gridcolor='#f3f4f6')
    
    # Ajuster l'espace en haut pour les labels
    max_devis = max(devis_current + devis_prev) if (devis_current and devis_prev) else 0
    if max_devis > 0:
        fig_devis.update_yaxes(range=[0, max_devis * 1.15])
    
    # Graphique 2 : Comparaison des commandes (seulement types de commandes)
    fig_commandes = go.Figure()
    
    fig_commandes.add_trace(go.Bar(
        name=f'{fiscal_year}',
        x=commandes_types,
        y=commandes_current,
        marker_color='#059669',
        text=[f'{v}<br>({p:.1f}%)' if v > 0 else '' for v, p in zip(commandes_current, commandes_current_pct)],
        textposition='outside',
        textfont=dict(size=10, color='#1f2937')
    ))
    
    fig_commandes.add_trace(go.Bar(
        name=f'{fiscal_year-1}',
        x=commandes_types,
        y=commandes_prev,
        marker_color='#86efac',
        text=[f'{v}<br>({p:.1f}%)' if v > 0 else '' for v, p in zip(commandes_prev, commandes_prev_pct)],
        textposition='outside',
        textfont=dict(size=10, color='#1f2937')
    ))
    
    fig_commandes.update_layout(
        title=f'✅ Comparaison des Commandes par Type: {fiscal_year} vs {fiscal_year-1}',
        xaxis_title='Type de commande',
        yaxis_title='Nombre de commandes',
        barmode='group',
        height=500,
        showlegend=True,
        font=dict(size=11),
        title_font_size=15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45)
    )
    fig_commandes.update_xaxes(gridcolor='#f3f4f6')
    fig_commandes.update_yaxes(gridcolor='#f3f4f6')
    
    # Ajuster l'espace en haut pour les labels
    max_commandes = max(commandes_current + commandes_prev) if (commandes_current and commandes_prev) else 0
    if max_commandes > 0:
        fig_commandes.update_yaxes(range=[0, max_commandes * 1.15])
    
    # Créer les tableaux détaillés
    devis_table_current = get_detailed_stats(df_devis_current, fiscal_year, 'Devis')
    devis_table_prev = get_detailed_stats(df_devis_prev, fiscal_year-1, 'Devis')
    commandes_table_current = get_detailed_stats(df_commandes_current, fiscal_year, 'Commandes')
    commandes_table_prev = get_detailed_stats(df_commandes_prev, fiscal_year-1, 'Commandes')
    
    return fig_devis, fig_commandes, (devis_table_current, devis_table_prev), (commandes_table_current, commandes_table_prev)

def create_gauge_chart(current_value, max_value, title, value_text=""):
    """Crée un graphique en jauge avec une seule valeur centrée"""
    percentage = (current_value / max_value * 100) if max_value > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b style='font-size:12px'>{title}</b><br><span style='font-size:10px; color:#6b7280'>{value_text}</span>", 
                 'font': {'size': 10}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'size': 8}},
            'bar': {'color': "#1f77b4", 'thickness': 0.25},
            'steps': [
                {'range': [0, 50], 'color': "#ffe6e6"},
                {'range': [50, 80], 'color': "#fff4e6"},
                {'range': [80, 100], 'color': "#e6f7e6"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': percentage}}))
    
    fig.update_layout(
        height=180, 
        font={'color': "darkblue", 'family': "Arial"},
        margin=dict(l=15, r=15, t=40, b=15)
    )
    
    # Une seule annotation au centre
    fig.add_annotation(
        x=0.5, y=0.35,
        xref="paper", yref="paper",
        text=f"<b style='font-size:18px; color:#1f77b4'>{percentage:.1f}%</b>",
        showarrow=False
    )
    
    return fig

def get_top_clients_by_orders(df_commandes, fiscal_year, n=10):
    """Obtient les top clients par valeur de commandes"""
    net_value_col = get_net_value_column(df_commandes, fiscal_year)
    if not net_value_col:
        return pd.DataFrame()
    
    top_clients = df_commandes.groupby(['SoldTo #', 'SoldTo Name'])[net_value_col].sum().reset_index()
    top_clients.columns = ['SoldTo #', 'Client', 'Valeur Commandes']
    return top_clients.nlargest(n, 'Valeur Commandes')[['Client', 'Valeur Commandes']]

def get_top_clients_by_quotes(df_devis, fiscal_year, n=10):
    """Obtient les top clients par valeur de devis"""
    net_value_col = get_net_value_column(df_devis, fiscal_year)
    if not net_value_col:
        return pd.DataFrame()
    
    top_clients = df_devis.groupby(['SoldTo #', 'SoldTo Name'])[net_value_col].sum().reset_index()
    top_clients.columns = ['SoldTo #', 'Client', 'Valeur Devis']
    return top_clients.nlargest(n, 'Valeur Devis')[['Client', 'Valeur Devis']]

def extract_managed_group_name(managed_group_value):
    """Extrait le nom du groupe à partir de la colonne SoldTo Managed Group"""
    if pd.isna(managed_group_value) or str(managed_group_value).strip() == '-':
        return None
    
    value_str = str(managed_group_value).strip()
    if '-' in value_str:
        # Prendre tout ce qui est avant le dernier tiret
        parts = value_str.split('-')
        if len(parts) > 1:
            return '-'.join(parts[:-1])  # Tout sauf la dernière partie
    
    return None  # Si pas de tiret, on ignore

def get_top_managed_groups_by_type(df, fiscal_year, n=15, data_type="commandes"):
    """Obtient les top groupes par valeur pour devis ou commandes normales"""
    if df.empty or 'SoldTo Managed Group' not in df.columns:
        return pd.DataFrame()
    
    # Ajouter la colonne du nom du groupe
    df_copy = df.copy()
    df_copy['Groupe_Name'] = df_copy['SoldTo Managed Group'].apply(extract_managed_group_name)
    
    # Filtrer seulement les lignes avec un groupe valide
    df_groups = df_copy[df_copy['Groupe_Name'].notna()].copy()
    
    if df_groups.empty:
        return pd.DataFrame()
    
    # Déterminer la colonne de valeur selon le type
    if data_type == "devis":
        net_value_col = get_net_value_column(df_groups, fiscal_year)
        if not net_value_col:
            return pd.DataFrame()
        value_col = net_value_col
        result_col_name = 'Valeur Devis'
    else:  # commandes
        net_value_col = get_net_value_column(df_groups, fiscal_year)
        if not net_value_col:
            return pd.DataFrame()
        value_col = net_value_col
        result_col_name = 'Valeur Commandes'
    
    # Grouper par groupe et sommer les valeurs
    top_groups = df_groups.groupby('Groupe_Name')[value_col].sum().reset_index()
    top_groups.columns = ['Groupe', result_col_name]
    
    return top_groups.nlargest(n, result_col_name)

def get_top_managed_groups_real_orders(df_real_orders, n=15):
    """Obtient les top groupes pour les commandes réelles (avec attribution)"""
    if df_real_orders.empty or 'SoldTo Managed Group' not in df_real_orders.columns:
        return pd.DataFrame()
    
    # Ajouter la colonne du nom du groupe
    df_copy = df_real_orders.copy()
    df_copy['Groupe_Name'] = df_copy['SoldTo Managed Group'].apply(extract_managed_group_name)
    
    # Filtrer seulement les lignes avec un groupe valide
    df_groups = df_copy[df_copy['Groupe_Name'].notna()].copy()
    
    if df_groups.empty:
        return pd.DataFrame()
    
    # Pour les commandes réelles, utiliser directement 'Net Value'
    if 'Net Value' not in df_groups.columns:
        return pd.DataFrame()
    
    # Grouper par groupe et sommer les valeurs
    top_groups = df_groups.groupby('Groupe_Name')['Net Value'].sum().reset_index()
    top_groups.columns = ['Groupe', 'Valeur Commandes Réelles']
    
    return top_groups.nlargest(n, 'Valeur Commandes Réelles')

@st.cache_data
def load_attribution_file(uploaded_file):
    """Charge le fichier d'attribution réelle des commandes"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Nettoyer les noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Vérifier les colonnes nécessaires
        required_cols = ['Order Document #', 'Created By Header']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes dans le fichier attribution : {', '.join(missing_cols)}")
            return None
        
        # Nettoyer les données
        df['Order Document #'] = df['Order Document #'].astype(str).str.strip()
        df['Created By Header'] = df['Created By Header'].astype(str).str.strip()
        
        # Supprimer les doublons (garder la première occurrence)
        df = df.drop_duplicates(subset=['Order Document #'], keep='first')
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier attribution : {str(e)}")
        return None

def create_real_orders_data(df_commandes, df_attribution, fiscal_year):
    """
    VERSION CORRIGÉE POUR ÊTRE IDENTIQUE AU CODE PYTHON
    """
    
    try:
        # ✅ CORRECTION 1 : Vérifier qu'on a des données
        if df_commandes.empty or df_attribution.empty:
            return pd.DataFrame()
        
        # ✅ CORRECTION 2 : Nettoyer les IDs AVANT le traitement
        df_commandes_clean = df_commandes.copy()
        df_commandes_clean['Sales Document #'] = df_commandes_clean['Sales Document #'].astype(str).str.strip()
        
        df_attribution_clean = df_attribution.copy()
        df_attribution_clean['Order Document #'] = df_attribution_clean['Order Document #'].astype(str).str.strip()
        df_attribution_clean = df_attribution_clean.drop_duplicates(subset=['Order Document #'], keep='first')
        
        # ✅ CORRECTION 3 : Utiliser la logique exacte du code Python
        # Traiter TOUS les commerciaux en même temps, puis filtrer si nécessaire
        
        # Détecter et nettoyer la colonne Net Value
        df_with_net_value = _detect_net_value_column(df_commandes_clean, fiscal_year)
        
        # Agrégation par document
        aggs = aggregate_orders_by_doc(df_with_net_value)
        
        # Attribution (INNER JOIN)
        attrib_unique = df_attribution_clean[["Order Document #","Created By Header"]].drop_duplicates("Order Document #")
        step1 = aggs.merge(attrib_unique, left_on="Sales Document #", right_on="Order Document #", how="inner")
        step1["Commercial Réel"] = step1["Created By Header"]

        # IC non présentes dans attribution
        deja = set(step1["Sales Document #"])
        step2 = aggs[(aggs["HasIC"] == True) & (~aggs["Sales Document #"].isin(deja))].copy()
        step2["Commercial Réel"] = step2["Created By Line"]

        # Union finale
        df_final = pd.concat([step1, step2], ignore_index=True)
        
        # Renommer les colonnes pour correspondre aux attentes Streamlit
        column_mapping = {
            'FY': 'Année_Fiscale',
            'FiscalMonth': 'Mois_Fiscal',
            'MonthName': 'Nom_Mois'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_final.columns:
                df_final = df_final.rename(columns={old_col: new_col})
        
        return df_final
        
    except Exception as e:
        st.error(f"Erreur dans create_real_orders_data: {str(e)}")
        return pd.DataFrame()
    
def categorize_real_commercials(df_real_orders, df_mapping_commerciaux, df_mapping_saisie):
    """Catégorise les commerciaux réels entre commerciaux et commerciaux de saisie"""
    if df_real_orders.empty:
        return [], []
    all_real_commercials = df_real_orders['Commercial Réel'].dropna().astype(str).unique()

    
    commerciaux_reels_list = []
    commerciaux_saisie_reels_list = []
    
    # Commerciaux avec objectifs
    if df_mapping_commerciaux is not None:
        commerciaux_avec_objectifs = df_mapping_commerciaux['User Sap'].astype(str).unique()
        for commercial in all_real_commercials:
            if commercial in commerciaux_avec_objectifs:
                commerciaux_reels_list.append(commercial)
    
    # Commerciaux de saisie
    if df_mapping_saisie is not None:
        commerciaux_saisie_mapping = df_mapping_saisie['User Sap'].astype(str).unique()
        for commercial in all_real_commercials:
            if commercial in commerciaux_saisie_mapping:
                commerciaux_saisie_reels_list.append(commercial)
    
    return commerciaux_reels_list, commerciaux_saisie_reels_list

def calculate_real_kpis(df_real_orders, df_objectifs=None, df_mapping=None, fiscal_year=None, period_months=None, df_objectifs_personnalises=None):
    """Calcule les KPIs pour les données réelles d'attribution"""
    kpis = {}
    
    if df_real_orders.empty:
        return {
            'nb_commandes_reelles': 0,
            'valeur_commandes_reelles': 0,
            'nb_commerciaux_reels': 0,
            'nb_clients_commandes_reels': 0,
            'moy_nb_commandes_par_commercial_reel': 0,
            'moy_valeur_commandes_par_commercial_reel': 0,
            'objectif_total_reel': 0,
            'pct_objectif_atteint_reel': 0
        }
    
    # KPIs de base
    kpis['nb_commandes_reelles'] = df_real_orders['Sales Document #'].nunique()
    kpis['valeur_commandes_reelles'] = df_real_orders['Net Value'].sum()
    kpis['nb_clients_commandes_reels'] = df_real_orders['SoldTo #'].nunique()
    
    # Nombre de commerciaux réels
    if df_mapping is not None:
        # Prendre les commerciaux du mapping qui ont des données
        commercials_with_data = df_real_orders['Commercial Réel'].dropna().astype(str).unique()

        commercials_in_mapping = df_mapping['User Sap'].astype(str).unique()
        valid_commercials = [c for c in commercials_with_data if c in commercials_in_mapping]
        nb_commerciaux_reels = len(valid_commercials)
    else:
        valid_commercials = df_real_orders['Created By Line'].dropna().astype(str).unique()
        nb_commerciaux_reels = len(valid_commercials)
    
    kpis['nb_commerciaux_reels'] = nb_commerciaux_reels
    
    # Ajuster le nombre si on filtre sur un commercial spécifique
    commercials_in_data = df_real_orders['Commercial Réel'].dropna().astype(str).unique()
    if len(commercials_in_data) == 1:
        nb_commerciaux_reels = 1
        kpis['nb_commerciaux_reels'] = 1
    
    # Moyennes par commercial
    if nb_commerciaux_reels > 0:
        kpis['moy_nb_commandes_par_commercial_reel'] = kpis['nb_commandes_reelles'] / nb_commerciaux_reels
        kpis['moy_valeur_commandes_par_commercial_reel'] = kpis['valeur_commandes_reelles'] / nb_commerciaux_reels
    else:
        kpis['moy_nb_commandes_par_commercial_reel'] = 0
        kpis['moy_valeur_commandes_par_commercial_reel'] = 0
    
    # Objectif réel (même logique que pour les KPIs normaux)
    if df_objectifs is not None and fiscal_year and period_months and nb_commerciaux_reels > 0:
        objectif_total_reel = 0
        
        # Déterminer quels commerciaux considérer
        if len(commercials_in_data) == 1:
            commercials_to_calculate = list(commercials_in_data)
        else:
            commercials_to_calculate = valid_commercials
        
        # Calculer l'objectif pour chaque commercial
        for commercial in commercials_to_calculate:
            objectif_commercial_periode = 0
            for month_fiscal in period_months:
                objectif_mois = get_objectif_priorite(commercial, month_fiscal, fiscal_year, df_objectifs_personnalises, df_objectifs)
                objectif_commercial_periode += objectif_mois
            objectif_total_reel += objectif_commercial_periode
        
        kpis['objectif_total_reel'] = objectif_total_reel
        kpis['pct_objectif_atteint_reel'] = (kpis['valeur_commandes_reelles'] / kpis['objectif_total_reel'] * 100) if kpis['objectif_total_reel'] > 0 else 0
    else:
        kpis['objectif_total_reel'] = 0
        kpis['pct_objectif_atteint_reel'] = 0
    
    return kpis

def filter_real_orders_data(df_real_orders, fiscal_year=None, month=None, period=None, commercial=None):
    """Filtre les données réelles selon les critères sélectionnés"""
    if df_real_orders.empty:
        return df_real_orders
    
    filtered_df = df_real_orders.copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        filtered_df = filtered_df[filtered_df['Created By Line'] == commercial]
    
    return filtered_df

def filter_real_orders_by_commercial_list_CORRECTED(df_real_orders, commercials_list, fiscal_year=None, month=None, period=None, commercial=None):
    """
    VERSION CORRIGÉE : Filtre sur 'Commercial Réel' (pas 'Created By Line')
    """
    if df_real_orders.empty:
        return df_real_orders
    
    # ✅ CORRECTION: Filtrer sur 'Commercial Réel' au lieu de 'Created By Line'
    filtered_df = df_real_orders[df_real_orders['Commercial Réel'].isin(commercials_list)].copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        # ✅ CORRECTION: Filtrer sur 'Commercial Réel' au lieu de 'Created By Line'
        filtered_df = filtered_df[filtered_df['Commercial Réel'] == commercial]
    
    return filtered_df

def filter_data_by_commercial_list_for_saisie(df, commercials_list, fiscal_year=None, month=None, period=None, commercial=None):
    """Filtre spécialement pour les commerciaux de saisie - VERSION CORRIGÉE"""
    # Créer une copie et convertir les types
    df_copy = df.copy()
    df_copy['Created By Line'] = df_copy['Created By Line'].astype(str)
    
    # Convertir la liste des commerciaux en string
    commercials_list_str = [str(x) for x in commercials_list]
    
    # Filtrer par la liste des commerciaux autorisés
    filtered_df = df_copy[df_copy['Created By Line'].isin(commercials_list_str)].copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        filtered_df = filtered_df[filtered_df['Created By Line'] == str(commercial)]
    
    return filtered_df

def filter_real_orders_by_commercial_list(df_real_orders, commercials_list, fiscal_year=None, month=None, period=None, commercial=None):
    """Filtre les données réelles selon la liste des commerciaux et autres critères"""
    if df_real_orders.empty:
        return df_real_orders
    
    # D'abord filtrer par la liste des commerciaux autorisés
    filtered_df = df_real_orders[df_real_orders['Created By Line'].isin(commercials_list)].copy()
    
    if fiscal_year:
        filtered_df = filtered_df[filtered_df['Année_Fiscale'] == fiscal_year]
    
    if month and not period:
        filtered_df = filtered_df[filtered_df['Mois_Fiscal'] == month]
    elif period and not month:
        start_month, end_month = period
        filtered_df = filtered_df[
            (filtered_df['Mois_Fiscal'] >= start_month) & 
            (filtered_df['Mois_Fiscal'] <= end_month)
        ]
    
    if commercial and commercial != "Tous les commerciaux":
        filtered_df = filtered_df[filtered_df['Created By Line'] == commercial]
    
    return filtered_df

def create_monthly_evolution_charts(df_devis, df_commandes, fiscal_year, vue_type=None, df_real_orders=None):
    """Crée les graphiques d'évolution mensuelle"""
    # Filtrer par année fiscale
    devis_year = df_devis[df_devis['Année_Fiscale'] == fiscal_year]
    commandes_year = df_commandes[df_commandes['Année_Fiscale'] == fiscal_year]
    
    # Identifier les mois qui ont des données
    mois_avec_donnees = sorted(list(set(
        list(devis_year['Mois_Fiscal'].unique()) + 
        list(commandes_year['Mois_Fiscal'].unique())
    )))
    
    # Si aucune donnée, retourner des graphiques vides
    if not mois_avec_donnees:
        fig_empty = go.Figure()
        fig_empty.add_annotation(
            text="Aucune donnée disponible pour cette période",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig_empty, fig_empty, fig_empty
    
    # Déterminer les colonnes de valeur
    net_value_col_devis = get_net_value_column(devis_year, fiscal_year)
    net_value_col_commandes = get_net_value_column(commandes_year, fiscal_year)
    
    # Préparation des données selon le type de vue
    if vue_type == "Commercial" and df_real_orders is not None:
        # Pour Commercial : utiliser les commandes réelles
        real_orders_year = df_real_orders[df_real_orders['Année_Fiscale'] == fiscal_year]
        mois_avec_donnees_real = sorted(list(set(
            list(devis_year['Mois_Fiscal'].unique()) + 
            list(real_orders_year['Mois_Fiscal'].unique())
        )))
        mois_avec_donnees = mois_avec_donnees_real
    else:
        real_orders_year = None

    # Agrégation mensuelle selon le type de vue
    monthly_stats = []
    for month in mois_avec_donnees:
        devis_month = devis_year[devis_year['Mois_Fiscal'] == month]
        
        if vue_type == "Commercial" and real_orders_year is not None:
            # Pour Commercial : utiliser les commandes réelles
            commandes_month = real_orders_year[real_orders_year['Mois_Fiscal'] == month]
            
            stats = {
                'Mois': month,
                'Clients devis': devis_month['SoldTo #'].nunique(),
                'Clients commandes': commandes_month['SoldTo #'].nunique(),
                'Nombre devis': devis_month['Sales Document #'].nunique(),
                'Nombre commandes': commandes_month['Sales Document #'].nunique(),
                'Valeur devis': devis_month[net_value_col_devis].sum() if net_value_col_devis else 0,
                'Valeur commandes': commandes_month['Net Value'].sum()  # Commandes réelles
            }
        elif vue_type == "Commercial de saisie":
            # Pour Commercial de saisie : seulement les commandes (pas de devis)
            commandes_month = commandes_year[commandes_year['Mois_Fiscal'] == month]
            
            stats = {
                'Mois': month,
                'Clients devis': 0,  # Pas de devis pour Commercial de saisie
                'Clients commandes': commandes_month['SoldTo #'].nunique(),
                'Nombre devis': 0,  # Pas de devis pour Commercial de saisie
                'Nombre commandes': commandes_month['Sales Document #'].nunique(),
                'Valeur devis': 0,  # Pas de devis pour Commercial de saisie
                'Valeur commandes': commandes_month[net_value_col_commandes].sum() if net_value_col_commandes else 0
            }
        else:
            # Pour Vue globale : logique normale
            commandes_month = commandes_year[commandes_year['Mois_Fiscal'] == month]
            
            stats = {
                'Mois': month,
                'Clients devis': devis_month['SoldTo #'].nunique(),
                'Clients commandes': commandes_month['SoldTo #'].nunique(),
                'Nombre devis': devis_month['Sales Document #'].nunique(),
                'Nombre commandes': commandes_month['Sales Document #'].nunique(),
                'Valeur devis': devis_month[net_value_col_devis].sum() if net_value_col_devis else 0,
                'Valeur commandes': commandes_month[net_value_col_commandes].sum() if net_value_col_commandes else 0
            }
        
        monthly_stats.append(stats)
        
    df_monthly = pd.DataFrame(monthly_stats)
    
    # Noms des mois
    month_names = ['Août', 'Sept', 'Oct', 'Nov', 'Déc', 'Jan', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin', 'Juil']
    df_monthly['Mois_Nom'] = df_monthly['Mois'].apply(lambda x: month_names[x-1])
    
    # Graphique 1 : Évolution du nombre de clients
    fig1 = go.Figure()

    # Ajouter la ligne des devis seulement si ce n'est pas Commercial de saisie
    if vue_type != "Commercial de saisie":
        label_devis = 'Clients avec devis'
        fig1.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Clients devis'], 
                                mode='lines+markers', name=label_devis, 
                                line=dict(color='#8b5cf6', width=3), marker=dict(size=8)))

    # Ajouter la ligne des commandes avec le bon label
    if vue_type == "Commercial":
        label_commandes = 'Clients avec commandes réelles'
    else:
        label_commandes = 'Clients avec commandes'

    fig1.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Clients commandes'], 
                            mode='lines+markers', name=label_commandes, 
                            line=dict(color='#b45309', width=3), marker=dict(size=8)))
    fig1.update_layout(title='Évolution mensuelle du nombre de clients',
                      xaxis_title='Mois', yaxis_title='Nombre de clients',
                      height=300,
                      font=dict(size=11),
                      title_font_size=14,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    fig1.update_xaxes(gridcolor='#f3f4f6')
    fig1.update_yaxes(gridcolor='#f3f4f6')
    
    # Graphique 2 : Évolution du nombre de documents
    fig2 = go.Figure()

    # Ajouter la ligne des devis seulement si ce n'est pas Commercial de saisie
    if vue_type != "Commercial de saisie":
        fig2.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Nombre devis'], 
                                mode='lines+markers', name='Devis', line=dict(color='#ea580c', width=2)))

    # Ajouter la ligne des commandes avec le bon label
    if vue_type == "Commercial":
        label_commandes = 'Commandes réelles'
    else:
        label_commandes = 'Commandes'

    fig2.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Nombre commandes'], 
                            mode='lines+markers', name=label_commandes, line=dict(color='#059669', width=2)))
    fig2.update_layout(title='Évolution mensuelle du nombre de commandes',
                      xaxis_title='Mois', yaxis_title='Nombre', height=300,
                      font=dict(size=11),
                      title_font_size=14,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    fig2.update_xaxes(gridcolor='#f3f4f6')
    fig2.update_yaxes(gridcolor='#f3f4f6')
    
    # Graphique 3 : Évolution des valeurs
    fig3 = go.Figure()

    # Ajouter la ligne des devis seulement si ce n'est pas Commercial de saisie
    if vue_type != "Commercial de saisie":
        fig3.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Valeur devis'], 
                                mode='lines+markers', name='Valeur devis', 
                                line=dict(color='#8b5cf6', width=3), marker=dict(size=8)))

    # Ajouter la ligne des commandes avec le bon label
    if vue_type == "Commercial":
        label_valeur = 'Valeur commandes réelles'
    else:
        label_valeur = 'Valeur commandes'

    fig3.add_trace(go.Scatter(x=df_monthly['Mois_Nom'], y=df_monthly['Valeur commandes'], 
                            mode='lines+markers', name=label_valeur, 
                            line=dict(color='#b45309', width=3), marker=dict(size=8)))
    fig3.update_layout(title='Évolution mensuelle des valeurs',
                      xaxis_title='Mois', yaxis_title='Valeur (€)',
                      height=300,
                      font=dict(size=11),
                      title_font_size=14,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    fig3.update_xaxes(gridcolor='#f3f4f6')
    fig3.update_yaxes(gridcolor='#f3f4f6')
    
    return fig1, fig2, fig3

# Initialisation des variables de session
if 'df_devis' not in st.session_state:
    st.session_state.df_devis = None
if 'df_commandes' not in st.session_state:
    st.session_state.df_commandes = None
if 'df_objectifs' not in st.session_state:
    st.session_state.df_objectifs = None
if 'df_mapping' not in st.session_state:
    st.session_state.df_mapping = None
if 'df_mapping_saisie' not in st.session_state:
    st.session_state.df_mapping_saisie = None
if 'df_objectifs_personnalises' not in st.session_state:
    st.session_state.df_objectifs_personnalises = None
if 'df_attribution' not in st.session_state:
    st.session_state.df_attribution = None    
if 'files_loaded' not in st.session_state:
    st.session_state.files_loaded = False

# Interface principale - Titre centré et stylisé
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 50%, #1e3a5f 100%);
                padding: 15px 30px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
                border: 2px solid #c9a961;
                margin: 10px 0 25px 0;
                width: 100%;'>
        <h1 style='color: white;
                   font-size: 32px;
                   font-weight: 700;
                   margin: 0;
                   text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
                   letter-spacing: 1px;'>
            🎯 Dashboard Performance Commerciale
        </h1>
        <h4 style='color: #c9a961;
                  font-size: 16px;
                  margin-top: 8px;
                  font-weight: 400;
                  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);'>
            Analyse des Conversions Devis → Commandes
        </h4>
    </div>
    """, unsafe_allow_html=True)

# Si les fichiers ne sont pas encore chargés, afficher la page d'import
if not st.session_state.files_loaded:
    st.markdown("#### 📁 Chargement des fichiers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📄 Fichier des devis *")
        devis_file = st.file_uploader("Sélectionnez votre fichier Excel", type=['xlsx', 'xls'], key="devis_uploader", label_visibility="collapsed")
        
        st.markdown("##### 🎯 Fichier des objectifs *")
        objectifs_file = st.file_uploader("Fichier avec les objectifs mensuels", type=['xlsx', 'xls'], key="objectifs_uploader", label_visibility="collapsed")
        
        st.markdown("##### 👥 Fichier mapping commerciaux *")
        mapping_file = st.file_uploader("Fichier avec les noms des commerciaux", type=['xlsx', 'xls'], key="mapping_uploader", label_visibility="collapsed")
    
    with col2:
        st.markdown("##### 📋 Fichier des commandes *")
        commandes_file = st.file_uploader("Sélectionnez votre fichier Excel", type=['xlsx', 'xls'], key="commandes_uploader", label_visibility="collapsed")
        
        st.markdown("##### 📝 Fichier mapping commerciaux de saisie *")
        mapping_saisie_file = st.file_uploader("Fichier avec les noms des commerciaux de saisie", type=['xlsx', 'xls'], key="mapping_saisie_uploader", label_visibility="collapsed")
        
        st.markdown("##### 🎯 Fichier objectifs personnalisés")
        objectifs_personnalises_file = st.file_uploader("Fichier avec les objectifs personnalisés par commercial", type=['xlsx', 'xls'], key="objectifs_personnalises_uploader", label_visibility="collapsed")
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("##### 🔍 Fichier attribution réelle commandes *")
        attribution_file = st.file_uploader("Fichier avec l'attribution réelle des commandes aux commerciaux", type=['xlsx', 'xls'], key="attribution_uploader", label_visibility="collapsed")
    
    # Vérifier que tous les fichiers sont chargés
    if devis_file and commandes_file and objectifs_file and mapping_file and mapping_saisie_file and objectifs_personnalises_file and attribution_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                with st.spinner("⏳ Traitement des données en cours..."):
                    # Charger les fichiers
                    df_devis = load_and_process_file(devis_file)
                    
                    df_commandes = load_and_process_file(commandes_file)
                    
                    df_objectifs = load_objectifs_file(objectifs_file)
                    
                    df_mapping = load_commerciaux_mapping(mapping_file)
                    
                    df_mapping_saisie = load_commerciaux_saisie_mapping(mapping_saisie_file)

                    df_objectifs_personnalises = load_objectifs_personnalises(objectifs_personnalises_file)

                    df_attribution = load_attribution_file(attribution_file)

                    if df_devis is not None and df_commandes is not None and df_objectifs is not None and df_mapping is not None and df_mapping_saisie is not None and df_objectifs_personnalises is not None and df_attribution is not None:
                        st.session_state.df_devis = df_devis
                        st.session_state.df_commandes = df_commandes
                        st.session_state.df_objectifs = df_objectifs
                        st.session_state.df_mapping = df_mapping
                        st.session_state.df_mapping_saisie = df_mapping_saisie
                        st.session_state.df_objectifs_personnalises = df_objectifs_personnalises
                        st.session_state.df_attribution = df_attribution

                        st.session_state.files_loaded = True

                        st.success("✅ Tous les fichiers ont été chargés avec succès!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors du chargement d'un ou plusieurs fichiers. Veuillez vérifier vos fichiers et réessayer.")
    else:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; color: #6b7280;'>
            <p style='font-size: 1.1rem;'>👆 Veuillez sélectionner tous les fichiers Excel pour commencer</p>
            <p style='font-size: 0.9rem; margin-top: 1rem; color: #dc3545;'>* Tous les fichiers sont obligatoires</p>
        </div>
        """, unsafe_allow_html=True)

# Si les fichiers sont chargés, afficher l'interface principale
else:
    df_devis = st.session_state.df_devis
    df_commandes = st.session_state.df_commandes
    df_objectifs = st.session_state.df_objectifs
    df_mapping = st.session_state.df_mapping
    df_mapping_saisie = st.session_state.df_mapping_saisie
    df_objectifs_personnalises = st.session_state.df_objectifs_personnalises
    df_attribution = st.session_state.df_attribution

    # Calculer l'année fiscale actuelle
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year

    # Utiliser la même logique que parse_date_column
    current_fiscal_year = current_year + 1 if current_month >= 8 else current_year

    # Calcul du mois fiscal actuel
    current_fiscal_month = ((current_month - 8) % 12) + 1
    current_fiscal_month = current_month - 7 if current_month >= 8 else current_month + 5
    
    # Catégoriser les commerciaux
    commerciaux_list, commerciaux_saisie_list = categorize_commercials(
        df_devis, df_commandes, df_mapping, df_mapping_saisie
    )
    
    
    # Sidebar pour les filtres
    with st.sidebar:
        # Bouton pour charger de nouveaux fichiers
        if st.button("📁 Nouveaux fichiers", use_container_width=True):
            st.session_state.files_loaded = False
            st.session_state.df_devis = None
            st.session_state.df_commandes = None
            st.session_state.df_objectifs = None
            st.session_state.df_mapping = None
            st.session_state.df_mapping_saisie = None
            st.session_state.df_objectifs_personnalises = None

            st.rerun()
        
        st.divider()
        
        # Choix entre Vue globale, Commercial et Commercial de saisie
        vue_type = st.radio(
            "Type de vue", 
            ["Vue globale", "Commercial", "Commercial de saisie"],
            help="Vue globale: tous les commerciaux | Commercial: avec objectifs | Commercial de saisie: sans objectifs"
        )
        
        # Année fiscale (obligatoire)
        fiscal_years = sorted(list(set(
            list(df_devis['Année_Fiscale'].dropna().astype(int).unique()) + 
            list(df_commandes['Année_Fiscale'].dropna().astype(int).unique())
        )))
        
        selected_fiscal_year = st.selectbox(
            "Année fiscale", 
            fiscal_years,
            index=fiscal_years.index(current_fiscal_year) if current_fiscal_year in fiscal_years else 0
        )
        # Création des données réelles (maintenant que selected_fiscal_year est défini)
        df_real_orders = create_real_orders_data(df_commandes, df_attribution, selected_fiscal_year)
        df_real_orders_prev = create_real_orders_data(df_commandes, df_attribution, selected_fiscal_year - 1)

        # Catégoriser les commerciaux réels
        commerciaux_reels_list, commerciaux_saisie_reels_list = categorize_real_commercials(
            df_real_orders, df_mapping, df_mapping_saisie
)
        
        # Choix entre Mois ou Période
        filter_type = st.radio("Type de filtre temporel", ["Mois", "Période"])
        
        selected_month = None
        selected_period = None
        
        if filter_type == "Mois":
            months = list(range(1, 13))
            month_names = ['Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre', 
                          'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet']
            selected_month = st.selectbox("Mois", [None] + months, 
                                        format_func=lambda x: "Tous les mois" if x is None else month_names[x-1])
        else:
            # Sélection de période avec slider
            st.markdown("**Sélection de période personnalisée**")
            month_names = ['Août', 'Sept', 'Oct', 'Nov', 'Déc', 'Jan', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin', 'Juil']
            
            # Slider pour sélectionner la plage de mois
            period_range = st.slider(
                "Période",
                min_value=1,
                max_value=12,
                value=(1, 12),  # Par défaut toute l'année
                format="%d",
                help="Glissez pour sélectionner la période désirée"
            )
            
            # Afficher la période sélectionnée
            start_month_name = month_names[period_range[0]-1]
            end_month_name = month_names[period_range[1]-1]
            st.info(f"📅 Période sélectionnée: **{start_month_name}** à **{end_month_name}**")
            
            selected_period = period_range
        
        # Commercial (seulement si on a choisi Commercial ou Commercial de saisie)
        selected_commercial = "Tous les commerciaux"
        
        if vue_type == "Commercial":
            if commerciaux_list:
                commercial_options = ["Tous les commerciaux"] + commerciaux_list
                
                # Afficher les noms complets si mapping disponible
                commercial_display = {"Tous les commerciaux": "Tous les commerciaux"}
                for commercial in commerciaux_list:
                    nom_complet = get_commercial_name(commercial, df_mapping)
                    commercial_display[commercial] = nom_complet
                
                selected_commercial = st.selectbox(
                    "Commercial", 
                    commercial_options,
                    format_func=lambda x: commercial_display.get(x, x)
                )
            else:
                st.warning("Aucun commercial avec objectifs trouvé")
                
        elif vue_type == "Commercial de saisie":
            if commerciaux_saisie_list:
                commercial_options = ["Tous les commerciaux"] + commerciaux_saisie_list
                
                # Afficher les noms complets si mapping disponible
                commercial_display = {"Tous les commerciaux": "Tous les commerciaux"}
                for commercial in commerciaux_saisie_list:
                    nom_complet = get_commercial_name(commercial, df_mapping_saisie)
                    commercial_display[commercial] = nom_complet
                
                selected_commercial = st.selectbox(
                    "Commercial de saisie", 
                    commercial_options,
                    format_func=lambda x: commercial_display.get(x, x)
                )
            else:
                st.warning("Aucun commercial de saisie trouvé")
    
    # Déterminer la période pour les calculs
    if selected_month:
        period_months = [selected_month]
    elif selected_period:
        if selected_period[0] <= selected_period[1]:
            period_months = list(range(selected_period[0], selected_period[1] + 1))
        else:
            # Cas où la période traverse l'année (ex: novembre à février)
            period_months = list(range(selected_period[0], 13)) + list(range(1, selected_period[1] + 1))
    else:
        # Si aucun filtre, prendre jusqu'au mois actuel pour l'année en cours
        if selected_fiscal_year == current_fiscal_year:
            period_months = list(range(1, current_fiscal_month + 1))
        else:
            period_months = list(range(1, 13))
    
    # Filtrage des données selon le type de vue
    if vue_type == "Vue globale":
        # Vue globale : tous les commerciaux
        df_devis_filtered = filter_data(df_devis, selected_fiscal_year, selected_month, selected_period, None)
        df_commandes_filtered = filter_data(df_commandes, selected_fiscal_year, selected_month, selected_period, None)
        
        # Données pour comparaison année précédente
        df_devis_prev = filter_data(df_devis, selected_fiscal_year - 1, selected_month, selected_period, None)
        df_commandes_prev = filter_data(df_commandes, selected_fiscal_year - 1, selected_month, selected_period, None)
        
        # Données réelles
        df_real_orders_filtered = filter_real_orders_data(df_real_orders, selected_fiscal_year, selected_month, selected_period, None)
        df_real_orders_prev_filtered = filter_real_orders_data(df_real_orders_prev, selected_fiscal_year - 1, selected_month, selected_period, None)
        
    elif vue_type == "Commercial":
        # Vue commerciaux avec objectifs
        df_devis_filtered = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year, selected_month, selected_period, selected_commercial)
        df_commandes_filtered = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year, selected_month, selected_period, selected_commercial)
        
        # Données pour comparaison année précédente
        df_devis_prev = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year - 1, selected_month, selected_period, selected_commercial)
        df_commandes_prev = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year - 1, selected_month, selected_period, selected_commercial)
        
        # Données réelles
        df_real_orders_filtered = filter_real_orders_by_commercial_list_CORRECTED(df_real_orders, commerciaux_reels_list, selected_fiscal_year, selected_month, selected_period, selected_commercial)
        df_real_orders_prev_filtered = filter_real_orders_by_commercial_list_CORRECTED(df_real_orders_prev, commerciaux_reels_list, selected_fiscal_year - 1, selected_month, selected_period, selected_commercial)
            

    else:  # vue_type == "Commercial de saisie"
        df_devis_filtered = pd.DataFrame()  # ← VIDE : pas de devis pour commerciaux de saisie
        df_commandes_filtered = filter_data_by_commercial_list_for_saisie(
            df_commandes, commerciaux_saisie_list, selected_fiscal_year, 
            selected_month, selected_period, selected_commercial
        )
        
        # Données pour comparaison année précédente
        df_devis_prev = pd.DataFrame()  # ← VIDE : pas de devis pour commerciaux de saisie
        df_commandes_prev = filter_data_by_commercial_list_for_saisie(
            df_commandes, commerciaux_saisie_list, selected_fiscal_year - 1, 
            selected_month, selected_period, selected_commercial
        )
        
        # Pour Commercial de saisie : PAS de données réelles
        df_real_orders_filtered = pd.DataFrame()
        df_real_orders_prev_filtered = pd.DataFrame()
    
    # Calcul des KPIs pour l'année actuelle et précédente
    if vue_type == "Commercial":
        # Utiliser les objectifs seulement pour les commerciaux
        kpis_current = calculate_kpis(df_devis_filtered, df_commandes_filtered, df_objectifs, df_mapping, selected_fiscal_year, period_months, df_objectifs_personnalises)
        kpis_prev = calculate_kpis(df_devis_prev, df_commandes_prev, df_objectifs, df_mapping, selected_fiscal_year - 1, period_months, df_objectifs_personnalises)
        kpis_real_current = calculate_real_kpis(df_real_orders_filtered, df_objectifs, df_mapping, selected_fiscal_year, period_months, df_objectifs_personnalises)
        kpis_real_prev = calculate_real_kpis(df_real_orders_prev_filtered, df_objectifs, df_mapping, selected_fiscal_year - 1, period_months, df_objectifs_personnalises)
        current_mapping = df_mapping
    elif vue_type == "Commercial de saisie":
        # ✅ CORRECTION : Utiliser la fonction spécifique pour commerciaux de saisie
        kpis_current = calculate_kpis_saisie_only(df_commandes_filtered, df_mapping_saisie, selected_fiscal_year, period_months)
        kpis_prev = calculate_kpis_saisie_only(df_commandes_prev, df_mapping_saisie, selected_fiscal_year - 1, period_months)
        
        # Pas de données réelles pour commerciaux de saisie
        kpis_real_current = {
            'nb_commandes_reelles': 0, 'valeur_commandes_reelles': 0, 'nb_commerciaux_reels': 0,
            'nb_clients_commandes_reels': 0, 'moy_nb_commandes_par_commercial_reel': 0,
            'moy_valeur_commandes_par_commercial_reel': 0, 'objectif_total_reel': 0, 'pct_objectif_atteint_reel': 0
        }
        kpis_real_prev = kpis_real_current.copy()
        current_mapping = df_mapping_saisie
    else:
        # Vue globale
        kpis_current = calculate_kpis(df_devis_filtered, df_commandes_filtered, None, df_mapping, selected_fiscal_year, period_months)
        kpis_prev = calculate_kpis(df_devis_prev, df_commandes_prev, None, df_mapping, selected_fiscal_year - 1, period_months)
        kpis_real_current = calculate_real_kpis(df_real_orders_filtered, None, df_mapping, selected_fiscal_year, period_months)
        kpis_real_prev = calculate_real_kpis(df_real_orders_prev_filtered, None, df_mapping, selected_fiscal_year - 1, period_months)
        current_mapping = df_mapping


    # Message d'encouragement personnalisé pour les commerciaux
    if selected_commercial != "Tous les commerciaux":
        if vue_type == "Commercial":
            commercial_name = get_commercial_name(selected_commercial, df_mapping)
            encouragements = [
                "💪 Vous êtes sur la bonne voie, continuez comme ça !",
                "🌟 Votre détermination fait la différence !",
                "🚀 Chaque défi est une opportunité de briller !",
                "⭐ Votre talent et votre persévérance portent leurs fruits !",
                "🎯 Vous avez tout ce qu'il faut pour atteindre vos objectifs !",
                "🔥 Votre énergie positive inspire toute l'équipe !",
                "💎 Vous transformez chaque opportunité en succès !",
                "🌈 Votre sourire et votre professionnalisme font la différence !",
                "🏆 Chaque client satisfait est la preuve de votre excellence.",
                "📈 Vos résultats sont à l’image de votre engagement : impressionnants !",
                "💡 Votre créativité ouvre de nouvelles portes chaque jour.",
                "🛠️ Vous construisez des relations solides qui dureront dans le temps.",
                "🎉 Votre passion est contagieuse, elle entraîne toute l’équipe vers le haut.",
                "📊 Vous transformez les objectifs en réussites concrètes.",
                "🌍 Votre travail rayonne bien au-delà de votre portefeuille client.",
                "💥 Vous avez cette force tranquille qui fait toute la différence.",
                "🎯 Vous visez juste, et ça se voit dans vos résultats.",
                "🚀 Grâce à vous, Signals continue de monter en puissance."
            ]

            
            
            encouragement_aleatoire = random.choice(encouragements)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px 20px;
                        border-radius: 10px;
                        text-align: center;
                        margin: 20px 0;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                <h3 style='color: white; margin: 0; font-size: 22px;'>
                    👋 Bonjour {commercial_name} !
                </h3>
                <p style='color: #f0f8ff; margin: 10px 0 0 0; font-size: 16px; font-style: italic;'>
                    {encouragement_aleatoire}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif vue_type == "Commercial de saisie":
            commercial_name = get_commercial_name(selected_commercial, df_mapping_saisie)
            encouragements_saisie = [
                "💪 Votre travail de saisie est essentiel à notre succès !",
                "🌟 Chaque commande saisie contribue à notre réussite collective !",
                "🚀 Votre précision et votre efficacité font toute la différence !",
                "⭐ Vous êtes un maillon indispensable de notre chaîne de succès !",
                "🎯 Votre rigueur dans la saisie nous aide à atteindre nos objectifs !",
                "🔥 Votre engagement quotidien est remarquable !",
                "💎 La qualité de votre travail se reflète dans nos résultats !",
                "🌈 Votre contribution est précieuse et appréciée par tous !",
                "🖋️ Vous inscrivez la réussite de Signals, ligne après ligne.",
                "📑 Chaque saisie impeccable nous rapproche de l’excellence.",
                "🔍 Votre attention au détail est un atout inestimable.",
                "📦 Grâce à vous, chaque commande est traitée avec soin et rapidité.",
                "⏱️ Votre efficacité fait gagner un temps précieux à toute l’équipe.",
                "💼 Votre professionnalisme garantit la fluidité de nos opérations.",
                "🛡️ Vous veillez à la qualité de nos données, et c’est inestimable.",
                "📋 Vous faites de la précision une véritable signature personnelle.",
                "🌟 Vous êtes la colonne vertébrale invisible de notre réussite.",
                "🧩 Chaque information que vous traitez complète notre puzzle commun."
            ]

            
            encouragement_aleatoire = random.choice(encouragements_saisie)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                        padding: 15px 20px;
                        border-radius: 10px;
                        text-align: center;
                        margin: 20px 0;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                <h3 style='color: white; margin: 0; font-size: 22px;'>
                    👋 Bonjour {commercial_name} !
                </h3>
                <p style='color: #f0fff0; margin: 10px 0 0 0; font-size: 16px; font-style: italic;'>
                    {encouragement_aleatoire}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Affichage des KPIs principaux avec comparaison
    st.markdown("### 📊 Vue d'ensemble des performances")

    # Section 1 : Volumes et Valeurs avec comparaison (seulement pour Vue globale et Commercial)
    if vue_type != "Commercial de saisie":
        st.markdown("""
        <div class="kpi-section">
            <div class="kpi-title" style="color: #374151;">📊 Volumes et Valeurs - Comparaison avec l'année précédente</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Calculer la moyenne globale une seule fois si un commercial est sélectionné
    if selected_commercial != "Tous les commerciaux":
        # Calculer la moyenne globale (tous commerciaux confondus dans la catégorie)
        if vue_type == "Commercial":
            all_devis_filtered = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year, selected_month, selected_period, None)
            all_commandes_filtered = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year, selected_month, selected_period, None)
            all_commercials = commerciaux_list
        else:  # Commercial de saisie
            all_devis_filtered = filter_data_by_commercial_list(df_devis, commerciaux_saisie_list, selected_fiscal_year, selected_month, selected_period, None)
            all_commandes_filtered = filter_data_by_commercial_list(df_commandes, commerciaux_saisie_list, selected_fiscal_year, selected_month, selected_period, None)
            all_commercials = commerciaux_saisie_list
        
        nb_all_commercials = len(all_commercials)
        
        if nb_all_commercials > 0:
            total_devis_global = all_devis_filtered['Sales Document #'].nunique()
            total_commandes_global = all_commandes_filtered['Sales Document #'].nunique()
            moyenne_globale_devis = total_devis_global / nb_all_commercials
            moyenne_globale_commandes = total_commandes_global / nb_all_commercials
        else:
            moyenne_globale_devis = 0
            moyenne_globale_commandes = 0
        
    if vue_type == "Vue globale":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_devis = kpis_current['nb_devis'] - kpis_prev['nb_devis']
            delta_pct_devis = (delta_devis / kpis_prev['nb_devis'] * 100) if kpis_prev['nb_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Nombre de devis</div>
                <div class="metric-value" style="color: #4f46e5;">{kpis_current['nb_devis']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_devis']:,} ({delta_pct_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_commandes = kpis_current['nb_commandes'] - kpis_prev['nb_commandes']
            delta_pct_commandes = (delta_commandes / kpis_prev['nb_commandes'] * 100) if kpis_prev['nb_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Nombre de commandes</div>
                <div class="metric-value" style="color: #059669;">{kpis_current['nb_commandes']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_commandes']:,} ({delta_pct_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            delta_val_devis = kpis_current['valeur_devis'] - kpis_prev['valeur_devis']
            delta_pct_val_devis = (delta_val_devis / kpis_prev['valeur_devis'] * 100) if kpis_prev['valeur_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Valeur totale devis</div>
                <div class="metric-value" style="color: #7c3aed;">{kpis_current['valeur_devis']:,.0f} €</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['valeur_devis']:,.0f} € ({delta_pct_val_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            delta_val_commandes = kpis_current['valeur_commandes'] - kpis_prev['valeur_commandes']
            delta_pct_val_commandes = (delta_val_commandes / kpis_prev['valeur_commandes'] * 100) if kpis_prev['valeur_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Valeur totale commandes</div>
                <div class="metric-value" style="color: #dc2626;">{kpis_current['valeur_commandes']:,.0f} €</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['valeur_commandes']:,.0f} € ({delta_pct_val_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
    elif vue_type == "Commercial":
        # Pour Commercial : seulement 2 KPIs (Nombre de devis et Valeur totale devis)
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col2:
            delta_devis = kpis_current['nb_devis'] - kpis_prev['nb_devis']
            delta_pct_devis = (delta_devis / kpis_prev['nb_devis'] * 100) if kpis_prev['nb_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Nombre de devis</div>
                <div class="metric-value" style="color: #4f46e5;">{kpis_current['nb_devis']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_devis']:,} ({delta_pct_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            delta_val_devis = kpis_current['valeur_devis'] - kpis_prev['valeur_devis']
            delta_pct_val_devis = (delta_val_devis / kpis_prev['valeur_devis'] * 100) if kpis_prev['valeur_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Valeur totale devis</div>
                <div class="metric-value" style="color: #7c3aed;">{kpis_current['valeur_devis']:,.0f} €</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['valeur_devis']:,.0f} € ({delta_pct_val_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
    # Section spéciale pour Commercial de saisie - Seulement les commandes
    if vue_type == "Commercial de saisie":
        st.markdown("""
        <div class="kpi-section">
            <div class="kpi-title" style="color: #374151;">📊 Performance Commandes - Comparaison avec l'année précédente</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta_commandes = kpis_current['nb_commandes'] - kpis_prev['nb_commandes']
            delta_pct_commandes = (delta_commandes / kpis_prev['nb_commandes'] * 100) if kpis_prev['nb_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Nombre de commandes</div>
                <div class="metric-value" style="color: #059669;">{kpis_current['nb_commandes']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_commandes']:,} ({delta_pct_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_val_commandes = kpis_current['valeur_commandes'] - kpis_prev['valeur_commandes']
            delta_pct_val_commandes = (delta_val_commandes / kpis_prev['valeur_commandes'] * 100) if kpis_prev['valeur_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Valeur totale commandes</div>
                <div class="metric-value" style="color: #dc2626;">{kpis_current['valeur_commandes']:,.0f} €</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['valeur_commandes']:,.0f} € ({delta_pct_val_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
    # Section Attribution Réelle (seulement pour Vue globale et Commercial)
    if vue_type != "Commercial de saisie":
        st.markdown("""
        <div class="kpi-section" style="background: #f0f9ff; border-left: 4px solid #3b82f6;">
            <div class="kpi-title" style="color: #1e40af;">🎯 Attribution Réelle des Commandes - Comparaison avec l'année précédente</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_real_commandes = kpis_real_current['nb_commandes_reelles'] - kpis_real_prev['nb_commandes_reelles']
            delta_pct_real_commandes = (delta_real_commandes / kpis_real_prev['nb_commandes_reelles'] * 100) if kpis_real_prev['nb_commandes_reelles'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container" style="border-left: 3px solid #3b82f6;">
                <div class="metric-label">Commandes réelles</div>
                <div class="metric-value" style="color: #1e40af;">{kpis_real_current['nb_commandes_reelles']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_real_prev['nb_commandes_reelles']:,} ({delta_pct_real_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            delta_real_valeur = kpis_real_current['valeur_commandes_reelles'] - kpis_real_prev['valeur_commandes_reelles']
            delta_pct_real_valeur = (delta_real_valeur / kpis_real_prev['valeur_commandes_reelles'] * 100) if kpis_real_prev['valeur_commandes_reelles'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container" style="border-left: 3px solid #059669;">
                <div class="metric-label">Devis Convertis</div>
                <div class="metric-value" style="color: #047857;">{kpis_real_current['valeur_commandes_reelles']:,.0f} €</div>
                <small style="color: #6b7280;">N-1: {kpis_real_prev['valeur_commandes_reelles']:,.0f} € ({delta_pct_real_valeur:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            delta_real_clients = kpis_real_current['nb_clients_commandes_reels'] - kpis_real_prev['nb_clients_commandes_reels']
            delta_pct_real_clients = (delta_real_clients / kpis_real_prev['nb_clients_commandes_reels'] * 100) if kpis_real_prev['nb_clients_commandes_reels'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container" style="border-left: 3px solid #7c3aed;">
                <div class="metric-label">Clients réels</div>
                <div class="metric-value" style="color: #6d28d9;">{kpis_real_current['nb_clients_commandes_reels']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_real_prev['nb_clients_commandes_reels']:,} ({delta_pct_real_clients:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            if vue_type == "Commercial" and kpis_real_current['objectif_total_reel'] > 0:
                delta_real_objectif = kpis_real_current['pct_objectif_atteint_reel'] - kpis_real_prev['pct_objectif_atteint_reel']
                objectif_color = "#059669" if kpis_real_current['pct_objectif_atteint_reel'] >= 90 else "#dc2626" if kpis_real_current['pct_objectif_atteint_reel'] < 70 else "#d97706"
                st.markdown(f"""
                <div class="metric-container" style="border-left: 3px solid #dc2626;">
                    <div class="metric-label">% Objectif réel atteint</div>
                    <div class="metric-value" style="color: {objectif_color};">{kpis_real_current['pct_objectif_atteint_reel']:.1f}%</div>
                    <small style="color: #6b7280;">N-1: {kpis_real_prev['pct_objectif_atteint_reel']:.1f}% (Écart: {delta_real_objectif:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                delta_real_moy = kpis_real_current['moy_valeur_commandes_par_commercial_reel'] - kpis_real_prev['moy_valeur_commandes_par_commercial_reel']
                delta_pct_real_moy = (delta_real_moy / kpis_real_prev['moy_valeur_commandes_par_commercial_reel'] * 100) if kpis_real_prev['moy_valeur_commandes_par_commercial_reel'] > 0 else 0
                st.markdown(f"""
                <div class="metric-container" style="border-left: 3px solid #dc2626;">
                    <div class="metric-label">Moy. Valeur/Commercial réel</div>
                    <div class="metric-value" style="color: #dc2626;">{kpis_real_current['moy_valeur_commandes_par_commercial_reel']:,.0f} €</div>
                    <small style="color: #6b7280;">N-1: {kpis_real_prev['moy_valeur_commandes_par_commercial_reel']:,.0f} € ({delta_pct_real_moy:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)

    # Section 2 : Jauges de Performance
    if vue_type == "Vue globale":
        st.markdown("""
        <div class="kpi-section" style="background: #f3f4f6;">
            <div class="kpi-title" style="color: #374151;">⚡ Jauges de Performance</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)  # ← CHANGÉ DE 3 À 2 COLONNES
        
        with col1:
            # Jauge de conversion valeur (commandes/devis)
            conversion_pct = (kpis_current['valeur_commandes'] / kpis_current['valeur_devis'] * 100) if kpis_current['valeur_devis'] > 0 else 0
            value_text = f"{kpis_current['valeur_commandes']:,.0f}€ / {kpis_current['valeur_devis']:,.0f}€"
            fig_gauge1 = create_gauge_chart(conversion_pct, 100, "Taux de Conversion Valeur", value_text)
            st.plotly_chart(fig_gauge1, use_container_width=True)

        with col2:  # ← CHANGÉ DE col3 À col2
            # Jauge d'attribution réelle
            if kpis_real_current['valeur_commandes_reelles'] > 0:
                ratio_real = (kpis_real_current['valeur_commandes_reelles'] / kpis_current['valeur_commandes'] * 100) if kpis_current['valeur_commandes'] > 0 else 0
                value_text = f"{kpis_real_current['valeur_commandes_reelles']:,.0f}€ / {kpis_current['valeur_commandes']:,.0f}€"
                fig_gauge3 = create_gauge_chart(ratio_real, 100, "Réel vs Saisie", value_text)
                st.plotly_chart(fig_gauge3, use_container_width=True)
        

    elif vue_type == "Commercial":
        # Pour Commercial seulement : afficher la jauge
        st.markdown("""
        <div class="kpi-section" style="background: #f3f4f6;">
            <div class="kpi-title" style="color: #374151;">⚡ Jauges de Performance</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])  # ← CENTRER LA JAUGE

        with col2:  # ← UNE SEULE JAUGE CENTRÉE
            # Jauge d'objectif réel atteint
            if kpis_real_current['objectif_total_reel'] > 0:
                value_text = f"{kpis_real_current['valeur_commandes_reelles']:,.0f}€ / {kpis_real_current['objectif_total_reel']:,.0f}€"
                fig_gauge4 = create_gauge_chart(kpis_real_current['pct_objectif_atteint_reel'], 100, "% Objectif Réel Atteint", value_text)
            else:
                fig_gauge4 = create_gauge_chart(0, 100, "% Objectif Réel Atteint", "Objectif non défini")
            st.plotly_chart(fig_gauge4, use_container_width=True)
    
    # Pas de jauges pour Commercial de saisie
    
    # Section 3 : Autres KPIs (ajustée selon le type de vue)
    if vue_type == "Vue globale":
        st.markdown("""
        <div class="kpi-section" style="background: #f9fafb;">
            <div class="kpi-title" style="color: #374151;">💼 Clients</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta_clients_devis = kpis_current['nb_clients_devis'] - kpis_prev['nb_clients_devis']
            delta_pct_clients_devis = (delta_clients_devis / kpis_prev['nb_clients_devis'] * 100) if kpis_prev['nb_clients_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Clients avec devis</div>
                <div class="metric-value" style="color: #2563eb;">{kpis_current['nb_clients_devis']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_clients_devis']:,} ({delta_pct_clients_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_clients_commandes = kpis_current['nb_clients_commandes'] - kpis_prev['nb_clients_commandes']
            delta_pct_clients_commandes = (delta_clients_commandes / kpis_prev['nb_clients_commandes'] * 100) if kpis_prev['nb_clients_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Clients Commandes Directes</div>
                <div class="metric-value" style="color: #047857;">{kpis_current['nb_clients_commandes']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_clients_commandes']:,} ({delta_pct_clients_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
    
    elif vue_type == "Commercial":
        # Pour Commercial : KPIs orientés données réelles
        st.markdown("""
        <div class="kpi-section" style="background: #f9fafb;">
            <div class="kpi-title" style="color: #374151;">💼 Clients et Moyennes</div>
        </div>
        """, unsafe_allow_html=True)
                
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_clients_devis = kpis_current['nb_clients_devis'] - kpis_prev['nb_clients_devis']
            delta_pct_clients_devis = (delta_clients_devis / kpis_prev['nb_clients_devis'] * 100) if kpis_prev['nb_clients_devis'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Clients avec devis</div>
                <div class="metric-value" style="color: #2563eb;">{kpis_current['nb_clients_devis']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_clients_devis']:,} ({delta_pct_clients_devis:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if selected_commercial != "Tous les commerciaux":
                # Performance du commercial sélectionné vs moyenne globale
                commercial_devis = kpis_current['nb_devis']  # Nombre de devis du commercial
                delta_vs_moyenne = commercial_devis - moyenne_globale_devis
                delta_pct_vs_moyenne = (delta_vs_moyenne / moyenne_globale_devis * 100) if moyenne_globale_devis > 0 else 0
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Commercial vs Moy. Globale Devis</div>
                    <div class="metric-value" style="color: #6d28d9;">{commercial_devis:.0f}</div>
                    <small style="color: #6b7280;">Moy. globale: {moyenne_globale_devis:.1f} (Écart: {delta_pct_vs_moyenne:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Moyenne par commercial pour tous les commerciaux
                delta_moy_devis = kpis_current['moy_nb_devis_par_commercial'] - kpis_prev['moy_nb_devis_par_commercial']
                delta_pct_moy_devis = (delta_moy_devis / kpis_prev['moy_nb_devis_par_commercial'] * 100) if kpis_prev['moy_nb_devis_par_commercial'] > 0 else 0
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Moy. Devis/Commercial</div>
                    <div class="metric-value" style="color: #6d28d9;">{kpis_current['moy_nb_devis_par_commercial']:.1f}</div>
                    <small style="color: #6b7280;">N-1: {kpis_prev['moy_nb_devis_par_commercial']:.1f} ({delta_pct_moy_devis:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            # Moyenne commandes réelles par commercial
            delta_moy_real = kpis_real_current['moy_nb_commandes_par_commercial_reel'] - kpis_real_prev['moy_nb_commandes_par_commercial_reel']
            delta_pct_moy_real = (delta_moy_real / kpis_real_prev['moy_nb_commandes_par_commercial_reel'] * 100) if kpis_real_prev['moy_nb_commandes_par_commercial_reel'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Moy. Commandes Réelles/Commercial</div>
                <div class="metric-value" style="color: #047857;">{kpis_real_current['moy_nb_commandes_par_commercial_reel']:.1f}</div>
                <small style="color: #6b7280;">N-1: {kpis_real_prev['moy_nb_commandes_par_commercial_reel']:.1f} ({delta_pct_moy_real:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
    
    elif vue_type == "Commercial de saisie":
        st.markdown("""
        <div class="kpi-section" style="background: #f9fafb;">
            <div class="kpi-title" style="color: #374151;">💼 Clients Commandes</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta_clients_commandes = kpis_current['nb_clients_commandes'] - kpis_prev['nb_clients_commandes']
            delta_pct_clients_commandes = (delta_clients_commandes / kpis_prev['nb_clients_commandes'] * 100) if kpis_prev['nb_clients_commandes'] > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Clients avec commandes</div>
                <div class="metric-value" style="color: #047857;">{kpis_current['nb_clients_commandes']:,}</div>
                <small style="color: #6b7280;">N-1: {kpis_prev['nb_clients_commandes']:,} ({delta_pct_clients_commandes:+.1f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if selected_commercial != "Tous les commerciaux":
                commercial_commandes = kpis_current['nb_commandes']
                delta_vs_moyenne = commercial_commandes - moyenne_globale_commandes
                delta_pct_vs_moyenne = (delta_vs_moyenne / moyenne_globale_commandes * 100) if moyenne_globale_commandes > 0 else 0
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Commercial vs Moyenne</div>
                    <div class="metric-value" style="color: #be123c;">{commercial_commandes:.0f}</div>
                    <small style="color: #6b7280;">Moy. globale: {moyenne_globale_commandes:.1f} (Écart: {delta_pct_vs_moyenne:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                delta_moy_commandes = kpis_current['moy_nb_commandes_par_commercial'] - kpis_prev['moy_nb_commandes_par_commercial']
                delta_pct_moy_commandes = (delta_moy_commandes / kpis_prev['moy_nb_commandes_par_commercial'] * 100) if kpis_prev['moy_nb_commandes_par_commercial'] > 0 else 0
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Moy. Commandes/Commercial</div>
                    <div class="metric-value" style="color: #be123c;">{kpis_current['moy_nb_commandes_par_commercial']:.1f}</div>
                    <small style="color: #6b7280;">N-1: {kpis_prev['moy_nb_commandes_par_commercial']:.1f} ({delta_pct_moy_commandes:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Section 4 : Objectifs (seulement pour Commercial)
    if vue_type == "Commercial" and (kpis_current['objectif_total'] > 0 or selected_commercial != "Tous les commerciaux"):
        st.markdown("""
        <div class="kpi-section" style="background: #f0f9ff;">
            <div class="kpi-title" style="color: #374151;">🎯 Objectifs et Performance</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration des colonnes selon le contexte
        if selected_commercial != "Tous les commerciaux":
            # Calculer les moyennes globales pour tous les commerciaux pour la même période
            all_devis_for_avg = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year, selected_month, selected_period, None)
            all_commandes_for_avg = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year, selected_month, selected_period, None)
            all_kpis_for_avg = calculate_kpis(all_devis_for_avg, all_commandes_for_avg, df_objectifs, df_mapping, selected_fiscal_year, period_months)
            
            # Calculer les moyennes par commercial
            moyenne_globale_valeur_devis = all_kpis_for_avg['moy_valeur_devis_par_commercial']
            moyenne_globale_valeur_commandes = all_kpis_for_avg['moy_valeur_commandes_par_commercial']
            
            # Afficher 4 colonnes pour le commercial spécifique
            col1, col2, col3, col4 = st.columns(4)
            
            # KPI 1: Valeur devis vs moyenne globale
            with col1:
                valeur_devis_commercial = kpis_current['valeur_devis']
                ecart_devis = valeur_devis_commercial - moyenne_globale_valeur_devis
                ecart_pct_devis = (ecart_devis / moyenne_globale_valeur_devis * 100) if moyenne_globale_valeur_devis > 0 else 0
                color_devis = "#059669" if ecart_pct_devis >= 0 else "#dc2626"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Valeur Devis vs Moyenne</div>
                    <div class="metric-value" style="color: {color_devis};">{valeur_devis_commercial:,.0f} €</div>
                    <small style="color: #6b7280;">Moy. globale: {moyenne_globale_valeur_devis:,.0f} € ({ecart_pct_devis:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
            
            # KPI 2: Valeur commandes vs moyenne globale
            with col2:
                valeur_commandes_commercial = kpis_current['valeur_commandes']
                ecart_commandes = valeur_commandes_commercial - moyenne_globale_valeur_commandes
                ecart_pct_commandes = (ecart_commandes / moyenne_globale_valeur_commandes * 100) if moyenne_globale_valeur_commandes > 0 else 0
                color_commandes = "#059669" if ecart_pct_commandes >= 0 else "#dc2626"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Valeur Commandes vs Moyenne</div>
                    <div class="metric-value" style="color: {color_commandes};">{valeur_commandes_commercial:,.0f} €</div>
                    <small style="color: #6b7280;">Moy. globale: {moyenne_globale_valeur_commandes:,.0f} € ({ecart_pct_commandes:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
            
            # KPI 3: Objectif total période (si disponible)
            with col3:
                if kpis_current['objectif_total'] > 0:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Objectif période</div>
                        <div class="metric-value" style="color: #4338ca;">{kpis_current['objectif_total']:,.0f} €</div>
                        <small style="color: #6b7280;">{len(period_months)} mois</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Objectif période</div>
                        <div class="metric-value" style="color: #6b7280;">Non défini</div>
                        <small style="color: #6b7280;">{len(period_months)} mois</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # KPI 4: % Objectif atteint (si disponible)
            with col4:
                if kpis_current['objectif_total'] > 0:
                    objectif_color = "#059669" if kpis_real_current['pct_objectif_atteint_reel'] >= 90 else "#dc2626" if kpis_real_current['pct_objectif_atteint_reel'] < 70 else "#d97706"
                    ecart_pct = kpis_real_current['pct_objectif_atteint_reel'] - kpis_real_prev['pct_objectif_atteint_reel']
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">% Objectif Réel Atteint</div>
                        <div class="metric-value" style="color: {objectif_color};">{kpis_real_current['pct_objectif_atteint_reel']:.1f}%</div>
                        <small style="color: #6b7280;">N-1: {kpis_real_prev['pct_objectif_atteint_reel']:.1f}% (Écart: {ecart_pct:+.1f}%)</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">% Objectif atteint</div>
                        <div class="metric-value" style="color: #6b7280;">N/A</div>
                        <small style="color: #6b7280;">Objectif non défini</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Configuration originale pour tous les commerciaux
            col1, col2, col3, col4 = st.columns(4)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Objectif total période</div>
                    <div class="metric-value" style="color: #4338ca;">{kpis_current['objectif_total']:,.0f} €</div>
                    <small style="color: #6b7280;">{len(period_months)} mois × {kpis_current['nb_commerciaux']} commerciaux</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                objectif_color = "#059669" if kpis_real_current['pct_objectif_atteint_reel'] >= 90 else "#dc2626" if kpis_real_current['pct_objectif_atteint_reel'] < 70 else "#d97706"
                ecart_pct = kpis_real_current['pct_objectif_atteint_reel'] - kpis_real_prev['pct_objectif_atteint_reel']
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">% Objectif Réel Atteint</div>
                    <div class="metric-value" style="color: {objectif_color};">{kpis_real_current['pct_objectif_atteint_reel']:.1f}%</div>
                    <small style="color: #6b7280;">N-1: {kpis_real_prev['pct_objectif_atteint_reel']:.1f}% (Écart: {ecart_pct:+.1f}%)</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Section 5 : Types de commandes avec comparaison
    st.markdown("""
    <div class="kpi-section" style="background: #fef9c3;">
        <div class="kpi-title" style="color: #374151;">📞 Répartition par Type de Commande - Comparaison Annuelle</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Créer les graphiques de comparaison
    if vue_type == "Commercial":
        # Pour Commercial : vérifier si les données réelles ont la colonne Purchase Order Type
        # Sinon utiliser les données normales de commandes
        if not df_real_orders_filtered.empty and 'Purchase Order Type' in df_real_orders_filtered.columns:
            result = create_order_type_comparison_charts(
                df_devis_filtered, 
                df_real_orders_filtered, 
                df_devis_prev, 
                df_real_orders_prev_filtered, 
                selected_fiscal_year
            )
        else:
            # Fallback : utiliser les données normales de commandes
            result = create_order_type_comparison_charts(
                df_devis_filtered, 
                df_commandes_filtered, 
                df_devis_prev, 
                df_commandes_prev, 
                selected_fiscal_year
            )
    elif vue_type == "Commercial de saisie":
        # Pour Commercial de saisie : seulement commandes
        result = create_order_type_comparison_charts(
            pd.DataFrame(), 
            df_commandes_filtered, 
            pd.DataFrame(), 
            df_commandes_prev, 
            selected_fiscal_year
        )
    else:
        # Pour Vue globale : données normales
        result = create_order_type_comparison_charts(
            df_devis_filtered, 
            df_commandes_filtered, 
            df_devis_prev, 
            df_commandes_prev, 
            selected_fiscal_year
        )

    if len(result) == 4:
        fig_devis_comp, fig_commandes_comp, devis_tables, commandes_tables = result
        
        # Afficher le graphique des devis SEULEMENT si pas Commercial de saisie
        if vue_type != "Commercial de saisie":
            st.plotly_chart(fig_devis_comp, use_container_width=True)
        
        # Tableau détaillé pour les devis SEULEMENT si pas Commercial de saisie
        if vue_type != "Commercial de saisie" and devis_tables and len(devis_tables) == 2:
            devis_table_current, devis_table_prev = devis_tables
            with st.expander(f"📋 Détail des Devis par Type - {selected_fiscal_year} vs {selected_fiscal_year-1}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Année {selected_fiscal_year}**")
                    if not devis_table_current.empty:
                        # Colorer toutes les colonnes
                        styled_table = devis_table_current.style.format({
                            'Valeur (€)': '{:,.0f} €',
                            '% Devis': '{:.1f}%',
                            '% Valeur': '{:.1f}%'
                        })
                        
                        # Colorer la colonne Type
                        styled_table = styled_table.apply(lambda x: ['background-color: #e0e7ff' for _ in x], subset=['Type'])
                        
                        # Colorer les colonnes numériques
                        numeric_cols = ['Nb Devis', 'Nb Clients', 'Valeur (€)', '% Devis', '% Valeur']
                        available_cols = [col for col in numeric_cols if col in devis_table_current.columns]
                        
                        if 'Nb Devis' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Devis'], cmap='Oranges')
                        if 'Nb Clients' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Clients'], cmap='Greens')
                        if 'Valeur (€)' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Valeur (€)'], cmap='Blues')
                        if '% Devis' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Devis'], cmap='Purples')
                        if '% Valeur' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Valeur'], cmap='Reds')
                        
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée disponible")
                
                with col2:
                    st.markdown(f"**Année {selected_fiscal_year-1}**")
                    if not devis_table_prev.empty:
                        # Colorer toutes les colonnes
                        styled_table = devis_table_prev.style.format({
                            'Valeur (€)': '{:,.0f} €',
                            '% Devis': '{:.1f}%',
                            '% Valeur': '{:.1f}%'
                        })
                        
                        # Colorer la colonne Type
                        styled_table = styled_table.apply(lambda x: ['background-color: #e0e7ff' for _ in x], subset=['Type'])
                        
                        # Colorer les colonnes numériques
                        numeric_cols = ['Nb Devis', 'Nb Clients', 'Valeur (€)', '% Devis', '% Valeur']
                        available_cols = [col for col in numeric_cols if col in devis_table_prev.columns]
                        
                        if 'Nb Devis' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Devis'], cmap='Oranges')
                        if 'Nb Clients' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Clients'], cmap='Greens')
                        if 'Valeur (€)' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Valeur (€)'], cmap='Blues')
                        if '% Devis' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Devis'], cmap='Purples')
                        if '% Valeur' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Valeur'], cmap='Reds')
                        
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée disponible")
        # Afficher le graphique des commandes
        st.plotly_chart(fig_commandes_comp, use_container_width=True)
        
        # Tableau détaillé pour les commandes
        if commandes_tables and len(commandes_tables) == 2:
            commandes_table_current, commandes_table_prev = commandes_tables
            with st.expander(f"✅ Détail des Commandes par Type - {selected_fiscal_year} vs {selected_fiscal_year-1}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Année {selected_fiscal_year}**")
                    if not commandes_table_current.empty:
                        # Colorer toutes les colonnes
                        styled_table = commandes_table_current.style.format({
                            'Valeur (€)': '{:,.0f} €',
                            '% Commandes': '{:.1f}%',
                            '% Valeur': '{:.1f}%'
                        })
                        
                        # Colorer la colonne Type
                        styled_table = styled_table.apply(lambda x: ['background-color: #e0e7ff' for _ in x], subset=['Type'])
                        
                        # Colorer les colonnes numériques
                        numeric_cols = ['Nb Commandes', 'Nb Clients', 'Valeur (€)', '% Commandes', '% Valeur']
                        available_cols = [col for col in numeric_cols if col in commandes_table_current.columns]
                        
                        if 'Nb Commandes' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Commandes'], cmap='Oranges')
                        if 'Nb Clients' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Clients'], cmap='Greens')
                        if 'Valeur (€)' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Valeur (€)'], cmap='Blues')
                        if '% Commandes' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Commandes'], cmap='Purples')
                        if '% Valeur' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Valeur'], cmap='Reds')
                        
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée disponible")
                
                with col2:
                    st.markdown(f"**Année {selected_fiscal_year-1}**")
                    if not commandes_table_prev.empty:
                        # Colorer toutes les colonnes
                        styled_table = commandes_table_prev.style.format({
                            'Valeur (€)': '{:,.0f} €',
                            '% Commandes': '{:.1f}%',
                            '% Valeur': '{:.1f}%'
                        })
                        
                        # Colorer la colonne Type
                        styled_table = styled_table.apply(lambda x: ['background-color: #e0e7ff' for _ in x], subset=['Type'])
                        
                        # Colorer les colonnes numériques
                        numeric_cols = ['Nb Commandes', 'Nb Clients', 'Valeur (€)', '% Commandes', '% Valeur']
                        available_cols = [col for col in numeric_cols if col in commandes_table_prev.columns]
                        
                        if 'Nb Commandes' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Commandes'], cmap='Oranges')
                        if 'Nb Clients' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Nb Clients'], cmap='Greens')
                        if 'Valeur (€)' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['Valeur (€)'], cmap='Blues')
                        if '% Commandes' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Commandes'], cmap='Purples')
                        if '% Valeur' in available_cols:
                            styled_table = styled_table.background_gradient(subset=['% Valeur'], cmap='Reds')
                        
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée disponible")
    else:
        # Affichage des graphiques simples si les tableaux ne sont pas disponibles
        fig_devis_comp, fig_commandes_comp = result[:2]
        st.plotly_chart(fig_devis_comp, use_container_width=True)
        st.plotly_chart(fig_commandes_comp, use_container_width=True)
    
    # Section 6 : Alertes performance par commercial (seulement pour Commercial avec tous les commerciaux)
    if vue_type == "Commercial" and selected_commercial == "Tous les commerciaux":
        st.markdown("### 🚨 Alertes Performance par Commercial")
        
        df_stats, alerts = calculate_commercial_performance(
            df_devis_filtered, df_commandes_filtered, df_objectifs, df_mapping, 
            selected_fiscal_year, period_months, df_objectifs_personnalises, df_real_orders_filtered, vue_type
        )

        
        # Afficher les alertes dans des menus déroulants pour économiser l'espace
        if len(alerts) > 0:
            critical_alerts = [a for a in alerts if a['niveau'] == 'critical']
            warning_alerts = [a for a in alerts if a['niveau'] == 'warning']
            success_alerts = [a for a in alerts if a['niveau'] == 'success']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with st.expander(f"🔴 En difficulté (< 60%) - {len(critical_alerts)} commerciaux", expanded=False):
                    if critical_alerts:
                        for alert in critical_alerts:
                            if alert['objectif'] > 0:
                                st.markdown(f"""
                                <div class="alert-box alert-critical">
                                    <strong>{alert['commercial']}</strong><br>
                                    {alert['pct_atteint']:.1f}% de l'objectif<br>
                                    <small>{alert['valeur_commandes']:,.0f}€ / {alert['objectif']:,.0f}€</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="alert-box alert-critical">
                                    <strong>{alert['commercial']}</strong><br>
                                    Objectif non défini<br>
                                    <small>Commandes: {alert['valeur_commandes']:,.0f}€</small>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun commercial en difficulté")
            
            with col2:
                with st.expander(f"🟡 À surveiller (60-90%) - {len(warning_alerts)} commerciaux", expanded=False):
                    if warning_alerts:
                        for alert in warning_alerts:
                            st.markdown(f"""
                            <div class="alert-box alert-warning">
                                <strong>{alert['commercial']}</strong><br>
                                {alert['pct_atteint']:.1f}% de l'objectif<br>
                                <small>{alert['valeur_commandes']:,.0f}€ / {alert['objectif']:,.0f}€</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun commercial à surveiller")
            
            with col3:
                with st.expander(f"🟢 Performants (> 90%) - {len(success_alerts)} commerciaux", expanded=False):
                    if success_alerts:
                        for alert in success_alerts:
                            st.markdown(f"""
                            <div class="alert-box alert-success">
                                <strong>{alert['commercial']}</strong><br>
                                {alert['pct_atteint']:.1f}% de l'objectif<br>
                                <small>{alert['valeur_commandes']:,.0f}€ / {alert['objectif']:,.0f}€</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun commercial dans cette catégorie")
        else:
            st.warning("Aucune alerte disponible - Vérifiez la configuration des objectifs")
    
    # Section 7 : Tableau détaillé des performances (seulement pour tous les commerciaux)
    if (vue_type == "Commercial" or vue_type == "Commercial de saisie") and selected_commercial == "Tous les commerciaux":
        st.markdown("### 📋 Tableau Détaillé des Performances")
        
        # Calculer les performances selon le type
        if vue_type == "Commercial":
            df_stats, _ = calculate_commercial_performance(
                df_devis_filtered, df_commandes_filtered, df_objectifs, df_mapping, 
                selected_fiscal_year, period_months, df_objectifs_personnalises, df_real_orders_filtered, vue_type
            )
        else:  # Commercial de saisie
            df_stats, _ = calculate_commercial_performance_saisie(
                df_commandes_filtered, df_mapping_saisie, selected_fiscal_year
            )
        # Formater le dataframe pour l'affichage selon le type
        if vue_type == "Commercial":
            columns_to_keep = ['Commercial', 'Nb Devis', 'Valeur Devis', 'Nb Clients Devis',
                              'Nb Commandes Réelles', 'Valeur Commandes Réelles', 'Nb Clients Réels',
                              'Objectif', '% Objectif Réel Atteint']
        else:  # Commercial de saisie
            columns_to_keep = ['Commercial', 'Nb Commandes', 'Valeur Commandes', 'Nb Clients Commandes']

        
        df_display = df_stats[columns_to_keep].copy()
        
        # RÉORGANISER seulement l'ordre des colonnes
        if vue_type == "Commercial":
            nouvel_ordre = [
                'Commercial',
                'Objectif', 
                'Valeur Commandes Réelles',           # ← En 3ème position  
                '% Objectif Réel Atteint',            # ← En 4ème position
                'Nb Devis',
                'Valeur Devis',
                'Nb Clients Devis',
                'Nb Commandes Réelles',
                'Nb Clients Réels'
            ]
        else:  # Commercial de saisie  
            nouvel_ordre = [
                'Commercial',
                'Nb Commandes',
                'Valeur Commandes',
                'Nb Clients Commandes'
            ]

        # Appliquer le nouvel ordre (garder seulement les colonnes qui existent)
        colonnes_existantes = [col for col in nouvel_ordre if col in df_display.columns]
        df_display = df_display[colonnes_existantes]
        # Formater le dataframe pour l'affichage
        format_dict = {
            'Valeur Devis': '{:,.0f} €',
            'Valeur Commandes': '{:,.0f} €',
            'Valeur Commandes Réelles': '{:,.0f} €',
            '% Conversion Clients': '{:.1f}%'
        }
        
        if vue_type == "Commercial":
            format_dict.update({
                'Objectif': lambda x: '{:,.0f} €'.format(x) if pd.notna(x) else 'N/A',
                '% Objectif Atteint': lambda x: '{:.1f}%'.format(x) if pd.notna(x) else 'N/A',
                '% Objectif Réel Atteint': lambda x: '{:.1f}%'.format(x) if pd.notna(x) else 'N/A'
                                        })
        
        styled_df = df_display.style.format(format_dict)
        
        # Appliquer un code couleur sur le % d'objectif atteint (seulement pour Commercial)
        def color_objectif(val):
            if pd.isna(val):
                return ''
            if val < 60:
                return 'background-color: #fecaca'
            elif val < 85:
                return 'background-color: #fed7aa'
            else:
                return 'background-color: #bbf7d0'
        
        # Appliquer des couleurs à toutes les colonnes numériques
        def apply_all_colors(styler, df):
            # Colorer la colonne Commercial en bleu clair
            if 'Commercial' in df.columns:
                styler = styler.apply(lambda x: ['background-color: #dbeafe' for _ in x], subset=['Commercial'])
            
            # Colorer les colonnes numériques avec des dégradés
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for i, col in enumerate(numeric_cols):
                if col in ['% Objectif Atteint', '% Objectif Réel Atteint'] and vue_type == "Commercial":  # ← CORRECTION
                    styler = styler.map(color_objectif, subset=[col])
                else:
                    # Utiliser différentes couleurs pour chaque colonne
                    colors = ['Oranges', 'Greens', 'Blues', 'Purples', 'Reds', 'YlOrRd']
                    color_map = colors[i % len(colors)]
                    styler = styler.background_gradient(subset=[col], cmap=color_map)
            
            return styler
        if vue_type == "Commercial" and '% Objectif Atteint' in df_display.columns:
            styled_df = apply_all_colors(styled_df, df_display)
        else:
            styled_df = apply_all_colors(styled_df, df_display)
        
        # Trier par Valeur Commandes décroissante
        if vue_type == "Commercial":
            # Pour Commercial, trier par Valeur Commandes Réelles
            if 'Valeur Commandes Réelles' in df_display.columns:
                df_display_sorted = df_display.sort_values('Valeur Commandes Réelles', ascending=False)
            else:
                df_display_sorted = df_display.copy()
        else:
            # Pour Commercial de saisie, trier par Valeur Commandes
            if 'Valeur Commandes' in df_display.columns:
                df_display_sorted = df_display.sort_values('Valeur Commandes', ascending=False)
            else:
                df_display_sorted = df_display.copy()
        styled_df = df_display_sorted.style.format(format_dict)
        
        styled_df = apply_all_colors(styled_df, df_display_sorted)
        
        st.dataframe(styled_df, use_container_width=True, height=350, hide_index=True)
    
    # Section 8 : Évolution mensuelle ou comparaison
    if selected_month is None and selected_period and len(period_months) > 3:
        # Afficher les graphiques d'évolution mensuelle pour une période étendue
        st.markdown("### 📈 Évolution Mensuelle")
        
        # Filtrer les données pour l'année complète (pas de filtre mois) selon le type de vue
        if vue_type == "Vue globale":
            if selected_commercial != "Tous les commerciaux":
                df_devis_charts = filter_data(df_devis, selected_fiscal_year, None, None, selected_commercial)
                df_commandes_charts = filter_data(df_commandes, selected_fiscal_year, None, None, selected_commercial)
            else:
                df_devis_charts = filter_data(df_devis, selected_fiscal_year, None, None, None)
                df_commandes_charts = filter_data(df_commandes, selected_fiscal_year, None, None, None)
        elif vue_type == "Commercial":
            if selected_commercial != "Tous les commerciaux":
                df_devis_charts = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year, None, None, selected_commercial)
                df_commandes_charts = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year, None, None, selected_commercial)
            else:
                df_devis_charts = filter_data_by_commercial_list(df_devis, commerciaux_list, selected_fiscal_year, None, None, None)
                df_commandes_charts = filter_data_by_commercial_list(df_commandes, commerciaux_list, selected_fiscal_year, None, None, None)
        else:  # Commercial de saisie
            if selected_commercial != "Tous les commerciaux":
                df_devis_charts = filter_data_by_commercial_list(df_devis, commerciaux_saisie_list, selected_fiscal_year, None, None, selected_commercial)
                df_commandes_charts = filter_data_by_commercial_list(df_commandes, commerciaux_saisie_list, selected_fiscal_year, None, None, selected_commercial)
            else:
                df_devis_charts = filter_data_by_commercial_list(df_devis, commerciaux_saisie_list, selected_fiscal_year, None, None, None)
                df_commandes_charts = filter_data_by_commercial_list(df_commandes, commerciaux_saisie_list, selected_fiscal_year, None, None, None)
        
        # Passer les données réelles pour Commercial
        if vue_type == "Commercial":
            df_real_orders_charts = filter_real_orders_by_commercial_list_CORRECTED(df_real_orders, commerciaux_reels_list, selected_fiscal_year, None, None, None if selected_commercial == "Tous les commerciaux" else selected_commercial)
            fig1, fig2, fig3 = create_monthly_evolution_charts(df_devis_charts, df_commandes_charts, selected_fiscal_year, vue_type, df_real_orders_charts)
        else:
            fig1, fig2, fig3 = create_monthly_evolution_charts(df_devis_charts, df_commandes_charts, selected_fiscal_year, vue_type)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Section 9 : Top clients
    st.markdown("### ⭐ Top 10 des Clients")

    if vue_type == "Commercial de saisie":
        # Pour Commercial de saisie : seulement top clients commandes
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("##### 🏆 Top Clients par Valeur de Commandes")
            
            # Calculer directement le top clients avec les données filtrées
            if not df_commandes_filtered.empty:
                net_value_col = get_net_value_column(df_commandes_filtered, selected_fiscal_year)
                if net_value_col and net_value_col in df_commandes_filtered.columns and 'SoldTo Name' in df_commandes_filtered.columns:
                    # Grouper par client et sommer les valeurs
                    df_top_orders = df_commandes_filtered.groupby('SoldTo #').agg({
                        net_value_col: 'sum',
                        'SoldTo Name': 'first'  # Prendre le premier nom pour chaque SoldTo #
                    }).reset_index()
                    df_top_orders.columns = ['SoldTo #', 'Valeur Commandes', 'Client']
                    # Réorganiser les colonnes pour avoir Client en premier
                    df_top_orders = df_top_orders[['Client', 'SoldTo #', 'Valeur Commandes']]
                    df_top_orders['SoldTo #'] = df_top_orders['SoldTo #'].astype(int)

                    # Trier par valeur décroissante et prendre le top 10
                    df_top_orders = df_top_orders.sort_values('Valeur Commandes', ascending=False).head(10)
                    
                    if not df_top_orders.empty:
                        # Colorer la colonne Client
                        styled_orders = df_top_orders.style.format({
                            'Valeur Commandes': '{:,.0f} €',
                             'SoldTo #': '{:d}'  # Format entier
                        })
                        
                        # Colorer la colonne Client en bleu clair
                        styled_orders = styled_orders.apply(lambda x: ['background-color: #dbeafe' for _ in x], subset=['Client'])
                        # Colorer la colonne SoldTo # en gris clair
                        styled_orders = styled_orders.apply(lambda x: ['background-color: #f3f4f6' for _ in x], subset=['SoldTo #'])
                        # Colorer la colonne Valeur Commandes avec un dégradé
                        styled_orders = styled_orders.background_gradient(subset=['Valeur Commandes'], cmap='Greens')
                        
                        st.dataframe(
                            styled_orders,
                            use_container_width=True,
                            hide_index=True,
                            height=350
                        )
                    else:
                        st.info("Aucune donnée de commandes après filtrage")
                else:
                    st.info("Colonnes nécessaires introuvables")
            else:
                st.info("Aucune donnée de commandes pour la période sélectionnée")
    else:
    
        col1, col2 = st.columns(2)
        
        with col1:
            if vue_type == "Commercial":
                st.markdown("##### 🏆 Top Clients par Valeur de Commandes Réelles")
                
                # Utiliser les commandes réelles pour Commercial
                if not df_real_orders_filtered.empty:
                    if 'Net Value' in df_real_orders_filtered.columns and 'SoldTo Name' in df_real_orders_filtered.columns:
                        # Grouper par client et sommer les valeurs réelles
                        df_top_orders = df_real_orders_filtered.groupby('SoldTo #').agg({
                            'Net Value': 'sum',
                            'SoldTo Name': 'first'  # Prendre le premier nom pour chaque SoldTo #
                        }).reset_index()
                        df_top_orders.columns = ['SoldTo #', 'Valeur Commandes', 'Client']
                        # Réorganiser les colonnes pour avoir Client en premier
                        df_top_orders = df_top_orders[['Client', 'SoldTo #', 'Valeur Commandes']]
                        df_top_orders['SoldTo #'] = df_top_orders['SoldTo #'].astype(int)

                        # Trier par valeur décroissante et prendre le top 10
                        df_top_orders = df_top_orders.sort_values('Valeur Commandes', ascending=False).head(10)
                        
                        if not df_top_orders.empty:
                            # Colorer la colonne Client
                            styled_orders = df_top_orders.style.format({
                                'Valeur Commandes': '{:,.0f} €'
                            })
                            
                            # Colorer la colonne Client en bleu clair
                            styled_orders = styled_orders.apply(lambda x: ['background-color: #dbeafe' for _ in x], subset=['Client'])
                            # Colorer la colonne SoldTo # en gris clair
                            styled_orders = styled_orders.apply(lambda x: ['background-color: #f3f4f6' for _ in x], subset=['SoldTo #'])
                            # Colorer la colonne Valeur Commandes avec un dégradé
                            styled_orders = styled_orders.background_gradient(subset=['Valeur Commandes'], cmap='Greens')
                            
                            st.dataframe(
                                styled_orders,
                                use_container_width=True,
                                hide_index=True,
                                height=350
                            )
                        else:
                            st.info("Aucune donnée de commandes réelles après filtrage")
                    else:
                        st.info("Colonnes nécessaires introuvables pour les commandes réelles")
                else:
                    st.info("Aucune donnée de commandes réelles pour la période sélectionnée")
            else:
                st.markdown("##### 🏆 Top Clients par Valeur de Commandes")
                
                # Calculer directement le top clients avec les données filtrées
                if not df_commandes_filtered.empty:
                    net_value_col = get_net_value_column(df_commandes_filtered, selected_fiscal_year)
                    if net_value_col and net_value_col in df_commandes_filtered.columns and 'SoldTo Name' in df_commandes_filtered.columns:
                        # Grouper par client et sommer les valeurs
                        df_top_orders = df_commandes_filtered.groupby('SoldTo #').agg({
                            net_value_col: 'sum',
                            'SoldTo Name': 'first'  # Prendre le premier nom pour chaque SoldTo #
                        }).reset_index()
                        df_top_orders.columns = ['SoldTo #', 'Valeur Commandes', 'Client']
                        # Réorganiser les colonnes pour avoir Client en premier
                        df_top_orders = df_top_orders[['Client', 'SoldTo #', 'Valeur Commandes']]
                        df_top_orders['SoldTo #'] = df_top_orders['SoldTo #'].astype(int)

                        # Trier par valeur décroissante et prendre le top 10
                        df_top_orders = df_top_orders.sort_values('Valeur Commandes', ascending=False).head(10)
                        
                        if not df_top_orders.empty:
                            # Colorer la colonne Client
                            styled_orders = df_top_orders.style.format({
                                'Valeur Commandes': '{:,.0f} €'
                            })
                            
                            # Colorer la colonne Client en bleu clair
                            styled_orders = styled_orders.apply(lambda x: ['background-color: #dbeafe' for _ in x], subset=['Client'])
                            # Colorer la colonne SoldTo # en gris clair
                            styled_orders = styled_orders.apply(lambda x: ['background-color: #f3f4f6' for _ in x], subset=['SoldTo #'])

                            # Colorer la colonne Valeur Commandes avec un dégradé
                            styled_orders = styled_orders.background_gradient(subset=['Valeur Commandes'], cmap='Greens')
                            
                            st.dataframe(
                                styled_orders,
                                use_container_width=True,
                                hide_index=True,
                                height=350
                            )
                        else:
                            st.info("Aucune donnée de commandes après filtrage")
                    else:
                        st.info("Colonnes nécessaires introuvables")
                else:
                    st.info("Aucune donnée de commandes pour la période sélectionnée")
        
        with col2:
            st.markdown("##### 📈 Top Clients par Valeur de Devis")
            
            # Calculer directement le top clients avec les données filtrées
            if not df_devis_filtered.empty:
                net_value_col = get_net_value_column(df_devis_filtered, selected_fiscal_year)
                if net_value_col and net_value_col in df_devis_filtered.columns and 'SoldTo Name' in df_devis_filtered.columns:
                    # Grouper par client et sommer les valeurs
                    df_top_quotes = df_devis_filtered.groupby('SoldTo #').agg({
                        net_value_col: 'sum',
                        'SoldTo Name': 'first'  # Prendre le premier nom pour chaque SoldTo #
                    }).reset_index()
                    df_top_quotes.columns = ['SoldTo #', 'Valeur Devis', 'Client']
                    # Réorganiser les colonnes pour avoir Client en premier
                    df_top_quotes = df_top_quotes[['Client', 'SoldTo #', 'Valeur Devis']]
                    df_top_quotes['SoldTo #'] = df_top_quotes['SoldTo #'].astype(int)

                    # Trier par valeur décroissante et prendre le top 10
                    df_top_quotes = df_top_quotes.sort_values('Valeur Devis', ascending=False).head(10)
                    
                    if not df_top_quotes.empty:
                        # Colorer la colonne Client
                        styled_quotes = df_top_quotes.style.format({
                            'Valeur Devis': '{:,.0f} €',
                            'SoldTo #': '{:d}'  # Format entier

                        })
                        
                        # Colorer la colonne Client en bleu clair
                        styled_quotes = styled_quotes.apply(lambda x: ['background-color: #dbeafe' for _ in x], subset=['Client'])
                        # Colorer la colonne SoldTo # en gris clair
                        styled_quotes = styled_quotes.apply(lambda x: ['background-color: #f3f4f6' for _ in x], subset=['SoldTo #'])
                        # Colorer la colonne Valeur Devis avec un dégradé
                        styled_quotes = styled_quotes.background_gradient(subset=['Valeur Devis'], cmap='Blues')
                        
                        st.dataframe(
                            styled_quotes,
                            use_container_width=True,
                            hide_index=True,
                            height=350
                        )
                    else:
                        st.info("Aucune donnée de devis après filtrage")
                else:
                    st.info("Colonnes nécessaires introuvables")
            else:
                st.info("Aucune donnée de devis pour la période sélectionnée")
                
    # Section 10 : Top 15 Managed Groups
    st.markdown("### 🏢 Top 15 des Groupes (Managed Groups)")

    if vue_type == "Commercial de saisie":
        # Pour Commercial de saisie : seulement top groupes commandes
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("##### 🏆 Top Groupes par Valeur de Commandes")
            
            if not df_commandes_filtered.empty:
                df_top_groups_orders = get_top_managed_groups_by_type(df_commandes_filtered, selected_fiscal_year, 15, "commandes")
                
                if not df_top_groups_orders.empty:
                    # Styler le tableau
                    styled_groups_orders = df_top_groups_orders.style.format({
                        'Valeur Commandes': '{:,.0f} €'
                    })
                    
                    # Appliquer les couleurs
                    styled_groups_orders = styled_groups_orders.apply(
                        lambda x: ['background-color: #dbeafe' for _ in x], subset=['Groupe']
                    )
                    styled_groups_orders = styled_groups_orders.background_gradient(
                        subset=['Valeur Commandes'], cmap='Greens'
                    )
                    
                    st.dataframe(
                        styled_groups_orders,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                else:
                    st.info("Aucun groupe trouvé pour la période sélectionnée")
            else:
                st.info("Aucune donnée de commandes pour la période sélectionnée")

    else:
        # Pour Vue globale et Commercial : 2 colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            if vue_type == "Commercial":
                # Pour Commercial : utiliser les commandes réelles (avec attribution)
                st.markdown("##### 🏆 Top Groupes par Valeur de Commandes Réelles")
                
                if not df_real_orders_filtered.empty:
                    df_top_groups_orders = get_top_managed_groups_real_orders(df_real_orders_filtered, 15)
                    
                    if not df_top_groups_orders.empty:
                        # Styler le tableau
                        styled_groups_orders = df_top_groups_orders.style.format({
                            'Valeur Commandes Réelles': '{:,.0f} €'
                        })
                        
                        # Appliquer les couleurs
                        styled_groups_orders = styled_groups_orders.apply(
                            lambda x: ['background-color: #dbeafe' for _ in x], subset=['Groupe']
                        )
                        styled_groups_orders = styled_groups_orders.background_gradient(
                            subset=['Valeur Commandes Réelles'], cmap='Greens'
                        )
                        
                        st.dataframe(
                            styled_groups_orders,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    else:
                        st.info("Aucun groupe trouvé dans les commandes réelles")
                else:
                    st.info("Aucune donnée de commandes réelles pour la période sélectionnée")
            else:
                # Pour Vue globale : utiliser les commandes normales
                st.markdown("##### 🏆 Top Groupes par Valeur de Commandes")
                
                if not df_commandes_filtered.empty:
                    df_top_groups_orders = get_top_managed_groups_by_type(df_commandes_filtered, selected_fiscal_year, 15, "commandes")
                    
                    if not df_top_groups_orders.empty:
                        # Styler le tableau
                        styled_groups_orders = df_top_groups_orders.style.format({
                            'Valeur Commandes': '{:,.0f} €'
                        })
                        
                        # Appliquer les couleurs
                        styled_groups_orders = styled_groups_orders.apply(
                            lambda x: ['background-color: #dbeafe' for _ in x], subset=['Groupe']
                        )
                        styled_groups_orders = styled_groups_orders.background_gradient(
                            subset=['Valeur Commandes'], cmap='Greens'
                        )
                        
                        st.dataframe(
                            styled_groups_orders,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    else:
                        st.info("Aucun groupe trouvé pour les commandes")
                else:
                    st.info("Aucune donnée de commandes pour la période sélectionnée")
        
        with col2:
            # Pour Vue globale et Commercial : toujours afficher les groupes devis
            st.markdown("##### 📈 Top Groupes par Valeur de Devis")
            
            if not df_devis_filtered.empty:
                df_top_groups_quotes = get_top_managed_groups_by_type(df_devis_filtered, selected_fiscal_year, 15, "devis")
                
                if not df_top_groups_quotes.empty:
                    # Styler le tableau
                    styled_groups_quotes = df_top_groups_quotes.style.format({
                        'Valeur Devis': '{:,.0f} €'
                    })
                    
                    # Appliquer les couleurs
                    styled_groups_quotes = styled_groups_quotes.apply(
                        lambda x: ['background-color: #dbeafe' for _ in x], subset=['Groupe']
                    )
                    styled_groups_quotes = styled_groups_quotes.background_gradient(
                        subset=['Valeur Devis'], cmap='Blues'
                    )
                    
                    st.dataframe(
                        styled_groups_quotes,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                else:
                    st.info("Aucun groupe trouvé pour les devis")
            else:
                st.info("Aucune donnée de devis pour la période sélectionnée")  
                
    # Footer personnalisé Signals
    st.markdown("---")

    st.markdown("""
    <div style='text-align: center; padding: 20px; margin: 20px 0;'>
        <div style='font-style: italic; color: #4f46e5; font-size: 16px; font-weight: 500; line-height: 1.5;'>
            ✨ Chez Signals, chacun brille à sa façon, tous progressent ensemble… et notre objectif est clair : devenir les meilleurs, avec le sourire en prime 😊
        </div>
    </div>
    """, unsafe_allow_html=True)
