import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Complaint Prediction Tool",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional and beautiful UI
st.markdown("""
    <style>
    /* Import a professional font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Main background and font settings */
    .stApp {
        background-color: #f0f7ff;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Header with a subtle gradient */
    .header {
        background: linear-gradient(90deg, #153647 0%, #1a4d64 100%);
        /* Changed padding to move the header up */
        padding: 5px 20px;
        border-radius: 0 0 15px 15px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 6px 15px rgba(21, 54, 71, 0.4);
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Main title in header */
    .header-title {
        color: white;
        font-size: 36px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        text-align: center;
        color: #16d49b;
        font-size: 20px;
        margin-top: 15px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    /* Section headers with a more refined look */
    .stMarkdown h3 {
        color: #153647;
        border-bottom: 2px solid #16d49b;
        padding-bottom: 8px;
        margin-top: 2rem;
        font-weight: 600;
    }

    /* File uploader styling */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        border: 2px dashed #16d49b;
        box-shadow: inset 0 0 10px rgba(22, 212, 155, 0.1);
    }
    
    /* Button styling - Aptia teal */
    .stButton > button {
        background-color: #16d49b;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 4px 6px rgba(22, 212, 155, 0.2);
    }
    
    .stButton > button:hover {
        background-color: #12b585;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(22, 212, 155, 0.4);
    }
    
    /* Download button styling - Aptia blue */
    .stDownloadButton > button {
        background-color: #153647;
        color: white;
        box-shadow: 0 4px 6px rgba(21, 54, 71, 0.2);
    }
    
    .stDownloadButton > button:hover {
        background-color: #0f2a38;
        box-shadow: 0 6px 12px rgba(21, 54, 71, 0.4);
    }
    
    /* Metric boxes - Now without a background color */
    .metric-card {
        background: none;
        border: 2px solid #153647;
        color: #153647;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(21, 54, 71, 0.1);
    }
    
    .metric-title {
        font-size: 16px;
        font-weight: 400;
        margin-bottom: 5px;
        opacity: 0.9;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }

    /* Custom CSS for the green line */
    .green-line {
        border-top: 2px solid #16d49b;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Style for DataFrame headers */
    .stDataFrame table th {
        background-color: #153647;
        color: white;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 20px;
        padding: 15px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(21, 54, 71, 0.1);
    }
    
    /* Logo image styling */
    .logo-img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Load the Aptia logo
try:
    aptia_logo = Image.open("Aptialogo.png")
except FileNotFoundError:
    aptia_logo = None

# Sidebar with simplified information
with st.sidebar:
    if aptia_logo:
        st.image(aptia_logo, use_container_width=True)
    else:
        st.markdown("""
            <div class="logo-container">
                <h3 style="color: #153647; margin: 0; text-align: center;">Aptia</h3>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ℹ️ About This Tool")
    st.write("This tool uses machine learning to predict which customer inquiries are likely to escalate into formal complaints. Leverage this powerful tool to **proactively manage customer satisfaction** and streamline your complaint resolution process.")
    
    st.markdown("#### How to Use")
    st.write("1. **Upload your data**: Drag and drop a `.xlsx` or `.csv` file containing customer inquiry information.")
    st.write("2. **Get predictions**: The tool will automatically analyze the data and generate predictions.")
    st.write("3. **View results**: See which cases have a high probability of becoming a formal complaint.")
    st.write("4. **Take action**: Download the results to prioritize your team's workload and address high-risk cases.")

# Main content
col_spacer, col_header, col_spacer_right = st.columns([0.1, 1, 0.1])
with col_header:
    st.markdown("""
        <div class="header">
            <h1 class="header-title">Complaint Prediction Tool</h1>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="subtitle-text">Empower your team with data-driven insights to predict and prevent customer complaints.</p>', unsafe_allow_html=True)
    
st.markdown("### 📤 Upload Customer Inquiry Data")
st.markdown("Start by uploading your data file. The system will then use a pre-trained machine learning model to predict the likelihood of each case escalating into a formal complaint.")
    
# File uploader
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['xlsx', 'xls', 'csv'],
    label_visibility="collapsed"
)

# ... [rest of the code remains the same] ...

# Load pre-trained models
@st.cache_resource
def load_models():
    try:
        return joblib.load("all_xgb_models.pkl")
    except FileNotFoundError:
        st.error("Model file `all_xgb_models.pkl` not found. Please ensure the file is in the same directory.")
        return None

# Load lookup tables containing historical features
@st.cache_data
def load_lookup_tables():
    try:
        lookup_nino = pd.read_csv("nino_lookup_features.csv")
        lookup_procgroup = pd.read_csv("nino_procgroup_lookup_features.csv")

        # Clean keys for consistency
        lookup_nino['Unique Identifier (NINO Encrypted)'] = (
            lookup_nino['Unique Identifier (NINO Encrypted)'].astype(str).str.strip().str.upper()
        )
        lookup_procgroup['Unique Identifier (NINO Encrypted)'] = (
            lookup_procgroup['Unique Identifier (NINO Encrypted)'].astype(str).str.strip().str.upper()
        )
        lookup_procgroup['Process Group'] = (
            lookup_procgroup['Process Group'].astype(str).str.strip()
        )
        return lookup_nino, lookup_procgroup
    except FileNotFoundError:
        st.error("Historical data files not found. Please ensure `nino_lookup_features.csv` and `nino_procgroup_lookup_features.csv` are in the same directory.")
        return None, None

# Cleaning NINO
def clean_nino_column(df):
    df['Unique Identifier (NINO Encrypted)'] = (
        df['Unique Identifier (NINO Encrypted)'].astype(str).str.strip().str.upper()
    )
    return df

# Add nino_monthly_frequency grouped by exact Report_Date
def add_nino_frequency_exact_date(df):
    df['Report_Date'] = pd.to_datetime(df['Report_Date'])
    freq = df.groupby(['Unique Identifier (NINO Encrypted)', 'Report_Date']).size().reset_index(name='nino_monthly_frequency')
    df = df.merge(freq, on=['Unique Identifier (NINO Encrypted)', 'Report_Date'], how='left')
    return df

# Function to preprocess and encode data as per your pipeline
def preprocess_data_and_drop(df, drop_columns, target_column=None):
    df = df.copy()
    df['Report_Date'] = pd.to_datetime(df['Report_Date'])
    df['year_month'] = df['Report_Date'].dt.to_period('M').astype(str)
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        y = None
        X = df

    X = X.fillna("Missing")

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders

# Expected feature order during training
train_features_order = [
    'Title', 'Portfolio', 'Location', 'Team Name', 'Process Name', 'Process Group',
    'Onshore/Offshore', 'Days to Target', 'Scan+2', 'Site', 'Manual/RPA', 'Forthcoming Event',
    'Within SLA', 'Vulnerable Customer', 'No of Days', 'Mercer Days', 'nino_monthly_frequency',
    'nino_cum_complaints', 'nino_total_enquiries', 'nino_procgroup_enquiries', 'year_month'
]

if uploaded_file is not None:
    # Read the uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            user_df = pd.read_csv(uploaded_file)
        else:
            user_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file. Please ensure it is a valid CSV or Excel format. Error: {e}")
        user_df = None

    if user_df is not None:
        st.markdown('<div class="green-line"></div>', unsafe_allow_html=True)
    
        st.markdown("### 📋 Data Preview")
        st.dataframe(user_df.head(), use_container_width=True)
    
        with st.spinner("Loading prediction models and historical data..."):
            models = load_models()
            lookup_nino, lookup_procgroup = load_lookup_tables()
        
        if models is not None and lookup_nino is not None and lookup_procgroup is not None:
            with st.spinner("Processing data and generating predictions..."):
                user_df = clean_nino_column(user_df)
                df_merged = user_df.merge(lookup_nino, on='Unique Identifier (NINO Encrypted)', how='left')
                df_merged = df_merged.merge(lookup_procgroup, on=['Unique Identifier (NINO Encrypted)', 'Process Group'], how='left')
                
                for col in ['nino_cum_complaints', 'nino_total_enquiries', 'nino_procgroup_enquiries']:
                    if col in df_merged.columns:
                        df_merged[col] = df_merged[col].fillna(0).astype(int)

                df_merged = add_nino_frequency_exact_date(df_merged)

                if 'will_file_complaint_in_future' not in df_merged.columns:
                    df_merged['will_file_complaint_in_future'] = 0

                columns_to_drop = columns_to_drop = [
    'Case ID',
    'Unique Identifier (NINO Encrypted)',
    'ClientName',
    'OneCode',
    'Current Activity User',
    'Report_Date',
    'Start Date',
    'Create Date',
    'Mercer Consented',
    'Pend Case',
    #'Scan+2',
    'Source',
    'Pend Case',
    'Operational Location',
    'Case Indicator',
    'Flag_Scheme',
    #'Portfolio',
    'Team Name'
    #'Location',
    'Process Name',
    'Process Group',
    'Critical',
    #'Onshore/Offshore',
    'Current Outsourcing Team',
    'Event Type',
    #'Site',#
    'Completes',
    'Consented/Non consented',
    'Scheme'

]
                X_user, _, _ = preprocess_data_and_drop(df_merged, columns_to_drop, target_column='will_file_complaint_in_future')

                for f in train_features_order:
                    if f not in X_user.columns:
                        X_user[f] = 0

                X_user = X_user[train_features_order]

                model = models['model_jan_feb_mar_apr']

                #df_merged['Predicted_Complaint'] = model.predict(X_user)
                #df_merged['Complaint_Probability'] = model.predict_proba(X_user)[:, 1]

                df_merged['Predicted_Complaint'] = model.predict(X_user)
                df_merged['Complaint_Probability'] = model.predict_proba(X_user)[:, 1]
                print(model.get_booster().feature_names)
                print(f"Number of features trained on: {len(model.get_booster().feature_names)}")
                print("Are columns unique?", len(X_user.columns) == len(set(X_user.columns)))
                X_user = X_user.apply(pd.to_numeric, errors='coerce').fillna(0)
            st.markdown("### 📊 Prediction Summary")
            
            total_cases = len(df_merged)
            complaint_cases = sum(df_merged['Predicted_Complaint'] == 1)
            complaint_percentage = (complaint_cases / total_cases * 100) if total_cases > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Inquiries</div>
                        <div class="metric-value">{total_cases}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">High-Risk Cases</div>
                        <div class="metric-value">{complaint_cases}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Potential Complaint Rate</div>
                        <div class="metric-value">{complaint_percentage:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 High-Risk Cases Predicted")
            
            if complaint_cases > 0:
                high_risk_cases = df_merged[df_merged['Predicted_Complaint'] == 1].sort_values('Complaint_Probability', ascending=False)
                
                display_columns = ['Case ID', 'Unique Identifier (NINO Encrypted)', 'Title', 
                                   'Process Group', 'Complaint_Probability']
                available_columns = [col for col in display_columns if col in high_risk_cases.columns]
                
                st.dataframe(high_risk_cases[available_columns], use_container_width=True)
                
                csv_data = high_risk_cases.to_csv(index=False)
                st.download_button(
                    "📥 Download Prediction Results", 
                    csv_data, 
                    file_name='predicted_complaints.csv', 
                    mime='text/csv'
                )
            else:
                st.markdown(f"<p style='text-align:center; color:#153647; font-weight:bold;'>No cases predicted to turn into complaints. Great job!</p>", unsafe_allow_html=True)