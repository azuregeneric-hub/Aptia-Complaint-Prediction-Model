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

# Custom CSS for Aptia-themed UI
st.markdown("""
    <style>
    /* Main background with lighter shade of Aptia blue */
    .stApp {
        background-color: #f0f7ff;
        font-family: 'Arial', sans-serif;
    }
    
    /* Header with Aptia blue */
    .header {
        background-color: #153647;
        padding: 10px 20px; /* Reduced vertical padding for shorter height */
        border-radius: 0 0 15px 15px;
        margin-bottom: 10px;
        color: white;
        box-shadow: 0 4px 12px rgba(21, 54, 71, 0.3);
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%; /* Ensure it spans the column width */
    }
    
    /* Main title in header */
    .header-title {
        color: white;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Subtitle styling now on the main background */
    .subtitle-text {
        text-align: center;
        color: #16d49b; /* Green color */
        font-size: 18px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        color: #153647;
        border-bottom: 2px solid #16d49b;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Card styling for sections */
    .card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(21, 54, 71, 0.1);
        margin-bottom: 25px;
        border-left: 4px solid #16d49b;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #16d49b;
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
    }
    
    .stButton > button:hover {
        background-color: #12b585;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(22, 212, 155, 0.3);
    }
    
    /* Download button styling - Aptia blue */
    .stDownloadButton > button {
        background-color: #153647;
        color: white; /* Make the text white */
    }
    
    .stDownloadButton > button:hover {
        background-color: #0f2a38;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(21, 54, 71, 0.1);
    }

    /* Dataframe header styling */
    /* Column headers row background */
    .stDataFrame thead tr {
        background-color: #153647;
    }
    
    /* Column headers text color */
    .stDataFrame thead th {
        color: white;
        font-weight: 600;
    }
    
    /* Column pane background to match the header */
    .stDataFrame th {
        background-color: #153647;
    }

    /* Column labels text color in the pane */
    .stDataFrame th > div {
        color: white;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #e6f7ee;
        color: #0d6833;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #16d49b;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #e6f0ff;
        color: #153647;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #153647;
    }
    
    /* Custom metric boxes - Aptia theme */
    .metric-card {
        background: linear-gradient(135deg, #153647 0%, #0f2a38 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(21, 54, 71, 0.2);
    }
    
    .metric-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f7ff 0%, #e6f0ff 100%);
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 20px;
        padding: 15px;
        background-color: white;
        border-radius: 10px;
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
except:
    aptia_logo = None

# Sidebar with simplified information
with st.sidebar:
    # Add Aptia logo with proper alignment
    if aptia_logo:
        st.image(aptia_logo, use_container_width=True)
    else:
        st.markdown("""
            <div class="logo-container">
                <h3 style="color: #153647; margin: 0; text-align: center;">Aptia</h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Create a card-like container using native Streamlit elements
    st.markdown("### ℹ️ About This Tool")
    st.write("This tool uses machine learning to predict which customer inquiries are likely to escalate into formal complaints.")
    
    st.markdown("#### How to Use")
    st.write("• Upload your customer inquiry data file")
    st.write("• The system will analyze it automatically")
    st.write("• See which cases might become complaints")
    st.write("• Download the results for your team")
    
    st.markdown("#### File Types")
    st.write("• Excel files (.xlsx, .xls)")
    st.write("• CSV files (.csv)")

# Main content
# Use columns to create an offset for the header to align with the sidebar
col_spacer, col_header, col_spacer_right = st.columns([1, 6, 1]) # Adjust ratios as needed

with col_header:
    st.markdown("""
        <div class="header">
            <h1 class="header-title">Complaint Prediction Tool</h1>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="subtitle-text">Identify which customer inquiries are likely to escalate into formal complaints</p>', unsafe_allow_html=True)
    
st.markdown("### 📤 Upload Customer Inquiry Data")
st.markdown("Drag and drop your Excel or CSV file below to analyze which inquiries might turn into complaints.")
    
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
    return joblib.load("all_xgb_models.pkl")

# Load lookup tables containing historical features
@st.cache_data
def load_lookup_tables():
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
    # Add year_month from Report_Date
    df['Report_Date'] = pd.to_datetime(df['Report_Date'])
    df['year_month'] = df['Report_Date'].dt.to_period('M').astype(str)

    # Drop specified columns
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        y = None
        X = df

    # Fill missing values
    X = X.fillna("Missing")

    # Label encode categorical columns
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
    if uploaded_file.name.endswith('.csv'):
        user_df = pd.read_csv(uploaded_file)
    else:
        user_df = pd.read_excel(uploaded_file)
    
    # Success message
    st.success(f"✅ Successfully uploaded {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
    
    # Preview section in a card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Data Preview")
        st.dataframe(user_df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models and lookup tables
    with st.spinner("Loading prediction models and historical data..."):
        models = load_models()
        lookup_nino, lookup_procgroup = load_lookup_tables()
    
    # Process the data
    with st.spinner("Processing data and generating predictions..."):
        # Clean NINO in uploaded data
        user_df = clean_nino_column(user_df)

        # Merge NINO-level historical features
        df_merged = user_df.merge(lookup_nino, on='Unique Identifier (NINO Encrypted)', how='left')

        # Merge Process Group-level features
        df_merged = df_merged.merge(lookup_procgroup, on=['Unique Identifier (NINO Encrypted)', 'Process Group'], how='left')

        # Fill missing historical feature values with 0
        for col in ['nino_cum_complaints', 'nino_total_enquiries', 'nino_procgroup_enquiries']:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0).astype(int)

        # Add nino_monthly_frequency column grouped by exact Report_Date
        df_merged = add_nino_frequency_exact_date(df_merged)

        # Add dummy label column if missing
        if 'will_file_complaint_in_future' not in df_merged.columns:
            df_merged['will_file_complaint_in_future'] = 0

        # Columns to drop same as your training preprocessing
        columns_to_drop =  [
            'Case ID', 'Unique Identifier (NINO Encrypted)', 'ClientName', 'OneCode',
            'Current Activity User', 'Report_Date', 'Start Date', 'Create Date', 'Mercer Consented',
            'Pend Case', 'Source', 'Pend Case', 'Operational Location', 'Case Indicator',
            'Flag_Scheme', 'Team Name', 'Process Name', 'Critical', 'Current Outsourcing Team',
            'Event Type', 'Completes', 'Consented/Non consented', 'Scheme'
        ]

        # Preprocess for prediction
        X_user, _, _ = preprocess_data_and_drop(df_merged, columns_to_drop, target_column='will_file_complaint_in_future')

        # Add any missing features expected by the model with default values 0
        for f in train_features_order:
            if f not in X_user.columns:
                X_user[f] = 0

        # Re-order columns exactly as trained model expects
        X_user = X_user[train_features_order]

        # Select model
        model = models['model_jan_feb_mar_apr']

        # Predict complaints and probabilities
        df_merged['Predicted_Complaint'] = model.predict(X_user)
        df_merged['Complaint_Probability'] = model.predict_proba(X_user)[:, 1]
    
    # Results section in a card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Metrics row
        total_cases = len(df_merged)
        complaint_cases = sum(df_merged['Predicted_Complaint'] == 1)
        complaint_percentage = (complaint_cases / total_cases * 100) if total_cases > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Cases</div>
                    <div class="metric-value">{total_cases}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Potential Complaints</div>
                    <div class="metric-value">{complaint_cases}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Complaint Rate</div>
                    <div class="metric-value">{complaint_percentage:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Display predicted complaint rows
        st.markdown("### 🎯 Rows Predicted to Turn into Complaints")
        
        if complaint_cases > 0:
            # Filter and sort by probability
            high_risk_cases = df_merged[df_merged['Predicted_Complaint'] == 1].sort_values('Complaint_Probability', ascending=False)
            
            # Display important columns
            display_columns = ['Case ID', 'Unique Identifier (NINO Encrypted)', 'Title', 
                               'Process Group', 'Complaint_Probability']
            
            # Filter to only include columns that exist
            available_columns = [col for col in display_columns if col in high_risk_cases.columns]
            
            st.dataframe(high_risk_cases[available_columns], use_container_width=True)
            
            # Download button
            csv_data = high_risk_cases.to_csv(index=False)
            st.download_button(
                "📥 Download Prediction Results", 
                csv_data, 
                file_name='predicted_complaints.csv', 
                mime='text/csv'
            )
        else:
            st.info("No cases predicted to turn into complaints. Great job!")
        
        st.markdown('</div>', unsafe_allow_html=True)