import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page config and UI CSS styling
st.set_page_config(page_title="Complaint Prediction Tool", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
.stApp { background-color: #f0f7ff; font-family: 'Roboto', sans-serif; }
.header {
    background: linear-gradient(90deg, #153647 0%, #1a4d64 100%);
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
.header-title { font-size: 36px; font-weight: 700; letter-spacing: 1px; margin: 0; }
.subtitle-text {
    text-align: center;
    color: #16d49b;
    font-size: 20px;
    margin-top: 15px;
    margin-bottom: 30px;
    font-weight: 500;
}
.stFileUploader > div > div {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    border: 2px dashed #16d49b;
    box-shadow: inset 0 0 10px rgba(22, 212, 155, 0.1);
}
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
.stDownloadButton > button {
    background-color: #153647;
    color: white;
    box-shadow: 0 4px 6px rgba(21, 54, 71, 0.2);
}
.stDownloadButton > button:hover {
    background-color: #0f2a38;
    box-shadow: 0 6px 12px rgba(21, 54, 71, 0.4);
}
.metric-card {
    background: none;
    border: 2px solid #153647;
    color: #153647;
    padding: 10px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(21, 54, 71, 0.1);
}
.metric-title { font-size: 16px; font-weight: 400; margin-bottom: 5px; opacity: 0.9; }
.metric-value { font-size: 24px; font-weight: 700; }
.green-line { border-top: 2px solid #16d49b; margin-top: 15px; margin-bottom: 15px; }
.stDataFrame table th { background-color: #153647; color: white; }
.sidebar .sidebar-content { background: #f8f9fa; border-right: 1px solid #e0e0e0; }
.logo-container {
    text-align: center;
    margin-bottom: 20px;
    padding: 15px;
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(21, 54, 71, 0.1);
}
.logo-img { max-width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

# Load logo
try:
    aptia_logo = Image.open("Aptialogo.png")
except FileNotFoundError:
    aptia_logo = None

with st.sidebar:
    if aptia_logo:
        st.image(aptia_logo, use_container_width=True)
    else:
        st.markdown('<div class="logo-container"><h3 style="color:#153647;text-align:center;margin:0;">Aptia</h3></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This Tool")
    st.write("This tool uses machine learning to predict which customer inquiries are likely to escalate into formal complaints...")
    st.markdown("#### How to Use")
    st.write("1. Upload your data (.csv or .xlsx).")
    st.write("2. Get predictions.")
    st.write("3. View results.")
    st.write("4. Download predictions.")

col1, col2, col3 = st.columns([0.1,1,0.1])
with col2:
    st.markdown('<div class="header"><h1 class="header-title">Complaint Prediction Tool</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Empower your team with data-driven insights to predict and prevent customer complaints.</p>', unsafe_allow_html=True)
st.markdown("### 📤 Upload Customer Inquiry Data")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'], label_visibility="collapsed")

@st.cache_data
def load_models():
    # model = pickle.load(open(r'C:/Users/Sumit/Downloads/AptiaComplaintPredictModel2 -w0lag/Pythoncodes/model.pkl', 'rb'))
    # le_dict = pickle.load(open(r'C:/Users/Sumit/Downloads/AptiaComplaintPredictModel2 -w0lag/Pythoncodes/le_dict.pkl', 'rb'))
    # lookup_cum = pd.read_csv(r'C:/Users/Sumit/Downloads/AptiaComplaintPredictModel2 -w0lag/Pythoncodes/nino_lookup_cumulative.csv')
    # lookup_monthly = pd.read_csv(r'C:/Users/Sumit/Downloads/AptiaComplaintPredictModel2 -w0lag/Pythoncodes/nino_lookup_monthly_freq.csv')
    model = pickle.load(open('model.pkl', 'rb'))
    le_dict = pickle.load(open('le_dict.pkl', 'rb'))
    lookup_cum = pd.read_csv('nino_lookup_cumulative.csv')
    lookup_monthly = pd.read_csv('nino_lookup_monthly_freq.csv')

    return model, le_dict, lookup_cum, lookup_monthly

model, le_dict, lookup_cum, lookup_monthly = load_models()

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        user_df = pd.read_csv(uploaded_file)
    else:
        user_df = pd.read_excel(uploaded_file)

    st.markdown("### 📋 Uploaded Data Preview")
    st.dataframe(user_df.head(), use_container_width=True)

    uid_col = "Unique Identifier (NINO Encrypted)"

    user_df['Create Date'] = pd.to_datetime(user_df['Create Date'], errors='coerce')
    user_df['YearMonth'] = user_df['Create Date'].dt.to_period('M').astype(str)

    for col in ['Past_Case_Count', 'Past_Complaints_Cum', 'Complaint_Ratio',
                'Complaint_30d_Rolling', 'Vulnerable_CumCount']:
        user_df[col] = 0
    user_df['Monthly_Query_Count_Past'] = 0

    user_df = user_df.merge(lookup_cum, how='left', on=uid_col, suffixes=('', '_lookup'))
    for col in ['Past_Case_Count', 'Past_Complaints_Cum', 'Complaint_Ratio',
                'Complaint_30d_Rolling', 'Vulnerable_CumCount']:
        user_df[col] = user_df[col + '_lookup'].fillna(0)
        user_df.drop(columns=[col + '_lookup'], inplace=True)

    user_df = user_df.merge(lookup_monthly, how='left', on=[uid_col, 'YearMonth'], suffixes=('', '_lookup'))
    user_df['Monthly_Query_Count_Past'] = user_df['Monthly_Query_Count_Past_lookup'].fillna(0)
    user_df.drop(columns=['Monthly_Query_Count_Past_lookup'], inplace=True)

    user_df['is_complaint'] = 0

    user_df["Vulnerable_Flag"] = user_df["Vulnerable Customer"].map(lambda x: 1 if str(x).strip().upper() == "Y" else 0)
    user_df["Vulnerable_CumCount_User"] = user_df.groupby(uid_col)["Vulnerable_Flag"].cumsum()
    user_df["Past_Case_Count_User"] = user_df.groupby(uid_col).cumcount()
    user_df["Vulnerable_Ratio"] = user_df["Vulnerable_CumCount_User"] / user_df["Past_Case_Count_User"].replace(0, np.nan)
    user_df["Vulnerable_Ratio"] = user_df["Vulnerable_Ratio"].fillna(0)
    user_df.drop(columns=["Vulnerable_Flag", "Vulnerable_CumCount_User", "Past_Case_Count_User"], inplace=True)

    def fix_title(row, mode_title_per_uid):
        if pd.isna(row["Title"]) or row["Title"] == "#ERROR!":
            return mode_title_per_uid.get(row[uid_col], "Unknown")
        return row["Title"]

    valid_titles = user_df.loc[~user_df["Title"].isin(["#ERROR!"]), [uid_col, "Title"]].dropna()
    mode_title_per_uid = valid_titles.groupby(uid_col)["Title"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    user_df["Title"] = user_df.apply(lambda row: fix_title(row, mode_title_per_uid), axis=1)
    user_df["Title"] = user_df["Title"].fillna("Unknown").str.upper().str.replace(r"\.", "", regex=True)

    male_titles = {"MR", "MSTR", "SIR", "LORD", "REV", "REVD", "CAPT", "BAR"}
    female_titles = {"MRS", "MISS", "MS", "LADY", "DAME"}
    neutral_titles = {"DR", "MX"}

    def map_gender(title):
        if title in male_titles:
            return "Male"
        elif title in female_titles:
            return "Female"
        elif title in neutral_titles or title == "UNKNOWN":
            return "Unknown"
        else:
            return "Unknown"

    user_df["Gender"] = user_df["Title"].apply(map_gender)

    user_df['Create_DayOfWeek'] = user_df['Create Date'].dt.dayofweek
    user_df['Create_Month'] = user_df['Create Date'].dt.month
    user_df['Consented_Flag'] = user_df['Consented/Non consented'].apply(lambda x: 1 if str(x).lower() == 'consented' else 0)
    user_df['Event_Location'] = user_df['Event Type'].astype(str) + '_' + user_df['Location'].astype(str)

    columns_to_drop = [
        'Report_Date', 'Case ID', 'Source', 'OneCode', 'Location', 'ClientName',
        'Process Group', 'Completes', 'Pend Case', 'Operational Location', 'Current Activity User',
        'Consented/Non consented', 'Start Date', 'Case Indicator', 'Mercer Consented',
        'Onshore/Offshore', 'Within SLA', 'Flag_Scheme', 'Title',
        'Vulnerable Customer', 'Event Type', 'Process Name', 'YearMonth', 'Vulnerable_CumCount'
    ]
    user_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    X_user = user_df.drop([uid_col, 'Create Date', 'is_complaint'], axis=1, errors='ignore')
    model_features = [
        'Portfolio', 'Scheme', 'Team Name', 'Current Outsourcing Team', 'Days to Target', 'Scan+2',
        'Site', 'Manual/RPA', 'Critical', 'Forthcoming Event', 'No of Days', 'Mercer Days',
        'Past_Case_Count', 'Vulnerable_Ratio', 'Gender', 'Create_DayOfWeek', 'Create_Month',
        'Past_Complaints_Cum', 'Complaint_Ratio', 'Event_Location', 'Complaint_30d_Rolling',
        'Consented_Flag', 'Monthly_Query_Count_Past'
    ]

    X_user = X_user[model_features]

    for col, le in le_dict.items():
        if col in X_user.columns:
            X_user[col] = X_user[col].astype(str)
            known_classes = set(le.classes_)
            X_user[col] = X_user[col].apply(lambda x: x if x in known_classes else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            X_user[col] = le.transform(X_user[col])

    X_user = X_user.replace([np.inf, -np.inf], np.nan).fillna(0)

    preds_proba = model.predict_proba(X_user)[:, 1]

    threshold = 0.97
    preds_label = (preds_proba > threshold).astype(int)

    user_df['Complaint_Probability'] = preds_proba
    user_df['Prediction'] = preds_label

    high_prob_df = user_df[user_df['Complaint_Probability'] >= threshold]

    # KPIs: total, potential complaints, complaint rate
    total_cases = len(user_df)
    potential_complaints = len(high_prob_df)
    complaint_rate = (potential_complaints / total_cases * 100) if total_cases > 0 else 0
    st.markdown(f"### Prediction Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Inquiries</div><div class="metric-value">{total_cases}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Potential Complaints</div><div class="metric-value">{potential_complaints}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Complaint Rate</div><div class="metric-value">{complaint_rate:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown(f"### Complaints Data")
    st.dataframe(high_prob_df[[uid_col, 'Complaint_Probability']], use_container_width=True)

    csv = high_prob_df.to_csv(index=False).encode()
    st.download_button(label="Download Predictions CSV", data=csv, file_name='complaint_predictions.csv', mime='text/csv')
