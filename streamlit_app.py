import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- Configuration ---
st.set_page_config(page_title="NSL-KDD Intrusion Detector", layout="wide")

# Define the list of selected features based on provided importance
selected_features = [
    "byte_ratio",
    "src_bytes",
    "byte_diff",
    "service",
    "diff_srv_rate",
    "flag",
    "same_srv_rate",
    "dst_bytes",
    "dst_host_diff_srv_rate",
    "dst_host_srv_count",
    "count"
]

# Numerical features for sliders
numerical_features = [
    "byte_ratio",
    "src_bytes",
    "byte_diff",
    "diff_srv_rate",
    "same_srv_rate",
    "dst_bytes",
    "dst_host_diff_srv_rate",
    "dst_host_srv_count",
    "count"
]

# Categorical features for dropdowns
categorical_features = ["service", "flag"]

# Define common values for dropdowns (based on NSL-KDD dataset)
service_options = [
    "http", "ftp", "telnet", "smtp", "finger", "domain_u", "auth", "pop_3",
    "ftp_data", "other", "private", "domain", "echo", "irc", "ssh"
]
flag_options = [
    "SF", "S0", "REJ", "RSTR", "RSTO", "S1", "S2", "S3", "OTH", "RSTOS0", "SH"
]

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    try:
        # Column names
        feature = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
                   "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                   "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
                   "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                   "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                   "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                   "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

        # Load dataset from URL
        train_url = "https://raw.githubusercontent.com/blurerjr/Improving_IDS_/refs/heads/master/KDDTrain%2B.txt"
        train_data = pd.read_csv(train_url, names=feature)
        
        # Drop 'difficulty' column
        train_data = train_data.drop(['difficulty'], axis=1)
        
        # Compute derived features
        train_data['byte_ratio'] = train_data['src_bytes'] / (train_data['dst_bytes'] + 1)  # Avoid division by zero
        train_data['byte_diff'] = train_data['src_bytes'] - train_data['dst_bytes']
        
        # Function to convert attack labels
        def change_label(df):
            df = df.copy()
            df['label'] = df['label'].replace(
                ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'], 'Dos')
            df['label'] = df['label'].replace(
                ['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
                 'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'], 'R2L')
            df['label'] = df['label'].replace(
                ['ipsweep','mscan','nmap','portsweep','saint','satan'], 'Probe')
            df['label'] = df['label'].replace(
                ['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'], 'U2R')
            return df
        
        train_data = change_label(train_data)
        
        # Create a copy for processing
        multi_data = train_data.copy()
        
        # Normalize numerical columns
        std_scaler = StandardScaler()
        numeric_col = multi_data.select_dtypes(include='number').columns
        for col in numeric_col:
            arr = multi_data[col].values
            multi_data[col] = std_scaler.fit_transform(arr.reshape(-1, 1))
        
        # Encode labels
        label_encoder = LabelEncoder()
        multi_data['intrusion'] = label_encoder.fit_transform(multi_data['label'])
        multi_data = multi_data.drop(labels=['label'], axis=1)
        
        # One-hot encode categorical columns
        multi_data = pd.get_dummies(multi_data, columns=['protocol_type','service','flag'], prefix="", prefix_sep="")
        
        # Calculate feature statistics for slider ranges
        stats_df = multi_data.select_dtypes(include='number').describe().loc[['min', 'max', '50%']].transpose()
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Separate features and labels
        X = multi_data.drop(labels=['intrusion'], axis=1)
        y = multi_data['intrusion']
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        
        return X, y, label_encoder, imputer, stats_df
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None, None, None

# Load and preprocess the data, and get feature statistics
data_load_state = st.info("Loading data and preprocessing...")
try:
    X, y_encoded, label_encoder, imputer, stats_df = load_and_preprocess_data()
    data_load_state.empty()
except Exception as e:
    data_load_state.error(f"An error occurred during data loading and preprocessing: {e}")
    X, y_encoded, label_encoder, imputer, stats_df = None, None, None, None, None

# Check if data loaded successfully
if X is not None and y_encoded is not None and label_encoder is not None and imputer is not None and stats_df is not None:
    st.title("🤖 NSL-KDD Intrusion Detector")
    st.write("Use the sidebar to enter the feature values and predict the type of activity.")

    # --- Optional: Display Raw Data ---
    with st.expander('Show Raw Data (from GitHub)'):
        st.write("This is the preprocessed data loaded from your GitHub repository.")
        st.dataframe(X.head())
        st.write("Label counts:")
        st.write(pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts())

    # --- Model Training ---
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)
        st.success("Model training complete!")
        return model

    # Train the model
    rf_model = train_model(X, y_encoded)

    # --- Feature Importance ---
    st.header("Feature Importance")
    st.write("Shows which features the model considers most important for classification.")

    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    st.dataframe(feature_importance_df.head(20), hide_index=True)

    st.subheader("Feature Importance Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
    ax.set_title('Top 20 Feature Importance from Random Forest')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)

    # --- Input Features Section (Sidebar, Using Sliders and Dropdowns) ---
    with st.sidebar:
        st.header("Input Features")
        st.write("Adjust the sliders and dropdowns below to get a prediction.")
        st.write("Numerical ranges based on training data statistics.")

        # Numerical inputs (sliders)
        input_data = {}
        valid_numerical_features = [f for f in numerical_features if f in X.columns and f in stats_df.index]
        if not valid_numerical_features:
            st.error("No valid numerical features available for input. Please check feature selection.")
        else:
            for feature in valid_numerical_features:
                min_val = float(stats_df.loc[feature, 'min'])
                max_val = float(stats_df.loc[feature, 'max'])
                median_val = float(stats_df.loc[feature, '50%'])
                if min_val == max_val:
                    input_data[feature] = st.number_input(f"{feature}", value=min_val, key=f"sidebar_input_{feature}")
                else:
                    input_data[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=median_val, key=f"sidebar_input_{feature}")

        # Categorical inputs (dropdowns)
        st.subheader("Categorical Features")
        categorical_inputs = {}
        categorical_inputs['service'] = st.selectbox("Service", options=service_options, key="sidebar_input_service")
        categorical_inputs['flag'] = st.selectbox("Flag", options=flag_options, key="sidebar_input_flag")

        # --- Prediction Button ---
        st.header("Detect Intrusion")
        st.info("Use the button below to detect intrusion in the input network data.")
        if st.button("Detect Activity"):
            # Prepare input data for prediction efficiently
            input_df = pd.DataFrame([input_data])
            # Create a DataFrame with all columns initialized to 0
            full_input_df = pd.DataFrame(0.0, index=[0], columns=X.columns)
            # Update numerical features
            for col in input_data:
                full_input_df[col] = input_data[col]
            # Update categorical features (set one-hot encoded columns)
            for feature, value in categorical_inputs.items():
                if feature == 'service':
                    col_name = value if value != 'other' else 'other'
                    if col_name in X.columns:
                        full_input_df[col_name] = 1.0
                elif feature == 'flag':
                    if value in X.columns:
                        full_input_df[value] = 1.0
            
            # Impute and predict
            input_imputed = imputer.transform(full_input_df)
            input_processed_df = pd.DataFrame(input_imputed, columns=X.columns)

            # Make prediction
            prediction_encoded = rf_model.predict(input_processed_df)
            predicted_label_raw = label_encoder.inverse_transform(prediction_encoded)[0]

            # Beautify and display prediction
            st.subheader("Detection Result")
            formatted_label = predicted_label_raw.replace('_', ' ').title()
            if predicted_label_raw == 'normal':
                st.success(f"Detected Activity: **{formatted_label}** ✅")
                st.info("The model detects normal, non-intrusive network activity.")
            else:
                st.warning(f"Detected Activity: **{formatted_label}** 🚨")
                st.info(f"The model detects an intrusion of type: **{formatted_label}**.")
                st.info("Proceed accordingly to mitigate the intrusion.")

else:
    st.error("App could not load data or train the model. Please check the data URL and file format.")
