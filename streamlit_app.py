import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model # For loading Keras models

# --- 1. Load Pre-trained Models and Preprocessors ---
# Define the paths where you saved your models and preprocessors
PREPROCESSOR_PATH = 'preprocessor.pkl'
DL_FEATURE_EXTRACTOR_PATH = 'dl_feature_extractor.h5'
ML_CLASSIFIER_PATH = 'ml_classifier.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

@st.cache_resource # Cache the loading of heavy resources
def load_assets():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        dl_feature_extractor = load_model(DL_FEATURE_EXTRACTOR_PATH)
        ml_classifier = joblib.load(ML_CLASSIFIER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return preprocessor, dl_feature_extractor, ml_classifier, label_encoder
    except FileNotFoundError:
        st.error("Error: Model or preprocessor files not found. "
                 "Please ensure 'preprocessor.pkl', 'dl_feature_extractor.h5', "
                 "'ml_classifier.pkl', and 'label_encoder.pkl' are in the same directory.")
        st.stop() # Stop the app execution if files are missing
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

preprocessor, dl_feature_extractor, ml_classifier, label_encoder = load_assets()

# Get the list of original feature names before one-hot encoding for the input form
# This assumes your original X DataFrame had these columns in this order
# If your original X was different, adjust this list!
original_feature_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Identify categorical features for Streamlit selectbox
categorical_features_for_streamlit = ['protocol_type', 'service', 'flag'] # These need select boxes
binary_features_for_streamlit = ['land', 'logged_in', 'is_host_login', 'is_guest_login',
                                 'root_shell', 'su_attempted'] # These are binary (0/1)

# List of all numerical features (excluding binary which are already handled)
numerical_features_for_streamlit = [col for col in original_feature_columns
                                    if col not in categorical_features_for_streamlit + binary_features_for_streamlit]

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="NIDS Attack Detector", layout="wide")

st.title("🛡️ Network Intrusion Detection System")
st.markdown("Enter network connection features to predict if it's normal or an attack type.")

# --- Input Form ---
st.header("Connection Details")

input_data = {}

# Layout inputs in columns for better organization
cols = st.columns(3) # Adjust number of columns as needed

# Function to get default values for numerical inputs
def get_default_value(feature_name):
    # These are arbitrary, you might want to calculate means from your training data
    if 'rate' in feature_name: return 0.0
    if 'bytes' in feature_name: return 0
    if 'count' in feature_name: return 1
    if feature_name == 'duration': return 0
    return 0 # Default for others

# Collect inputs
with cols[0]:
    st.subheader("General Metrics")
    input_data['duration'] = st.number_input("Duration (seconds)", value=0)
    input_data['src_bytes'] = st.number_input("Source Bytes", value=get_default_value('src_bytes'))
    input_data['dst_bytes'] = st.number_input("Destination Bytes", value=get_default_value('dst_bytes'))
    input_data['land'] = st.selectbox("Land (Same src/dst IP/Port)", [0, 1], index=0)
    input_data['wrong_fragment'] = st.number_input("Wrong Fragments", value=get_default_value('wrong_fragment'))
    input_data['urgent'] = st.number_input("Urgent Packets", value=get_default_value('urgent'))
    input_data['hot'] = st.number_input("Hot Indicators", value=get_default_value('hot'))
    input_data['num_failed_logins'] = st.number_input("Failed Logins", value=get_default_value('num_failed_logins'))
    input_data['logged_in'] = st.selectbox("Logged In", [0, 1], index=0)
    input_data['num_compromised'] = st.number_input("Compromised Conditions", value=get_default_value('num_compromised'))

with cols[1]:
    st.subheader("Service & Status")
    # You might want to get actual unique values from your training data for these
    # For now, providing common ones. Expand as needed.
    input_data['protocol_type'] = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
    input_data['service'] = st.selectbox("Service", ['http', 'ftp', 'smtp', 'dns', 'other', 'private', 'ecr_i', 'domain_udp', 'finger', 'telnet', 'auth', 'pop_3']) # Add more common services
    input_data['flag'] = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'OTH', 'SH', 'S1', 'S2', 'S3'])

    input_data['num_root'] = st.number_input("Root Accesses", value=get_default_value('num_root'))
    input_data['num_file_creations'] = st.number_input("File Creations", value=get_default_value('num_file_creations'))
    input_data['num_shells'] = st.number_input("Shells Opened", value=get_default_value('num_shells'))
    input_data['num_access_files'] = st.number_input("Access Files", value=get_default_value('num_access_files'))
    input_data['num_outbound_cmds'] = st.number_input("Outbound Commands", value=get_default_value('num_outbound_cmds'))
    input_data['is_host_login'] = st.selectbox("Is Host Login", [0, 1], index=0)
    input_data['is_guest_login'] = st.selectbox("Is Guest Login", [0, 1], index=0)

with cols[2]:
    st.subheader("Connection Statistics")
    input_data['count'] = st.number_input("Count of connections to same host", value=get_default_value('count'))
    input_data['srv_count'] = st.number_input("Count of connections to same service", value=get_default_value('srv_count'))
    input_data['serror_rate'] = st.number_input("SYN Error Rate", value=get_default_value('serror_rate'), format="%.4f")
    input_data['srv_serror_rate'] = st.number_input("SYN Error Rate (srv)", value=get_default_value('srv_serror_rate'), format="%.4f")
    input_data['rerror_rate'] = st.number_input("REJ Error Rate", value=get_default_value('rerror_rate'), format="%.4f")
    input_data['srv_rerror_rate'] = st.number_input("REJ Error Rate (srv)", value=get_default_value('srv_rerror_rate'), format="%.4f")
    input_data['same_srv_rate'] = st.number_input("Same Service Rate", value=get_default_value('same_srv_rate'), format="%.4f")
    input_data['diff_srv_rate'] = st.number_input("Diff Service Rate", value=get_default_value('diff_srv_rate'), format="%.4f")
    input_data['srv_diff_host_rate'] = st.number_input("Diff Host Rate (srv)", value=get_default_value('srv_diff_host_rate'), format="%.4f")
    input_data['dst_host_count'] = st.number_input("Dst Host Count", value=get_default_value('dst_host_count'))
    input_data['dst_host_srv_count'] = st.number_input("Dst Host Srv Count", value=get_default_value('dst_host_srv_count'))
    input_data['dst_host_same_srv_rate'] = st.number_input("Dst Host Same Srv Rate", value=get_default_value('dst_host_same_srv_rate'), format="%.4f")
    input_data['dst_host_diff_srv_rate'] = st.number_input("Dst Host Diff Srv Rate", value=get_default_value('dst_host_diff_srv_rate'), format="%.4f")
    input_data['dst_host_same_src_port_rate'] = st.number_input("Dst Host Same Src Port Rate", value=get_default_value('dst_host_same_src_port_rate'), format="%.4f")
    input_data['dst_host_srv_diff_host_rate'] = st.number_input("Dst Host Srv Diff Host Rate", value=get_default_value('dst_host_srv_diff_host_rate'), format="%.4f")
    input_data['dst_host_serror_rate'] = st.number_input("Dst Host Serror Rate", value=get_default_value('dst_host_serror_rate'), format="%.4f")
    input_data['dst_host_srv_serror_rate'] = st.number_input("Dst Host Srv Serror Rate", value=get_default_value('dst_host_srv_serror_rate'), format="%.4f")
    input_data['dst_host_rerror_rate'] = st.number_input("Dst Host Rerror Rate", value=get_default_value('dst_host_rerror_rate'), format="%.4f")
    input_data['dst_host_srv_rerror_rate'] = st.number_input("Dst Host Srv Rerror Rate", value=get_default_value('dst_host_srv_rerror_rate'), format="%.4f")


# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Ensure the order of columns in input_df matches the original training data's features (X)
# This is CRUCIAL for the preprocessor
input_df = input_df[original_feature_columns]

if st.button("Predict Attack Type"):
    st.subheader("Prediction Result")
    try:
        # --- 3. Apply Preprocessing ---
        # The preprocessor (ColumnTransformer) expects the original column names
        processed_input = preprocessor.transform(input_df)

        # --- 4. Deep Learning Feature Extraction ---
        dl_features = dl_feature_extractor.predict(processed_input)

        # --- 5. Machine Learning Classification ---
        prediction_numerical = ml_classifier.predict(dl_features)
        prediction_proba = ml_classifier.predict_proba(dl_features) # Get probabilities

        # Inverse transform the numerical prediction to the original class name
        predicted_category = label_encoder.inverse_transform(prediction_numerical)[0]

        st.success(f"Predicted Attack Category: **{predicted_category}**")

        st.subheader("Prediction Probabilities")
        # Display probabilities for each class
        probabilities_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': prediction_proba[0]
        }).sort_values(by='Probability', ascending=False)
        st.dataframe(probabilities_df)

        st.markdown("""
        **Interpretation:**
        - **Normal**: Legitimate network traffic.
        - **DoS (Denial of Service)**: Attempts to make a machine or network resource unavailable to its intended users.
        - **Probe (Surveillance/Scanning)**: Attempts to gather information about the network or systems.
        - **R2L (Remote to Local)**: Unauthorized access from a remote machine to a local user account.
        - **U2R (User to Root)**: Unauthorized access from a local user account to superuser (root) privileges.
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are valid and reflect typical network connection values.")

st.markdown("---")
st.write("Developed for Network Intrusion Detection using Deep Learning Feature Extraction + Machine Learning Classification.")
