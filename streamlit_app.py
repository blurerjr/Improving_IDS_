import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from io import StringIO # For handling string data as file

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
    "count",
    "protocol_type"
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
categorical_features = ["service", "flag", "protocol_type"]

# Define common values for dropdowns (based on NSL-KDD dataset)
service_options = [
    "http", "ftp", "telnet", "smtp", "finger", "domain_u", "auth", "pop_3",
    "ftp_data", "other", "private", "domain", "echo", "irc", "ssh"
]
flag_options = [
    "SF", "S0", "REJ", "RSTR", "RSTO", "S1", "S2", "S3", "OTH", "RSTOS0", "SH"
]
protocol_type_options = ["tcp", "udp", "icmp"]

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    try:
        # Column names
        feature_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
                         "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                         "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
                         "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                         "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                         "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                         "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                         "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

        # Load training dataset from URL
        train_url = "https://raw.githubusercontent.com/blurerjr/Improving_IDS_/refs/heads/master/KDDTrain%2B.txt"
        train_data = pd.read_csv(train_url, names=feature_names)
        
        # Load test dataset from URL (for simulating real-world data distribution)
        test_url = "https://raw.githubusercontent.com/blurerjr/Improving_IDS_/refs/heads/master/KDDTest%2B.txt"
        test_data = pd.read_csv(test_url, names=feature_names)

        # Combine train and test data for consistent preprocessing (excluding 'difficulty' for now)
        # We will separate them again later for training/testing.
        combined_data = pd.concat([train_data.drop(columns=['difficulty'], errors='ignore'), 
                                   test_data.drop(columns=['difficulty'], errors='ignore')], 
                                  ignore_index=True)
                                  
        # Compute derived features
        combined_data['byte_ratio'] = combined_data['src_bytes'] / (combined_data['dst_bytes'] + 1)  # Avoid division by zero
        combined_data['byte_diff'] = combined_data['src_bytes'] - combined_data['dst_bytes']
        
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
            
        combined_data = change_label(combined_data)
        
        # Create a copy for processing
        multi_data = combined_data.copy()
        
        # Identify numerical columns before encoding
        numeric_col = multi_data.select_dtypes(include=np.number).columns.tolist()
        # Remove 'intrusion' or 'label' if present in numeric_col
        if 'intrusion' in numeric_col:
            numeric_col.remove('intrusion')
        if 'label' in numeric_col: # Should not be here but a safeguard
            numeric_col.remove('label')

        # Initialize and fit StandardScaler on numerical data
        std_scaler = StandardScaler()
        multi_data[numeric_col] = std_scaler.fit_transform(multi_data[numeric_col])
        
        # Encode labels
        label_encoder = LabelEncoder()
        multi_data['intrusion'] = label_encoder.fit_transform(multi_data['label'])
        multi_data = multi_data.drop(labels=['label'], axis=1)
        
        # One-hot encode categorical columns
        # Temporarily store original categorical columns for proper one-hot encoding
        original_categorical_cols = ['protocol_type', 'service', 'flag']
        multi_data = pd.get_dummies(multi_data, columns=original_categorical_cols, prefix="", prefix_sep="")
        
        # Separate features and labels for training
        X_train = multi_data.iloc[:len(train_data)].drop(labels=['intrusion'], axis=1)
        y_train = multi_data.iloc[:len(train_data)]['intrusion']

        # Impute missing values (fit on X_train)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        
        # Calculate feature statistics for slider ranges (from the scaled training data)
        stats_df = X_train[numerical_features].describe().loc[['min', 'max', '50%']].transpose()
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle potential inf/NaN from scaling

        return X_train, y_train, label_encoder, imputer, stats_df, std_scaler, numeric_col # Return scaler and original numeric columns
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None, None, None, None, None

# Load and preprocess the data, and get feature statistics
data_load_state = st.info("Loading data and preprocessing...")
try:
    X, y_encoded, label_encoder, imputer, stats_df, std_scaler, original_numeric_cols = load_and_preprocess_data()
    data_load_state.empty()
except Exception as e:
    data_load_state.error(f"An error occurred during data loading and preprocessing: {e}")
    X, y_encoded, label_encoder, imputer, stats_df, std_scaler, original_numeric_cols = None, None, None, None, None, None, None

# Check if data loaded successfully
if X is not None and y_encoded is not None and label_encoder is not None and imputer is not None and stats_df is not None and std_scaler is not None and original_numeric_cols is not None:
    st.title("🤖 NSL-KDD Intrusion Detector")
    st.write("Use the sidebar to enter the feature values and predict the type of activity.")
    st.write("Tip: For attacks, try high values for `count`, `diff_srv_rate`, or select `service=private`, `flag=S0`, `protocol_type=tcp`.")

    # --- Optional: Display Raw Data ---
    with st.expander('Show Preprocessed Training Data Sample'):
        st.write("This is a sample of the preprocessed training data used for the model.")
        st.dataframe(X.head())
        st.write("Label counts in training data:")
        st.write(pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts())

    # --- Model Training ---
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
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
        st.write("Numerical ranges are normalized (scaled). High `count` or `diff_srv_rate` (original scale) may indicate attacks.")

        # Numerical inputs (sliders)
        input_data = {}
        # Filter for features that are both numerical and exist in the stats_df index
        valid_numerical_features_for_sliders = [f for f in numerical_features if f in X.columns and f in stats_df.index]
        
        if not valid_numerical_features_for_sliders:
            st.error("No valid numerical features available for input. Please check feature selection.")
        else:
            for feature in valid_numerical_features_for_sliders:
                min_val = float(stats_df.loc[feature, 'min'])
                max_val = float(stats_df.loc[feature, 'max'])
                median_val = float(stats_df.loc[feature, '50%'])
                if min_val == max_val: # Handle cases where range is zero (e.g., all values are same after scaling)
                    input_data[feature] = st.number_input(f"{feature} (scaled)", value=min_val, key=f"sidebar_input_{feature}")
                else:
                    input_data[feature] = st.slider(f"{feature} (scaled)", min_value=min_val, max_value=max_val, value=median_val, key=f"sidebar_input_{feature}")

        # Categorical inputs (dropdowns)
        st.subheader("Categorical Features")
        categorical_inputs = {}
        categorical_inputs['service'] = st.selectbox("Service", options=service_options, key="sidebar_input_service")
        categorical_inputs['flag'] = st.selectbox("Flag", options=flag_options, key="sidebar_input_flag")
        categorical_inputs['protocol_type'] = st.selectbox("Protocol Type", options=protocol_type_options, key="sidebar_input_protocol_type")

        # Test attack input button
        if st.button("Test Attack Input (Dos-like)"):
            # Define original, unscaled values for a Dos attack scenario
            unscaled_dos_input = {
                'byte_ratio': 5000.0, # Example: high src_bytes, low dst_bytes -> high ratio
                'src_bytes': 100000.0, # Example: large source bytes
                'byte_diff': 99000.0,  # Example: large difference
                'diff_srv_rate': 0.8, # Example: relatively high difference in service rate
                'same_srv_rate': 0.1, # Example: relatively low same service rate
                'dst_bytes': 100.0,   # Example: small destination bytes
                'dst_host_diff_srv_rate': 0.7, # Example: relatively high diff service rate on dest host
                'dst_host_srv_count': 5.0,  # Example: low service count on dest host
                'count': 250.0        # Example: high connection count
            }
            
            # Create a temporary DataFrame for scaling only the numerical values
            temp_df_for_scaling = pd.DataFrame([unscaled_dos_input])
            
            # Ensure all numerical columns present in `original_numeric_cols` are in `temp_df_for_scaling`
            # Pad with zeros for missing columns in the specific input scenario
            for col in original_numeric_cols:
                if col not in temp_df_for_scaling.columns:
                    temp_df_for_scaling[col] = 0.0
            
            # Reorder columns to match the scaler's fit order (CRITICAL!)
            temp_df_for_scaling = temp_df_for_scaling[original_numeric_cols]

            # Scale the numerical inputs using the original scaler
            scaled_dos_input_array = std_scaler.transform(temp_df_for_scaling)
            scaled_dos_input_dict = pd.DataFrame(scaled_dos_input_array, columns=original_numeric_cols).iloc[0].to_dict()

            # Update the input_data dictionary with the scaled values for the sliders
            for feature, value in scaled_dos_input_dict.items():
                if feature in input_data: # Only update if it's one of the selected features for sliders
                    input_data[feature] = value
                    # Set the slider value programmatically
                    st.session_state[f"sidebar_input_{feature}"] = value

            # Update categorical inputs for the dropdowns
            categorical_inputs['service'] = 'private'
            categorical_inputs['flag'] = 'S0'
            categorical_inputs['protocol_type'] = 'tcp'
            
            # Set the selectbox values programmatically
            st.session_state["sidebar_input_service"] = 'private'
            st.session_state["sidebar_input_flag"] = 'S0'
            st.session_state["sidebar_input_protocol_type"] = 'tcp'
            
            st.experimental_rerun() # Rerun to update the sidebar values

        # --- Prediction Button ---
        st.header("Detect Intrusion")
        st.info("Use the button below to detect intrusion in the input network data.")
        
        # Placeholder for the prediction result display in the main area
        prediction_result_placeholder = st.empty()

        if st.button("Detect Activity"):
            # Prepare input data for prediction efficiently
            # Create a DataFrame with all columns initialized to 0.0, based on the model's expected columns
            full_input_df = pd.DataFrame(0.0, index=[0], columns=X.columns)

            # Update numerical features (these are already scaled from sliders or test attack button)
            for col, value in input_data.items():
                if col in full_input_df.columns:
                    full_input_df[col] = value
                else:
                    st.warning(f"Numerical feature '{col}' from input_data not found in model's expected columns.")

            # Update categorical features (set one-hot encoded columns)
            for feature, value in categorical_inputs.items():
                if feature == 'service':
                    # NSL-KDD has 'other' service, ensure it's handled correctly
                    col_name = f"_{value}" if value != 'other' else '_other' # One-hot encoded column names are like '_http', '_ftp'
                    if col_name in full_input_df.columns:
                        full_input_df[col_name] = 1.0
                    else:
                        st.warning(f"One-hot encoded column '{col_name}' for service '{value}' not found.")
                elif feature == 'flag' or feature == 'protocol_type':
                    col_name = f"_{value}" # One-hot encoded column names are like '_SF', '_tcp'
                    if col_name in full_input_df.columns:
                        full_input_df[col_name] = 1.0
                    else:
                        st.warning(f"One-hot encoded column '{col_name}' for {feature} '{value}' not found.")

            # Impute and predict
            # Ensure the order of columns matches the order during training after imputation
            try:
                input_imputed = imputer.transform(full_input_df)
                input_processed_df = pd.DataFrame(input_imputed, columns=X.columns)
            except ValueError as ve:
                st.error(f"Error during imputation. This can happen if input columns do not match training columns. Details: {ve}")
                st.stop() # Stop execution if imputation fails

            # Make prediction
            prediction_encoded = rf_model.predict(input_processed_df)
            predicted_label_raw = label_encoder.inverse_transform(prediction_encoded)[0]

            # Display prediction
            with prediction_result_placeholder.container():
                st.subheader("Detection Result")
                formatted_label = predicted_label_raw.replace('_', ' ').title()
                if predicted_label_raw == 'normal':
                    st.success(f"Detected Activity: **{formatted_label}** ✅")
                    st.info("The model detects normal, non-intrusive network activity.")
                else:
                    st.warning(f"Detected Activity: **{formatted_label}** 🚨")
                    st.info(f"The model detects an intrusion of type: **{formatted_label}**.")
                    st.info("Proceed accordingly to mitigate the intrusion.")

                # Display prediction probabilities
                prediction_proba = rf_model.predict_proba(input_processed_df)
                proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
                st.write("Prediction Probabilities:")
                st.dataframe(proba_df)

else:
    st.error("App could not load data or train the model. Please check the data URLs and file formats.")
