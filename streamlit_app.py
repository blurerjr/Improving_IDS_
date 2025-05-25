import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # New import for SMOTE

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
        
        # Load test dataset from URL
        test_url = "https://raw.githubusercontent.com/blurerjr/Improving_IDS_/refs/heads/master/KDDTest%2B.txt"
        test_data = pd.read_csv(test_url, names=feature_names)

        # Drop 'difficulty' column from both datasets
        train_data = train_data.drop(['difficulty'], axis=1, errors='ignore')
        test_data = test_data.drop(['difficulty'], axis=1, errors='ignore')
                                  
        # Compute derived features for both datasets
        train_data['byte_ratio'] = train_data['src_bytes'] / (train_data['dst_bytes'] + 1)  # Avoid division by zero
        train_data['byte_diff'] = train_data['src_bytes'] - train_data['dst_bytes']
        
        test_data['byte_ratio'] = test_data['src_bytes'] / (test_data['dst_bytes'] + 1)
        test_data['byte_diff'] = test_data['src_bytes'] - test_data['dst_bytes']
        
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
            
        train_data_processed = change_label(train_data.copy())
        test_data_processed = change_label(test_data.copy())

        # Combine for consistent encoding/scaling, then split back
        # Identify all columns, including newly engineered ones, that will be numerical
        # Exclude 'label' temporarily from numeric cols list for scaling
        all_numeric_cols = [col for col in train_data_processed.select_dtypes(include=np.number).columns if col != 'label']
        all_categorical_cols = ['protocol_type','service','flag']

        # Fit StandardScaler on the training data's numerical columns
        std_scaler = StandardScaler()
        train_data_processed[all_numeric_cols] = std_scaler.fit_transform(train_data_processed[all_numeric_cols])
        # Transform test data using the scaler fitted on training data
        test_data_processed[all_numeric_cols] = std_scaler.transform(test_data_processed[all_numeric_cols])
        
        # Encode labels (fit on combined labels to ensure all unique labels are seen)
        label_encoder = LabelEncoder()
        # Fit on all unique labels from both train and test to avoid errors with unseen labels later
        all_labels = pd.concat([train_data_processed['label'], test_data_processed['label']]).unique()
        label_encoder.fit(all_labels)

        train_data_processed['intrusion'] = label_encoder.transform(train_data_processed['label'])
        test_data_processed['intrusion'] = label_encoder.transform(test_data_processed['label'])

        train_data_processed = train_data_processed.drop(labels=['label'], axis=1)
        test_data_processed = test_data_processed.drop(labels=['label'], axis=1)
        
        # One-hot encode categorical columns for both train and test
        # Use pd.concat and then get_dummies to ensure consistent columns after one-hot encoding across both datasets
        combined_ohe = pd.concat([train_data_processed, test_data_processed], ignore_index=True)
        combined_ohe = pd.get_dummies(combined_ohe, columns=all_categorical_cols, prefix="", prefix_sep="")

        # Separate X_train, y_train, X_test, y_test after one-hot encoding
        X_train = combined_ohe.iloc[:len(train_data_processed)].drop(labels=['intrusion'], axis=1)
        y_train = combined_ohe.iloc[:len(train_data_processed)]['intrusion']
        
        X_test = combined_ohe.iloc[len(train_data_processed):].drop(labels=['intrusion'], axis=1)
        y_test = combined_ohe.iloc[len(train_data_processed):]['intrusion']

        # Align columns - crucial for consistent feature sets
        # This handles cases where test data might have categories not present in train, or vice-versa
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        missing_in_test = list(train_cols - test_cols)
        for col in missing_in_test:
            X_test[col] = 0

        missing_in_train = list(test_cols - train_cols)
        for col in missing_in_train:
            X_train[col] = 0

        X_test = X_test[X_train.columns] # Ensure column order is the same

        # Impute missing values (fit on X_train, transform both)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        
        X_test_imputed = imputer.transform(X_test)
        X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
        
        # Calculate feature statistics for slider ranges (from the scaled training data)
        # Ensure only original numerical features are used for stats_df
        stats_df = X_train[numerical_features].describe().loc[['min', 'max', '50%']].transpose()
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Return all necessary components
        return X_train, y_train, X_test, y_test, label_encoder, imputer, stats_df, std_scaler, all_numeric_cols
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None, None, None, None, None, None, None

# Load and preprocess the data
data_load_state = st.info("Loading data and preprocessing...")
try:
    X_train, y_train, X_test, y_test, label_encoder, imputer, stats_df, std_scaler, original_numeric_cols = load_and_preprocess_data()
    data_load_state.empty()
except Exception as e:
    data_load_state.error(f"An error occurred during data loading and preprocessing: {e}")
    X_train, y_train, X_test, y_test, label_encoder, imputer, stats_df, std_scaler, original_numeric_cols = None, None, None, None, None, None, None, None, None

# Check if data loaded successfully
if X_train is not None and y_train is not None and X_test is not None and y_test is not None and label_encoder is not None and imputer is not None and stats_df is not None and std_scaler is not None and original_numeric_cols is not None:
    st.title("🤖 NSL-KDD Intrusion Detector")
    st.write("Use the sidebar to enter the feature values and predict the type of activity.")
    st.write("Tip: For attacks, try high values for `count`, `diff_srv_rate` (original scale), or select `service=private`, `flag=S0`, `protocol_type=tcp`.")

    # --- Optional: Display Raw Data ---
    with st.expander('Show Preprocessed Training Data Sample & Class Distribution'):
        st.write("This is a sample of the preprocessed training data used for the model.")
        st.dataframe(X_train.head())
        
        st.write("Training Data Class Distribution (Before SMOTE):")
        # Get actual labels from encoded labels
        actual_labels = label_encoder.inverse_transform(y_train)
        label_counts = pd.Series(actual_labels).value_counts()
        st.write(label_counts)
        if 'normal' in label_counts:
            normal_percentage = label_counts['normal'] / len(y_train) * 100
            st.write(f"Percentage of Normal activity in training data: **{normal_percentage:.2f}%**")
        else:
            st.info("No 'normal' class found in training data.")

    # --- Model Training ---
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model with SMOTE...")
        
        # Apply SMOTE - REMOVED n_jobs
        smote = SMOTE(random_state=42) 
        st.info("Applying SMOTE to balance the training data...")
        X_resampled, y_resampled = smote.fit_resample(features, target)
        st.success(f"SMOTE applied. Original samples: {len(features)}, Resampled samples: {len(X_resampled)}")
        
        st.info("Fitting Random Forest model on resampled data...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_resampled, y_resampled)
        st.success("Model training complete!")
        return model

    # Train the model
    rf_model = train_model(X_train, y_train)

    # --- Model Evaluation ---
    st.header("Model Performance on Test Data")
    st.write("Evaluating the model on unseen test data (`KDDTest+.txt`) to assess its generalization ability.")

    test_predictions_encoded = rf_model.predict(X_test)
    test_true_labels = label_encoder.inverse_transform(y_test)
    test_predicted_labels = label_encoder.inverse_transform(test_predictions_encoded)

    st.subheader("Classification Report")
    report = classification_report(test_true_labels, test_predicted_labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, subset=pd.IndexSlice[['precision', 'recall', 'f1-score'], :]))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(test_true_labels, test_predicted_labels, labels=label_encoder.classes_)
    
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix for Test Data')
    plt.tight_layout()
    st.pyplot(fig_cm)


    # --- Feature Importance ---
    st.header("Feature Importance")
    st.write("Shows which features the model considers most important for classification.")

    importances = rf_model.feature_importances_
    feature_names_model = X_train.columns # Use training data columns for feature names
    feature_importance_df = pd.DataFrame({'feature': feature_names_model, 'importance': importances})
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
        valid_numerical_features_for_sliders = [f for f in numerical_features if f in X_train.columns and f in stats_df.index]
        
        if not valid_numerical_features_for_sliders:
            st.error("No valid numerical features available for input. Please check feature selection.")
        else:
            for feature in valid_numerical_features_for_sliders:
                min_val = float(stats_df.loc[feature, 'min'])
                max_val = float(stats_df.loc[feature, 'max'])
                median_val = float(stats_df.loc[feature, '50%'])
                
                # Initialize session state for consistent slider values on rerun
                if f"sidebar_input_{feature}" not in st.session_state:
                    st.session_state[f"sidebar_input_{feature}"] = median_val

                if min_val == max_val: 
                    input_data[feature] = st.number_input(
                        f"{feature} (scaled)", 
                        value=st.session_state[f"sidebar_input_{feature}"], 
                        key=f"sidebar_input_{feature}"
                    )
                else:
                    input_data[feature] = st.slider(
                        f"{feature} (scaled)", 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=st.session_state[f"sidebar_input_{feature}"], 
                        key=f"sidebar_input_{feature}"
                    )

        # Categorical inputs (dropdowns)
        st.subheader("Categorical Features")
        categorical_inputs = {}
        
        if "sidebar_input_service" not in st.session_state:
            st.session_state["sidebar_input_service"] = service_options[0]
        if "sidebar_input_flag" not in st.session_state:
            st.session_state["sidebar_input_flag"] = flag_options[0]
        if "sidebar_input_protocol_type" not in st.session_state:
            st.session_state["sidebar_input_protocol_type"] = protocol_type_options[0]

        categorical_inputs['service'] = st.selectbox("Service", options=service_options, key="sidebar_input_service")
        categorical_inputs['flag'] = st.selectbox("Flag", options=flag_options, key="sidebar_input_flag")
        categorical_inputs['protocol_type'] = st.selectbox("Protocol Type", options=protocol_type_options, key="sidebar_input_protocol_type")

        # Test attack input button
        if st.button("Test Attack Input (Dos-like)"):
            # Define original, unscaled values for a Dos attack scenario
            unscaled_dos_input = {
                'byte_ratio': 5000.0,
                'src_bytes': 100000.0,
                'byte_diff': 99000.0,
                'diff_srv_rate': 0.8,
                'same_srv_rate': 0.1,
                'dst_bytes': 100.0,
                'dst_host_diff_srv_rate': 0.7,
                'dst_host_srv_count': 5.0,
                'count': 250.0
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

            # Update the session_state for the sliders with the scaled values
            for feature, value in scaled_dos_input_dict.items():
                if feature in input_data: # Only update if it's one of the features in numerical_features
                    st.session_state[f"sidebar_input_{feature}"] = value

            # Update session_state for categorical dropdowns
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
            # Create a DataFrame with all columns initialized to 0.0, based on the model's expected training columns
            full_input_df = pd.DataFrame(0.0, index=[0], columns=X_train.columns)

            # Update numerical features (these are already scaled from sliders or test attack button)
            for col, value in input_data.items():
                if col in full_input_df.columns:
                    full_input_df[col] = value
                # No warning here as it's expected that not all X_train columns are part of input_data
            
            # Update categorical features (set one-hot encoded columns)
            # Iterate through actual columns of X_train to ensure we match correct OHE names
            for feature, value in categorical_inputs.items():
                if feature == 'service':
                    col_name = f"_{value}"
                    if col_name in full_input_df.columns:
                        full_input_df[col_name] = 1.0
                    elif value == 'other': # Handle 'other' specifically if it might not have a leading underscore
                        if '_other' in full_input_df.columns:
                            full_input_df['_other'] = 1.0
                        else:
                            st.warning(f"One-hot encoded column for 'other' service not found.")
                    else:
                        st.warning(f"One-hot encoded column '{col_name}' for service '{value}' not found.")
                elif feature in ['flag', 'protocol_type']:
                    col_name = f"_{value}"
                    if col_name in full_input_df.columns:
                        full_input_df[col_name] = 1.0
                    else:
                        st.warning(f"One-hot encoded column '{col_name}' for {feature} '{value}' not found.")

            # Impute and predict
            try:
                input_imputed = imputer.transform(full_input_df)
                input_processed_df = pd.DataFrame(input_imputed, columns=X_train.columns)
            except ValueError as ve:
                st.error(f"Error during imputation. This can happen if input columns do not match training columns. Details: {ve}")
                st.stop() 

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
