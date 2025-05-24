import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# --- Configuration ---
st.set_page_config(page_title="NSL-KDD Intrusion Detector", layout="wide")

# Initialize selected_features as empty; will be set after model training
selected_features = []

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
        
        # Create a copy for multi-class labels
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

# Load and preprocess data
data_load_state = st.info("Loading data and preprocessing...")
X, y_encoded, label_encoder, imputer, stats_df = load_and_preprocess_data()
data_load_state.empty()

# Check if data loaded successfully
if X is not None and y_encoded is not None and label_encoder is not None and imputer is not None and stats_df is not None:
    st.title("🤖 NSL-KDD Intrusion Detector")
    st.write("Use the sidebar to enter feature values and predict the type of network activity.")

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
        # Update selected_features with top 10 important features
        global selected_features
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': features.columns, 'importance': importances})
        selected_features = feature_importance_df.sort_values('importance', ascending=False)['feature'].head(10).tolist()
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
    st.dataframe(feature_importance_df.head(10), hide_index=True)

    st.subheader("Feature Importance Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importance from Random Forest')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)

    # --- Model Evaluation ---
    st.header("Model Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    y_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm, x=label_encoder.classes_.tolist(), y=label_encoder.classes_.tolist(),
        colorscale='Blues', showscale=True
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
    st.plotly_chart(fig)

    st.subheader("ROC Curve")
    y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
    y_pred_proba = rf_model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(fig)
    auc_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
    st.write(f"AUC Score: {auc_score:.3f}")

    # --- Input Features Section (Sidebar, Using Sliders) ---
    with st.sidebar:
        st.header("Input Features")
        st.write("Adjust the values below to get a prediction.")
        st.write("Ranges based on training data statistics (normalized).")
        input_data = {}
        # Ensure only valid features are used
        valid_features = [f for f in selected_features if f in stats_df.index]
        if not valid_features:
            st.error("No valid features available for input. Please check feature importance.")
        else:
            for feature in valid_features:
                min_val = float(stats_df.loc[feature, 'min'])
                max_val = float(stats_df.loc[feature, 'max'])
                median_val = float(stats_df.loc[feature, '50%'])
                if min_val == max_val:
                    input_data[feature] = st.number_input(f"{feature}", value=min_val, key=f"input_{feature}")
                else:
                    input_data[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=median_val, key=f"input_{feature}")

            # --- Prediction Button ---
            st.header("Detect Intrusion")
            st.info("Use the button below to detect intrusion in the input network data.")
            if st.button("Detect Activity"):
                # Prepare input data for prediction
                input_df = pd.DataFrame([input_data])
                # Ensure input_df has all columns in X (fill missing with 0)
                for col in X.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[X.columns]  # Reorder to match training data
                input_imputed = imputer.transform(input_df)
                input_processed_df = pd.DataFrame(input_imputed, columns=X.columns)

                # Make prediction
                prediction_encoded = rf_model.predict(input_processed_df)
                predicted_label_raw = label_encoder.inverse_transform(prediction_encoded)[0]

                # Display prediction
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
