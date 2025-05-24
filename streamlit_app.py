import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# Streamlit app title
st.title("NSL-KDD Intrusion Detection with Random Forest")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42)
train_button = st.sidebar.button("Train Model")

# Function to load and preprocess data
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
        
        # Function to convert attack labels (avoiding inplace to prevent FutureWarning)
        def change_label(df):
            df = df.copy()  # Create a copy to avoid modifying the original
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
        le = LabelEncoder()
        multi_data['intrusion'] = le.fit_transform(multi_data['label'])
        multi_data = multi_data.drop(labels=['label'], axis=1)
        
        # One-hot encode categorical columns
        multi_data = pd.get_dummies(multi_data, columns=['protocol_type','service','flag'], prefix="", prefix_sep="")
        
        # Separate features and labels
        X_data = multi_data.drop(labels=['intrusion'], axis=1)
        y_data = multi_data['intrusion']
        
        return X_data, y_data, le
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None

# Load and preprocess data
X_data, y_data, label_encoder = load_and_preprocess_data()

# Check if data was loaded successfully
if X_data is None or y_data is None:
    st.stop()

# Display dataset info
st.subheader("Dataset Overview")
st.write("Shape of preprocessed data:", X_data.shape)
st.write("Label distribution:")
# Create a dictionary for label mapping to avoid TypeError
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
st.dataframe(pd.Series(y_data).value_counts().rename(index=label_mapping))

# Split data and train model when the user clicks the "Train Model" button
if train_button:
    with st.spinner("Training model..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)
            
            # RandomForest hyperparameter tuning
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
            
            rf_classifier = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, 
                                              n_iter=5, cv=3, verbose=1, n_jobs=-1, random_state=random_state)
            
            # Train model
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            
            # Display best parameters
            st.subheader("Model Results")
            st.write("Best parameters:", random_search.best_params_)
            
            # Evaluate model
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(
                z=cm, x=label_encoder.classes_.tolist(), y=label_encoder.classes_.tolist(),
                colorscale='Blues', showscale=True
            )
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
            st.plotly_chart(fig)
            
            # ROC Curve
            st.subheader("ROC Curve")
            y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
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
            
            # AUC Score
            auc_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
            st.write(f"AUC Score: {auc_score:.3f}")
        except Exception as e:
            st.error(f"Error during model training or evaluation: {str(e)}")
