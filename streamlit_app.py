import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA # Import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
st.set_page_config(page_title="Network Intrusion Detector", layout="wide")

# --- Define Dataset URL and Features ---
# IMPORTANT: Replace this with your actual dataset URL.
dataset_url = 'https://raw.githubusercontent.com/blurerjr/Improving_IDS_/refs/heads/master/KDDTrain%2B.txt' # REPLACED WITH YOUR PROVIDED URL

# Define ALL feature names in your dataset (including the target column)
all_dataset_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'
]

# Identify numerical and categorical features (excluding the target 'outcome')
numerical_features = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'level'
]
categorical_features = [
    'protocol_type', 'service', 'flag'
]
target_column = 'outcome'

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    try:
        # Added sep=',' to correctly parse the .txt file as a CSV
        # Added engine='python' and skipinitialspace=True for more robust parsing of .txt files
        df = pd.read_csv(dataset_url, names=all_dataset_columns, header=None, sep=',', engine='python', skipinitialspace=True)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # --- Preprocessing Pipeline with PCA ---
        # Numerical pipeline: Imputation -> Scaling -> PCA
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()), # It's good practice to scale before PCA
            ('pca', PCA(n_components=0.95)) # PCA to retain 95% of variance
        ])

        # Categorical pipeline: One-hot encoding
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create a preprocessor using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough'
        )

        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)

        # Get feature names after all transformations for the model
        # PCA components will be named 'pca__component_X'
        pca_component_names = [f'pca__component_{i}' for i in range(preprocessor.named_transformers_['num'].named_steps['pca'].n_components_)]
        onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        final_model_feature_names = pca_component_names + list(onehot_feature_names)

        X_processed_df = pd.DataFrame(X_processed, columns=final_model_feature_names)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Calculate feature statistics for slider ranges (for original numerical features)
        stats_df = X[numerical_features].describe().loc[['min', 'max', '50%']].transpose()
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X_processed_df, y_encoded, label_encoder, preprocessor, stats_df

    except Exception as e:
        st.error(f"Error loading or preprocessing data from {dataset_url}: {e}")
        return None, None, None, None, None

# Import Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline

# Load and preprocess the data
data_load_state = st.info("Loading and preprocessing dataset...")
try:
    X_model_input, y_encoded, label_encoder, preprocessor, stats_df = load_and_preprocess_data()
    data_load_state.empty()
except Exception as e:
    data_load_state.error(f"An error occurred during data loading or preprocessing: {e}")
    X_model_input, y_encoded, label_encoder, preprocessor, stats_df = None, None, None, None, None


# Check if data loaded successfully
if X_model_input is not None and y_encoded is not None and label_encoder is not None and preprocessor is not None and stats_df is not None:

    st.title("🤖 Network Intrusion Detector")
    st.write("Use the sidebar to enter network feature values and predict intrusion type.")

    # --- Optional: Display Preprocessed Data ---
    with st.expander('Show Preprocessed Data (Model Input)'):
        st.write("This is the head of the data after PCA and One-Hot Encoding, used for model training.")
        st.dataframe(X_model_input.head())
        st.write(f"Total features for model: {X_model_input.shape[1]}")
        st.write("Attack Type counts:")
        st.write(pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts())


    # --- Model Training ---
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model for intrusion detection...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)
        st.success("Model training complete!")
        return model

    # Train the model
    rf_model = train_model(X_model_input, y_encoded)

    # --- Feature Importance ---
    st.header("Feature Importance")
    st.write("Shows which transformed features (PCA components and one-hot encoded) the model considers most important for intrusion detection.")

    final_feature_names_for_importance = X_model_input.columns.tolist()

    if len(final_feature_names_for_importance) == rf_model.feature_importances_.shape[0]:
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': final_feature_names_for_importance, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        st.dataframe(feature_importance_df, hide_index=True)

        st.subheader("Feature Importance Bar Chart")
        # Display top 20 or adjust as needed, as there could be many one-hot encoded features
        fig, ax = plt.subplots(figsize=(12, min(len(feature_importance_df), 20) * 0.5)) # Dynamic height
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
        ax.set_title('Top Transformed Features Importance from Random Forest')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Could not match feature names for importance display. Check preprocessing steps.")


    # --- Input Features Section (Moved to Sidebar, Using Sliders/Selectboxes for ORIGINAL features) ---
    with st.sidebar:
        st.header("Input Network Features")
        st.write("Adjust the values below to get a prediction.")
        st.write("Ranges for numerical features are based on training data statistics.")

        input_data = {}

        # Input for Numerical Features
        st.subheader("Numerical Features")
        for feature in numerical_features:
            min_val = float(stats_df.loc[feature, 'min'])
            max_val = float(stats_df.loc[feature, 'max'])
            median_val = float(stats_df.loc[feature, '50%'])

            if min_val == max_val:
                 input_data[feature] = st.number_input(f"{feature}", value=min_val, key=f"sidebar_num_input_{feature}")
            else:
                 input_data[feature] = st.slider(f"{feature}",
                                                  min_value=min_val,
                                                  max_value=max_val,
                                                  value=median_val,
                                                  key=f"sidebar_slider_{feature}")

        # Input for Categorical Features
        st.subheader("Categorical Features")
        for feature in categorical_features:
            # Get unique categories from the preprocessor's fitted encoder
            # This is safer than hardcoding if categories vary
            # Access the OneHotEncoder directly from the preprocessor's pipeline
            cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_idx = categorical_features.index(feature)
            categories = cat_transformer.categories_[cat_idx].tolist()

            input_data[feature] = st.selectbox(f"{feature}", options=categories, key=f"sidebar_cat_input_{feature}")


        # --- Prediction Button (Moved to Sidebar) ---
        st.header("Get Prediction")
        predict_button_clicked = st.button("Predict Intrusion")


    # --- Prediction Result Display (Moved to Main Area) ---
    if predict_button_clicked:
        # --- Prepare Input Data for Prediction ---
        # Create a DataFrame from current input values
        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns in input_df matches the original X columns
        # before passing to the preprocessor
        input_df_ordered = input_df[numerical_features + categorical_features]

        # Apply the same preprocessor (imputer, scaler, pca, one-hot encoder) to the input data
        input_processed_for_prediction = preprocessor.transform(input_df_ordered)

        # The model expects a DataFrame with the same column names as X_model_input
        input_processed_df_for_prediction = pd.DataFrame(input_processed_for_prediction, columns=X_model_input.columns)


        # --- Make Prediction ---
        prediction_encoded = rf_model.predict(input_processed_df_for_prediction)
        predicted_label_raw = label_encoder.inverse_transform(prediction_encoded)[0]

        # --- Beautify and Display Prediction ---
        st.subheader("Prediction Result")
        formatted_label = predicted_label_raw.replace('_', ' ').title()

        if 'Normal' in formatted_label or 'Benign' in formatted_label: # Adjust based on your 'normal' class name
            st.success(f"Predicted Activity: **{formatted_label}** ✅")
            st.info("The model predicts normal network traffic.")
        else:
            st.error(f"Predicted Activity: **{formatted_label}** 🚨")
            st.info(f"The model predicts a network intrusion of type: **{formatted_label}**.")

        # Optional: Display prediction probabilities
        # prediction_proba = rf_model.predict_proba(input_processed_df_for_prediction)
        # proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
        # st.write("Prediction Probabilities:")
        # st.dataframe(proba_df)


else:
    st.warning("App is loading or encountered an error. Please ensure dataset URL and feature lists are correct.")

