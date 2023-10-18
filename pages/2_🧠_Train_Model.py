import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import os
import joblib 
import json
import re
import base64
import shap

# Get higher level functions
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from functions import data_demographics_fig, describe_dataframe, feature_importance
from preprocessing import preprocess, split_df
from models_classifier import generate_model


#--------------------------Functions--------------------------#
@st.cache_data
def cached_data_demographics_fig(df):
    fig = data_demographics_fig(df)
    return fig

@st.cache_data
def cached_describe_dataframe(df):
    numerical, categorical = describe_dataframe(df)
    return numerical, categorical

@st.cache_data
def cached_preprocess(df):
    df = preprocess(df)
    return df

@st.cache_resource
def cached_generate_model(df):
    best_model, best_params, accuracy, X_train = generate_model(df)
    return best_model, best_params, accuracy, X_train

# Define a function to train a simple model (you can replace this with your actual model training code)
def train_model():
    # Here, you can replace this with your model training logic
    model = "YourTrainedModel"
    return model

#     return shap_values


#--------------------------Page description--------------------------#
st.set_page_config(
    page_title="Train Model",
    page_icon="🧠",
)

# Main layout
st.title("Train model")
st.markdown("""This page allows user to train a model 
            and fine-tune its parameters. 
            It provides options for uploading new data, 
            selecting model algorithms, and adjusting training settings. 
            Start by uploding a dataset with TEG data on the left column.""")

#--------------------------Side bar--------------------------#
# Upload patient's data (Excel format only) button
uploaded_file = st.sidebar.file_uploader("Upload Data Set of Patient Data (XLSX)", type=["xlsx"])

#--------------------------Data description--------------------------#
if uploaded_file is not None:
    # Read the uploaded Excel file into a Pandas DataFrame
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Show data demographics
    fig = cached_data_demographics_fig(df)
    st.plotly_chart(fig, use_container_width=True)

    # Show data description
    numerical, categorical = cached_describe_dataframe(df)
    with st.expander("Analysis of numerical values"):
        st.dataframe(numerical)
    with st.expander("Analysis of categorical values"):
        st.dataframe(categorical)

    #--------------------------Parameters--------------------------#
    # Preprocess data
    df = cached_preprocess(df)

    # Generate model
    with st.spinner('Generating model first draft...'):
        # Make models to find contribution of each parameter
        best_model, best_params, accuracy, X_train = cached_generate_model(df)
        
        # Get feature importances from the XGBoost model in the pipeline
        feature_importance_df = feature_importance(best_model, X_train)

        teg_df , non_teg_df = split_df(feature_importance_df)
        teg_df.drop(columns=['Importance'], inplace=True)
        non_teg_df.drop(columns=['Importance'], inplace=True)


    # User selects parameters

    # Load the JSON file with a list of lists of strings
    # Replace 'your_json_file.json' with the actual path to your JSON file
    with open('TEG_collinear.json', 'r') as json_file:
        feature_groups = json.load(json_file)

    # Create a dictionary to store the feature groups
    grouped_features = {group: [] for group in feature_groups}

    # Get all the features
    features = teg_df['Feature'].values.tolist()

    # Group the features based on the information in the JSON file
    # feature_groups = {"group name":[teg value 1, teg value2...]}
    for group, teg_values in feature_groups.items():
        for teg_value in teg_values:
            for f in features:
                if f.startswith(teg_value):
                    grouped_features[group].append(f)
                    

    st.subheader("Select one of the related parameters")

    # Create an empty dictionary to store selected features for each group
    selected_features = {}

    # Iterate over feature groups and create expanders
    for title, group in grouped_features.items():
        
        with st.expander(f"{title}"):
            
            # Create a list of radio button labels with feature names and percentages
            radio_labels = [f"{row['Feature']} ({row['Percentage Contribution']}%)" for _, row in teg_df.iterrows() if row['Feature'] in group]

            # Create a radio button to select a feature from the group
            selected_feature = st.radio("", radio_labels, key=group)
            
            # Convert the group list to a tuple and store the selected feature in the dictionary
            selected_features[title] = selected_feature

 
    st.write("Parameters selected by the user")
    st.write(selected_features)
        
    with st.expander("Other parameters"):
        # Create a list of radio button labels with feature names and percentages
        st.dataframe(non_teg_df)
    

    # Train optimized model
    # Train and validate model button in side bar
    if st.sidebar.button("Train and validate"):

        # Create new dataframe
        selected_features_list = []

        # Process the values in the dictionary containing items selected by user. Convert to list
        for key, value in selected_features.items():
            # Remove patterns matching (float%)
            selected_features_list.append(re.sub(r'\(\d+\.\d+%\)', '', value).strip())
        
        # Extract the features mentioned in the "Feature" column of non_teg_df
        non_teg_features = non_teg_df['Feature'].tolist()
        selected_features_list =  selected_features_list + non_teg_features + ['Upcoming_event']
        # Filter the columns of the original DataFrame based on the selected features
        filtered_df = df[selected_features_list]

        st.text("Training the model...")
        best_model, best_params, score, X_train = cached_generate_model(filtered_df)
        st.text("Model trained successfully!")

        # Save the trained model to a file (e.g., using joblib)
        joblib.dump((best_model, best_params, score), "trained_model.pkl") # This is what is being dowloaded: best_model, best_params, score
        with open("trained_model.pkl", "rb") as model_file:
            model_binary = model_file.read()
        
        # Encode the model_binary in base64
        b64 = base64.b64encode(model_binary).decode()
        
        # Create a download link for the model file
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="trained_model.pkl">Download Model</a>'
        st.markdown(href, unsafe_allow_html=True)

            
    #--------------------------Model performance--------------------------#