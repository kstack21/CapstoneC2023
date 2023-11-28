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
from functions import *



#--------------------------Cached Functions--------------------------#

@st.cache_data
def import_json_values():
    # Import json values
    # Get the current working directory (base directory)
    base_directory = os.getcwd()

    # Boundary values
    filename = 'data_boundaries.json'
    file_path = os.path.join(base_directory, 'data', filename)
    with open(file_path, 'r') as json_file:
        boundaries = json.load(json_file)

    # Define the filename
    filename = 'timepoints.json'
    file_path = os.path.join(base_directory, 'data', filename)
    with open(file_path, 'r') as json_file:
        timepoints = json.load(json_file)
        
    #Collinear teg
    filename = 'TEG_collinear.json'
    file_path = os.path.join(base_directory, 'data', filename)
    with open(file_path, 'r') as json_file:
        collinearity = json.load(json_file)

    return boundaries, timepoints, collinearity

# Load general data
boundaries, timepoints, collinearity = import_json_values()

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


#--------------------------Side bar: Upload data --------------------------#
# Upload patient's data (Excel format only) button
uploaded_file = st.sidebar.file_uploader("Upload Data Set of Patient Data (XLSX)", type=["xlsx"])

#--------------------------Data description--------------------------#
if uploaded_file is not None:
    # Read the uploaded Excel file into a Pandas DataFrame
    data_frames = pd.read_excel(uploaded_file, engine="openpyxl")
    sheet_names = ['Baseline', 'TEG Values', 'Events']  # Replace with your sheet names

    # Access each sheet's data using the sheet name as the key
    baseline_df = data_frames[sheet_names[0]]
    tegValues_df = data_frames[sheet_names[1]]
    events_df = data_frames[sheet_names[2]]

    # Merge tables
    baseline_df, tegValues_df = merge_events_count(baseline_df, tegValues_df, events_df)

    # Perform data transformations
    clean_baseline_df, clean_TEG_df, tegValues = transform_data(baseline_df, tegValues_df, boundaries, timepoints)

    # Show data demographics
    fig = visualize_data(clean_baseline_df, clean_TEG_df)
    st.plotly_chart(fig, use_container_width=True)

    # Show data description

    #--------------------------Parameters--------------------------#
    #------------------------Side bar:  Toggle user chooses to extend data------------------------#
    user_extend_data = st.toggle("Calculate rates", value=True)

    if user_extend_data:
        extended_df, new_columns  = extend_df (clean_TEG_df, tegValues)
    else:
        new_columns = None
        extended_df = clean_TEG_df.copy()

    
    # Generate model
    with st.spinner('Generating model first draft...'):
        # Make models to find contribution of each parameter
        best_model_baseline, baseline_train = train_model(clean_baseline_df, 'Events', ['Record ID'])
        best_model_TEG1, TEG1_train = train_model(extended_df, 'Events', ['Record ID'])

        # Get feature importances from the XGBoost model in the pipeline
        importance_df_bsaeline, shap_values_baseline  = feature_importance(best_model_baseline, baseline_train)
        importance_df_TEG1, shap_values_TEG1 = feature_importance(best_model_TEG1, TEG1_train)

    # Plot SHAP summary plot
    shap.summary_plot(shap_values_baseline, baseline_train, plot_type="bar", show= False)
    shap.summary_plot(shap_values_TEG1, TEG1_train, plot_type="bar", show= False)

    # Get list of parameters for user to select
    columns_to_keep = user_options(extended_df, tegValues, new_columns, importance_df_TEG1, user_extend_data)
    user_TEG_df = extended_df.copy()
    user_TEG_df = user_TEG_df[columns_to_keep.keys()] # Keep only non repeated values

    # User selects non collinear parameters
    st.subheader("Select one of the related parameters")
    # Create empty dictionary to hold selection
    selected_features = {}

    # Use the dictionary with columns to keep to show user their options
    for group_name , elements in collinearity.items():
        with st.expander(f"{group_name}"):
            
            # Filter keys based on prefixes
            filtered_keys = [key for key in columns_to_keep.keys() if any(key.startswith(prefix) for prefix in elements)]

            # Create a list of strings by appending keys with values multiplied by 100
            radio_labels = [f"{key} ({round(columns_to_keep[key] * 100, 2)}%)" for key in filtered_keys]

            # Create a radio button to select a feature from the group
            selected_feature = st.radio("", radio_labels, key=group_name)
            selected_feature = radio_labels[0]

            # Convert the group list to a tuple and store the selected feature in the dictionary
            selected_features[group_name] = selected_feature

    # Display selection to user
    st.write("Parameters selected by the user")
    st.write(selected_features)
        
    with st.expander("Other parameters"):
        # Create a list of radio button labels with feature names and percentages
        st.write("Do we still want to show this?")
    

    # Train optimized model
    # #------------------------Side bar: Train and validate new model -----------------------#
    if st.sidebar.button("Train and validate"):

        columns_to_drop = user_selection(selected_features, columns_to_keep, collinearity)

        model2_df = user_TEG_df.copy()
        model2_df.drop(columns=columns_to_drop, inplace=True)

        with st.spinner('Generating optimized model...'):
            # Make model
            best_model_TEG2, TEG2_train = train_model(model2_df, 'Events', ['Record ID'])

            # New feature importance
            importance_df_TEG2, shap_values_TEG2 = feature_importance(best_model_TEG2, TEG2_train)

        # Plot SHAP summary plot
        shap.summary_plot(shap_values_TEG2, TEG2_train, plot_type="bar", show= False)


        # Save the trained model to a file (using joblib)
        joblib.dump((best_model_TEG2, TEG2_train), "trained_model.pkl")
        with open("trained_model.pkl", "rb") as model_file:
            model_binary = model_file.read()
        
        # Encode the model_binary in base64
        b64 = base64.b64encode(model_binary).decode()
        
        # Create a download link for the model file
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="trained_model.pkl">Download Model</a>'
        st.markdown(href, unsafe_allow_html=True)

            
    #--------------------------Model performance--------------------------#