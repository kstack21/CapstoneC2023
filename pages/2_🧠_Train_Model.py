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

#--------------------------Page description--------------------------#
st.set_page_config(
    page_title="Train Model",
    page_icon="🧠",
    #showPyplotGlobalUse = False,
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Main layout
st.title("Train a New Model")
st.markdown("""This page allows you to train a predictive model 
            and fine-tune its parameters. 
            It provides options for uploading new data, 
            selecting model algorithms, and adjusting training settings.""") 
st.markdown("Start by uploding a dataset in the correct format using the button to the left.")


#--------------------------Cached Functions--------------------------#

@st.cache_data
def import_json_values_cached():
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
boundaries, timepoints, collinearity = import_json_values_cached()

@st.cache_data
def upoload_data_cached(uploaded_file):
    # Read the uploaded Excel file into a Pandas DataFrame
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")

    sheet_names = ['Baseline', 'TEG Values', 'Events']

    # Access each sheet's data using the sheet name as the key
    baseline_df = pd.read_excel(xls, sheet_names[0])
    tegValues_df = pd.read_excel(xls, sheet_names[1])
    events_df = pd.read_excel(xls, sheet_names[2])

    # Merge tables
    baseline_df, tegValues_df = merge_events_count(baseline_df, tegValues_df, events_df)

    # Perform data transformations
    clean_baseline_df, clean_TEG_df, tegValues = transform_data(baseline_df, tegValues_df, boundaries, timepoints)

    # Show data demographics
    fig = visualize_data(clean_baseline_df, clean_TEG_df)
    
    return fig, clean_TEG_df, tegValues, clean_baseline_df

@st.cache_data
def extend_data_cached(clean_TEG_df, tegValues, user_extend_data):

    if user_extend_data:
        extended_df, new_columns  = extend_df (clean_TEG_df, tegValues)
    else:
        new_columns = None
        extended_df = clean_TEG_df.copy()

    return extended_df, new_columns

@st.cache_resource
def train_model_cached(df, target_column, drop_columns):
    best_pipeline, X_train, score = train_model(df, target_column, drop_columns)

    importance_df, shap_values = feature_importance(best_pipeline, X_train)
    return best_pipeline, X_train, score, importance_df, shap_values


@st.cache_data
def user_options_cached(extended_df, tegValues, new_columns, importance_df_TEG1, user_extend_data):
    columns_to_keep = user_options(extended_df, tegValues, new_columns, importance_df_TEG1, user_extend_data)
    user_TEG_df = extended_df.copy()
    user_TEG_df = user_TEG_df[columns_to_keep.keys()] # Keep only non repeated values

    return columns_to_keep, user_TEG_df

@st.cache_data
def user_selection_cached(user_TEG_df,selected_features, columns_to_keep, collinearity):
    columns_to_drop = user_selection(selected_features, columns_to_keep, collinearity)
    model2_df = user_TEG_df.copy()
    model2_df.drop(columns=columns_to_drop, inplace=True)

    return model2_df

@st.cache_data
def download_cached(_object_input):
    joblib.dump(_object_input, "trained_model.pkl")
    with open("trained_model.pkl", "rb") as model_file:
        model_binary = model_file.read()
    
    # Encode the model_binary in base64
    b64 = base64.b64encode(model_binary).decode()
    
    # Create a download link for the model file
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="trained_model.pkl">Download Model</a>'
        
    return href


#--------------------------Side bar: Upload data --------------------------#
# Upload patient's data (Excel format only) button
uploaded_file = st.sidebar.file_uploader("Upload Data Set of Patient Data (XLSX)", type=["xlsx"])

# Side bar:  Toggle user chooses to extend data
user_extend_data = st.sidebar.toggle("Calculate rates", value=True, disabled= uploaded_file is not None)
#-------------------------- Main page--------------------------#
st.markdown("""---""")
if uploaded_file is not None:
    
    # show data demographics
    st.subheader("Demographics of the Uploaded Dataset")

    # Load data
    fig, clean_TEG_df, tegValues, clean_baseline_df = upoload_data_cached(uploaded_file)
    
    # Show data description
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

    # show most influential factors
    st.subheader("The intial models have been created! The current most influential factors are...")

    # Extend data:
    extended_df, new_columns = extend_data_cached(clean_TEG_df, tegValues, user_extend_data)

    # store column names
    baselineColumns = list(clean_baseline_df.columns)
    tegColumns = list(extended_df.columns)
    
    # Generate model
    with st.spinner('Generating model first draft...'):
        # Make models to find contribution of each parameter
        best_model_baseline, baseline_train, baseline_score, importance_df_bsaeline, shap_values_baseline = train_model_cached(clean_baseline_df, 'Events', ['Record ID'])
        best_model_TEG1, TEG1_train, TEG1_score, importance_df_TEG1, shap_values_TEG1 = train_model_cached(extended_df, 'Events', ['Record ID'])

    # Plot SHAP summary plot
    st.subheader(":blue[General Patient Information:]")
    st.pyplot(shap.summary_plot(shap_values_baseline, baseline_train, plot_type="bar", show= False, max_display=10))
    st.subheader(":blue[Patient TEG factors:]")
    st.pyplot(shap.summary_plot(shap_values_TEG1, TEG1_train, plot_type="bar", show= False, max_display=10))

    # Get list of parameters for user to select
    columns_to_keep, user_TEG_df = user_options_cached(extended_df, tegValues, new_columns, importance_df_TEG1, user_extend_data)
    st.markdown("""---""")
    # User selects non collinear parameters

    st.subheader("Please select one parameter from each of the following groups")
    st.markdown("These groups of factors are related. If you have a preference for which ones are used to train the model, please choose them below. Otherwise, you can leave them as their default values. The reason this is necessary is because factors that are related can create a biased model.")

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

            # Convert the group list to a tuple and store the selected feature in the dictionary
            selected_features[group_name] = selected_feature

    # Display selection to user
    st.subheader("These are the parameters you have chosen:")
    st.table(selected_features)   
    st.markdown("""---""")

    # tell them to train the model
    st.subheader("If this looks good, click the 'Train and Validate' button to the left to train your predictive model!") 

    # Train optimized model
    # #------------------------Side bar: Train and validate new model -----------------------#
    if st.sidebar.button("Train and Validate"):

        # Get new dataframe with dropped values
        model2_df = user_selection_cached(user_TEG_df,selected_features, columns_to_keep, collinearity)

        with st.spinner('Generating optimized model...'):
            # Make model and find feature importance
            best_model_TEG2, TEG2_train, TEG2_score, importance_df_TEG2, shap_values_TEG2 = train_model_cached(model2_df, 'Events', ['Record ID'])

        # introduce new model
        st.markdown("""---""")
        st.subheader("Your predictive model has been created! Here are its validity scores:")

        tegScores1 = pd.DataFrame(TEG1_score, index=["TEG-based model (TEG model 1)"])
        tegScores2 = pd.DataFrame(TEG2_score, index=["TEG-based model (TEG model 2)"])
        baselineScores = pd.DataFrame(baseline_score, index=["General info based model"])
        st.table(pd.concat([tegScores1, tegScores2, baselineScores], sort=False))

        # Plot SHAP summary plot
        st.subheader("And here are the TEG factors that most influence your model's predictions:")
        st.pyplot(shap.summary_plot(shap_values_TEG2, TEG2_train, plot_type="bar", show= False, max_display=10))
        st.markdown("""---""")

        # Save the trained model to a file 
        toDownload1 = {"TEG model": best_model_TEG1,
                      "Baseline model": best_model_baseline,
                      "Training data": TEG1_train,
                      "TEG training data": clean_TEG_df,
                      "Baseline training data": clean_baseline_df,
                      "TEG column names": tegColumns,
                      "Baseline column names": baselineColumns}
        toDownload2 = {"TEG model": best_model_TEG2,
                       "Baseline model": best_model_baseline,
                       "Training data": TEG2_train,
                       "TEG training data": clean_TEG_df,
                       "Baseline training data": clean_baseline_df,
                       "TEG column names": tegColumns,
                       "Baseline column names": baselineColumns}
        st.subheader("Click one of the links below ('Download Model') to download your predictive model!")
        st.markdown("You will need this for the next page, where you can predict the risk of an individual patient.")

        href1 = download_cached(toDownload1)
        href2 = download_cached(toDownload2)

        st.write("With TEG-based model 1:")
        st.markdown(href1, unsafe_allow_html=True)
        st.write("With TEG-based model 2:")
        st.markdown(href2, unsafe_allow_html=True)