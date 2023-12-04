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
import ast
import pickle

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
def train_model_cached(df, target_column, drop_columns,quantile_ranges, param_grid, scoring):
    best_pipeline, X_train, score = train_model(df, target_column, drop_columns,quantile_ranges, param_grid, scoring)

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
def download_cached(_my_dict, my_variable,file_name):
    # Serialize the dictionary to bytes using pickle
    serialized_dict = pickle.dumps(_my_dict)

    # Encode the serialized bytes in base64
    b64_encoded = base64.b64encode(serialized_dict).decode()

    # Create a download link for the pickled dictionary
    href = f'<a href="data:application/octet-stream;base64,{b64_encoded}" download={file_name}>Download Model</a>'

    return href


#--------------------------Side bar: Upload data --------------------------#
with st.sidebar:
    # Upload patient's data (Excel format only) button
    uploaded_file = st.file_uploader("Upload Data Set of Patient Data (XLSX)", type=["xlsx"])

    # Side bar:  Toggle user chooses to extend data
    user_extend_data = st.toggle("Calculate rates", value=True, disabled= uploaded_file is not None)

    # Advanced settings
    
    # Preset values
    quantile_ranges = (5,95)
    param_grid = {
        'xgb_regressor__max_depth': [3, 4, 5],
        'xgb_regressor__gamma': [0, 0.1, 0.2],
        'xgb_regressor__min_child_weight': [1, 2, 5]
    }
    scoring = 'r2'


    st.markdown("""---""")
    advanced_settings = st.toggle("Advanced settings", value=False, disabled= uploaded_file is not None)

    if advanced_settings:

         # Modify quantile ranges 
        st.subheader("Robust scaling of target column:", help="[Why?](https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/)")
        qrc1, qrc2 = st.columns(2)
        with qrc1:
            new_min = st.number_input("Min Quantile:", min_value=float(0), value= float(quantile_ranges[0]))
        with qrc2:
            new_max = st.number_input("Max Quantile:", min_value=float(new_min), max_value=float(100), value= float(quantile_ranges[1]))
        quantile_ranges = (new_min, new_max)


        # Modify scoring function
        st.subheader("Grid search scoring function:", help="[What are the options?](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)")
        scoring = st.text_input("Scoring function", scoring)


        # Modify hyper parameter tunning
        st.subheader("Hyperparameter tunning:", help="[What are the options?](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)")
        new_param_grid = st.text_area("Enter Parameter Grid:", param_grid)
        try:
            # Attempt to parse the input as JSON
            param_grid = ast.literal_eval(new_param_grid)
        except :
            st.warning(f"Your input is in the wrong format.")
        param_grid


        # Modify collinearity
        st.subheader("Multicollinearity:", help="The following structure contains column names that might be collinear. Make sure column names are properly spelled. Used for generation of model 2")
        new_collinearity = st.text_area("Update collinearity:", collinearity)
        try:
            # Attempt to parse the input as JSON
            collinearity = ast.literal_eval(new_collinearity)
        except :
            st.warning(f"Your input is in the wrong format.")
        collinearity

    
#-------------------------- Main page--------------------------#
if uploaded_file is not None:
    st.markdown("""---""")
    # show data demographics
    st.subheader("Demographics of the Uploaded Dataset")

    # Load data
    fig, clean_TEG_df, tegValues, clean_baseline_df = upoload_data_cached(uploaded_file)
    
    # Show data description
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

    # Extend data:
    extended_df, new_columns = extend_data_cached(clean_TEG_df, tegValues, user_extend_data)

    # store column names
    baselineColumns = list(clean_baseline_df.columns)
    tegColumns1 = list(extended_df.columns)
    
    # Generate model
    with st.spinner('Generating model first draft...'):
        # Make models to find contribution of each parameter
        best_model_baseline, baseline_train, baseline_score, importance_df_bsaeline, shap_values_baseline = train_model_cached(clean_baseline_df, 'Events', ['Record ID'],quantile_ranges, param_grid, scoring)
        best_model_TEG1, TEG1_train, TEG1_score, importance_df_TEG1, shap_values_TEG1 = train_model_cached(extended_df, 'Events', ['Record ID'],quantile_ranges, param_grid, scoring)

    # show most influential factors
    st.subheader("The intial models have been created! The current most influential factors are...")

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

            # Sort filtered_keys based on values in descending order
            sorted_keys = sorted(filtered_keys, key=lambda key: columns_to_keep[key], reverse=True)

            # Create a list of strings by appending keys with values multiplied by 100
            radio_labels = [f"{key} ({round(columns_to_keep[key] * 100, 2)}%)" for key in sorted_keys]

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

            # Save column names
            tegColumns2 = list(model2_df.columns)
            # Make model and find feature importance
            best_model_TEG2, TEG2_train, TEG2_score, importance_df_TEG2, shap_values_TEG2 = train_model_cached(model2_df, 'Events', ['Record ID'],quantile_ranges, param_grid, scoring)

        # introduce new model
        st.subheader("Your predictive model has been created! Here are its validity scores:")
        # show mse and r2 scores for train and test data
        tegScores1 = pd.DataFrame(TEG1_score, index=["TEG-based model (TEG model 1)"])
        tegScores2 = pd.DataFrame(TEG2_score, index=["TEG-based model (TEG model 2)"])
        baselineScores = pd.DataFrame(baseline_score, index=["General info based model"])
        st.table(pd.concat([tegScores1, tegScores2, baselineScores], sort=False))

        st.markdown("""---""")
        # Plot SHAP summary plot
        st.subheader("And here are the TEG factors that most influence your model's predictions:")
        st.pyplot(shap.summary_plot(shap_values_TEG2, TEG2_train, plot_type="bar", show= False, max_display=10))
        st.markdown("""---""")

        # Save the trained model to a file 
        toDownload1 = {"TEG model": best_model_TEG1,
                      "Baseline model": best_model_baseline,
                      "TEG column names": tegColumns1,
                      "Baseline column names": baselineColumns,
                      "Extend data":user_extend_data}
        
        toDownload2 = {"TEG model": best_model_TEG2,
                       "Baseline model": best_model_baseline,
                       "TEG column names": tegColumns2,
                       "Baseline column names": baselineColumns,
                      "Extend data":user_extend_data}
        
        st.subheader("Click one of the links below ('Download Model') to download your predictive model!")
        st.markdown("You will need this for the next page, where you can predict the risk of an individual patient.")

        href1 = download_cached(toDownload1,tegColumns1,"CLOTWATCH_predictive_model_1.pkl")
        href2 = download_cached(toDownload2,tegColumns2, "CLOTWATCH_predictive_model_2.pkl")

        st.write("With TEG-based model 1:")
        st.markdown(href1, unsafe_allow_html=True)
        st.write("With TEG-based model 2:")
        st.markdown(href2, unsafe_allow_html=True)