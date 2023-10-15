import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import os

# Get higher level functions
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from functions import path_back_to, data_demographics_fig, describe_dataframe
from preprocessing import preprocess
from models_classifier import generate_model
import shap

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
    best_model, best_scaler = generate_model(df)
    return best_model, best_scaler

@st.cache_data
def cached_model_explainer(model, X_test):
    # Create an explainer
    explainer = shap.Explainer(model)

    # Calculate Shapley values for a specific instance or a set of instances
    shap_values = explainer(X_test)

    # Plot summary Shapley values
    shap.summary_plot(shap_values, X_test)
    return shap_values


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
# Upload patient's data (Excel format only)
uploaded_file = st.sidebar.file_uploader("Upload Data Set of Patient Data (XLSX)", type=["xlsx"])

# Test parameters
st.sidebar.button("Test parameters")

# Train and validate model
st.sidebar.button("Train and validate")

# Download 
st.sidebar.button("Download")

#--------------------------Data description--------------------------#
if uploaded_file is not None:
    # Read the uploaded Excel file into a Pandas DataFrame
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Display the DataFrame
    st.header("Data description")

    # Show data demographics
    fig = cached_data_demographics_fig(df)
    st.plotly_chart(fig, use_container_width=True)

    # Show data description
    numerical, categorical = cached_describe_dataframe(df)
    st.subheader("Analysis of numerical values")
    st.table(numerical)
    st.subheader("Analysis of categorical values")
    st.table(categorical)

    #--------------------------Parameters--------------------------#


    # Preprocess data
    df = cached_preprocess(df)

    # Make models to find contribution of each parameter
    best_model, best_scaler = generate_model(df)

    model = best_model.named_steps['classifier']
    st.write(model.feature_importances_)

# # Radio button widget
# st.subheader("Selection of Highly Correlated Paramters")

# #percentages should turn into variables pulled from the model
# radio_pivpa = st.radio("Choose one parameter", ["Platlet Inhibition (75%)", "Platelet Aggregation (76%)"])

# #Below is an example idk what would be correlated
# radio_gvs=st.radio("Chose one parameter", ["Gender (50%)","Smoking (85%)"])

# st.write(f"You selected: {radio_pivpa} and {radio_gvs}")


# #--------------------------Model performance--------------------------#

