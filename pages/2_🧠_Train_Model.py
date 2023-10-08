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

#--------------------------Functions--------------------------#
@st.cache_data
def pre_process_df(df):
    df = preprocess(df)
    return df

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
            selecting model algorithms, and adjusting training settings. """)

#--------------------------Data description--------------------------#

# Get file path of data DUMMY
data_path = path_back_to(["data", "DummyData_Extended.xlsx"])
df = pd.read_excel(data_path)

st.header("Data description")

# Show data demographics
fig = data_demographics_fig(df)
st.plotly_chart(fig, use_container_width=True)

# Show data description
numerical, categorical = describe_dataframe(df)
st.subheader("Analysis of numerical values")
st.table(numerical)
st.subheader("Analysis of categorical values")
st.table(categorical)

#--------------------------Parameters--------------------------#
# Preprocess data
df = pre_process_df(df)

# Make models to find contribution of each parameter

# Radio button widget
st.subheader("Selection of Highly Correlated Paramters")

#percentages should turn into variables pulled from the model
radio_pivpa = st.radio("Choose one parameter", ["Platlet Inhibition (75%)", "Platelet Aggregation (76%)"])

#Below is an example idk what would be correlated
radio_gvs=st.radio("Chose one parameter", ["Gender (50%)","Smoking (85%)"])

st.write(f"You selected: {radio_pivpa} and {radio_gvs}")


#--------------------------Model performance--------------------------#


#--------------------------Side bar--------------------------#
# Upload data set
st.sidebar.file_uploader("Upload Data Set")

# Test parameters
st.sidebar.button("Test parameters")

# Train and validate model
st.sidebar.button("Train and validate")

# Download 
st.sidebar.button("Download")


# Side bar layout
upload_file = st.sidebar.file_uploader("Upload Model") #Creates sidebar button for model upload
if upload_file is not None: # Checks for upload, Runs if a file is uploaded
    st.write("Model Uploaded Successfully") # indicates to user that Model File was uploaded
    st.write("Displaying Model") # Text that indicates Model is Being displayed 
    df = pd.DataFrame()

# The following line is where we will try to display the uploaded model
   # df = pd.DataFrame(np) 

upload_file = st.sidebar.file_uploader("Upload Patient Data") #creates button for Uploading Patient Data
if upload_file is not None:
    st.write("Patient Data Uploaded Successfully") # Creates text to notify user Patient Data has been uploaded

if st.sidebar.button("Export"): # Creates a button that states EXPORT, but what will export be 
    st.write("I AM LOSING MY SANITY") #output for button press 