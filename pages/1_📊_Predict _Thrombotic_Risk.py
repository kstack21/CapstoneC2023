import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

# Get higher level functions
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from functions import path_back_to

#--------------------------Page description--------------------------#
st.set_page_config(
    page_title="Predict Thrombotic Risk",
    page_icon="📊",
)

# Main layout
st.title("Page title")
st.markdown("""This page should be where users can input patient 
            data and receive predictions regarding thrombotic risk. 
            Users can input patient-specific information, 
            and the model will generate a risk assessment based on the input data. 
            This page should have a user-friendly interface for data input 
            and result display.""")
st.markdown("Patient data demographics overview:")

#--------------------------Model info--------------------------#


#--------------------------Patient info--------------------------#

#--------------------------Prediction--------------------------#
# Get data from folder
data_path = path_back_to(["data","DummyResult.xlsx"])

# Get 10 most influencial elements
prediction = pd.read_excel(data_path)
prediction = prediction.T.squeeze()
largest_contributor = prediction.nlargest(n=10, keep='first')
largest_contributor = pd.DataFrame({'Category': largest_contributor.index, 'Value': largest_contributor.values})

# Plot pie chart
fig = px.pie(largest_contributor, names='Category', values='Value', title='Parameters contribution to risk')
st.plotly_chart(fig, use_container_width=True)


#--------------------------Side bar--------------------------#
# Upload model
st.sidebar.file_uploader("Upload Data Set")

# Upload patient's data
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["csv", "xlsx"])
if uploaded_file != None:
    st.write("Patient data uploaded.")

# Download 
st.sidebar.button("Export results")

