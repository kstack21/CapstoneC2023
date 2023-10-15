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
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])
if uploaded_file != None:
    st.write("Patient data uploaded.")

# Download 
st.sidebar.button("Export results")

#--------------------------Patient info--------------------------#
# Get data from folder
data_path = path_back_to(["data","DummyResult.xlsx"])


## Patient 0 Header ##
c1, c2, c3 = st.columns(3)
with c2:
    st.header(':red[Patient 0]')

# Organizing text in columns
col1, col2, col3 = st.columns(3)

# Present General Patient Info
with col1:
    st.metric(label="Age", value=45)
    st.metric(label="Tobacco Use", value="Low")
    st.metric(label="Hypertension", value="Yes")
with col2:
    st.metric(label="Sex", value="Male")
    st.metric(label="Race", value="White")
    st.metric(label="Clotting Disorder", value="No")
with col3:
    st.metric(label="Affected Artery", value="Tibial")
    st.metric(label="BMI", value=28.5)
    st.metric(label="Diabetes", value="No")


## Patient 1 Header ##
c1, c2, c3 = st.columns(3)
with c2:
    st.header(':red[Patient 1]')

# Organizing text in columns
col1, col2, col3 = st.columns(3)

# Present General Patient Info
with col1:
    st.metric(label="Age", value=62)
    st.metric(label="Tobacco Use", value="Medium")
    st.metric(label="Hypertension", value="No")
with col2:
    st.metric(label="Sex", value="Female")
    st.metric(label="Race", value="White")
    st.metric(label="Clotting Disorder", value="Yes")
with col3:
    st.metric(label="Affected Artery", value="Femoral")
    st.metric(label="BMI", value=32.1)
    st.metric(label="Diabetes", value="Yes")


## Patient 2 Header ##
c1, c2, c3 = st.columns(3)
with c2:
    st.header(':red[Patient 2]')

# Organizing text in columns
col1, col2, col3 = st.columns(3)

# Present General Patient Info
with col1:
    st.metric(label="Age", value=57)
    st.metric(label="Tobacco Use", value="High")
    st.metric(label="Hypertension", value="Yes")
with col2:
    st.metric(label="Sex", value="Male")
    st.metric(label="Race", value="Not White")
    st.metric(label="Clotting Disorder", value="No")
with col3:
    st.metric(label="Affected Artery", value="Tibial")
    st.metric(label="BMI", value=25.8)
    st.metric(label="Diabetes", value="No")
