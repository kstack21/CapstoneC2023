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
from functions import path_back_to

#--------------------------Page description--------------------------#
st.set_page_config(
    page_title="Train Model",
    page_icon="🧠",
)

# Main layout
st.title("Page title")
st.markdown("""This page is for administrators who want to retrain the model 
            or fine-tune its parameters. 
            It can provide options for uploading new data, 
            selecting model algorithms, and adjusting training settings. """)

#--------------------------Data description--------------------------#

# Get file path of data 
data_path = path_back_to(["data", "DummyData.xlsx"])

# Show table
prediction = pd.read_excel(data_path)
st.table(prediction)
#data = uploaded_file.getvalue()

# Find patient dataset stats
df = pd.DataFrame(prediction)
st.write("Total number of patients: ", len(df))
st.write("Average patient age: ", df['Age'].mean())
st.write("Percent of white patients: ", (df[df.White == 1].shape[0])/len(df) * 100, "%")

# Make pie subplots
fig = make_subplots(2, 2, specs=[[{'type':'domain'}, {'type':'domain'}],[{'type':'xy'}, {'type':'xy'}]],
                    subplot_titles=['Gender Distribution', 'Ethnicity Distribution','Age', 'BMI'])

# Count binary values in the "Male" column
male_counts = df['Male'].value_counts()
male_labels = ['Male' if male_counts.index[0] else 'Female', 'Male' if not male_counts.index[0] else 'Female']
# Create a pie chart for "Male"
fig.add_trace(go.Pie(labels=male_labels, values=male_counts), row=1, col=1)

# Count binary values in the "White" column
white_counts = df['White'].value_counts()
white_labels = ['White' if white_counts.index[0] else 'Non-White', 'White' if not white_counts.index[0] else 'Non-White']
# Create a pie chart for "White"
fig.add_trace(go.Pie(labels=white_labels, values=white_counts), row=1, col=2)

# Age histogram
fig.add_trace(go.Histogram(x=df["Age"], name="Age"), row=2, col=1)

# BMI histogram
fig.add_trace(go.Histogram(x=df["BMI"], name="BMI"), row=2, col=2)

# Display histrograms
st.plotly_chart(fig, use_container_width=True)

#--------------------------Parameters--------------------------#
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