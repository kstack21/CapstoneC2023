import streamlit as st
import pandas as pd
import os
import plotly.express as px

#--------------------------Helper functions--------------------------#
def path_back_to(new_folder_name):
    # Get the directory name of the provided path
    directory_name = os.path.dirname(__file__)

    # Split the directory path into components
    directory_components = directory_name.split(os.path.sep)

    # Remove the last folder 
    if directory_components[-1]:
        directory_components.pop()

    # Add the new folder to the path
    for file in new_folder_name:
        directory_components.append(file)

    # Join the modified components to create the new path
    new_path = os.path.sep.join(directory_components)

    return new_path

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

st.bar_chart(df, x = "White")
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