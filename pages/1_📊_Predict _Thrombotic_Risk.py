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
st.set_page_config(
    page_title="Predict Thrombotic Risk",
    page_icon="📊",
)

#--------------------------Page description--------------------------#
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

