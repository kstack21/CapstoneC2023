import streamlit as st
import pandas as pd
import os

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


# Side bar layout

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])
st.write(type(uploaded_file))
if uploaded_file != None:
    st.write("Patient data uploaded.")

# Get file path of data (hardcoded, not the file uploaded to website)
path_pages = os.path.dirname(__file__)
path_pages = path_pages.replace("\\pages", "")
data_path = os.path.join(path_pages, "data", "DummyData.xlsx")

# Show table
prediction = pd.read_excel(data_path)
st.table(prediction)
#data = uploaded_file.getvalue()

# Find patient data stats
df = pd.DataFrame(prediction)
st.write("Total number of patients: ", len(df))
st.write("Average patient age: ", df['Age'].mean())
st.write("Percent of white patients: ", (df[df.White == 1].shape[0])/len(df) * 100, "%")

st.sidebar.file_uploader("Upload Patient Data and Usable Model")

