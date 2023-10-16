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
<<<<<<< HEAD
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["csv", "xlsx"])
#if uploaded_file != None:
#    st.write("Patient data uploaded.")
=======
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])
if uploaded_file != None:
    st.write("Patient data uploaded.")
>>>>>>> 5185c234555b5395fe26a1b8665dfb438c2f434e

# Download 
st.sidebar.button("Export results")

#--------------------------Patient info--------------------------#
# Get patient data from uploaded file
if uploaded_file != None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    #st.write(df)   # shows whole uploaded excel file

    # Patient Data Header #
    st.header(':green[Patient Data Uploaded]')

    # Organizing text in columns
    col1, col2, col3 = st.columns(3)

    # Present General Patient Info
    with col1:
        # Age
        if 'Age' in df:
            st.metric(label = "Age", value = df.at[0,'Age'])
        else:
            st.metric(label = ":red[Age]", value = "No column named 'Age'")

        # Tobacco Use
        if 'Tobacco Use' in df:
            if df.at[0,'Tobacco Use'] == 1:
                temp = "Low"
            elif df.at[0, 'Tobacco Use'] == 2:
                temp = "Medium"
            elif df.at[0, 'Tobacco Use'] >= 3:
                temp = "High"
            else:
                temp = "None"
            st.metric(label = "Tobacco Use", value = temp)
        else:
            st.metric(label = ":red[Tobacco Use]", value = "No column named 'Tobacco Use'")

        # Hypertension
        if 'Hypertension' in df:
            if df.at[0,'Hypertension']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label="Hypertension", value = temp)
        else:
            st.metric(label = ":red[Hypertension]", value = "No column named 'Hypertension'")

    with col2:
        # Sex
        if 'Male' in df:
            if df.at[0,'Male']:
                temp = "Male"
            else:
                temp = "Not Male (Female or other)"
            st.metric(label = "Sex", value = temp)
        else:
            st.metric(label = ":red[Sex]", value = "No column named 'Sex'")

        # Race (White vs Not White)
        if 'White' in df:
            if df.at[0,'White']:
                temp = "White"
            else:
                temp = "Not White"
            st.metric(label = "Race", value = temp)
        else:
            st.metric(label = ":red[White]", value = "No column named 'White'")

        # Clotting Disorder
        if 'Clotting Disorder' in df:
            if df.at[0,'Clotting Disorder']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label = "Clotting Disorder", value = temp)
        else:
            st.metric(label = ":red[Clotting Disorder]", value = "No column named 'Clotting Disorder'")

    with col3:
        
        # Extremity and Artery Affected
        if ('Extremity' in df) & ('Artery affected' in df):
            temp = df.at[0, 'Extremity'] + " " + df.at[0, 'Artery affected']
            st.metric(label="Affected Artery", value = temp)
        else:
            st.metric(label = ":red[Affected Artery]", value = "No column named 'Extremity' and/or 'Artery affected'")

        # BMI
        if 'BMI' in df:
            st.metric(label="BMI", value = df.at[0, 'BMI'])
        else:
            st.metric(label = ":red[BMI]", value = "No column named 'BMI'")

        # Diabetes
        if 'Diabetes' in df:
            if df.at[0, 'Diabetes']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label="Diabetes", value = temp)
        else:
            st.metric(label = ":red[Diabetes]", value = "No column named 'Diabetes'")

# display outline of patient data if nothing has been uploaded
else:
    # display dataframe
    st.header("Data description")

    # data header (no patient info)
    st.header(':red[No Patient Data Uploaded]')

    # organizing text in columns
    col1, col2, col3 = st.columns(3)

    # display empty sections
    with col1:
        st.metric(label=":red[Age]", value = '')
        st.metric(label=":red[Tobacco Use]", value = '')
        st.metric(label=":red[Hypertension]", value = '')
    with col2:
        st.metric(label=":red[Sex]", value = '')
        st.metric(label=":red[Race]", value = '')
        st.metric(label=":red[Clotting Disorder]", value = '')
    with col3:
        st.metric(label=":red[Affected Artery]", value = '')
        st.metric(label=":red[BMI]", value = '')
        st.metric(label=":red[Diabetes]", value = '')