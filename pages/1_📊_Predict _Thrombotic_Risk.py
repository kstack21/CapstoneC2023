import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
from fastapi import FastAPI
import joblib

# Get higher level functions
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from functions import path_back_to

exampleDataFrame = pd.DataFrame({'Age': [50], 'Diabetes': [1], 'BMI': [28.5], 'etc.': ['etc.']})
#--------------------------Page description--------------------------#
st.set_page_config(
    page_title="Predict Thrombotic Risk",
    page_icon="📊",
)

# Title and Instructions
st.title("Predict a Patient's Risk of Thrombosis")
st.markdown("""Upload a patient's file using the button 'Upload Patient Data'
            in the sidebar to see their predicted risk of thrombosis.""")
st.write("""Please make sure that the file is an Excel file in the following format:""")
st.dataframe(exampleDataFrame, width=500)
st.write("""Please also make sure that any yes/no values are indicated as 1/0, respectively,
         as illustrated in the 'Diabetes' column above.""")   
st.write("""Minimum expected factors: 'Age', 'Tobacco Use', 'Hypertension', 'Male', 'White',
            'Clotting Disorder', 'Extremity', 'Artery affected', 'BMI', and 'Diabetes'""")

#--------------------------Side bar--------------------------#
# Upload model
uploaded_model = st.sidebar.file_uploader("Upload Predictive Model", type = ["pkl"])

# Upload patient's data
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])

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
            st.metric(label = ":red[Age]", value = "n/a")
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
            st.metric(label = ":red[Tobacco Use]", value = "n/a")
        # Hypertension
        if 'Hypertension' in df:
            if df.at[0,'Hypertension']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label="Hypertension", value = temp)
        else:
            st.metric(label = ":red[Hypertension]", value = "n/a")

    with col2:
        # Sex
        if 'Male' in df:
            if df.at[0,'Male']:
                temp = "Male"
            else:
                temp = "Not Male (Female or other)"
            st.metric(label = "Sex", value = temp)
        else:
            st.metric(label = ":red[Sex]", value = "n/a")
        # Race (White vs Not White)
        if 'White' in df:
            if df.at[0,'White']:
                temp = "White"
            else:
                temp = "Not White"
            st.metric(label = "Race", value = temp)
        else:
            st.metric(label = ":red[White]", value = "n/a")
        # Clotting Disorder
        if 'Clotting Disorder' in df:
            if df.at[0,'Clotting Disorder']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label = "Clotting Disorder", value = temp)
        else:
            st.metric(label = ":red[Clotting Disorder]", value = "n/a")

    with col3:
        # Extremity and Artery Affected
        if ('Extremity' in df) & ('Artery affected' in df):
            temp = df.at[0, 'Extremity'] + " " + df.at[0, 'Artery affected']
            st.metric(label="Affected Artery", value = temp)
        elif ('Extremity' in df) & ('Artery affected' not in df):
            temp = df.at[0, 'Extremity'] + " Side"
            st.metric(label = "Affected Artery", value = temp)
        elif ('Extremity' not in df) & ('Artery affected' in df):
            temp = df.at[0, 'Artery affected']
            st.metric(label = "Affected Artery", value = temp)
        else:
            st.metric(label = ":red[Affected Artery]", value = "n/a")
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

    # display thrombosis risk
    st.header(":green[Patient's Calculated Risk of Thrombosis: ]")
    st.subheader(":red[No risk calculated yet]")
# display outline of patient data if nothing has been uploaded
else:
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

    # display thrombosis risk
    st.header(":red[Patient's Calculated Risk of Thrombosis: ]")
    st.subheader(":red[No risk calculated yet]")

#--------------------------Model info--------------------------#
#app = FastAPI()
#model = joblib.load(uploaded_model)


"""
#--------------------------Prediction--------------------------#
# Get data from folder
data_path = path_back_to(["data","DummyResult.xlsx"])

# Get 10 most influencial elements
prediction = pd.read_excel(data_path)
prediction = prediction.T.squeeze()
largest_contributor = prediction.nlargest(n=10, keep='first')
largest_contributor = pd.DataFrame({'Category': largest_contributor.index, 'Value': largest_contributor.values})

# Pie chart title
st.subheader("Predictive Model Details:")

# Plot pie chart
fig = px.pie(largest_contributor, names='Category', values='Value', title='Contribution of Top 10 Influential Factors')
st.plotly_chart(fig, use_container_width=True)
"""