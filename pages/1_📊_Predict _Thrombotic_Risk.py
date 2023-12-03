import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
import joblib

# Get higher level functions
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from functions import *


# Set page config
st.set_page_config(
    page_title="Predict Thrombotic Risk",
    page_icon="📊",
)

# Import boundary and timepoint values
base_directory = os.getcwd()
filename = 'data_boundaries.json'
file_path = os.path.join(base_directory, 'data', filename)
with open(file_path, 'r') as json_file:
    boundaries = json.load(json_file)

filename = 'timepoints.json'
file_path = os.path.join(base_directory, 'data', filename)
with open(file_path, 'r') as json_file:
    timepoints = json.load(json_file)

#--------------------------Page description--------------------------#
# Title and Instructions
st.title("Predict a Patient's Risk of Thrombosis")
st.write("Begin by uploading a patient's file using the button 'Upload Patient Data' in the sidebar!")
st.write("Please make sure that the file is in the same format as the downloadable template on the welcome page.")
st.write("Please also upload a trained predictive model (if you just did this on the 'Train Model' page, look in your downloads for a file called 'trained_model.pkl').")  

#--------------------------Side bar--------------------------#
# Upload model
uploadedDict = st.sidebar.file_uploader("Upload Predictive Model", type = ["pkl"])

# Upload patient's data
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])

# Download 
st.sidebar.button("Export results")

#--------------------------Patient info--------------------------#
# Get patient data from uploaded file
if uploaded_file != None:

    # Read patient data 
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    patientBaseline = pd.read_excel(uploaded_file, sheet_name = 0, engine = "openpyxl")
    patientTEG = pd.read_excel(uploaded_file, sheet_name = 1, engine = "openpyxl")

    # clean patient data
    cleanPatientBaseline, cleanPatientTEG, tegValues = transform_data(patientBaseline, patientTEG, boundaries, timepoints)

    # get IDs and timepoints of each patient TEG record
    tegRecordID = patientTEG['Record ID']
    tegTimepoint = patientTEG['Visit Timepoint']

    # Patient Data Header #
    st.header(':green[Patient Data Uploaded]')

    # Organizing text in columns
    col1, col2, col3, col4 = st.columns(4)

    # Present General Patient Info
    with col1:
        # Age
        if 'Age' in df:
            st.metric(label = "Age", value = df.at[0,'Age'])
        else:
            st.metric(label = ":red[Age]", value = "n/a")
        # Diabetes
        if 'Diabetes' in df:
            if df.at[0, 'Diabetes']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label="Diabetes", value = temp)
        else:
            st.metric(label = ":red[Diabetes]", value = "No column named 'Diabetes'")
        
    with col2:
        # Sex
        if 'Sex' in df:
            if df.at[0,'Sex']:
                temp = "Male"
            else:
                temp = "Not Male (Female or other)"
            st.metric(label = "Sex", value = temp)
        else:
            st.metric(label = ":red[Sex]", value = "n/a")
        # Hypertension
        if 'Hypertension' in df:
            if df.at[0,'Hypertension']:
                temp = "Yes"
            else:
                temp = "No"
            st.metric(label="Hypertension", value = temp)
        else:
            st.metric(label = ":red[Hypertension]", value = "n/a")

    with col3:
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
   
    with col4: 
        # BMI
        if 'BMI' in df:
            st.metric(label="BMI", value = round(df.at[0, 'BMI'], 2))
        else:
            st.metric(label = ":red[BMI]", value = "No column named 'BMI'")  

    # display thrombosis risk
    st.markdown("""---""")
    st.header(":green[Patient's Calculated Risk of Thrombosis: ]")
    if uploadedDict != None:

        # import training data and models from the uploaded pkl file
        trainingData = joblib.load(uploadedDict).get("Training data")
        trainingBaseline = joblib.load(uploadedDict).get("Baseline training data")
        trainingTEG = joblib.load(uploadedDict).get("TEG training data")
        trainedModelTEG = joblib.load(uploadedDict).get("TEG model")
        trainedModelBaseline = joblib.load(uploadedDict).get("Baseline model")
        trainingTEGColumns = joblib.load(uploadedDict).get("TEG column names")
        trainingBaselineColumns = joblib.load(uploadedDict).get("Baseline column names")

        # clean the imported data before using in predictions
        cleanPatientTEG_updated = check_column_names(cleanPatientTEG, trainingTEGColumns)
        cleanPatientBaseline_updated = check_column_names(cleanPatientBaseline, trainingBaselineColumns)

        # get prediction from baseline model, make string to display percentage
        baselineRisk = predict(cleanPatientBaseline_updated, ['Record ID', 'Events'], trainedModelBaseline)
        baselineRiskText = "".join([str(round(baselineRisk[0] * 100, 2)), "%"])
        st.subheader(":blue[Based on general patient info:]")
        st.subheader(baselineRiskText)

        # get prediction from TEG model, iterate over each TEG record and display percentage for each
        tegRisk = predict(cleanPatientTEG_updated, ['Record ID', 'Events'], trainedModelTEG)
        st.subheader(":blue[Based on TEG info:]")
        tegRiskText = []
        tegRiskTextID = []
        tegRiskTextNum = []
        for i in range (len(tegRisk)):
            tegRiskText.append(str(round(tegRisk[i] * 100, 2)))
        for i in range (len(tegRiskText)):
            tegRiskTextID.append("".join(["From record ", str(tegRecordID[i]), " at timepoint ", tegTimepoint[i], ":"]))
            tegRiskTextNum.append("".join([tegRiskText[i], "%"]))
            st.subheader(tegRiskTextID[i])
            st.subheader(tegRiskTextNum[i])
        st.markdown("""---""")

        # illustrate the training data demographics
        st.subheader("Demographics of the Training Data")
        st.write("""This is an overview of the demographics of the data used in the training of your predictive model. Please consider how your patient
                 fits in with these demographics; if there is little to no representation of their age, race, etc. the model's prediction may not
                 be as accurate.""")
        st.plotly_chart(visualize_data(trainingBaseline, trainingTEG), use_container_width=True)
    else:
        st.subheader(":red[No risk calculated yet]")

# display outline of page with no information
else:
    # data header (no patient info)
    st.header(':red[No Patient Data Uploaded]')

    # display thrombosis risk
    st.header(":red[Patient's Calculated Risk of Thrombosis: ]")
    st.subheader(":red[No risk calculated yet]")

# Get 10 most influencial elements
#prediction = pd.read_excel(data_path)
#prediction = prediction.T.squeeze()
#largest_contributor = prediction.nlargest(n=10, keep='first')
#largest_contributor = pd.DataFrame({'Category': largest_contributor.index, 'Value': largest_contributor.values})

# Pie chart title
#st.subheader("Predictive Model Details:")

# Plot pie chart
#fig = px.pie(largest_contributor, names='Category', values='Value', title='Contribution of Top 10 Influential Factors')
#st.plotly_chart(fig, use_container_width=True)