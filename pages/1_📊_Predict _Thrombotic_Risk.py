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

#--------------------------Page description--------------------------#
# Title and Instructions
st.title("Predict a Patient's Risk of Thrombosis")
st.write("Begin by uploading a patient's file using the button 'Upload Patient Data' in the sidebar!")
st.write("Please make sure that the file is in the same format as the downloadable template on the welcome page.")
st.write("Please also upload a trained predictive model (if you just did this on the 'Train Model' page, look in your downloads for a file called 'CLOTWATCH_predictive_model.pkl').")  

#--------------------------Cached Functions--------------------------#

# Import boundary and timepoint values
@st.cache_data
def import_json_values_cached():
    base_directory = os.getcwd()
    filename = 'data_boundaries.json'
    file_path = os.path.join(base_directory, 'data', filename)
    with open(file_path, 'r') as json_file:
        boundaries = json.load(json_file)
    
    return boundaries

# Load general data
boundaries = import_json_values_cached()

@st.cache_data
def input_data(uploaded_file):
    # Read the uploaded Excel file into a Pandas DataFrame
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_names = ['Baseline', 'TEG Values']  # Replace with your sheet names

    # Access each sheet's data using the sheet name as the key
    patientBaseline = pd.read_excel(xls, sheet_names[0])
    patientTEG = pd.read_excel(xls, sheet_names[1])
    
    # Save IDs
    baseline_id = patientBaseline["Record ID"].astype(str)
    tegValues_id = {"Patient": patientTEG["Record ID"].astype(str),
                    "Date":patientTEG["Date of TEG Collection"]}
    
    # clean patient data
    cleanPatientBaseline, cleanPatientTEG, tegValues = transform_data(patientBaseline, patientTEG, boundaries, training = False)

    # Get patients general info
    modified_df = cleanPatientBaseline[['Record ID','BMI','Is Male','White', 'Age', 'Clotting Disorder', 'Hypertension', 'Diabetes']].copy()
    # Round 'BMI' to 2 decimal places
    modified_df['BMI'] = modified_df['BMI'].round(2)
    # Convert boolean columns to 'Yes' or 'No'
    boolean_columns = ['White','Is Male', 'Hypertension', 'Diabetes']
    for column in boolean_columns:
        modified_df[column] = modified_df[column].map({True: 'Yes', False: 'No'})
    # Set 'Record ID' as the index
    modified_df.set_index('Record ID', inplace=True)

    return modified_df, cleanPatientBaseline, cleanPatientTEG, tegValues, baseline_id, tegValues_id

@st.cache_data
def calculate_risk(df, column_names, _model, ids, plot_name):

    # Check column names
    checked_df = check_column_names(df, column_names)

    #Get risk
    pred = predict(checked_df, _model)
    
    #Plot
    importance_df = iterate_importance(checked_df, _model, ids)
    fig = plot_importance(importance_df ,plot_name)

    return pred, fig
    

#--------------------------Side bar--------------------------#
# Upload patient's data
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])

# Upload model
uploaded_model_file = st.sidebar.file_uploader("Upload Predictive Model", type = ["pkl"])


#--------------------------Patient info--------------------------#
# Get patient data from uploaded file
if uploaded_file != None and uploaded_model_file != None :

    # import training data and models from the uploaded pkl file
    uploaded_model = joblib.load(uploaded_model_file)
    model_TEG = uploaded_model.get("TEG model")
    model_baseline = uploaded_model.get("Baseline model")
    column_TEG = uploaded_model.get("TEG column names")
    columns_baseline = uploaded_model.get("Baseline column names")
    scores_TEG = uploaded_model.get("TEG model scores")
    scores_baseline = uploaded_model.get("Baseline model score")
    extend_data = uploaded_model.get("Extend data")

    # Load patient data
    patient_data, cleanPatientBaseline, cleanPatientTEG, tegValues, baseline_id, tegValues_id = input_data(uploaded_file)

    # Patients info 
    st.subheader(':blue[Patients Uploaded:]')
    st.markdown("Please review if the information below is correct.")
    st.table(patient_data)

    # Calculate risk
    id_teg_list = [f"Patient {patient} {date}" for patient, date in zip(tegValues_id["Patient"], tegValues_id["Date"].astype(str))]
    pred_TEG, fig_TEG = calculate_risk(cleanPatientTEG, column_TEG,model_TEG, id_teg_list, "Most influencial factors from TEG model")
    
    pred_baseline, fig_baseline = calculate_risk(cleanPatientBaseline, columns_baseline, model_baseline, list(baseline_id), "Most influencial factors Gen. & Comorbid. model")
 
    # display thrombosis risk
    st.markdown("""---""")
    st.subheader(":blue[Patients' Calculated Risk of Thrombosis:]")
    st.table(pred_baseline)
    st.table(pred_TEG)

    st.subheader(":blue[Most influencial factors when calculating risk of thrombosis:]")
    st.plotly_chart(fig_baseline)
    st.plotly_chart(fig_TEG)
    
    
    # Note about training data demographics
    st.write("""Note: please be mindful of the demographics of the data used to train your predictive model.
             The more represented your patient is in the training data, the more reliable the prediction will be.""")

