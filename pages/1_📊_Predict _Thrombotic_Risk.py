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
def input_data(patientBaseline,patientTEG,user_extend_data=False):
    
    # Save IDs
    baseline_id = patientBaseline["Record ID"]
    tegValues_id = patientTEG[["Record ID","Date of TEG Collection"]]
    
    # clean patient data
    cleanPatientBaseline, cleanPatientTEG, tegValues = transform_data(patientBaseline, patientTEG, boundaries, training = False)

    # Extend data
    if user_extend_data:
        cleanPatientTEG, _ = extend_df(cleanPatientTEG,tegValues)

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
    pred = predict(checked_df, _model,ids)
    
    #Plot
    importance_df = iterate_importance(checked_df, _model, ids)
    fig = plot_importance(importance_df ,plot_name)

    return pred, fig

@st.cache_data
def plot_pred_cached(pred_TEG , pred_baseline):
    fig = plot_pred(pred_TEG , pred_baseline)
    return fig
    

#--------------------------Side bar--------------------------#
# Upload patient's data
uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["xlsx"])

# Upload model
uploaded_model_file = st.sidebar.file_uploader("Upload Predictive Model", type = ["pkl"])


#--------------------------Patient info--------------------------#
# If only model is uploaded show model info
if uploaded_model_file != None and uploaded_file == None :
    # import training data and models from the uploaded pkl file
    uploaded_model = joblib.load(uploaded_model_file)
    model_TEG = uploaded_model.get("TEG model")
    model_baseline = uploaded_model.get("Baseline model")
    column_TEG = uploaded_model.get("TEG column names")
    columns_baseline = uploaded_model.get("Baseline column names")
    scores_TEG = uploaded_model.get("TEG model scores")
    scores_baseline = uploaded_model.get("Baseline model score")
    extend_data = uploaded_model.get("Extend data")
    data_fig = uploaded_model.get("Data demographics")

    # Note about training data demographics
    st.write("""**Note:** please be mindful of the demographics of the data used to train your predictive model.
            The more represented your patient is in the training data, the more reliable the prediction will be.""")
        
    if extend_data:
        st.markdown("This TEG model has been trained avoiding collinear values.")
    else:
        st.markdown("This TEG model has been trained using all possible values, including some that might be collinear.")
        
        st.markdown("The data demographics of the data used to trained the models were:")
        st.plotly_chart(data_fig, use_container_width=True)

        st.markdown("Validity score")
        scores_TEG = pd.DataFrame(scores_TEG, index=["TEG-based model 2 (Selected factors)"])
        scores_baseline = pd.DataFrame(scores_baseline, index=["General info based model"])
        st.table(pd.concat([scores_baseline, scores_TEG], sort=False))


# Get patient data from uploaded file            
elif uploaded_file != None :
    try:
        # Read the uploaded Excel file into a Pandas DataFrame
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        sheet_names = ['Baseline', 'TEG Values']  # Replace with your sheet names

        # Access each sheet's data using the sheet name as the key
        patientBaseline = pd.read_excel(xls, sheet_names[0])
        patientTEG = pd.read_excel(xls, sheet_names[1])
        
    except:
        st.error("The uploaded file does not conform to the required format. Specifically, it should include the pages labeled 'Baseline', and 'TEG Values'. ", icon="🚨")
        st.stop()

    # Check if the file uploaded has the right columns
    missing_columns = check_columns(patientBaseline, "baseline")
    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)} in the Baseline sheet", icon="🚨")
        st.stop()
    missing_columns = check_columns(patientTEG, "teg")
    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)} in the TEG Values sheet", icon="🚨")
        st.stop()

        
    patient_data, _, _, _, _, _ = input_data(patientBaseline,patientTEG)

    # Patients info 
    st.subheader(':blue[Patients Uploaded:]')
    st.markdown("Please review if the information below is correct. The index of the table is the column called *Record ID*")
    st.table(patient_data)

    if uploaded_model_file != None :

        # import training data and models from the uploaded pkl file
        uploaded_model = joblib.load(uploaded_model_file)
        model_TEG = uploaded_model.get("TEG model")
        model_baseline = uploaded_model.get("Baseline model")
        column_TEG = uploaded_model.get("TEG column names")
        columns_baseline = uploaded_model.get("Baseline column names")
        scores_TEG = uploaded_model.get("TEG model scores")
        scores_baseline = uploaded_model.get("Baseline model score")
        extend_data = uploaded_model.get("Extend data")
        data_fig = uploaded_model.get("Data demographics")

        # Note about training data demographics
        st.write("""**Note:** please be mindful of the demographics of the data used to train your predictive model.
                The more represented your patient is in the training data, the more reliable the prediction will be.""")
        
        with st.expander("Model evaluation"):
            if extend_data:
                st.markdown("This TEG model has been trained avoiding collinear values.")
            else:
                st.markdown("This TEG model has been trained using all possible values, including some that might be collinear.")
            
            st.markdown("The data demographics of the data used to trained the models were:")
            st.plotly_chart(data_fig, use_container_width=True)

            st.markdown("Validity score")
            scores_TEG = pd.DataFrame(scores_TEG, index=["TEG-based model 2 (Selected factors)"])
            scores_baseline = pd.DataFrame(scores_baseline, index=["General info based model"])
            st.table(pd.concat([scores_baseline, scores_TEG], sort=False))

        # Reload patient data with extend_data
        patient_data, cleanPatientBaseline, cleanPatientTEG, tegValues, baseline_id, tegValues_id = input_data(patientBaseline,patientTEG, extend_data)

        # Calculate risk
        pred_TEG, fig_TEG = calculate_risk(cleanPatientTEG, column_TEG,model_TEG, tegValues_id, "Most influencial factors from TEG model")
        pred_baseline, fig_baseline = calculate_risk(cleanPatientBaseline, columns_baseline, model_baseline, baseline_id, "Most influencial factors Gen. & Comorbid. model")
    
        # Please change this, its going everwhere
        pred_TEG_todisp = pred_TEG.copy()
        pred_TEG_todisp['Date of TEG Collection'] = pd.to_datetime(pred_TEG['Date of TEG Collection']).dt.strftime('%Y-%m-%d')
    # Select elements to display
    # Select 'Record ID' using a Streamlit multiselect widget
        #selected_record_ids = st.multiselect("Select Record IDs", merged_df['Record ID'].unique())
        # Filter the DataFrame based on selected 'Record ID'
        #filtered_df = merged_df[merged_df['Record ID'].isin(selected_record_ids)]

        # display thrombosis risk
        st.markdown("""---""")
        st.subheader(":blue[Patients' Calculated Risk of Thrombosis:]")


        # Merge DataFrames on 'Record ID'
        #merged_df = pd.merge(pred_baseline, pred_TEG, on='Record ID', how='outer')

        # # Select 'Record ID' using a Streamlit multiselect widget
        # selected_record_ids = st.multiselect("Select Record IDs", merged_df['Record ID'].unique())

        # # Filter the DataFrame based on selected 'Record ID'
        # filtered_df = merged_df[merged_df['Record ID'].isin(selected_record_ids)]
        # pred_baseline_filtered_df = pred_baseline[pred_baseline['Record ID'].isin(selected_record_ids)]
        # pred_TEG_filtered_df = pred_TEG[pred_TEG['Record ID'].isin(selected_record_ids)]
        
        # # Convert 'Date of TEG Collection' to string format
        # filtered_df['Date of TEG Collection'] = pd.to_datetime(filtered_df['Date of TEG Collection']).dt.strftime('%Y-%m-%d')

        # Display predictions as text # This needs to be fixed
        for index, row in pred_baseline.iterrows():
            record_id = row['Record ID'].astype(str)
            prediction_baseline = row['Prediction']
            #st.write("".join([":blue[Baseline-based prediction] for patient ", str(record_id), ":"]))
            st.write(":blue[Baseline-based prediction for patient:]")
            st.subheader("".join([str(round(prediction_baseline,2)), "%"]))
            #st.markdown(f"""
                        #The baseline risk for patient **{record_id}** is **{round(prediction_baseline*1,2)}%**
                        
                        #According to their TEG results:
            #            """)

            st.write(":blue[TEG-based prediction(s) for patient:]")
            for indexT, rowT in pred_TEG_todisp.iterrows():
                prediction_TEG = rowT['Prediction']
                date_teg = rowT['Date of TEG Collection']
                st.write("".join(["Risk based on data from ", str(date_teg), ":"]))
                st.subheader("".join([str(round(prediction_TEG,2)), "%"]))
                #st.write("TEG prediction(s) for patient")
                #st.markdown(f"- Risk at {date_teg}: {round(prediction_TEG*1,2)}%")


        st.markdown("---")
        st.markdown("""In the following chart the dashed lines (--) represent
                    the prediction from the Gen. & Comorbid. model and the solid lines with points (o-)
                    are the predictions based on TEG values. The colors are grouped by *Record ID*. """)
        fig_pred = plot_pred_cached(pred_TEG , pred_baseline) # This could be updated
        st.plotly_chart(fig_pred)

        st.subheader(":blue[Most influencial factors when calculating risk of thrombosis:]")
        st.plotly_chart(fig_baseline)
        st.plotly_chart(fig_TEG)
    
    
    

else:
    st.write("Begin by uploading a patient's file using the button 'Upload Patient Data' in the sidebar!")
    st.write("Please make sure that the file is in the same format as the downloadable template on the welcome page.")
    st.write("Please also upload a trained predictive model (if you just did this on the 'Train Model' page, look in your downloads for a file called 'CLOTWATCH_predictive_model.pkl').")  
