"""
Module Name: Welcome.py
Description: This module contains the code necessary to define the user interface, 
             functionality, and layout of the web application
Run in terminal: streamlit run Welcome.py
"""
import streamlit as st
import pandas as pd
import os
import sys
import base64
import io

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

st.set_page_config(
    page_title="CLØT WATCH",
    page_icon="🏥",
)
title1 , title2 = st.columns([1,5])
with title1:
    st.image("CLOTWATCH_logo.png", width= 100)
with title2:
    st.title("Clot Watch")

st.markdown("""---""")
st.markdown(
    """
    Welcome to CLOT WATCH!
    CLOT WATCH predicts a patient's risk of thrombosis with the click of a few buttons!   
    
    To use the current predictive model to obtain a patient's predicted risk, use the side bar to navigate to page 1
    ('Predict Thrombotic Risk').    
    
    To upload data to train a new predictive model, use the side bar to navigate to page 2 ('Train Model').

    If you're new here (or if you want a refresher on what to do), refer to the instructions below!   
    """
)
st.markdown("""---""")
st.write("""
        :red[DISCLAIMER:    
         CLOT WATCH was developed as an academic project by seniors at Northeastern University in collaboration with
         Dr. Anahita Dua and her team at Massachusetts General Hospital. CLOT WATCH has been minimally tested and has
         not been reviewed by the FDA. It should not be used to definitively assume a patient's risk of thrombosis.
         There are many factors that may play into a patient's risk of thrombosis, and CLOT WATCH does not take all of
         them into account. Physicians should use their own judgement and analysis of the patient in addition to, or
         even in place of, the risk evaluation produced by CLOT WATCH.]
         """)
st.markdown("""---""")
st.subheader("Instructions for CLOT WATCH")
st.write("""

1. Click the button at the bottom of the page labeled 'Download Template' to download the data input template.
         
2. If you have a .pkl file for trained model and wish to only predict the thrombotic risk of a patient, skip to step 15. If you do not have a .pkl file for the trained model, and wish to train a new model, continue to step 3.  

:blue[TRAIN MODEL PAGE STEPS:]

3. Enter the training data (your full dataset) into the template file, making sure to follow the formatting of the example row.

4. Delete the example row once you have entered your data. 

5. Check the data file for typos and missing entries/columns. Save the file  with a new name when you are sure everything has been filled out properly.
        
6. From the “Welcome” page, navigate to the “Train Model” tab.

7. Upload your Excel file (saved in step 5) using the button on the left labeled 'Upload Data Set of Patient Data'. 

8. If an error message appears after the data is uploaded, return to step 3. Likely this is due to a data entry error. 

9. Review the initial data analysis that appears on the page. 

10. Select the desired parameters. It is alright to leave them as the default selections if you have no preference. 

11. Click the “Train and Validate” button on the sidebar to the left. 

12. Once training and validation has been completed, scroll through to review the model analysis. 

13. Click one of the “Download model” links at the bottom of the page. Further descriptions of the difference can be seen on the page. 

14. Verify the model has downloaded as a .pkl file.
         
:blue[PREDICT THROMBOTIC RISK PAGE STEPS:]

15. Navigate to the “Predict Thrombotic Risk” tab. This is where you will predict the thrombotic risk of one patient based on your trained predictive model.

16. Enter your patient's data into the template file you downloaded from this page (if you no longer have a clean version of this file, you can re-download it at the bottom of this page).

17. Delete the example row once you have entered your data.

18. Check the data file for typos and missing entries/columns. Save the file  with a new name when you are sure everything has been filled out properly.

19. In the sidebar to the left, upload your single patient data file under “Upload Patient Data”. 

20. Verify that your patient's data is uploaded and an overview is visible on the webpage. If an error message appears after the data is uploaded, return to step 16. 

21. In the sidebar, upload your model's .pkl file.

22. Your patient's risk will now be displayed on the page. One risk percentage will be calculated for each TEG record in their file, as well as one based on their general information.

20. Click the “export results” butotn on the sidebar to the left to export these results. 

21. If you wish to train a new model with a different selection of parameters, return to step 3. Otherwise, you can safely exit the website.

:red[Note: your patient data will be deleted from the website cache within 24 hours.]

""")

st.markdown("""---""")

st.write("Download the template file here to get started! The first line is filled in with example data; please follow its format and delete it when you have finished entering your data.")
# data entry template download
base_directory = os.getcwd()
filename = 'HeadersTemplate.xlsx'
file_path = os.path.join(base_directory, filename)

buffer = io.BytesIO()

baseline = pd.read_excel(file_path, sheet_name = 0, engine="openpyxl")
teg = pd.read_excel(file_path, sheet_name = 1, engine="openpyxl")
events = pd.read_excel(file_path, sheet_name = 2, engine="openpyxl")

with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    # Write each dataframe to a different worksheet.
    baseline.to_excel(writer, sheet_name="Baseline", index=False)
    teg.to_excel(writer, sheet_name="TEG Values", index=False)
    events.to_excel(writer, sheet_name="Events", index=False)

    writer.close()

    st.download_button(
        label="Download Template",
        data=buffer,
        file_name="CLOTWATCH_template.xlsx",
        mime="application/vnd.ms-excel"
    )

st.markdown("""---""")