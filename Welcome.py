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

st.title("CLØT WATCH")
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
st.markdown(
    """
    CLOT WATCH predicts a patient's risk of thrombosis with the click of a few buttons!   
    
    To use the current predictive model to obtain a patient's predicted risk, use the side bar to navigate to page 1
    ('Predict Thrombotic Risk').    
    
    To upload data to train a new predictive model, use the side bar to navigate to page 2 ('Train Model').

    If you're new here, refer to the instructions below!   
    """
)
st.markdown("""---""")
st.subheader("Instructions for CLØT WATCH")
st.write("""

1. If you have a .pkl file for trained model and wish to only predict the thrombotic risk of a patient, skip to step 14. If you do not have a .pkl file for the trained model, and wish to train a new model, continue to step 2. 

2. From the “Welcome” page, navigate to the “Train Model” tab. 

3. Follow the instructions on data formatting. 

4. Check the data file for typos and missing entries/columns. 

5. Upload patient training data as an excel file. 

6. If an error message appears after the data is uploaded, return to step 4. 

7. Review initial data analysis seen on page. 

8. Select desired parameters. 

9. Click “Train and Validate” on the sidebar. 

10. Scroll to the bottom of the page. 

11. Click the “download model” link 

12. Verify the model is downloaded as a .pkl file. 

13. Navigate to the “Predict Thrombotic Risk” tab. 

14. Follow instructions on data formatting. 

15. Check the data file for typos and missing entries/columns. 

16. In the sidebar, under “Upload Patient Data” upload your single patient data file. 

17. Verify patient data is uploaded and visible on webpage. If an error message appears after the data is uploaded, return to step 15. 

18. In the sidebar, upload your model’s .pkl file. 

19. Verify all page sections contain data. 

20. Click “export results” on the sidebar. 

13. If the user wishes to train a new model with a different selection of parameters, return to step 3. 

14. User may exit the website. """)


st.write("Download the template file here to get started! The first line is filled in with example data, please follow its format and delete it when you have finished entering your data.")
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
        label="Download template",
        data=buffer,
        file_name="CLOTWATCH_template.xlsx",
        mime="application/vnd.ms-excel"
    )