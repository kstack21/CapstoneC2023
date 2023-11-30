"""
Module Name: Welcome.py
Description: This module contains the code necessary to define the user interface, 
             functionality, and layout of the web application
Run in terminal: streamlit run Welcome.py
"""
import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="CLØT WATCH",
    page_icon="🏥",
)

st.title("CLØT WATCH")

st.markdown(
    """
    CLOT WATCH predicts a patient's risk of thrombosis with the click of a few buttons!   
    
    To use the current predictive model to obtain a patient's predicted risk, use the side bar to navigate to page 1
    ('Predict Thrombotic Risk').    
    
    To upload data to train a new predictive model, use the side bar to navigate to page 2 ('Train Model').   
    """
)

st.write("""
         :red[DISCLAIMER:    
         CLOT WATCH was developed as an academic project by seniors at Northeastern University in collaboration with
         Dr. Anahita Dua and her team at Massachusetts General Hospital. CLOT WATCH has been minimally tested and has
         not been reviewed by the FDA. It should not be used to definitively assume a patient's risk of thrombosis.
         There are many factors that may play into a patient's risk of thrombosis, and CLOT WATCH does not take all of
         them into account. Physicians should use their own judgement and analysis of the patient in addition to, or
         even in place of, the risk evaluation produced by CLOT WATCH.]
         """)

# Title for data template section
st.title("Data File Template Download")

# Information text for data template button``
st.markdown("""
    Click the button below to download the Excel data file template. 
    The data you upload should follow the format seen in this template.
""")

# Button widget for Data Template Download
if st.button("Download Data File Template"):
    # Code to handle the button click event can be added here
    st.write("You clicked the button!")