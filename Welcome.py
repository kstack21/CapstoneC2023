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

st.title("Welcome to MAGIC-CLØT! 👋")

st.markdown(
    """
    MAGIC-CLØT (Machine Learning Algorithm for General Identification of Clots in Lower-Extremity Obstruction and Thrombosis)
    Clot Watch
    CLOT WATCH (Computerized Learning and Observation Tool for Detecting Thrombosis in Cardiac and Limb Artery Health)
    """
)