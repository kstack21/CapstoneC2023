"""
Module Name: Welcome.py
Description: This module contains the code necessary to define the user interface, 
             functionality, and layout of the web application
Run in terminal: streamlit run Welcome.py
"""
import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="Welcome and Instructions",
    page_icon="🏥",
)

st.title("Welcome to [App name]! 👋")

st.markdown(
    """
    This page should provide a welcome to the users and offer 
    clear instructions on how to use the website. 
    You can include a brief overview of the project, 
    the importance of predicting thrombotic events, 
    and step-by-step guidance on how to proceed.
"""
)