# TO RUN: streamlit run 🏥_Welcome.py

import streamlit as st

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