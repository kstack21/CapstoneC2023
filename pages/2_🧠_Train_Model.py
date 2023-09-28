import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Train Model",
    page_icon="🧠",
)

# Main layout
st.title("Page title")
st.markdown("""This page is for administrators who want to retrain the model 
            or fine-tune its parameters. 
            It can provide options for uploading new data, 
            selecting model algorithms, and adjusting training settings. """)

# Radio button widget
st.subheader("Selection of Highly Correlated Paramters")
#percentages should turn into variables pulled from the model
radio_pivpa = st.radio("Choose one parameter", ["Platlet Inhibition (75%)", "Platelet Aggregation (76%)"])

#Below is an example idk what would be correlated
radio_gvs=st.radio("Chose one parameter", ["Gender (50%)","Smoking (85%)"])

st.write(f"You selected: {radio_pivpa} and {radio_gvs}")

# Side bar layout
st.sidebar.file_uploader("Upload Data Set")