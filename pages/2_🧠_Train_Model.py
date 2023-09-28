import streamlit as st
import pandas as pd
import os
import plotly.express as px


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

#PREDICTION
# Get data from folder
path_pages = os.path.dirname(__file__)
path_pages = path_pages.replace("/pages", "")
data_path = os.path.join(path_pages, "data", "DummyResult.xlsx")

# Get 10 most influencial elements
prediction = pd.read_excel(data_path)
prediction = prediction.T.squeeze()
largest_contributor = prediction.nlargest(n=10, keep='first')
largest_contributor = pd.DataFrame({'Category': largest_contributor.index, 'Value': largest_contributor.values})

# Plot pie chart
fig = px.pie(largest_contributor, names='Category', values='Value', title='Parameters contribution to risk')
st.plotly_chart(fig, use_container_width=True)

# Side bar layout
st.sidebar.file_uploader("Upload file X")