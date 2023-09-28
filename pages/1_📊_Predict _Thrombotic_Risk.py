import streamlit as st

st.set_page_config(
    page_title="Predict Thrombotic Risk",
    page_icon="📊",
)

# Main layout
st.title("Page title")
st.markdown("""This page should be where users can input patient 
            data and receive predictions regarding thrombotic risk. 
            Users can input patient-specific information, 
            and the model will generate a risk assessment based on the input data. 
            This page should have a user-friendly interface for data input 
            and result display.""")


# Side bar layout
st.sidebar.file_uploader("Upload Patient Data and Usable Model")