# streamlit run ./demos/streamlit_demo.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title
st.title("Streamlit Demo App")

# Header
st.header("This is a header")

# Subheader
st.subheader("This is a subheader")

# Text
st.write("This is some text.")

# Displaying data
st.write("Displaying data in a table:")
df = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))
st.table(df)

# Interactive widgets
st.sidebar.header("Interactive Widgets")

# Slider widget
slider_value = st.sidebar.slider("Select a value", 0, 100, 50)

# Button widget
if st.sidebar.button("Click me"):
    st.write(f"You clicked the button! Selected value: {slider_value}")

# Checkbox widget
checkbox_state = st.sidebar.checkbox("Check me")
if checkbox_state:
    st.write("Checkbox is checked")

# Selectbox widget
option = st.sidebar.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {option}")

# Radio button widget
radio_option = st.sidebar.radio("Choose one option", ["Option A", "Option B", "Option C"])
st.write(f"You selected: {radio_option}")

# File upload widget
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "txt"])
if uploaded_file is not None:
    st.write("You uploaded a file!")

# Plotting
st.write("Plotting a chart:")
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)

