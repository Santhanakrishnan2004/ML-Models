import pickle
import streamlit as st
import sklearn
import numpy as np

# Load the model from the file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.title("SPAM CLASSIFICATION MODEL")

# Subheader for instructions
st.subheader("Enter the message for spam classification:")

# Get the user input
message = st.text_input("Enter the message")

# If the user enters a message
if message:
    prediction = model.predict([message])
    # Show the result
    if prediction[0] == 1:
        st.subheader("This is a SPAM message.")
    else:
        st.subheader("This is NOT a SPAM message.")
