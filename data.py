import streamlit as st
import plotly_express as px 
import pandas as pd 

def app():
    st.title('Data')

    st.write("The following is the DataFrame of the `power outage` dataset.")



#uploid files
st.file_uploader(label="Upload your CSV or Excel file", type=['csv', 'xlsx'])