import streamlit as st
import pandas as pd 
import numpy as np


st.title('Customer Data Analytics Tool')

# File uploader – supports CSV and Excel files
uploaded_file = st.file_uploader("Input Customer Data: ", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')




st.dataframe(pd.DataFrame(df.columns, columns=["Column Names"]))


