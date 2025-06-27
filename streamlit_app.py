import streamlit as st
import pandas as pd 
import numpy as np


st.title('Customer Data Analytics Tool')

# File uploader – supports CSV and Excel files
"""uploaded_file = st.file_uploader("Input Customer Data: ", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

st.dataframe(pd.DataFrame(df.columns, columns=["Column Names"]))
"""

st.title("Dataset Loader")

# Step 1: User selects source
option = st.radio(
    "Choose data source:",
    ("Upload your own file", "Use preset dataset (daten.csv)")
)

df = None  # Initialize DataFrame

if option == "Upload your own file":
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("File successfully uploaded!")
        except Exception as e:
            st.error(f"Error reading the file: {e}")

elif option == "Use preset dataset (daten.csv)":
    try:
        # ⚠️ Make sure daten.csv is in the same folder as this script
        df = pd.read_csv("daten.csv")
        st.success("Loaded 'daten.csv' as preset dataset.")
    except FileNotFoundError:
        st.error("daten.csv not found. Please make sure it is in the same folder as this script.")
    except Exception as e:
        st.error(f"Error reading daten.csv: {e}")

# Step 2: Display if DataFrame is ready
if df is not None:
    st.write("### Column Names (Header):")
    st.dataframe(pd.DataFrame(df.columns, columns=["Column Names"]))

    st.write("### Data Preview (first 5 rows):")
    st.dataframe(df.head())

