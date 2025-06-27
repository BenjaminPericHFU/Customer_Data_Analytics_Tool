import streamlit as st
import pandas as pd 
import numpy as np


col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/Logo_01.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 5px;'>Customer Data Analytics Tool</h1>", unsafe_allow_html=True)

# Tabs hinzufuegen
tabs = st.tabs(["Daten", "Visualisierung", "ML-Training", "Auswertung"], width = "stretch")

##############################################################################################################
##############################################################################################################

with tabs[0]: 
    # Checkbox: use custom data or not
    st.subheader("Auswahl:")
    st.caption("Eigene Daten hinzufügen oder vorgefertigten Datensatz verwenden")
    
    use_custom = st.checkbox("Testdaten laden:", value=True)
    
    df = None
    
    if use_custom:
        st.info("Using predefined custom dataset: `daten.csv`")
    
        try:
            # Load from local file (make sure 'daten.csv' is in the same folder)
            df = pd.read_csv("data/daten.csv")
    
            # OR if using raw GitHub link, uncomment and set the correct URL:
            # url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/daten.csv"
            # df = pd.read_csv(url)
    
            st.success("Custom dataset loaded successfully.")
    
        except Exception as e:
            st.error(f"Could not load custom dataset: {e}")
    
    else:
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])
    
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
    
                st.success("File successfully uploaded.")
    
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")
    
    # If df was loaded successfully, show info
    if df is not None:
        st.write("### Column Names:")
        st.dataframe(pd.DataFrame(df.columns, columns=["Columns"]))
    
        st.write("### Data Preview:")
        st.dataframe(df.head())

