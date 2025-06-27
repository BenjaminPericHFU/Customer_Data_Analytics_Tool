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
    st.subheader("Auswahl:")
    st.caption("Eigene Daten hinzufügen oder vorgefertigten Datensatz verwenden")

    use_custom = st.checkbox("Testdaten laden:", value=True)

    df_work = None

    if use_custom:
        st.info("Using predefined custom dataset: `daten.csv`")
        try:
            df = pd.read_csv("data/daten.csv")
            # Nur die ersten 10 Spalten behalten
            df_work = df.iloc[:, :10].copy()
            st.success("Custom dataset loaded successfully with max. 10 columns.")
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
                df_work = df.iloc[:, :10].copy()  # ebenfalls auf max. 10 Spalten limitieren
                st.success("File successfully uploaded with max. 10 columns.")
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")

    if df_work is not None:
        st.write("### Spaltennamen (max. 10):")
        st.write(df_work.columns.tolist())

        st.write("### Datenvorschau:")
        st.write(df_work.head())

