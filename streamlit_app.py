import streamlit as st
import pandas as pd 
import numpy as np
from collections import Counter

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
    st.subheader("Datensatz einlesen:")
    # st.markdown("Eigene Daten hinzufügen oder vorgefertigten Datensatz verwenden")

    use_custom = st.checkbox("Testdaten laden? (Eigene Daten hinzufügen oder vorgefertigten Datensatz verwenden)", value=True)

    df_work = None
    n_cols = 10  # maximale Spaltenanzahl für Anzeige

    if use_custom:
        st.info("Testdatensatz names `daten.csv` wird verwendet.")
        try:
            df = pd.read_csv("data/daten.csv", sep=';')
            df_work = df.copy()
            st.success("Datensatz wurde erfolgreich eingeladen.")
        except Exception as e:
            st.error(f"Could not load custom dataset: {e}")
    else:
        uploaded_file = st.file_uploader("Upload your CSV file (nur CSV mit ';' als Separator)", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                df_work = df.copy()
                st.success("Datensatz wurde erfolgreich eingeladen.")
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")

    if df_work is not None:

        df_work = df_work.dropna()
        st.divider()
        st.subheader("Datensatz als Tabelle:")
        max_cols = min(n_cols, len(df_work.columns))
        st.dataframe(df_work.iloc[:, :].head())
        st.divider()
        st.subheader("Statistik zu Datensatz:")
        st.dataframe(df_work.describe())


##############################################################################################################
##############################################################################################################
with tabs[1]:
    st.subheader("Daten visualisieren")
    
    if df_work is None:
        st.warning("Bitte lade zuerst einen Datensatz im Tab 'Daten' hoch.")
    else:
        # Spalten nach Typ trennen (anhand des ersten Werts als Heuristik)
        numeric_cols = []
        categorical_cols = []

        for col in df_work.columns:
            first_val = df_work[col].dropna().iloc[0] if not df_work[col].dropna().empty else None
            if isinstance(first_val, (int, float)):
                numeric_cols.append(col)
            elif isinstance(first_val, str):
                categorical_cols.append(col)

        # Auswahl für X und Y aus numerischen Spalten
        st.markdown("### Wähle X-Achse (numerisch):")
        x_choice = st.selectbox("X-Achse", options=numeric_cols, key="x_axis")

        st.markdown("### Wähle Y-Achse (numerisch):")
        y_choice = st.selectbox("Y-Achse", options=numeric_cols, key="y_axis")

        # Auswahl für Hue (optional) aus kategorialen Spalten
        hue_choice = None
        if categorical_cols:
            st.markdown("### Optional: Wähle eine Gruppierungsvariable (Hue):")
            use_hue = st.checkbox("Hue verwenden?")
            if use_hue:
                hue_choice = st.selectbox("Hue", options=categorical_cols, key="hue_selector")

        # Plot
        if x_choice and y_choice:
            if x_choice == y_choice:
                st.warning("Bitte wähle unterschiedliche Spalten für X und Y.")
            else:
                fig, ax = plt.subplots()
                if hue_choice:
                    for group in df_work[hue_choice].dropna().unique():
                        subset = df_work[df_work[hue_choice] == group]
                        ax.scatter(subset[x_choice], subset[y_choice], label=str(group), alpha=0.7)
                    ax.legend(title=hue_choice)
                else:
                    ax.scatter(df_work[x_choice], df_work[y_choice], alpha=0.7)

                ax.set_xlabel(x_choice)
                ax.set_ylabel(y_choice)
                ax.set_title(f"Scatterplot: {x_choice} vs {y_choice}")
                st.pyplot(fig)
