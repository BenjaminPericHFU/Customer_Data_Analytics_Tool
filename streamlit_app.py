import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

        
        # Klassifikation nach Anzahl eindeutiger Werte (für spätere Visualisierung)
        column_classification = {"xy": [], "hue": []}
        for col in df_work.columns:
            nunique = df_work[col].nunique()
            if pd.api.types.is_numeric_dtype(df_work[col]) and nunique > 5:
                column_classification["xy"].append(col)
            elif 1 < nunique <= 5:
                column_classification["hue"].append(col)


##############################################################################################################
##############################################################################################################
with tabs[1]:
    if df_work is None or not column_classification["xy"]:
        st.warning("Keine geeigneten numerischen Spalten für Visualisierung gefunden.")
    else:
        st.header("📊 Interaktive Visualisierung")

        x_col = st.selectbox("X-Achse wählen:", column_classification["xy"], key="x")
        y_col = st.selectbox("Y-Achse wählen:", column_classification["xy"], key="y")
        hue_col = st.selectbox("Gruppierung (Hue, optional):", ["Keine"] + column_classification["hue"], key="hue")

        plot_type = st.radio(
            "Diagrammtyp wählen:",
            ["Scatterplot", "Boxplot", "Lineplot", "Areaplot", "Barplot"]
        )

        if plot_type in ["Lineplot", "Areaplot", "Barplot"]:
            st.info(f"{plot_type} wird nur mit X als Index und Y als Wert angezeigt (ohne Gruppierung).")

            if x_col != y_col:
                plot_data = df_work[[x_col, y_col]].copy()
                plot_data = plot_data.set_index(x_col).sort_index()

                if plot_type == "Lineplot":
                    st.line_chart(plot_data)
                elif plot_type == "Areaplot":
                    st.area_chart(plot_data)
                elif plot_type == "Barplot":
                    st.bar_chart(plot_data)
            else:
                st.error("X- und Y-Achse dürfen nicht identisch sein.")
        else:
            # Für Scatterplot, Boxplot usw. → wie vorher mit matplotlib/seaborn
            fig, ax = plt.subplots()

            if plot_type == "Scatterplot":
                sns.scatterplot(data=df_work, x=x_col, y=y_col, hue=None if hue_col == "Keine" else hue_col, ax=ax)
            elif plot_type == "Boxplot":
                sns.boxplot(data=df_work, x=hue_col if hue_col != "Keine" else x_col, y=y_col, ax=ax)

            st.pyplot(fig)






with tabs[2]:  # Visualisierung
    st.header("📊 Interaktive Visualisierung")

    # Auswahl über Dropdown-Menüs
    x_col = st.selectbox("X-Achse wählen:", column_classification["xy"])
    y_col = st.selectbox("Y-Achse wählen:", column_classification["xy"])
    hue_col = st.selectbox("Gruppierung (Hue, optional):", ["Keine"] + column_classification["hue"])

    # Art der Visualisierung auswählen
    plot_type = st.radio("Diagrammtyp wählen:", ["Scatterplot", "Boxplot", "Lineplot"])

    # Plot erstellen
    fig, ax = plt.subplots()

    if plot_type == "Scatterplot":
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=None if hue_col == "Keine" else hue_col, ax=ax)
    elif plot_type == "Boxplot":
        sns.boxplot(data=df, x=hue_col if hue_col != "Keine" else x_col, y=y_col, ax=ax)
    elif plot_type == "Lineplot":
        sns.lineplot(data=df, x=x_col, y=y_col, hue=None if hue_col == "Keine" else hue_col, ax=ax)

    st.pyplot(fig)
