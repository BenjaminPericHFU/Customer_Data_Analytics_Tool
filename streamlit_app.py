import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px


col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/Logo_01.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 5px;'>Customer Data Analytics Tool</h1>", unsafe_allow_html=True)

# Tabs hinzufuegen
tabs = st.tabs(["Daten", "Visualisierung", "ML-Training", "Auswertung"], width = "stretch")

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
with tabs[1]:
    st.header("📊 Interaktive Visualisierung")

    # --- Verteilung einer Einzelvariablen ---
    st.subheader("Verteilung der Daten analysieren")

    dist_col = st.selectbox(
        "Variable für Verteilung wählen:",
        column_classification["xy"],
        key="dist_col_tab1"
    )
    group_col = st.selectbox(
        "Farbliche Gruppierung (optional):",
        ["Keine"] + column_classification["hue"],
        key="group_tab1"
    )
    color_arg_dist = None if group_col == "Keine" else group_col

    # Gruppierungsvariable als Kategorie casten (wichtig!)
    if color_arg_dist is not None:
        df[color_arg_dist] = df[color_arg_dist].astype("category")

    fig_dist = px.histogram(
        df,
        x=dist_col,
        color=color_arg_dist,
        color_discrete_sequence=custom_colors
    )
    fig_dist.update_layout(barmode='overlay')
    fig_dist.update_traces(opacity=0.75)

    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # --- Vergleich zweier Variablen ---
    st.subheader("Beziehung zwischen Variablen")

    x_col = st.selectbox(
        "X-Achse wählen:",
        column_classification["xy"],
        key="x_axis_tab1"
    )
    y_col = st.selectbox(
        "Y-Achse wählen:",
        column_classification["xy"],
        key="y_axis_tab1"
    )
    hue_col = st.selectbox(
        "Farbliche Gruppierung (optional):",
        ["Keine"] + column_classification["hue"],
        key="hue_tab1"
    )

    plot_type = st.radio(
        "Diagrammtyp wählen:",
        ["Balkendiagramm", "Scatterplot", "Liniendiagramm"],
        key="plot_type_tab1"
    )
    color_arg = None if hue_col == "Keine" else hue_col

    # Gruppierungsvariable als Kategorie casten (wichtig!)
    if color_arg is not None:
        df[color_arg] = df[color_arg].astype("category")

    # Optional: Nur die Top-10-Kategorien anzeigen (bei sehr vielen Kategorien)
    # if color_arg is not None:
    #     top_cats = df[color_arg].value_counts().index[:10]
    #     df = df[df[color_arg].isin(top_cats)]
    #     df[color_arg] = df[color_arg].astype("category")

    if plot_type == "Balkendiagramm":
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            color_discrete_sequence=custom_colors
        )
    elif plot_type == "Scatterplot":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            color_discrete_sequence=custom_colors
        )
    elif plot_type == "Liniendiagramm":
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            color_discrete_sequence=custom_colors
        )

    st.plotly_chart(fig, use_container_width=True)



with tabs[2]:
    st.header("📊 Interaktive Visualisierung")

    # --- Verteilung einer Einzelvariablen ---
    st.subheader("Verteilung der Daten analysieren")

    dist_col = st.selectbox("Variable für Verteilung wählen:", column_classification["xy"], key="dist_col_tab1")
    group_col = st.selectbox("Farbliche Gruppierung (optional):", ["Keine"] + column_classification["hue"], key="group_tab1")
    color_arg_dist = None if group_col == "Keine" else group_col

    fig_dist = px.histogram(df, x=dist_col, color=color_arg_dist)
    fig_dist.update_layout(barmode='overlay')  # oder 'group', je nach Präferenz
    fig_dist.update_traces(opacity=0.75)  # bessere Lesbarkeit bei überlagerung

    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Trennlinie ---
st.divider()

# --- Vergleich zweier Variablen ---
st.subheader("Beziehung zwischen Variablen")

x_col = st.selectbox("X-Achse wählen:", column_classification["xy"], key="x_axis_tab1")
y_col = st.selectbox("Y-Achse wählen:", column_classification["xy"], key="y_axis_tab1")
hue_col = st.selectbox("Farbliche Gruppierung (optional):", ["Keine"] + column_classification["hue"], key="hue_tab1")

plot_type = st.radio("Diagrammtyp wählen:", ["Balkendiagramm", "Scatterplot", "Liniendiagramm"], key="plot_type_tab1")
color_arg = None if hue_col == "Keine" else hue_col

# Benutzerdefinierte, klare Farbpalette – z. B. Plotly, D3 oder Tableau-Schema
custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
    "#bcbd22", "#17becf"
]

if plot_type == "Balkendiagramm":
    fig = px.bar(df, x=x_col, y=y_col, color=color_arg, color_discrete_sequence=custom_colors)
elif plot_type == "Scatterplot":
    fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, color_discrete_sequence=custom_colors)
elif plot_type == "Liniendiagramm":
    fig = px.line(df, x=x_col, y=y_col, color=color_arg, color_discrete_sequence=custom_colors)

st.plotly_chart(fig, use_container_width=True)


with tabs[3]:
    st.header("📊 Interaktive Visualisierung")

    x_col = st.selectbox("X-Achse wählen:", column_classification["xy"], key="x_axis_tab1")
    y_col = st.selectbox("Y-Achse wählen:", column_classification["xy"], key="y_axis_tab1")
    hue_col = st.selectbox("Farbliche Gruppierung (optional):", ["Keine"] + column_classification["hue"], key="hue_tab1")

    plot_type = st.radio("Diagrammtyp wählen:", ["Balkendiagramm", "Scatterplot", "Liniendiagramm"], key="plot_type_tab1")

    color_arg = None if hue_col == "Keine" else hue_col

    if plot_type == "Balkendiagramm":
        fig = px.bar(df, x=x_col, y=y_col, color=color_arg)
    elif plot_type == "Scatterplot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_arg)
    elif plot_type == "Liniendiagramm":
        fig = px.line(df, x=x_col, y=y_col, color=color_arg)

    st.plotly_chart(fig, use_container_width=True)






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
