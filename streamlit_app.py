import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px


# Logo ganz oben zentriert
# st.image("assets/Kompetenzzentrum_Logo.png", width=500)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/Logo_01.png", width=200)
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
    st.subheader("Beziehung zwischen Variablen visualisieren")
    
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



# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

with tabs[2]:
    st.header("🔍 K-Means Clustering – einfach erklärt")
    st.markdown("🕐 **Lesezeit: ca. 5 Minuten**")

    # Abschnitt: Lernarten
    st.subheader("📚 Supervised vs. Unsupervised Learning")
    st.markdown("""
    Beim **Supervised Learning** lernst du mit bekannten Antworten – z. B. ob ein Kunde gekauft hat.  
    ➕ Du hast Daten **mit Labels** und trainierst ein Modell, um Vorhersagen zu treffen.

    Beim **Unsupervised Learning** gibt es **keine vorgegebenen Antworten**.  
    ➕ Das Ziel ist, **Muster oder Gruppen** in den Daten selbstständig zu entdecken.
    """)

    # Abschnitt: K-Means
    st.subheader("🎯 Was ist K-Means Clustering?")
    st.markdown("""
    K-Means ist ein einfacher Algorithmus, der versucht, Daten in **K Gruppen** zu unterteilen – so,  
    dass Punkte innerhalb einer Gruppe **möglichst ähnlich** sind.

    🔢 **Beispiel**: Du hast viele Kunden, aber keine Infos über deren Verhalten.  
    K-Means hilft dir, sie automatisch in Gruppen einzuteilen, etwa:
    - Kunden, die viel kaufen
    - Kunden, die selten kommen
    - Gelegenheitskäufer

    **Wie funktioniert das?**
    1. Wähle K zufällige Gruppenzentren
    2. Ordne alle Punkte dem nächstgelegenen Zentrum zu
    3. Berechne neue Gruppenzentren
    4. Wiederhole, bis sich die Gruppen nicht mehr ändern

    🧠 Der Algorithmus findet also **Strukturen in unbeschrifteten Daten** – ohne dass du ihm sagen musst, wie viele Gruppen es „wirklich“ gibt.
    """)

    # Interaktives Beispiel
    st.subheader("🧪 Probiere es selbst aus")

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import pandas as pd
    import plotly.express as px

    k = st.slider("Wähle die Anzahl der Cluster (K)", min_value=1, max_value=6, value=3, key="kmeans_slider")

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    df = pd.DataFrame(X, columns=["x", "y"])
    df["Cluster"] = labels.astype(str)  # Cluster als Kategorie für Plotly

    fig = px.scatter(df, x="x", y="y", color="Cluster",
                     title="K-Means Clustering Ergebnis",
                     color_discrete_sequence=px.colors.qualitative.Tab10)
    
    st.plotly_chart(fig, use_container_width=True)

    # Vorteile & Grenzen
    st.subheader("✅ Vorteile & 🛠️ Grenzen von K-Means")

    st.markdown("""
    **Vorteile:**
    - Einfach zu verstehen
    - Schnell bei großen Datenmengen
    - Gut, wenn Gruppen „rund“ verteilt sind

    **Grenzen:**
    - Du musst K vorher angeben
    - Funktioniert schlechter bei unregelmäßigen Gruppen
    - Reagiert empfindlich auf Ausreißer
    """)

    # Fazit
    st.subheader("🧩 Fazit")
    st.markdown("""
    K-Means ist ein **leicht verständlicher Einstieg** ins Clustering und ideal,  
    um **verborgene Strukturen** in unbeschrifteten Daten sichtbar zu machen.

    📊 Besonders nützlich in Bereichen wie Kundensegmentierung, Produktempfehlung oder Marktforschung.
    """)

    st.success("🎉 Tipp: Bewege den Schieberegler oben, um zu sehen, wie sich die Gruppierung verändert.")
    
    st.divider()
    st.video("https://www.youtube.com/watch?v=4b5d3muPQmA&t=33s")








with tabs[3]:
    st.header("🔍 K-Means Clustering – einfach erklärt")
    st.markdown("🕐 **Lesezeit: ca. 5 Minuten**")

    # Abschnitt: Lernarten
    st.subheader("📚 Supervised vs. Unsupervised Learning")
    st.markdown("""
    Beim **Supervised Learning** lernst du mit bekannten Antworten – z. B. ob ein Kunde gekauft hat.  
    ➕ Du hast Daten **mit Labels** und trainierst ein Modell, um Vorhersagen zu treffen.

    Beim **Unsupervised Learning** gibt es **keine vorgegebenen Antworten**.  
    ➕ Das Ziel ist, **Muster oder Gruppen** in den Daten selbstständig zu entdecken.
    """)

    # Abschnitt: K-Means
    st.subheader("🎯 Was ist K-Means Clustering?")
    st.markdown("""
    K-Means ist ein einfacher Algorithmus, der versucht, Daten in **K Gruppen** zu unterteilen – so,  
    dass Punkte innerhalb einer Gruppe **möglichst ähnlich** sind.

    🔢 **Beispiel**: Du hast viele Kunden, aber keine Infos über deren Verhalten.  
    K-Means hilft dir, sie automatisch in Gruppen einzuteilen, etwa:
    - Kunden, die viel kaufen
    - Kunden, die selten kommen
    - Gelegenheitskäufer

    **Wie funktioniert das?**
    1. Wähle K zufällige Gruppenzentren
    2. Ordne alle Punkte dem nächstgelegenen Zentrum zu
    3. Berechne neue Gruppenzentren
    4. Wiederhole, bis sich die Gruppen nicht mehr ändern

    🧠 Der Algorithmus findet also **Strukturen in unbeschrifteten Daten** – ohne dass du ihm sagen musst, wie viele Gruppen es „wirklich“ gibt.
    """)

    # Interaktives Beispiel
    st.subheader("🧪 Probiere es selbst aus")

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    k = st.slider("Wähle die Anzahl der Cluster (K)", min_value=1, max_value=6, value=3, key="kmeans_slider")

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=40)
    ax.set_title("K-Means Clustering Ergebnis")
    st.pyplot(fig)

    # Vorteile & Grenzen
    st.subheader("✅ Vorteile & 🛠️ Grenzen von K-Means")

    st.markdown("""
    **Vorteile:**
    - Einfach zu verstehen
    - Schnell bei großen Datenmengen
    - Gut, wenn Gruppen „rund“ verteilt sind

    **Grenzen:**
    - Du musst K vorher angeben
    - Funktioniert schlechter bei unregelmäßigen Gruppen
    - Reagiert empfindlich auf Ausreißer
    """)

    # Fazit
    st.subheader("🧩 Fazit")
    st.markdown("""
    K-Means ist ein **leicht verständlicher Einstieg** ins Clustering und ideal,  
    um **verborgene Strukturen** in unbeschrifteten Daten sichtbar zu machen.

    📊 Besonders nützlich in Bereichen wie Kundensegmentierung, Produktempfehlung oder Marktforschung.
    """)

    st.success("🎉 Tipp: Bewege den Schieberegler oben, um zu sehen, wie sich die Gruppierung verändert.")
    
    st.divider()
    st.video("https://www.youtube.com/watch?v=4b5d3muPQmA&t=33s")






