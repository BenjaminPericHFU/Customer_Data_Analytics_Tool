import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px


# Logo ganz oben zentriert
# st.image("assets/Kompetenzzentrum_Logo.png", width=500)
st.set_page_config(page_title="Meine App", layout="centered")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/Logo_01.png", width=200)
with col2:
    st.markdown("<h1 style='margin-top: 5px;'>Customer Data Analytics Tool</h1>", unsafe_allow_html=True)

# Tabs hinzufuegen
tabs = st.tabs(["Daten", "Visualisierung", "ML-Tutorial", "Datenvorverarbeitung", "Anwendung & Auswertung"], width = "stretch")




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
    Beim **Supervised Learning** werden Modelle mit bekannten Klassen und Zielgrößen trainiert – z.b. untergliedert man Kundendaten in Kundengruppen mit einer hohen und niedrigen Kaufhäufigkeit.  
    ➕ Es liegen also Daten **mit Labels** vor, die als Grundlage für Vorhersagen dienen.

    Beim **Unsupervised Learning** hingegen gibt es **keine vorgegebenen Klassen**.  
    ➕ Ziel ist es, neue **Muster oder Strukturen** in den Daten selbstständig zu erkennen – etwa bei einem Datensatz mit Kundendaten, bei dem nicht bekannt ist, ob und wie viele Käufe stattgefunden haben.
    """)

    # Abschnitt: K-Means
    st.subheader("🎯 Was ist K-Means Clustering?")
    st.markdown("""
    K-Means ist ein einfacher Algorithmus, der Daten in eine vorher festgelegte Anzahl von **K Gruppen (Clustern)** unterteilt – und zwar so, dass sich die Einträge innerhalb einer Gruppe **möglichst ähnlich** sind.

    🔢 **Beispiel**: Ein Unternehmen verfügt über viele Kundendaten, aber keine Informationen über das Kaufverhalten.  
    K-Means kann helfen, automatisch Gruppierungen zu identifizieren, etwa:
    - Kunden, die viel kaufen  
    - Kunden, die selten kommen  
    - Gelegenheitskäufer  

    💡 **Wichtig**: Welche Gruppierungen tatsächlich entstehen, ist **zuvor nicht bekannt**.  
    Der Algorithmus nutzt ausschließlich **mathematische Ähnlichkeiten** zwischen Datenpunkten – die daraus resultierenden Gruppen können sehr unterschiedlich ausfallen.  
    Es könnten durchaus die oben genannten Kundentypen sein, **aber auch ganz andere Strukturen**: z. B. eine Trennung nach Alter, Körpergröße oder Wohnregion.  
    **Erst durch eine anschließende Analyse** lässt sich einschätzen, was die einzelnen Gruppen inhaltlich verbindet.

    **Wie funktioniert der Algorithmus?**
    1. Auswahl von K Gruppierungen (Clustern)  
    2. Mathematische Zuordnung jedes Punkts zum nächstgelegenen Cluster-Zentrum  
    3. Neuberechnung der Gruppenzentren  
    4. Wiederholung, bis sich die Gruppenzugehörigkeiten nicht mehr ändern  

    🧠 K-Means erkennt dementsprechend **Strukturen in unbeschrifteten Daten**, ohne dass im Vorfeld definiert werden muss, welche konkreten Klassen vorliegen. 
    """)

    # Interaktives Beispiel
    st.subheader("🧪 Probieren Sie es selbst aus")

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np

    k = st.slider("Wählen Sie die Anzahl der Cluster (K)", min_value=1, max_value=6, value=3, key="kmeans_slider")

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_colors = ['blue', 'green', 'grey', 'red', 'orange', 'purple']
    colors = np.array(cluster_colors * (k // len(cluster_colors) + 1))[:k]

    fig, ax = plt.subplots()
    for i in range(k):
        points = X[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=40, color=colors[i], label=f"Cluster {i+1}")

    ax.set_title("K-Means Clustering Ergebnis")
    ax.legend()
    st.pyplot(fig)

    # Vorteile & Grenzen
    st.subheader("✅ Vorteile & 🛠️ Grenzen von K-Means")

    st.markdown("""
    **Vorteile:**
    - Intuitiv und einfach zu implementieren  
    - Schnell und effizient auch bei großen Datensätzen  
    - Funktioniert gut bei klar voneinander trennbaren, runden Gruppen  

    **Grenzen:**
    - Die Anzahl der Cluster (K) muss im Voraus gewählt werden  
    - Schwächen bei unregelmäßig geformten oder unterschiedlich großen Gruppen  
    - Anfällig gegenüber Ausreißern  
    """)

    # Fazit
    st.subheader("🧩 Fazit")
    st.markdown("""
    K-Means ist ein **anschaulicher und leistungsfähiger Einstieg** in das unüberwachte Lernen – besonders dann,  
    wenn es darum geht, **verborgene Strukturen** in umfangreichen, unbeschrifteten Datensätzen sichtbar zu machen.

    📊 Anwendungsgebiete sind unter anderem Kundensegmentierung, Produktempfehlungen oder explorative Datenanalysen in der Marktforschung.
    """)

    st.success("🎉 Tipp: Bewegen Sie den Schieberegler oben, um zu sehen, wie sich die Gruppierung verändert.")

    st.divider()
    st.video("https://www.youtube.com/watch?v=4b5d3muPQmA&t=33s")




# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

with tabs[3]:
    st.header("🔍 Ausreißer-Erkennung mit Six Sigma Kontrolle")

    if df_work is None:
        st.warning("Bitte lade zuerst einen Datensatz im Tab 'Daten' hoch.")
    else:
        sigma_level = st.radio("Wähle das Sigma-Level für die Ausreißer-Erkennung:", options=[2, 3, 6], index=2, horizontal=True)

        st.markdown("### Spalten auswählen, bei denen die automatische Six Sigma Filterung angewandt wird:")

        selected_columns = []
        outlier_counts = {}

        # Ausreißeranzahl pro Spalte berechnen
        for col in column_classification["xy"]:
            mean = df_work[col].mean()
            std = df_work[col].std()
            lower_bound = mean - sigma_level * std
            upper_bound = mean + sigma_level * std

            mask_outliers = (df_work[col] < lower_bound) | (df_work[col] > upper_bound)
            outlier_counts[col] = mask_outliers.sum()

        # Für jede Spalte eine einzelne Zeile mit Checkbox links und Text rechts
        for col_name in column_classification["xy"]:
            c1, c2 = st.columns([1, 3])
            with c1:
                checked = st.checkbox(col_name, value=False, key=f"chk_{col_name}")  # hier value=False
            with c2:
                st.markdown(
                    f"<div style='text-align: right;'>"
                    f"Ausreißer in Spalte {col_name}: {outlier_counts[col_name]}</div>",
                    unsafe_allow_html=True,
                )
            if checked:
                selected_columns.append(col_name)

        if len(selected_columns) == 0:
            st.info("Bitte mindestens eine Spalte auswählen.")
        else:
            df_filtered = df_work.copy()
            outlier_indices = set()
            cols_with_outliers = []

            for col in selected_columns:
                mean = df_filtered[col].mean()
                std = df_filtered[col].std()
                lower_bound = mean - sigma_level * std
                upper_bound = mean + sigma_level * std

                mask_outliers = (df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)
                outlier_idx_col = df_filtered[mask_outliers].index

                if len(outlier_idx_col) > 0:
                    outlier_indices.update(outlier_idx_col)
                    cols_with_outliers.append(col)

            st.write(f"**Gesamtzahl eindeutiger Ausreißer (über alle ausgewählten Spalten): {len(outlier_indices)}**")
            df_filtered = df_filtered.drop(index=outlier_indices)
            st.write(f"Datensatz nach Entfernung der Ausreißer enthält {len(df_filtered)} Zeilen statt {len(df_work)}")

            # Visualisierung der Spalten mit Ausreißern
            st.subheader("Scatterplots mit Ausreißer-Markierung (y=1)")

            for col in cols_with_outliers:
                mean = df_work[col].mean()
                std = df_work[col].std()
                lower_bound = mean - sigma_level * std
                upper_bound = mean + sigma_level * std

                mask_outliers = (df_work[col] < lower_bound) | (df_work[col] > upper_bound)

                fig, ax = plt.subplots(figsize=(8, 2))
                ax.scatter(df_work.loc[~mask_outliers, col], [1]*sum(~mask_outliers),
                           color="blue", label="Normal", alpha=0.6)
                ax.scatter(df_work.loc[mask_outliers, col], [1]*sum(mask_outliers),
                           color="red", label="Ausreißer", alpha=0.8)

                ax.set_yticks([1])
                ax.set_yticklabels([""])
                ax.set_xlabel(col)
                ax.set_title(f"Ausreißererkennung für '{col}' ({sigma_level}σ)")

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                st.pyplot(fig)





