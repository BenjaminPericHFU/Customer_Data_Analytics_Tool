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

daten_eingeladen = False

# CSS für blaue Sidebar (#00B0DB)
st.markdown("""
    <style>
        /* Sidebar-Hintergrund auf Hauptfarbe Blau */
        [data-testid="stSidebar"] {
            background-color: #00B0DB;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        /* Weißer Text für guten Kontrast */
        [data-testid="stSidebar"] * {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("assets/Logo_Negativ.png", use_container_width=True)
    st.markdown("## Anleitung")
    st.markdown("""
    1. **Daten** hochladen oder laden  
    2. **Visualisierung** der Rohdaten  
    3. **Vorverarbeitung** der Daten  
    4. **ML-Tutorial** zur Einführung  
    5. **K-Means** zur Segmentierung  
    6. **Analyse** der Ergebnisse 
    """)
    st.divider()

    # Spacer, der den Footer nach unten drückt
    st.markdown("<div style='flex-grow:1'></div>", unsafe_allow_html=True)

    # Footer ganz unten
    st.markdown(
        """
        <div style="color: white; font-size: 0.9em;">
        ⚙️ Developed by <a href="https://smart-service-bw.de/" target="_blank" style="color: white; text-decoration: underline;">Kompetenzzentrum Smart Services</a>
        </div>
        """,
        unsafe_allow_html=True
    )
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/Logo_02_1.png", width=200)
with col2:
    st.markdown("<h1 style='margin-top: 5px;'>Customer Data Analytics Tool</h1>", unsafe_allow_html=True)

# Tabs hinzufuegen
tabs = st.tabs(["Daten", "Visualisierung", "Datenvorverarbeitung", "ML-Tutorial", "K-Means Clustering", "Auswertung & Analyse"], width = "stretch")




# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
import io
import csv

def detect_csv_separator(file):
    try:
        file.seek(0)
        sample = file.read(1024).decode('utf-8', errors='ignore')
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ';'  # Fallback

def load_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith('.csv'):
        sep = detect_csv_separator(uploaded_file)
        df = pd.read_csv(uploaded_file, sep=sep)
    elif filename.endswith('.txt'):
        # Versuch Tab als Standard-Trenner
        df = pd.read_csv(uploaded_file, sep='\t')
    else:
        raise ValueError("Dateiformat wird nicht unterstützt.")
    return df

with tabs[0]:
    st.subheader("Datensatz einlesen:")

    dataset_source = st.radio(
        "**Wähle Datenquelle:**",
        options=["Testdaten verwenden", "Eigenen Datensatz hochladen"],
        index=0,
        horizontal=True
    )

    df_work = None
    n_cols = 10

    if dataset_source == "Testdaten verwenden":
        st.info("Testdatensatz namens daten.csv wird verwendet.")
        try:
            # Hier kannst du auch detect_csv_separator nutzen, wenn gewünscht
            df = pd.read_csv("data/daten.csv", sep=';')
            df_work = df.copy()
            st.success("Datensatz wurde erfolgreich eingeladen.")
            daten_eingeladen = True
        except Exception as e:
            st.error(f"Fehler beim Laden des Testdatensatzes: {e}")
            daten_eingeladen = False

    elif dataset_source == "Eigenen Datensatz hochladen":
        uploaded_file = st.file_uploader("Datei hochladen (csv, xls, xlsx, txt)", type=["csv", "xls", "xlsx", "txt"])
        if uploaded_file is not None:
            try:
                df = load_file(uploaded_file)
                df_work = df.copy()
                st.success("Datensatz wurde erfolgreich eingeladen.")
                daten_eingeladen = True
            except Exception as e:
                st.error(f"Fehler beim Lesen der Datei: {e}")
                daten_eingeladen = False

    if df_work is not None:
        df_work = df_work.dropna()
        st.divider()
        st.subheader("Datensatz als Tabelle:")
        st.dataframe(df_work.iloc[:, :])
        st.divider()
        st.subheader("Statistik zu Datensatz:")
        st.dataframe(df_work.describe())

        # Spaltenklassifikation
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
    if daten_eingeladen == False:
        st.warning("Daten wurden noch nicht hochgeladen.")
    else:
        st.header("📊 Interaktive Visualisierung")
    
        # --- Verteilung einer Einzelvariablen ---
        st.subheader("Verteilung der Daten analysieren")
    
        dist_col = st.selectbox("Variable für Verteilung wählen:", column_classification["xy"], key="dist_col_tab1")
        group_col = st.selectbox("Farbliche Gruppierung (optional):", ["Keine"] + column_classification["hue"], key="group_tab1")
        color_arg_dist = None if group_col == "Keine" else group_col
    
        df_dist = df.copy()
        if color_arg_dist is not None and color_arg_dist in df_dist.columns:
            if pd.api.types.is_numeric_dtype(df_dist[color_arg_dist]):
                df_dist[color_arg_dist] = df_dist[color_arg_dist].astype(str)
    
        fig_dist = px.histogram(df_dist, x=dist_col, color=color_arg_dist)
        fig_dist.update_layout(barmode='overlay')  # oder 'group', je nach Präferenz
        fig_dist.update_traces(opacity=0.75)
    
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
    
        df_plot = df.copy()
    
        if color_arg is not None and color_arg in df_plot.columns:
            if pd.api.types.is_numeric_dtype(df_plot[color_arg]):
                df_plot[color_arg] = df_plot[color_arg].astype(str)
    
        if plot_type in ["Balkendiagramm", "Scatterplot"]:
            if pd.api.types.is_numeric_dtype(df_plot[x_col]):
                df_plot[x_col] = df_plot[x_col].astype(str)
    
        if plot_type == "Balkendiagramm":
            fig = px.bar(df_plot, x=x_col, y=y_col, color=color_arg)
        elif plot_type == "Scatterplot":
            fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_arg)
        elif plot_type == "Liniendiagramm":
            fig = px.line(df_plot, x=x_col, y=y_col, color=color_arg)
    
        st.plotly_chart(fig, use_container_width=True)
    
        

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
with tabs[2]:
    if daten_eingeladen == False:
        st.warning("Daten wurden noch nicht hochgeladen.")
    else:
        st.header("🔍 Autonome Ausreißer-Filterung mit Six Sigma Methode")
        
        st.markdown("""
            ### 🔢 **Was bedeuten die Sigma-Level?**
            
            Das **Sigma-Level** bestimmt, wie streng die Ausreißer-Erkennung ist, also wie stark sich Werte vom Durchschnitt unterscheiden müssen, um als Ausreißer erkannt zu werden:
            
            - **2 Sigma (±2 Standardabweichungen)**  
              ⚠️ **Strenger Filter** – Es werden schon viele Werte als Ausreißer erkannt, auch wenn sie nur etwas ungewöhnlich sind.  
              👉 Gut, wenn du möglichst viele Auffälligkeiten finden möchtest.
            
            - **3 Sigma (±3 Standardabweichungen)**  
              ✔️ **Üblicher Standard** – Es werden nur Werte entfernt, die wirklich deutlich anders sind als der Durchschnitt.  
              🔄 Häufig die beste Wahl, weil es eine gute Balance bietet.
            
            - **6 Sigma (±6 Standardabweichungen)**  
              🛡️ **Sehr lockerer Filter** – Nur sehr extreme Ausreißer werden erkannt, alles, was sich leicht unterscheidet, bleibt drin.  
              🧘‍♂️ Ideal, wenn du nur ganz große Ausreißer entfernen möchtest.
            """, unsafe_allow_html=True)
    
        
    
        # Mapping von Anzeige-Label zu numerischem Sigma-Level
        sigma_options = {
            "±2 σ": 2,
            "±3 σ": 3,
            "±4 σ": 4,
            "±5 σ": 5,
            "±6 σ": 6
        }
        
        # Radio-Buttons mit formatierten Labels
        selection = st.radio(
            label="**Wähle das Sigma-Level für die Ausreißer-Erkennung:**",
            options=list(sigma_options.keys()),
            index=1,  # entspricht ±3σ
            horizontal=True
        )
        
        # Zugriff auf den numerischen Wert
        sigma_level = sigma_options[selection]
        
        st.divider()
        
        st.markdown("### Spalten auswählen, bei denen die automatische Six Sigma Filterung angewandt wird:")
    
        selected_columns = []
    
        for col_name in column_classification["xy"]:
            c1, c2 = st.columns([3, 2])
    
            with c1:
                checked = st.checkbox(col_name, value=False, key=f"chk_{col_name}")
    
            with c2:
                if checked:
                    mean = df_work[col_name].mean()
                    std = df_work[col_name].std()
                    lower_bound = mean - sigma_level * std
                    upper_bound = mean + sigma_level * std
    
                    mask_outliers = (df_work[col_name] < lower_bound) | (df_work[col_name] > upper_bound)
                    count_outliers = mask_outliers.sum()
                else:
                    count_outliers = 0
    
                st.markdown(
                    f"<div style='text-align: left;'>"
                    f"{col_name} — Ausreißer: <span style='font-weight: bold;'>{count_outliers}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    
            if checked:
                selected_columns.append(col_name)
    
        # Falls keine Spalten ausgewählt wurden, setze df_filtered = df_work.copy()
        if len(selected_columns) == 0:
            st.info("Visualisierungen sind zu sehen, sobald eine Spalte ausgewählt wird.")
            df_filtered = df_work.copy()
        else:
            df_filtered = df_work.copy()
            outlier_indices = set()
    
            for col in selected_columns:
                mean = df_filtered[col].mean()
                std = df_filtered[col].std()
                lower_bound = mean - sigma_level * std
                upper_bound = mean + sigma_level * std
    
                mask_outliers = (df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)
                outlier_idx_col = df_filtered[mask_outliers].index
    
                if len(outlier_idx_col) > 0:
                    outlier_indices.update(outlier_idx_col)
    
            st.write(f"**Gesamtzahl eindeutiger Ausreißer (über alle ausgewählten Spalten): {len(outlier_indices)}**")
            df_filtered = df_filtered.drop(index=outlier_indices)
            st.write(f"Datensatz nach Entfernung der Ausreißer enthält {len(df_filtered)} Zeilen statt {len(df_work)}")
            
            st.divider()
    
            # Visualisierung der ausgewählten Spalten mit Ausreißern
            st.subheader("Scatterplots mit Ausreißer-Markierung")
    
            for col in selected_columns:
                mean = df_work[col].mean()
                std = df_work[col].std()
                lower_bound = mean - sigma_level * std
                upper_bound = mean + sigma_level * std
    
                mask_outliers = (df_work[col] < lower_bound) | (df_work[col] > upper_bound)
    
                if mask_outliers.any():
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
    
        st.markdown("---")
        st.markdown("### 🧹 **Irrelevante Spalten vor der Analyse ausschließen**")
        
        columns_to_exclude = st.multiselect(
            label="Wähle Sie Spalten aus, die **irrelevant** für die Ausreißer-Erkennung sind:",
            options=df_filtered.columns.tolist(),
            default=[],
            help="Diese Spalten werden in der Analyse ignoriert – z. B. IDs, konstante Werte oder irrelevante Merkmale."
        )
        
        df_filtered = df_filtered.drop(columns=columns_to_exclude)
        
        st.markdown("### 📊 Vorschau des bereinigten Datensatzes")
        st.dataframe(df_filtered.head())
    

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
with tabs[3]:
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
with tabs[4]:
    if daten_eingeladen == False:
        st.warning("Daten wurden noch nicht hochgeladen.")
    else:
        st.header("📊 K-Means Clustering & Interaktive Visualisierung")
    
        if df_filtered is None or df_filtered.empty:
            st.warning("Bitte lade und filtere zuerst die Daten im vorherigen Tab.")
        else:
            df_cluster = df_filtered.copy()
    
            # Anzahl Cluster auswählen
            k = st.slider(
                "Wähle die Anzahl der Cluster (K)",
                min_value=1,
                max_value=6,
                value=3,
                key="kmeans_slider_tab4"
            )
    
            # Kategorische Spalten label-encoden
            for col in df_cluster.columns:
                if df_cluster[col].dtype == "object" or df_cluster[col].dtype.name == "category":
                    df_cluster[col] = df_cluster[col].astype("category").cat.codes
    
            # Numerische Spalten für Clustering
            numeric_df = df_cluster.select_dtypes(include=["number"])
    
            # KMeans Clustering durchführen
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=k, n_init="auto", random_state=42)
            df_cluster["cluster"] = model.fit_predict(numeric_df)
    
            st.success(f"✅ Clustering mit **{k} Clustern** durchgeführt.")
    
            # Auswahl der Achsen für Scatterplot
            st.markdown("### 📊 Interaktive Cluster-Visualisierung (Scatterplot)")
    
            x_col = st.selectbox(
                "X-Achse wählen:",
                options=numeric_df.columns.tolist(),
                index=0,
                key="scatter_x_col"
            )
            y_col = st.selectbox(
                "Y-Achse wählen:",
                options=numeric_df.columns.tolist(),
                index=1 if len(numeric_df.columns) > 1 else 0,
                key="scatter_y_col"
            )
    
            import plotly.express as px
    
            fig = px.scatter(
                df_cluster,
                x=x_col,
                y=y_col,
                color=df_cluster["cluster"].astype(str),
                labels={"color": "Cluster"},
                title=f"Cluster-Visualisierung: {x_col} vs. {y_col}",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
            # Datensatz mit Cluster-Zuweisung anzeigen, Cluster als erste Spalte
            df_cluster = df_cluster.reset_index(drop=True)
            cols_order = ["cluster"] + [col for col in df_cluster.columns if col != "cluster"]
            st.markdown("### 🔍 Datensatz mit Cluster-Zuweisung:")
            st.dataframe(df_cluster[cols_order])
            st.markdown(f"📋 **Datensatz enthält {df_cluster.shape[0]} Zeilen und {df_cluster.shape[1]} Spalten.**")


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
with tabs[5]:
    if daten_eingeladen == False:
        st.warning("Daten wurden noch nicht hochgeladen.")
    else:
        st.header("📊 Boxplot-Visualisierung der Cluster")
    
        if df_cluster is None or "cluster" not in df_cluster.columns:
            st.warning("Bitte lade zuerst einen Datensatz mit der Spalte 'cluster' hoch oder führe die Clusteranalyse aus.")
        else:
            st.markdown("### Spalten zur Visualisierung auswählen:")
            selected_columns = st.multiselect(
                label="Wähle eine oder mehrere Spalten (XY-Werte) für Boxplots:",
                options=column_classification["xy"],
                default=[]
            )
    
            if len(selected_columns) == 0:
                st.info("Bitte wähle mindestens eine Spalte aus, um Boxplots zu sehen.")
            else:
                for col in selected_columns:
                    st.subheader(f"Boxplot für {col}")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(
                        data=df_cluster,
                        x="cluster",
                        y=col,
                        ax=ax,
                        palette="Set2"  # Farbpalette für Cluster-Gruppen
                    )
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel(col)
                    st.pyplot(fig)
    
            st.divider()
    
            # Mittelwerte berechnen (nur numerische Spalten)
            group_means_all = df_cluster.groupby("cluster").mean(numeric_only=True).round(3)
    
            # Fehlende Spalten mit '-' auffüllen (für nicht numerische Spalten)
            for col in df_cluster.columns:
                if col not in group_means_all.columns and col != "cluster":
                    group_means_all[col] = "-"
    
            # Spalten sortieren wie im Original DataFrame (ohne 'cluster')
            group_means_all = group_means_all.reindex(columns=[col for col in df_cluster.columns if col != "cluster"])
    
            # Tabelle transponieren, sodass Cluster die Spalten sind und Features die Zeilen
            group_means_all_T = group_means_all.transpose()
            
    
            st.markdown("### Mittelwerte aller Spalten je Cluster")
            st.dataframe(group_means_all_T)
            
            st.divider()
            
            st.markdown("### Datensatz mit Cluster-Gruppierungen downloaden")
            
            csv_data = df_cluster.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Cluster-Daten als CSV herunterladen",
                data=csv_data,
                file_name="cluster_data.csv",
                mime="text/csv"
            )
