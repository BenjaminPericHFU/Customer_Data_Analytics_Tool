import streamlit as st

#st.title('Customer Data Analytics')

#st.write('Hello world!')



st.title("Drag and Drop CSV Uploader")

# File uploader – supports CSV and Excel files
uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xls", "xlsx"]
)

# If a file is uploaded
if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File successfully uploaded!")
        st.write("### Preview of DataFrame:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error reading the file: {e}")
