import streamlit as st

#st.title('Customer Data Analytics')

#st.write('Hello world!')



st.title("Drag and Drop CSV Uploader")

# File uploader (drag and drop enabled)
uploaded_file = st.file_uploader("Upload your CSV file here", type="csv")

# If a file is uploaded
if uploaded_file is not None:
    # Read the CSV as DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")
        st.write("### Preview of DataFrame:")
        st.dataframe(df)  # Display the DataFrame
    except Exception as e:
        st.error(f"Error reading the file: {e}")
