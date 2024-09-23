import streamlit as st
import pandas as pd
import numpy as np

# Function to load data from uploaded CSV file
def load_data(file):
    try:
        return pd.read_csv(file, on_bad_lines='skip')  # Skips lines with parsing issues
    except pd.errors.ParserError as e:
        st.error(f"Error loading file: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function for classifying news articles
def classify_news(df, text_column, case_sensitive, keywords):
    if not case_sensitive:
        keywords = [kw.lower() for kw in keywords]
    
    conditions = []
    choices = ['True', 'Mostly True', 'Half True', 'Mostly False', 'False']
    
    # Generate conditions based on keywords
    for i, keyword in enumerate(keywords):
        if not case_sensitive:
            conditions.append(df[text_column].str.contains(keyword.lower(), na=False))
        else:
            conditions.append(df[text_column].str.contains(keyword, na=False))

    # Assign classification based on the first match
    df['Classification'] = np.select(conditions, choices, default='Unknown')
    return df

# Streamlit app layout
st.title("Fake and True News Detection Model")
st.write("Upload your CSV file containing news articles to analyze and classify them. The file should have a column with text articles.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file)

    if df is not None:
        # Display the DataFrame
        st.write("Data Preview:")
        st.dataframe(df)

        # Allow the user to select which column to classify
        text_column = st.selectbox("Select the column containing news articles:", df.columns)

        # User-defined parameters
        case_sensitive = st.checkbox("Case Sensitive Classification", value=False)
        custom_keywords = st.text_area("Enter custom keywords (comma-separated):", 
                                        "true, mostly true, half true, mostly false, false")
        
        # Split keywords into a list and strip any whitespace
        keywords = [kw.strip() for kw in custom_keywords.split(",")]

        # Custom labels for classification
        custom_labels = st.text_input("Enter custom labels (comma-separated):", 
                                       "True, Mostly True, Half True, Mostly False, False")
        labels = [label.strip() for label in custom_labels.split(",")]

        # Check if the selected column has valid string data
        if df[text_column].dtype != 'object' or df[text_column].isnull().all():
            st.error("The selected column must contain text data.")
        else:
            # Classify news articles
            classified_df = classify_news(df, text_column, case_sensitive, keywords)

            # Update the classification column with custom labels
            classified_df['Classification'] = classified_df['Classification'].replace(
                ['True', 'Mostly True', 'Half True', 'Mostly False', 'False'], labels
            )

            # Display options
            show_raw_data = st.checkbox("Show Raw Data Alongside Classification Results", value=True)

            # Display the classification results
            st.write("Classification Results:")
            if show_raw_data:
                st.dataframe(classified_df[[text_column, 'Classification']])
            else:
                st.dataframe(classified_df['Classification'])

            # Display classification summary
            st.write("Classification Summary:")
            st.bar_chart(classified_df['Classification'].value_counts())

            # Provide an option to download the classified results
            output_format = st.selectbox("Select output format:", ["CSV", "Excel"])
            if output_format == "CSV":
                csv = classified_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Classified Results as CSV", csv, "classified_results.csv", "text/csv")
            elif output_format == "Excel":
                excel = classified_df.to_excel(index=False, engine='xlsxwriter')
                st.download_button("Download Classified Results as Excel", excel, "classified_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
