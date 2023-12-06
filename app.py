import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATASET_URL = "https://raw.githubusercontent.com/pramit910/Receipt_Count_Prediction/main/data_daily.csv"

st.title('Receipt Count Trend')

def load_data():
    data = pd.read_csv(DATASET_URL, header=0, names=['Date', 'Receipt_Count'])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# View Raw Data
st.subheader("Raw Data")
st.write(data.head(50))

# Small viz
st.subheader("Receipt Count trend in year 2021")
st.line_chart(data=data, y="Receipt_Count")