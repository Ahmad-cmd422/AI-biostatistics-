# Rbiostatitics 📊

A Streamlit application for biostatistical data analysis.

## Features

- 📁 **File Upload**: Support for CSV and Excel (.xlsx) files
- 👀 **Data Preview**: View the first 5 rows of your dataset
- 📋 **Column Summary**: Detailed information about data types and missing values
- 🔍 **Data Analysis**: Descriptive statistics and data type distribution
- ⚠️ **Error Handling**: Robust handling of empty files and corrupted data

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Use the sidebar to navigate between pages:
   - **Home**: Overview and introduction
   - **Data Upload**: Upload and preview your data files
   - **Analysis**: View descriptive statistics and data insights

## Supported File Formats

- CSV (`.csv`)
- Excel (`.xlsx`)

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- openpyxl >= 3.1.0
