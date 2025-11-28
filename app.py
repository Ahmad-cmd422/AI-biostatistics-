import streamlit as st
import pandas as pd
import io
import pingouin as pg
import scipy.stats as stats
import numpy as np
import google.generativeai as genai
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Configure Gemini API
# In a real app, use st.secrets
GENAI_API_KEY = "AIzaSyCbcJlXb03XpGAEw82icxDU2-mFAAjG9go"
genai.configure(api_key=GENAI_API_KEY)

def get_ai_explanation(test_name, result_df, p_value=None):
    """Generates and streams AI explanation for statistical results."""
    st.markdown("### 🤖 AI Explanation")
    
    prompt = f"""
    You are a biostatistics expert. Explain this {test_name} result to a medical student in plain English.
    
    Key Data:
    - Test: {test_name}
    - P-value: {p_value if p_value is not None else 'N/A'}
    - Full Result: 
    {result_df.to_string()}
    
    Please provide a concise explanation covering:
    1. Is the result statistically significant?
    2. What does this mean in practical terms?
    3. If applicable, what does the confidence interval imply?
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, stream=True)
        
        def stream_generator():
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        st.write_stream(stream_generator())
        
    except Exception as e:
        st.error(f"AI Error: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Rbiostatitics",
    page_icon="📊",
    layout="wide"
)

# Auto-load dataset for stress testing
local_path = "/Users/ahmadtarek/Downloads/healthcare_dataset.csv"
if 'df' not in st.session_state and os.path.exists(local_path):
    try:
        df = pd.read_csv(local_path)
        # Clean data immediately
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype(str).str.title().str.strip()
            except:
                pass
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'admission' in col.lower() or 'discharge' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        st.session_state['df'] = df
        st.session_state['filename'] = "healthcare_dataset.csv"
        # Feature Engineering
        adm_col = next((c for c in df.columns if 'admission' in c.lower() and 'date' in c.lower()), None)
        dis_col = next((c for c in df.columns if 'discharge' in c.lower() and 'date' in c.lower()), None)
        if adm_col and dis_col:
            df['Length of Stay'] = (df[dis_col] - df[adm_col]).dt.days
            
    except Exception as e:
        st.error(f"Auto-load failed: {e}")

# Sidebar for navigation
st.sidebar.title("📊 Rbiostatitics")
st.sidebar.markdown("---")
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Upload", "Analysis", "Statistical Tests", "Visualization", "Machine Learning", "AI Chatbot"]
)
st.sidebar.markdown("---")
st.sidebar.info("Upload your data files (.csv or .xlsx) to get started!")

# Main content
if page == "Home":
    st.title("Welcome to Rbiostatitics 📊")
    st.markdown("""
    ### A Streamlit Application for Biostatistical Analysis
    
    This application allows you to:
    - 📁 Upload CSV and Excel files
    - 👀 Preview your data
    - 📋 View data summaries
    - 🔍 Analyze data quality
    
    **Get started by navigating to the 'Data Upload' page using the sidebar!**
    """)

elif page == "Data Upload":
    st.title("📁 Data Upload")
    st.markdown("Upload your dataset to begin analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload a .csv or .xlsx file containing your data"
    )
    
    # Auto-load for testing/demo
    local_path = "/Users/ahmadtarek/Downloads/healthcare_dataset.csv"
    if os.path.exists(local_path):
        if st.button("🚀 Load Local Dataset (healthcare_dataset.csv)"):
            try:
                df = pd.read_csv(local_path)
                st.session_state['df'] = df
                st.session_state['filename'] = "healthcare_dataset.csv"
                st.success("Loaded local dataset!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load local file: {e}")
    
    if uploaded_file is not None:
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            # Check if dataframe is empty
            if df.empty:
                st.error("⚠️ The uploaded file is empty. Please upload a file with data.")
            else:
                # Data Cleaning & Preprocessing
                # 1. Convert object columns to string and title case
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = df[col].astype(str).str.title().str.strip()
                    except:
                        pass
                
                # 2. Date Parsing
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower() or 'admission' in col.lower() or 'discharge' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass

                # 3. Feature Engineering: Length of Stay
                # Check for common admission/discharge column names
                adm_col = next((c for c in df.columns if 'admission' in c.lower() and 'date' in c.lower()), None)
                dis_col = next((c for c in df.columns if 'discharge' in c.lower() and 'date' in c.lower()), None)
                
                if adm_col and dis_col:
                    df['Length of Stay'] = (df[dis_col] - df[adm_col]).dt.days
                
                # Store dataframe in session state
                st.session_state['df'] = df
                st.session_state['filename'] = uploaded_file.name
                
                st.success(f"✅ Successfully loaded and cleaned: **{uploaded_file.name}**")
                
                # 2. Date Parsing
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower() or 'admission' in col.lower() or 'discharge' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass

                # 3. Feature Engineering: Length of Stay
                # Check for common admission/discharge column names
                adm_col = next((c for c in df.columns if 'admission' in c.lower() and 'date' in c.lower()), None)
                dis_col = next((c for c in df.columns if 'discharge' in c.lower() and 'date' in c.lower()), None)
                
                if adm_col and dis_col:
                    df['Length of Stay'] = (df[dis_col] - df[adm_col]).dt.days
                
                # Store dataframe in session state
                st.session_state['df'] = df
                st.session_state['filename'] = uploaded_file.name
                
                st.success(f"✅ Successfully loaded and cleaned: **{uploaded_file.name}**")
                
                # Display basic information
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Display first 5 rows
                st.markdown("---")
                st.subheader("📋 Data Preview (First 5 Rows)")
                st.dataframe(df.head(5), use_container_width=True)
                
                # Display column summary
                st.markdown("---")
                st.subheader("📊 Column Summary")
                
                # Create summary dataframe
                summary_data = []
                for col in df.columns:
                    summary_data.append({
                        'Column Name': col,
                        'Data Type': str(df[col].dtype),
                        'Missing Values': df[col].isna().sum(),
                        'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.2f}%",
                        'Unique Values': df[col].nunique()
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Highlight columns with missing values
                if summary_df['Missing Values'].sum() > 0:
                    st.warning(f"⚠️ Found {summary_df['Missing Values'].sum()} missing values across {(summary_df['Missing Values'] > 0).sum()} columns")
                else:
                    st.success("✅ No missing values detected!")
                
        except pd.errors.EmptyDataError:
            st.error("⚠️ The uploaded file is empty. Please upload a file with data.")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your file is properly formatted and not corrupted.")
    else:
        st.info("👆 Please upload a file to get started")

elif page == "Analysis":
    st.title("🔍 Data Analysis")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success(f"Working with: **{st.session_state['filename']}**")
        
        st.markdown("---")
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔢 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
        
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Statistical Tests":
    st.title("🧪 Statistical Tests")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success(f"Working with: **{st.session_state['filename']}**")
        
        st.markdown("---")
        
        # Test Selection
        test_type = st.selectbox(
            "Select Statistical Test",
            ["T-Test (Independent)", "T-Test (Paired)", "Chi-Square Test", "ANOVA", "Pearson Correlation"]
        )
        
        st.markdown("---")
        
        if test_type == "T-Test (Independent)":
            st.subheader("Independent T-Test")
            st.info("Compare means of two independent groups.")
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Select Grouping Column (Categorical)", df.columns)
            with col2:
                value_col = st.selectbox("Select Value Column (Numeric)", df.select_dtypes(include=np.number).columns)
                
            if st.button("Run T-Test"):
                try:
                    # Check if group column has exactly 2 unique values
                    unique_groups = df[group_col].unique()
                    if len(unique_groups) != 2:
                        st.error(f"Grouping column must have exactly 2 unique values. Found {len(unique_groups)}: {unique_groups}")
                    else:
                        group1 = df[df[group_col] == unique_groups[0]][value_col]
                        group2 = df[df[group_col] == unique_groups[1]][value_col]
                        
                        res = pg.ttest(group1, group2, correction=True)
                        st.dataframe(res, use_container_width=True)
                        
                        p_val = res['p-val'].values[0]
                        if p_val < 0.05:
                            st.success(f"Significant difference found (p < 0.05). P-value: {p_val:.4f}")
                        else:
                            st.info(f"No significant difference found (p >= 0.05). P-value: {p_val:.4f}")
                            
                        # AI Explanation
                        get_ai_explanation("Independent T-Test", res, p_val)
                        
                except Exception as e:
                    st.error(f"Error running test: {str(e)}")

        elif test_type == "T-Test (Paired)":
            st.subheader("Paired T-Test")
            st.info("Compare means of two related groups (e.g., before vs after).")
            
            col1, col2 = st.columns(2)
            numeric_cols = df.select_dtypes(include=np.number).columns
            with col1:
                col_a = st.selectbox("Select Column A (Numeric)", numeric_cols, key='pair_a')
            with col2:
                col_b = st.selectbox("Select Column B (Numeric)", numeric_cols, key='pair_b')
                
            if st.button("Run Paired T-Test"):
                if col_a == col_b:
                    st.error("Please select two different columns.")
                else:
                    try:
                        res = pg.ttest(df[col_a], df[col_b], paired=True)
                        st.dataframe(res, use_container_width=True)
                        
                        p_val = res['p-val'].values[0]
                        if p_val < 0.05:
                            st.success(f"Significant difference found (p < 0.05). P-value: {p_val:.4f}")
                        else:
                            st.info(f"No significant difference found (p >= 0.05). P-value: {p_val:.4f}")
                            
                        # AI Explanation
                        get_ai_explanation("Paired T-Test", res, p_val)
                        
                    except Exception as e:
                        st.error(f"Error running test: {str(e)}")

        elif test_type == "Chi-Square Test":
            st.subheader("Chi-Square Test of Independence")
            st.info("Test for relationship between two categorical variables.")
            
            col1, col2 = st.columns(2)
            with col1:
                cat_col1 = st.selectbox("Select Row Variable", df.columns, key='chi_row')
            with col2:
                cat_col2 = st.selectbox("Select Column Variable", df.columns, key='chi_col')
                
            if st.button("Run Chi-Square Test"):
                try:
                    expected, observed, stats_res = pg.chi2_independence(df, x=cat_col1, y=cat_col2)
                    st.write("### Test Statistics")
                    st.dataframe(stats_res, use_container_width=True)
                    
                    st.write("### Observed Frequencies")
                    st.dataframe(observed, use_container_width=True)
                    
                    # AI Explanation
                    # For Chi-Square, p-value is in stats_res
                    p_val = stats_res.loc[stats_res['test'] == 'pearson', 'p'].values[0]
                    get_ai_explanation("Chi-Square Test", stats_res, p_val)
                    
                except Exception as e:
                    st.error(f"Error running test: {str(e)}")

        elif test_type == "ANOVA":
            st.subheader("One-way ANOVA")
            st.info("Compare means across three or more groups.")
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Select Grouping Column (Categorical)", df.columns, key='anova_group')
            with col2:
                value_col = st.selectbox("Select Dependent Variable (Numeric)", df.select_dtypes(include=np.number).columns, key='anova_val')
                
            if st.button("Run ANOVA"):
                try:
                    res = pg.anova(data=df, dv=value_col, between=group_col)
                    st.dataframe(res, use_container_width=True)
                    
                    p_val = res['p-unc'].values[0]
                    if p_val < 0.05:
                            st.success(f"Significant difference found (p < 0.05). P-value: {p_val:.4f}")
                    else:
                        st.info(f"No significant difference found (p >= 0.05). P-value: {p_val:.4f}")
                        
                    # AI Explanation
                    get_ai_explanation("One-way ANOVA", res, p_val)
                    
                except Exception as e:
                    st.error(f"Error running test: {str(e)}")

        elif test_type == "Pearson Correlation":
            st.subheader("Pearson Correlation")
            st.info("Measure linear relationship between two continuous variables.")
            
            col1, col2 = st.columns(2)
            numeric_cols = df.select_dtypes(include=np.number).columns
            with col1:
                col_x = st.selectbox("Select X Variable", numeric_cols, key='corr_x')
            with col2:
                col_y = st.selectbox("Select Y Variable", numeric_cols, key='corr_y')
                
            if st.button("Run Correlation"):
                try:
                    res = pg.corr(df[col_x], df[col_y])
                    st.dataframe(res, use_container_width=True)
                    
                    r_val = res['r'].values[0]
                    p_val = res['p-val'].values[0]
                    
                    st.metric("Correlation Coefficient (r)", f"{r_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success(f"Significant correlation (p < 0.05). P-value: {p_val:.4f}")
                    else:
                        st.info(f"No significant correlation (p >= 0.05). P-value: {p_val:.4f}")
                        
                    # AI Explanation
                    get_ai_explanation("Pearson Correlation", res, p_val)
                    
                except Exception as e:
                    st.error(f"Error running test: {str(e)}")

    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Visualization":
    st.title("📊 Data Visualization")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success(f"Working with: **{st.session_state['filename']}**")
        
        st.markdown("---")
        
        # Chart Selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Boxplot", "Scatter Plot", "Heatmap"]
        )
        
        st.markdown("---")
        
        if chart_type == "Boxplot":
            st.subheader("📦 Boxplot")
            st.info("Visualize distribution of numerical data across groups.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis (Categorical/Group)", df.columns, key='box_x')
            with col2:
                y_col = st.selectbox("Y-axis (Numeric)", df.select_dtypes(include=np.number).columns, key='box_y')
            with col3:
                color_col = st.selectbox("Color (Optional)", [None] + list(df.columns), key='box_color')
                
            if st.button("Generate Boxplot"):
                try:
                    fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Boxplot of {y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")

        elif chart_type == "Scatter Plot":
            st.subheader("🔵 Scatter Plot")
            st.info("Visualize relationship between two numerical variables.")
            
            col1, col2, col3 = st.columns(3)
            numeric_cols = df.select_dtypes(include=np.number).columns
            with col1:
                x_col = st.selectbox("X-axis (Numeric)", numeric_cols, key='scatter_x')
            with col2:
                y_col = st.selectbox("Y-axis (Numeric)", numeric_cols, key='scatter_y')
            with col3:
                color_col = st.selectbox("Color (Optional)", [None] + list(df.columns), key='scatter_color')
                
            if st.button("Generate Scatter Plot"):
                try:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {y_col} vs {x_col}", hover_data=df.columns)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")

        elif chart_type == "Heatmap":
            st.subheader("🔥 Correlation Heatmap")
            st.info("Visualize correlation matrix of numerical variables.")
            
            if st.button("Generate Heatmap"):
                try:
                    numeric_df = df.select_dtypes(include=np.number)
                    if numeric_df.empty:
                        st.error("No numeric columns found for correlation.")
                    else:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Correlation Matrix Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")

    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")
