import streamlit as st
import pandas as pd
import io
import pingouin as pg
import scipy.stats as stats
import numpy as np
import google.generativeai as genai
import os
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Import new engines
from logic_engine import LogicEngine
from viz_engine import VizEngine

# Initialize Engines
logic_engine = LogicEngine()
viz_engine = VizEngine()

# Optional imports for advanced features
try:
    from tableone import TableOne
    TABLEONE_AVAILABLE = True
except ImportError:
    TABLEONE_AVAILABLE = False
    
try:
    import statsmodels.stats.api as sms
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ROBUST MODEL LOADING (Fixes 404 Error)
def get_generative_model():
    """
    Dynamically selects the best available Gemini model.
    Prioritizes 2.0/2.5 Flash variants, then 1.5 Flash, then Pro.
    """
    # Configure Gemini API from secrets
    if "GENAI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GENAI_API_KEY"])
    else:
        st.warning("⚠️ Gemini API key not found in secrets. AI features will be disabled.")
        return None

    preferred_order = [
        'gemini-2.0-flash',
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro'
    ]
    
    try:
        # Attempt to list models available to this API key
        available_models = [m.name.replace('models/', '') for m in genai.list_models()]
        
        # Pick the first preferred model that exists in the available list
        for model_name in preferred_order:
            if model_name in available_models:
                return genai.GenerativeModel(model_name)
    except Exception as e:
        # If listing fails (e.g. permission issue), just try the newest one
        pass

    # Fallback: Default to gemini-2.0-flash if we can't check, 
    # as 1.5-flash is reported deprecated for some keys.
    return genai.GenerativeModel('gemini-2.0-flash')

# Academic Citations Library (for Methods sections)
CITATIONS = {
    'scipy': """Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.""",
    'pingouin': """Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source Software, 3(31), 1026.""",
    'pandas': """McKinney, W. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).""",
    'sklearn': """Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.""",
    'numpy': """Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357-362."""
}

def generate_citation(test_name, libraries_used):
    """Generate academic citation for Methods section."""
    lib_map = {
        'T-Test (Independent)': ['pingouin', 'scipy'],
        'T-Test (Paired)': ['pingouin'],
        'Chi-Square Test': ['pingouin'],
        'ANOVA': ['pingouin'],
        'Pearson Correlation': ['pingouin'],
        'Machine Learning': ['sklearn', 'pandas'],
        'Meta-Analysis': ['numpy', 'scipy']
    }
    
    libs = libraries_used if libraries_used else lib_map.get(test_name, ['scipy'])
    
    citation_text = f"""**Methods Section Citation:**

Statistical analysis was performed using Python programming language (version 3.11) with the following libraries: {', '.join(libs)}. """
    
    if test_name == 'T-Test (Independent)':
        citation_text += """Independent samples t-test was used to compare continuous variables between two groups. """
    elif test_name == 'T-Test (Paired)':
        citation_text += """Paired samples t-test was used to compare related measurements. """
    elif test_name == 'ANOVA':
        citation_text += """One-way analysis of variance (ANOVA) was used to compare means across multiple groups. """
    elif test_name == 'Chi-Square Test':
        citation_text += """Chi-square test of independence was used to examine associations between categorical variables. """
    elif test_name == 'Pearson Correlation':
        citation_text += """Pearson correlation coefficient was calculated to assess linear relationships. """
    
    citation_text += """Statistical significance was set at α = 0.05.\n\n**References:**\n"""
    
    for lib in libs:
        if lib in CITATIONS:
            citation_text += f"- {CITATIONS[lib]}\n"
    
    return citation_text

def validate_test_logic(test_name, col1_type, col2_type=None, col1_name="", col2_name=""):
    """Use AI to validate if test is appropriate for data types."""
    try:
        data_summary = f"Column '{col1_name}' is {col1_type}"
        if col2_type:
            data_summary += f". Column '{col2_name}' is {col2_type}"
        
        prompt = f"""You are a biostatistics expert. Answer ONLY with YES or NO followed by one brief sentence.

Test to run: {test_name}
Data types: {data_summary}

Question: Is this test statistically appropriate for these data types?
Format: YES/NO - [one sentence reason]
"""
        
        model = get_generative_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "YES - Unable to validate, proceeding with user selection."

        return "YES - Unable to validate, proceeding with user selection."

# detect_variable_type, smart_ttest, and smart_anova have been moved to logic_engine.py

# Helper function for AI explanations
def get_ai_explanation(test_name, result_df, p_value=None, logic_reason=None, effect_size=None, ci=None):
    """
    Get AI interpretation with strict NEJM-level standards.
    """
    st.markdown("### 🤖 AI Interpretation")
    
    with st.spinner("Consulting senior biostatistician..."):
        try:
            # Build context from logic
            logic_context = f"\n\nTest Selection Logic: {logic_reason}" if logic_reason else ""
            effect_context = f"\n\nEffect Size: {effect_size}" if effect_size else ""
            ci_context = f"\n\nConfidence Interval: {ci}" if ci else ""
            
            # STRICT NEJM PROMPT
            prompt = f"""You are a senior biostatistician reviewing a manuscript submission for the New England Journal of Medicine (NEJM).

Your task is to interpret the following statistical result with EXTREME rigor:

Test: {test_name}
P-value: {p_value if p_value else 'See result table'}
{logic_context}{effect_context}{ci_context}

Result Table:
{result_df.to_string()}

CRITICAL INSTRUCTIONS:
1. NEVER use the phrase "trend toward significance" or "approaching significance". If p ≥ 0.05, state clearly: "No statistically significant difference was observed."
2. ALWAYS cite the Effect Size first, then the Confidence Interval, and only then mention the p-value.
3. If the Confidence Interval is wide (e.g., crosses the null value or spans a large range), explicitly state: "The estimate is imprecise due to a wide confidence interval."
4. Interpret CLINICAL relevance, not just statistical significance. Ask: "Does this effect size matter to patients?"
5. If the logic reason indicates auto-switching (e.g., from T-Test to Mann-Whitney), acknowledge this and explain why it strengthens the validity of the result.
6. Keep your response to 3-4 sentences maximum. Be precise, not verbose.

Write your interpretation now:"""
            
            model = get_generative_model()
            response = model.generate_content(prompt)
            
            st.info(response.text)
            
        except Exception as e:
            st.error(f"Error generating AI interpretation: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Rbiostatistics - AI Research Copilot",
    page_icon="🧬",
    layout="wide"
)

# Robust Data Loading Function
def load_data():
    """Ensures data is loaded from session state."""
    if 'df' in st.session_state:
        return st.session_state['df']
    return None

# Initialize session state for workflow
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'copilot' not in st.session_state:
    st.session_state.copilot = None

# Load data at startup
df = load_data()

# Initialize Research Copilot
from logic_engine import ResearchCopilot
if df is not None and st.session_state.copilot is None:
    try:
        model = get_generative_model()
        st.session_state.copilot = ResearchCopilot(df=df, gemini_model=model)
    except:
        st.session_state.copilot = ResearchCopilot(df=df, gemini_model=None)

# Workflow Steps Definition
WORKFLOW_STEPS = {
    1: {"name": "Ingest & Inspect", "icon": "📥"},
    2: {"name": "Clean & Prepare", "icon": "🧹"},
    3: {"name": "The Hypothesis", "icon": "🔬"},
    4: {"name": "The Analysis", "icon": "📊"},
    5: {"name": "The Manuscript", "icon": "📝"}
}

# Sidebar Navigation with Wizard + Research Assistant
with st.sidebar:
    st.title("🧬 Rbiostatistics")
    st.markdown("### AI Research Copilot")
    st.markdown("---")
    
    # Wizard Navigation
    st.markdown("#### 🗺️ Research Workflow")
    
    # Create step menu
    step_options = [f"{WORKFLOW_STEPS[i]['icon']} {i}. {WORKFLOW_STEPS[i]['name']}" for i in range(1, 6)]
    
    selected_step = option_menu(
        menu_title=None,
        options=step_options,
        icons=["1-circle", "2-circle", "3-circle", "4-circle", "5-circle"],
        menu_icon="cast",
        default_index=st.session_state.current_step - 1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#2E86C1", "font-size": "14px"}, 
            "nav-link": {
                "font-size": "13px",
                "text-align": "left",
                "margin":"2px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": "#2E86C1"},
        }
    )
    
    # Update current step based on selection
    step_number = int(selected_step.split('.')[0].split()[-1])
    
    # GUARDRAIL: Prevent skipping steps if no data loaded
    if step_number > 1 and 'df' not in st.session_state:
        st.warning("⚠️ Please upload data in Step 1 first!")
        st.session_state.current_step = 1
    else:
        st.session_state.current_step = step_number
        
    # Clear Data Button (For Testing/Resetting)
    if 'df' in st.session_state:
        st.markdown("---")
        if st.button("🗑️ Clear Data & Restart", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_step = 1
            st.rerun()
    
    st.markdown("---")
    
    # Research Assistant Section
    st.markdown("#### 🤖 Research Assistant")
    
    if df is None:
        st.info("💤 Waiting for data... I will analyze your variables once you upload.")
    elif st.session_state.copilot and st.session_state.current_step == 1:
        st.success("✅ Data loaded! Analyzing your dataset...")
        
        if st.button("🔮 Suggest Research Questions", use_container_width=True):
            with st.spinner("Consulting AI biostatistician..."):
                suggestions = st.session_state.copilot.get_research_suggestions()
                st.session_state['ai_suggestions'] = suggestions
        
        # Display suggestions if available
        if 'ai_suggestions' in st.session_state:
            st.markdown("**💡 Suggested Questions:**")
            for i, sug in enumerate(st.session_state['ai_suggestions'][:3], 1):
                with st.expander(f"Q{i}: {sug['question'][:50]}...", expanded=False):
                    st.write(sug['question'])
                    st.caption(f"**Test:** {sug['suggested_test']}")
    else:
        st.info(f"📍 Step {st.session_state.current_step}: {WORKFLOW_STEPS[st.session_state.current_step]['name']}")

# Map step number to page name
page = WORKFLOW_STEPS[st.session_state.current_step]['name']

# Main content
# Main content based on wizard step
if page == "Ingest & Inspect":
    st.title("📥 Step 1: Ingest & Inspect Data")
    st.markdown("Upload your research data and preview the structure")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload a .csv or .xlsx file containing your data"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['df'] = df
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
            
            # Reinitialize copilot with new data
            try:
                model = get_generative_model()
                st.session_state.copilot = ResearchCopilot(df=df, gemini_model=model)
            except:
                st.session_state.copilot = ResarchCopilot(df=df, gemini_model=None)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    elif 'df' in st.session_state:
        df = st.session_state['df']
        st.info("Using previously loaded data")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Please upload a dataset to begin")
    
    # Next button
    st.markdown("---")
    if st.button("➡️ Next: Clean & Prepare Data", type="primary", use_container_width=True):
        if 'df' in st.session_state:
            st.session_state.current_step = 2
            st.rerun()
        else:
            st.error("Please upload data first!")

elif page == "Clean & Prepare":
    st.title("🧹 Step 2: Clean & Prepare Data")
    st.markdown("Handle missing data and prepare variables for analysis")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go back to Step 1.")
        if st.button("⬅️ Back to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        df = st.session_state['df']
        
        # Missing data analysis
        st.markdown("### 🔍 Missing Data Analysis")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        }).query('`Missing Count` > 0')
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("### 🛠️ Imputation Options")
            st.info("For now, you can drop rows with missing values. Advanced imputation coming soon!")
            
            if st.button("Drop Rows with Missing Values"):
                df_clean = df.dropna()
                st.session_state['df'] = df_clean
                st.success(f"✅ Dropped {len(df) - len(df_clean)} rows. New dataset has {len(df_clean)} rows.")
                st.rerun()
        else:
            st.success("✅ No missing data detected!")
        
        # Variable type confirmation
        st.markdown("### 📋 Variable Types")
        st.markdown("Review auto-detected variable types (you can adjust in next step)")
        
        type_data = []
        for col in df.columns[:10]:
            vtype = logic_engine.detect_variable_type(df[col])
            type_data.append({"Variable": col, "Type": vtype, "Unique Values": df[col].nunique()})
        
        st.dataframe(pd.DataFrame(type_data), use_container_width=True)
        
        # Next/Previous buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Ingest Data"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("➡️ Next: Define Hypothesis", type="primary"):
                st.session_state.current_step = 3
                st.rerun()

elif page == "The Hypothesis":
    st.title("🔬 Step 3: Define Your Hypothesis")
    st.markdown("Select variables and let AI suggest the appropriate statistical test")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go back to Step 1.")
    else:
        df = st.session_state['df']
        
        # Variable selection
        st.markdown("### 🎯 Select Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            independent_var = st.selectbox("Independent Variable (Predictor)", df.columns, key="ind_var")
        with col2:
            dependent_var = st.selectbox("Dependent Variable (Outcome)", df.columns, key="dep_var")
        
        # AI Test Suggestion
        if st.button("🤖 Suggest Statistical Test", use_container_width=True):
            if st.session_state.copilot:
                test_name, reason = st.session_state.copilot.suggest_test(independent_var, dependent_var, df)
                st.session_state['suggested_test'] = test_name
                st.session_state['test_reason'] = reason
                
                st.success(f"**Suggested Test:** {test_name}")
                st.info(f"**Reason:** {reason}")
            else:
                st.warning("AI assistant not available")
        
        # Display suggestion if available
        if 'suggested_test' in st.session_state:
            st.markdown("### 💡 AI Recommendation")
            st.markdown(f"**Test:** `{st.session_state['suggested_test']}`")
            st.markdown(f"**Rationale:** {st.session_state['test_reason']}")
        
        # Next/Previous buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Clean Data"):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("➡️ Next: Run Analysis", type="primary"):
                # PERSIST SELECTION explicitly to prevent widget cleanup loss
                st.session_state['confirmed_ind_var'] = independent_var
                st.session_state['confirmed_dep_var'] = dependent_var
                
                st.session_state.current_step = 4
                st.rerun()

elif page == "The Analysis":
    st.title("📊 Step 4: Run Statistical Analysis")
    st.markdown("Execute the test and visualize results")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go back to Step 1.")
    else:
        df = st.session_state['df']
        
        # Check for persistent variables first
        if 'confirmed_ind_var' in st.session_state and 'confirmed_dep_var' in st.session_state:
            ind_var = st.session_state['confirmed_ind_var']
            dep_var = st.session_state['confirmed_dep_var']
            
            st.info(f"**Testing:** {ind_var} → {dep_var}")
            
            # Run comparison if both are set
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    try:
                        result = logic_engine.compare_groups_automator(df, ind_var, dep_var)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.session_state['analysis_result'] = result
                            st.success("✅ Analysis complete!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
            
            # Display results if available
            if 'analysis_result' in st.session_state:
                result = st.session_state['analysis_result']
                
                st.markdown("### 📈 Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Used", result['Test Name'])
                with col2:
                    st.metric("P-value", result['P-value'])
                with col3:
                    st.metric("Effect Size", result['Effect Size'].split('(')[0].strip())
                
                st.markdown(f"**Logic:** {result['Logic Reason']}")
                
                # AI Interpretation
                st.markdown("### 🤖 AI Interpretation")
                try:
                    model = get_generative_model()
                    prompt = f"""Interpret this statistical result for a medical student:
Test: {result['Test Name']}
P-value: {result['P-value']}
Effect Size: {result['Effect Size']}
Logic: {result['Logic Reason']}

Provide a 2-3 sentence interpretation focusing on clinical significance."""
                    
                    if model:
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    else:
                         st.warning("AI unavailable (check API key)")
                except:
                    st.warning("AI interpretation unavailable")
        else:
            st.warning("Please define your hypothesis in Step 3 first")
            if st.button("⬅️ Go to Step 3"):
                st.session_state.current_step = 3
                st.rerun()

elif page == "The Manuscript":
    st.title("📝 Step 5: Generate Manuscript Sections")
    st.markdown("Create Table 1 and Methods section for publication")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ No data loaded. Please complete the workflow from Step 1.")
    else:
        df = st.session_state['df']
        
        # Table 1 Generator
        st.markdown("### 📊 Table 1 Generator")
        
        study_design = st.radio("Study Design", ["Observational", "RCT"], horizontal=True)
        group_col = st.selectbox("Grouping Variable (e.g., Treatment)", df.columns)
        
        if st.button("Generate Table 1"):
            try:
                table1 = logic_engine.generate_table1(df, group_col, study_design)
                st.session_state['table1'] = table1
                st.text(str(table1))
                st.success("✅ Table 1 generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Methods Section
        st.markdown("### 📖 Methods Section")
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            methods_text = f"""
**Statistical Analysis**

Continuous variables were assessed for normality using the Shapiro-Wilk test. 
{result['Logic Reason']} The {result['Test Name']} was used to compare groups.
Effect sizes are reported as {result['Effect Size'].split('(')[1].strip(')')} 
with 95% confidence intervals. Statistical significance was set at p < 0.05.
All analyses were performed using Python (version 3.11) with the Pingouin library.
"""
            
            st.code(methods_text.strip(), language=None)
            
            if st.button("📋 Copy Methods Section"):
                st.success("Methods section ready to copy!")
        else:
            st.info("Complete Step 4 (Analysis) to generate methods section")
        
        # Restart button
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Analysis"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis", type="primary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key not in ['current_step']:
                        del st.session_state[key]
                st.session_state.current_step = 1
                st.rerun()
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

elif page == "Variable Setup":
    st.title("🔍 Variable Setup & Confirmation")
    st.info("**CRITICAL STEP**: Confirm variable types to ensure accurate statistical analysis.")
    
    df = load_data()
    if df is not None:
        st.success(f"Working with: **{st.session_state['filename']}**")
        st.markdown("---")
        
        st.markdown("### 🤖 Auto-Detection Results")
        st.markdown("Review and confirm the detected variable types below. This prevents statistical errors.")
        
        # Auto-detect all variables
        if 'var_types' not in st.session_state:
            st.session_state['var_types'] = {}
            st.session_state['var_roles'] = {}
        
        # Create detection table
        detection_data = []
        for col in df.columns:
            auto_type = logic_engine.detect_variable_type(df[col])
            detection_data.append({
                'Variable Name': col,
                'Auto-Detected Type': auto_type,
                'Sample Values': str(df[col].head(3).tolist())[:50] + '...',
                'Unique Values': df[col].nunique()
            })
        
        detection_df = pd.DataFrame(detection_data)
        st.dataframe(detection_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ✅ Confirm Variable Types & Roles")
        
        # User confirmation interface
        confirmed_vars = {}
        var_roles = {}
        
        for col in df.columns:
            with st.expander(f"📊 {col}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                auto_type = logic_engine.detect_variable_type(df[col])
                
                with col1:
                    confirmed_type = st.selectbox(
                        f"Confirm Type for '{col}':",
                        ["Continuous", "Categorical", "Ordinal", "Date", "ID (Exclude)"],
                        index=0 if "Continuous" in auto_type else (1 if "Categorical" in auto_type else 2),
                        key=f"type_{col}"
                    )
                    confirmed_vars[col] = confirmed_type
                
                with col2:
                    if confirmed_type not in ["ID (Exclude)", "Date"]:
                        role = st.selectbox(
                            f"Role:",
                            ["Measure (Outcome)", "Grouping", "Exclude"],
                            key=f"role_{col}"
                        )
                        var_roles[col] = role
                    else:
                        st.info("Excluded")
                        var_roles[col] = "Exclude"
                
                with col3:
                    st.metric("Unique Values", df[col].nunique())
        
        # Save button
        if st.button("💾 Save Variable Configuration", type="primary"):
            st.session_state['var_types'] = confirmed_vars
            st.session_state['var_roles'] = var_roles
            st.success("✅ Variable configuration saved! You can now proceed to 'Research Question'.")
            st.balloons()
            
            # Summary
            measure_vars = [k for k, v in var_roles.items() if v == "Measure (Outcome)"]
            group_vars = [k for k, v in var_roles.items() if v == "Grouping"]
            
            st.markdown("### 📋 Configuration Summary")
            st.write(f"**Outcome Variables**: {', '.join(measure_vars) if measure_vars else 'None'}")
            st.write(f"**Grouping Variables**: {', '.join(group_vars) if group_vars else 'None'}")
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Research Question":
    st.title("🎯 Research Question Builder")
    st.info("**Natural Language Approach**: Tell us your research question, we'll pick the right test.")
    
    df = load_data()
    if df is not None and 'var_types' in st.session_state:
        st.success(f"Working with: **{st.session_state['filename']}**")
        st.markdown("---")
        
        # Get continuous and categorical variables
        var_types = st.session_state['var_types']
        var_roles = st.session_state['var_roles']
        
        continuous_vars = [k for k, v in var_types.items() if v == "Continuous" and var_roles[k] != "Exclude"]
        categorical_vars = [k for k, v in var_types.items() if v in ["Categorical", "Ordinal"] and var_roles[k] != "Exclude"]
        
        st.markdown("### 📝 Build Your Research Question")
        st.markdown("*Fill in the blanks to describe your research question:*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**I want to see if...**")
            dependent_var = st.selectbox(
                "Outcome Variable (what you're measuring):",
                continuous_vars + categorical_vars,
                help="The variable you want to analyze"
            )
        
        with col2:
            st.markdown("**...is different based on...**")
            independent_var = st.selectbox(
                "Grouping Variable (what defines groups):",
                categorical_vars + continuous_vars,
                help="The variable that defines groups or categories"
            )
        
        if st.button("🔍 Find Recommended Test", type="primary"):
            # HARD-CODED LOGIC ENGINE (Python is King!)
            dep_type = var_types[dependent_var]
            ind_type = var_types[independent_var]
            num_groups = df[independent_var].nunique() if ind_type in ["Categorical", "Ordinal"] else None
            
            recommended_test = None
            reason = ""
            
            # Decision Tree
            if dep_type == "Continuous" and ind_type in ["Categorical", "Ordinal"]:
                if num_groups == 2:
                    recommended_test = "Independent T-Test"
                    reason = f"Comparing a continuous variable ({dependent_var}) across 2 groups in {independent_var}."
                elif num_groups >= 3:
                    recommended_test = "ANOVA"
                    reason = f"Comparing a continuous variable ({dependent_var}) across {num_groups} groups in {independent_var}."
            elif dep_type in ["Categorical", "Ordinal"] and ind_type in ["Categorical", "Ordinal"]:
                recommended_test = "Chi-Square Test"
                reason = f"Testing association between two categorical variables: {dependent_var} and {independent_var}."
            elif dep_type == "Continuous" and ind_type == "Continuous":
                recommended_test = "Pearson Correlation"
                reason = f"Measuring linear relationship between two continuous variables: {dependent_var} and {independent_var}."
            else:
                recommended_test = "No Standard Test"
                reason = "This combination of variable types doesn't match a standard statistical test. Please review your variable types."
            
            # Display Recommendation
            if recommended_test != "No Standard Test":
                st.success(f"### ✅ Recommended Test: **{recommended_test}**")
                st.info(f"**Reason**: {reason}")
                
                st.markdown("### 🚀 Ready to Run?")
                st.markdown(f"Navigate to **'Statistical Tests'** and select **{recommended_test}** to run your analysis.")
                st.markdown(f"- **Outcome**: {dependent_var} ({dep_type})")
                st.markdown(f"- **Grouping**: {independent_var} ({ind_type})")
            else:
                st.error(f"❌ {reason}")
                st.info("💡 Try going back to 'Variable Setup' to adjust variable types.")
    
    elif df is not None:
        st.warning("⚠️ Please complete 'Variable Setup' first to define your variable types.")
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Table 1":
    st.title("📋 Table 1 Generator")
    
    if not TABLEONE_AVAILABLE:
        st.error("⚠️ The 'tableone' library is not available due to a version conflict. Please install compatible versions manually.")
        st.code("pip install 'statsmodels<0.15' tableone", language="bash")
        st.stop()
    
    st.info("Automatically generate a 'Table 1' (Demographics) for your study.")
    
    df = load_data()
    if df is not None:
        st.success(f"Working with: **{st.session_state['filename']}**")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            groupby_col = st.selectbox("Group By (e.g., Treatment Group)", [None] + list(df.columns))
        with col2:
            categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
            numeric_cols = list(df.select_dtypes(include=np.number).columns)
            all_cols = categorical_cols + numeric_cols
            selected_vars = st.multiselect("Select Variables to Include", all_cols, default=all_cols[:5])
            
        if st.button("Generate Table 1"):
            if not selected_vars:
                st.error("Please select at least one variable.")
            else:
                try:
                    # Define categorical variables for TableOne
                    cats = [c for c in selected_vars if c in categorical_cols]
                    
                    # Create TableOne
                    table1 = TableOne(
                        df, 
                        columns=selected_vars, 
                        categorical=cats, 
                        groupby=groupby_col, 
                        pval=True if groupby_col else False
                    )
                    
                    st.write(table1.tabulate(tablefmt="github"))
                    st.dataframe(table1.tableone, use_container_width=True)
                    
                    # Download
                    csv = table1.tableone.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Table 1 as CSV",
                        data=csv,
                        file_name='table1.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Error generating Table 1: {str(e)}")
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Statistical Wizard":
    st.title("🧙‍♂️ Statistical Wizard")
    st.info("Not sure which test to run? Answer 3 questions and I'll guide you.")
    
    df = load_data()
    if df is not None:
        st.markdown("### Step 1: Describe your data")
        
        q1 = st.radio("What kind of data are you comparing?", ["Numerical (e.g., Age, BP)", "Categorical (e.g., Gender, Disease Status)"])
        
        if q1.startswith("Numerical"):
            q2 = st.radio("How many groups are you comparing?", ["2 Groups (e.g., Drug vs Placebo)", "3+ Groups (e.g., Low vs Med vs High Dose)"])
            
            if q2.startswith("2 Groups"):
                q3 = st.radio("Are the groups paired?", ["No (Independent groups)", "Yes (Same patients before/after)"])
                
                if q3.startswith("No"):
                    st.success("✅ Recommendation: **Independent T-Test** (or Mann-Whitney U if not normal)")
                    if st.button("Go to T-Test"):
                        # In a real app we might redirect, but here we just point them
                        st.info("Navigate to 'Statistical Tests' > 'Independent T-Test' in the sidebar.")
                else:
                    st.success("✅ Recommendation: **Paired T-Test** (or Wilcoxon Signed-Rank)")
                    if st.button("Go to Paired T-Test"):
                        st.info("Navigate to 'Statistical Tests' > 'Paired T-Test' in the sidebar.")
            else:
                st.success("✅ Recommendation: **One-Way ANOVA** (or Kruskal-Wallis)")
                if st.button("Go to ANOVA"):
                    st.info("Navigate to 'Statistical Tests' > 'ANOVA' in the sidebar.")
                    
        else: # Categorical
            q2_cat = st.radio("What are you looking for?", ["Association between 2 variables", "Agreement between raters"])
            if q2_cat.startswith("Association"):
                st.success("✅ Recommendation: **Chi-Square Test** (or Fisher's Exact)")
                if st.button("Go to Chi-Square"):
                    st.info("Navigate to 'Statistical Tests' > 'Chi-Square Test' in the sidebar.")
            else:
                st.success("✅ Recommendation: **Cohen's Kappa**")
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "Machine Learning":
    st.title("🤖 Machine Learning")
    
    df = load_data()
    if df is not None:
        st.success(f"Working with: **{st.session_state['filename']}**")
        
        st.markdown("---")
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

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
                # PRE-TEST VALIDATION
                validation_result = validate_test_logic(
                    "Independent T-Test", 
                    "numeric",
                    "categorical (2 groups)",
                    value_col,
                    group_col
                )
                
                if "NO" in validation_result.upper():
                    st.error(f"⚠️ **Methodology Warning:** {validation_result}")
                    st.warning("This test may produce invalid results. Consider reviewing your data or choosing a different test.")
                
                try:
                    # Check if group column has exactly 2 unique values
                    unique_groups = df[group_col].unique()
                    if len(unique_groups) != 2:
                        st.error(f"Grouping column must have exactly 2 unique values. Found {len(unique_groups)}: {unique_groups}")
                    else:
                        group1 = df[df[group_col] == unique_groups[0]][value_col]
                        group2 = df[df[group_col] == unique_groups[1]][value_col]
                        
                        # USE NEJM-LEVEL AUTOMATOR
                        st.markdown("#### 🔍 NEJM-Level Assumption Checks & Auto-Selection")
                        
                        automator_result = logic_engine.compare_groups_automator(df, group_col, value_col)
                        
                        if "error" in automator_result:
                            st.error(automator_result["error"])
                        else:
                            # Display Reasons
                            if automator_result["Detailed Reasons"]:
                                for reason in automator_result["Detailed Reasons"]:
                                    st.warning(f"⚠️ {reason}")
                            else:
                                st.success("✅ Data meets all parametric assumptions (Normality & Variance).")
                            
                            st.info(f"👉 **Decision:** {automator_result['Logic Reason']}")

                            # RESULTS DISPLAY (Enhanced)
                            st.markdown("---")
                            st.markdown("### 📊 RESULTS")
                            
                            col_res1, col_res2, col_res3 = st.columns(3)
                            col_res1.metric("Test Used", automator_result["Test Name"])
                            col_res2.metric("P-Value", automator_result["P-value"])
                            col_res3.metric("Effect Size", automator_result["Effect Size"])
                            
                            st.dataframe(automator_result["Raw Result"], use_container_width=True)
                            
                            # Prepare variables for downstream (AI explanation etc)
                            test_used = automator_result["Test Name"]
                            res = automator_result["Raw Result"]
                            p_val = automator_result["Numeric P"]
                        
                        
                        # BOLD P-VALUE DISPLAY
                        # p_val is already extracted above from automator_result
                        st.markdown("### 🎯 Key Result")
                        st.markdown(f"# **p-value: {p_val:.4f}**")
                        
                        if p_val < 0.05:
                            st.success(f"✅ **Significant difference found** (p < 0.05)")
                        else:
                            st.info(f"ℹ️ **No significant difference found** (p >= 0.05)")
                        
                        st.markdown("---")
                        
                        # CODE TRANSPARENCY
                        with st.expander("📋 View Calculation Code (For Academic Transparency)"):
                            st.markdown("**Exact Python code used for this calculation:**")
                            if test_used == "Mann-Whitney U Test":
                                code_snippet = f"""import pingouin as pg

# Extract groups
group1 = df[df['{group_col}'] == '{unique_groups[0]}']['{value_col}']
group2 = df[df['{group_col}'] == '{unique_groups[1]}']['{value_col}']

# Check normality
norm1 = pg.normality(group1)
norm2 = pg.normality(group2)

# Data not normal - Use Mann-Whitney U Test
result = pg.mwu(group1, group2)

# Result: 
# p-value: {p_val:.4f}"""
                            else:
                                code_snippet = f"""import pingouin as pg

# Extract groups
group1 = df[df['{group_col}'] == '{unique_groups[0]}']['{value_col}']
group2 = df[df['{group_col}'] == '{unique_groups[1]}']['{value_col}']

# Data is normal - Use Independent T-Test
result = pg.ttest(group1, group2, correction=True)

# Result: 
# T-statistic: {res['T'].values[0] if 'T' in res.columns else 'N/A'}
# p-value: {p_val:.4f}"""
                            st.code(code_snippet, language='python')
                            st.info("💡 This code uses the **Pingouin** library, which is peer-reviewed and cited in academic publications.")
                        
                        st.markdown("---")
                        
                        # AI INTERPRETATION (Enhanced with NEJM logic)
                        get_ai_explanation(
                            test_used, 
                            res, 
                            p_val,
                            logic_reason=automator_result['Logic Reason'],
                            effect_size=automator_result['Effect Size']
                        )
                        
                        st.markdown("---")
                        
                        # ACADEMIC CITATION (Updated for auto-switching)
                        st.markdown("### 📚 Methods Section Citation")
                        citation_text = f"""**Methods Section Text:**

Statistical analysis was performed using Python programming language (version 3.11) with the Pingouin library (Vallat, 2018). Data normality was assessed using the Shapiro-Wilk test. {'Since data followed a normal distribution, an independent samples t-test was used' if test_used == 'Independent T-Test' else 'As the data violated normality assumptions, a non-parametric Mann-Whitney U test was used'} to compare continuous variables between two groups. Statistical significance was set at α = 0.05.

**References:**
- Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source Software, 3(31), 1026.
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.
"""
                        st.markdown(citation_text)
                        st.info("💡 Copy the text above for your paper's Methods section. It documents the automatic test selection based on assumption violations.")
                        
                        # AI Report Writing (Optional)
                        if st.button("📝 Generate APA Report"):
                            with st.spinner("Writing report..."):
                                try:
                                    prompt = f"""
                                    Write a standardized APA style results paragraph for an Independent T-Test.
                                    
                                    Context:
                                    - Variable: {value_col}
                                    - Groups: {unique_groups[0]} vs {unique_groups[1]}
                                    - T-statistic: {res['T'].values[0]:.2f}
                                    - Degrees of Freedom: {res['dof'].values[0]:.2f}
                                    - P-value: {p_val:.4f}
                                    - Mean Group 1: {group1.mean():.2f} (SD: {group1.std():.2f})
                                    - Mean Group 2: {group2.mean():.2f} (SD: {group2.std():.2f})
                                    
                                    Output ONLY the paragraph.
                                    """
                                    model = get_generative_model()
                                    response = model.generate_content(prompt)
                                    st.text_area("APA Results Paragraph", response.text, height=150)
                                except Exception as e:
                                    st.error(f"Error generating report: {e}")
                        
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
                # PRE-TEST VALIDATION
                validation_result = validate_test_logic(
                    "One-way ANOVA",
                    "numeric",
                    "categorical (3+ groups)",
                    value_col,
                    group_col
                )
                
                if "NO" in validation_result.upper():
                    st.error(f"⚠️ **Methodology Warning:** {validation_result}")
                    st.warning("This test may produce invalid results. Consider reviewing your data or choosing a different test.")
                
                try:
                    # USE SMART ANOVA (Auto-switching)
                    st.markdown("#### 🔍 Assumption Checks & Auto-Selection")
                    res, test_used, test_reason, messages = logic_engine.smart_anova(df, value_col, group_col)
                    
                    # Display messages from LogicEngine
                    for msg in messages:
                        if msg['type'] == 'warning':
                            st.warning(msg['text'])
                        elif msg['type'] == 'success':
                            st.success(msg['text'])
                    
                    # RESULTS DISPLAY (Enhanced)
                    st.markdown("---")
                    st.markdown("### 📊 RESULTS")
                    st.markdown(f"**Test Used:** {test_used}")
                    st.markdown(f"**Reason:** {test_reason}")
                    
                    st.dataframe(res, use_container_width=True)
                    
                    # BOLD P-VALUE DISPLAY
                    p_val = res['p-unc'].values[0] if 'p-unc' in res.columns else res['p-val'].values[0]
                    st.markdown("### 🎯 Key Result")
                    st.markdown(f"# **p-value: {p_val:.4f}**")
                    
                    if p_val < 0.05:
                        st.success(f"✅ **Significant difference found** (p < 0.05)")
                    else:
                        st.info(f"ℹ️ **No significant difference found** (p >= 0.05)")
                    
                    st.markdown("---")
                    
                    # CODE TRANSPARENCY
                    with st.expander("📋 View Calculation Code (For Academic Transparency)"):
                        st.markdown("**Exact Python code used for this calculation:**")
                        if test_used == "Kruskal-Wallis Test":
                            code_snippet = f"""import pingouin as pg

# Check assumptions
levene = pg.homoscedasticity(df, dv='{value_col}', group='{group_col}')
norm = pg.normality(df, dv='{value_col}', group='{group_col}')

# Assumptions violated - Use Kruskal-Wallis Test
result = pg.kruskal(df, dv='{value_col}', between='{group_col}')

# Result:
# p-value: {p_val:.4f}"""
                        else:
                            code_snippet = f"""import pingouin as pg

# Assumptions met - Use One-way ANOVA
result = pg.anova(data=df, dv='{value_col}', between='{group_col}')

# Result:
# F-statistic: {res['F'].values[0] if 'F' in res.columns else 'N/A'}
# p-value: {p_val:.4f}"""
                        st.code(code_snippet, language='python')
                        st.info("💡 This code uses the **Pingouin** library, which is peer-reviewed and cited in academic publications.")
                    
                    st.markdown("---")
                    
                    # AI INTERPRETATION (Separate Section)
                    get_ai_explanation(test_used, res, p_val)
                    
                    st.markdown("---")
                    
                    # ACADEMIC CITATION (Updated for auto-switching)
                    st.markdown("### 📚 Methods Section Citation")
                    citation_text = f"""**Methods Section Text:**

Statistical analysis was performed using Python programming language (version 3.11) with the Pingouin library (Vallat, 2018). Assumptions of normality and homogeneity of variance were assessed using Shapiro-Wilk and Levene's tests, respectively. {'Since assumptions were met, a one-way ANOVA was used' if test_used == 'One-way ANOVA' else 'As the data violated parametric assumptions, a non-parametric Kruskal-Wallis test was used'} to compare means across multiple groups. Statistical significance was set at α = 0.05.

**References:**
- Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source Software, 3(31), 1026.
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.
"""
                    st.markdown(citation_text)
                    st.info("💡 Copy the text above for your paper's Methods section. It documents the automatic test selection based on assumption violations.")
                    
                    # AI Report Writing (Optional)
                    if st.button("📝 Generate APA Report"):
                        with st.spinner("Writing report..."):
                            try:
                                prompt = f"""
                                Write a standardized APA style results paragraph for a One-Way ANOVA.
                                
                                Context:
                                - Dependent Variable: {value_col}
                                - Grouping Variable: {group_col}
                                - F-statistic: {res['F'].values[0]:.2f}
                                - P-value: {p_val:.4f}
                                - Degrees of Freedom: {res['ddof1'].values[0]}, {res['ddof2'].values[0]}
                                
                                Output ONLY the paragraph.
                                """
                                model = get_generative_model()
                                response = model.generate_content(prompt)
                                st.text_area("APA Results Paragraph", response.text, height=150)
                            except Exception as e:
                                st.error(f"Error generating report: {e}")
                                
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

elif page == "Meta-Analysis":
    st.title("📊 Meta-Analysis Generator")
    st.info("Perform a basic meta-analysis from summary data.")
    
    st.markdown("### 1. Enter Study Data")
    st.markdown("Enter the Odds Ratio (OR) and 95% Confidence Intervals for each study.")
    
    # Default data for demo
    default_data = pd.DataFrame({
        'Study Name': ['Study A', 'Study B', 'Study C'],
        'OR': [1.5, 2.0, 0.8],
        'Lower CI': [1.1, 1.5, 0.5],
        'Upper CI': [2.0, 2.8, 1.2]
    })
    
    edited_df = st.data_editor(default_data, num_rows="dynamic")
    
    if st.button("Run Meta-Analysis"):
        try:
            # Calculate Log OR and SE
            # SE = (ln(Upper) - ln(Lower)) / 3.92
            edited_df['log_or'] = np.log(edited_df['OR'])
            edited_df['log_lo'] = np.log(edited_df['Lower CI'])
            edited_df['log_up'] = np.log(edited_df['Upper CI'])
            edited_df['se'] = (edited_df['log_up'] - edited_df['log_lo']) / 3.92
            
            # Inverse Variance Weighting (Fixed Effects for simplicity, or basic Random)
            # Using statsmodels for robust calculation if possible, else manual
            # Let's use a simple IV method here for "Zero Coding" speed
            edited_df['weight'] = 1 / (edited_df['se'] ** 2)
            
            # Pooled Effect (Log Scale)
            pooled_log_or = np.sum(edited_df['weight'] * edited_df['log_or']) / np.sum(edited_df['weight'])
            pooled_se = np.sqrt(1 / np.sum(edited_df['weight']))
            
            pooled_or = np.exp(pooled_log_or)
            pooled_lo = np.exp(pooled_log_or - 1.96 * pooled_se)
            pooled_up = np.exp(pooled_log_or + 1.96 * pooled_se)
            
            # Heterogeneity (Q and I2)
            # Q = sum(w * (y - theta)^2)
            q_stat = np.sum(edited_df['weight'] * (edited_df['log_or'] - pooled_log_or)**2)
            df_q = len(edited_df) - 1
            i2 = max(0, (q_stat - df_q) / q_stat) * 100 if q_stat > 0 else 0
            
            st.success(f"✅ Pooled Odds Ratio: {pooled_or:.2f} [{pooled_lo:.2f}, {pooled_up:.2f}]")
            st.info(f"Heterogeneity: I² = {i2:.1f}%")
            
            # Forest Plot using VizEngine
            fig = viz_engine.create_forest_plot(edited_df, pooled_or, pooled_lo, pooled_up)
            st.pyplot(fig)
            
            st.text_area("Methods Citation", generate_citation("Meta-Analysis", ['numpy', 'matplotlib']), height=150)
            
        except Exception as e:
            st.error(f"Error running meta-analysis: {str(e)}")

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
                    fig = viz_engine.create_box_plot(df, x=x_col, y=y_col, color=color_col, title=f"Boxplot of {y_col} by {x_col}")
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
                    fig = viz_engine.create_scatter_plot(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {y_col} vs {x_col}", hover_data=df.columns)
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
                        fig = viz_engine.create_heatmap(
                            corr_matrix, 
                            title="Correlation Matrix Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")

    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")

elif page == "AI Chatbot":
    st.title("💬 Chat with Data")
    
    df = load_data()
    if df is not None:
        st.success(f"Working with: **{st.session_state['filename']}**")
        
        st.info("Ask questions about your data in plain English. The AI will generate code to answer you.")
        
        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question (e.g., 'What is the average age?', 'Plot gender distribution')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Construct Prompt
                    schema_info = df.dtypes.to_string()
                    head_info = df.head(3).to_string()
                    
                    full_prompt = f"""
                    You are a Python Data Analyst. You have a pandas DataFrame named 'df'.
                    
                    Data Schema:
                    {schema_info}
                    
                    First 3 rows:
                    {head_info}
                    
                    User Question: {prompt}
                    
                    Write Python code to answer the question. 
                    - If the user asks for a plot, use 'st.bar_chart', 'st.line_chart', or 'plotly.express'.
                    - If the user asks for a value, print it or use 'st.write'.
                    - DO NOT create a new dataframe, use the existing 'df'.
                    - Wrap your code in a block like this:
                    ```python
                    # code here
                    ```
                    - Do not include any explanations outside the code block. Just the code.
                    """
                    
                    model = get_generative_model()
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    
                    # Extract Code
                    if "```python" in ai_response:
                        code_to_run = ai_response.split("```python")[1].split("```")[0].strip()
                        
                        # Display Code (Optional, for transparency)
                        with st.expander("View Generated Code"):
                            st.code(code_to_run, language='python')
                        
                        # Execute Code
                        # We need to pass 'df', 'st', 'pd', 'px', 'plt' to the exec environment
                        local_vars = {
                            'df': df,
                            'st': st,
                            'pd': pd,
                            'px': px,
                            'plt': plt
                        }
                        
                        # Capture output
                        try:
                            exec(code_to_run, globals(), local_vars)
                            st.session_state.messages.append({"role": "assistant", "content": "Executed code to answer your question."})
                        except Exception as exec_error:
                            st.error(f"Error executing code: {exec_error}")
                            st.session_state.messages.append({"role": "assistant", "content": f"I tried to run code but failed: {exec_error}"})
                            
                    else:
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                except Exception as e:
                    st.error(f"AI Error: {str(e)}")
    else:
        st.warning("⚠️ No data loaded. Please upload a file in the 'Data Upload' page first.")
