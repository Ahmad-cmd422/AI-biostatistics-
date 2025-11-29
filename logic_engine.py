import pandas as pd
import numpy as np
import pingouin as pg
import scipy.stats as stats
from scipy.stats import shapiro, levene, mannwhitneyu, kruskal, chi2_contingency

class LogicEngine:
    """
    The Brain of the Operation.
    Handles all statistical logic, decision trees, and assumption checking.
    Enforces 'Python is King' - deterministic calculations only.
    """
    
    def __init__(self):
        pass

    def detect_variable_type(self, column):
        """
        Smart variable detection logic.
        """
        if pd.api.types.is_numeric_dtype(column):
            unique_count = column.nunique()
            if column.name.lower() in ['id', 'patient_id', 'subject_id', 'record_id']:
                return "ID (exclude from analysis)"
            elif unique_count < 10:
                return "Categorical (numeric codes)"
            elif unique_count < 30:
                return "Ordinal (or Continuous)"
            else:
                return "Continuous"
        else:
            if pd.api.types.is_datetime64_any_dtype(column):
                return "Date (exclude or convert)"
            else:
                return "Categorical (text)"

    def check_normality(self, data):
        """
        Performs Shapiro-Wilk test for normality.
        Returns (is_normal: bool, p_value: float)
        """
        stat, p = shapiro(data)
        return p > 0.05, p

    def check_homogeneity(self, data, group_col, value_col):
        """
        Performs Levene's test for homogeneity of variance.
        Returns (is_homogeneous: bool, p_value: float)
        """
        groups = [group[value_col].dropna() for name, group in data.groupby(group_col)]
        stat, p = levene(*groups)
        return p > 0.05, p

    def smart_ttest(self, group1, group2):
        """
        Smart T-Test with automatic switching to Mann-Whitney if assumptions violated.
        Returns: (result_df, test_used, test_reason, messages)
        """
        messages = []
        
        # Check normality with Shapiro-Wilk
        norm1 = pg.normality(group1)
        norm2 = pg.normality(group2)
        p_norm1 = norm1['pval'].values[0]
        p_norm2 = norm2['pval'].values[0]
        
        # Auto-switch logic
        if p_norm1 < 0.05 or p_norm2 < 0.05:
            # DATA NOT NORMAL - AUTO-SWITCH
            messages.append({
                "type": "warning",
                "text": "🔄 **Auto-Switch Activated**: Data is not normally distributed (Shapiro-Wilk p < 0.05). Automatically switched from T-Test to **Mann-Whitney U Test** for accuracy."
            })
            result = pg.mwu(group1, group2)
            test_used = "Mann-Whitney U Test"
            test_reason = "Non-parametric alternative used due to violated normality assumption"
        else:
            # DATA IS NORMAL - USE T-TEST
            messages.append({
                "type": "success",
                "text": "✅ Data is normally distributed (Shapiro-Wilk p > 0.05). Using **Independent T-Test**."
            })
            result = pg.ttest(group1, group2, correction=True)
            test_used = "Independent T-Test"
            test_reason = "Parametric test appropriate for normally distributed data"
        
        return result, test_used, test_reason, messages

    def smart_anova(self, df, dv, group_col):
        """
        Smart ANOVA with automatic switching to Kruskal-Wallis if assumptions violated.
        Returns: (result_df, test_used, test_reason, messages)
        """
        messages = []
        
        # Check assumptions
        # 1. Homogeneity of variance
        levene = pg.homoscedasticity(df, dv=dv, group=group_col)
        p_levene = levene['pval'].values[0]
        
        # 2. Normality per group
        norm = pg.normality(df, dv=dv, group=group_col)
        normality_violated = any(norm['pval'] < 0.05)
        
        # Auto-switch logic
        if p_levene < 0.05 or normality_violated:
            # ASSUMPTIONS VIOLATED - AUTO-SWITCH
            reasons = []
            if p_levene < 0.05:
                reasons.append("unequal variances (Levene's p < 0.05)")
            if normality_violated:
                reasons.append("non-normal distribution (Shapiro-Wilk p < 0.05)")
            
            messages.append({
                "type": "warning",
                "text": f"🔄 **Auto-Switch Activated**: Data violates assumptions ({', '.join(reasons)}). Automatically switched from ANOVA to **Kruskal-Wallis Test** for accuracy."
            })
            result = pg.kruskal(df, dv=dv, between=group_col)
            test_used = "Kruskal-Wallis Test"
            test_reason = f"Non-parametric alternative used due to: {', '.join(reasons)}"
        else:
            # ASSUMPTIONS MET - USE ANOVA
            messages.append({
                "type": "success",
                "text": "✅ Assumptions met (equal variances + normal distribution). Using **One-way ANOVA**."
            })
            result = pg.anova(data=df, dv=dv, between=group_col)
            test_used = "One-way ANOVA"
            test_reason = "Parametric test appropriate for data meeting assumptions"
        
        return result, test_used, test_reason, messages

    def compare_groups_automator(self, df, group_col, value_col):
        """
        NEJM-Level Group Comparison Automator.
        Automatically selects between Welch's T-Test and Mann-Whitney U based on rigorous assumption checks.
        """
        results = {}
        reasons = []
        
        # 0. Basic Validation
        if group_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns not found: {group_col}, {value_col}"}
            
        if group_col == value_col:
            return {"error": "Independent and Dependent variables cannot be the same."}
        
        # Get groups
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            return {"error": f"Grouping variable '{group_col}' must have exactly 2 levels (found {len(groups)}: {groups}). Use ANOVA for 3+ groups."}
            
        g1 = df[df[group_col] == groups[0]][value_col].dropna()
        g2 = df[df[group_col] == groups[1]][value_col].dropna()
        
        # Check sample sizes
        if len(g1) < 3 or len(g2) < 3:
            return {"error": f"Insufficient sample size (Group 1: {len(g1)}, Group 2: {len(g2)}). Need at least 3 per group."}
            
        # Check for constant values (Zero Variance)
        if g1.nunique() <= 1 and g2.nunique() <= 1:
             return {"error": "Both groups have constant values (zero variance). Statistical testing is impossible."}
        
        # 1. Normality Check
        is_normal = True
        for g, name in zip([g1, g2], groups):
            n = len(g)
            
            # Handle constant group for normality check
            if g.nunique() <= 1:
                is_normal = False
                reasons.append(f"Group '{name}' is constant (non-normal).")
                continue
                
            try:
                skewness = stats.skew(g)
                if n < 50:
                    stat, p_norm = stats.shapiro(g)
                    test_name = "Shapiro-Wilk"
                else:
                    stat, p_norm = stats.normaltest(g)
                    test_name = "D'Agostino-Pearson"
                
                # NEJM Logic: Significant p AND substantial skew required to reject normality
                if p_norm < 0.05 and abs(skewness) > 1:
                    is_normal = False
                    reasons.append(f"Group '{name}' is non-normal ({test_name} p={p_norm:.3f}, Skew={skewness:.2f})")
            except Exception as e:
                is_normal = False
                reasons.append(f"Could not test normality for '{name}': {str(e)}")
        
        # 2. Variance Check (Brown-Forsythe)
        try:
            stat_var, p_var = stats.levene(g1, g2, center='median')
            if p_var < 0.05:
                reasons.append(f"Unequal variances (Brown-Forsythe p={p_var:.3f})")
        except:
            pass # Levene might fail with constant inputs, ignore
        
        # 3. Select Test
        try:
            if not is_normal:
                # Mann-Whitney U
                res = pg.mwu(g1, g2)
                test_name = "Mann-Whitney U Test"
                p_val = res['p-val'].values[0]
                eff_size = res['RBC'].values[0]
                eff_label = "Rank-Biserial Correlation"
                logic_reason = "Data is non-normal (p < 0.05 & |skew| > 1). Using non-parametric test."
            else:
                # Welch's T-Test (Always preferred over Student's T per modern guidelines)
                res = pg.ttest(g1, g2, correction=True)
                test_name = "Welch's T-Test"
                p_val = res['p-val'].values[0]
                eff_size = res['cohen-d'].values[0]
                eff_label = "Cohen's d"
                logic_reason = "Data is normal. Using Welch's T-Test (robust to unequal variance)."
    
            # Formatting
            p_fmt = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
            
            return {
                "Test Name": test_name,
                "P-value": p_fmt,
                "Numeric P": p_val,
                "Effect Size": f"{eff_size:.2f} ({eff_label})",
                "Logic Reason": logic_reason,
                "Detailed Reasons": reasons,
                "Raw Result": res
            }
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}

    def generate_table1(self, df, group_col, study_design='Observational'):
        """
        Generates a publication-ready Table 1 using the tableone library.
        
        Parameters:
        - study_design: 'RCT' (No p-val, SMD=True) or 'Observational' (P-val=True)
        """
        from tableone import TableOne
        
        # 1. Auto-Detect Variable Types & Normality
        columns = list(df.columns)
        categorical = []
        nonnormal = []
        
        for col in columns:
            if col == group_col: continue
            
            # Check if categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10:
                    categorical.append(col)
                else:
                    # Check normality for continuous vars
                    stat, p = stats.shapiro(df[col].dropna())
                    if p < 0.05:
                        nonnormal.append(col)
            else:
                categorical.append(col)
        
        # 2. Configure Table Settings based on Design
        if study_design == 'RCT':
            pval = False
            smd = True
        else: # Observational
            pval = True
            smd = False
            
        # 3. Generate Table
        mytable = TableOne(
            df, 
            columns=columns, 
            categorical=categorical, 
            groupby=group_col, 
            nonnormal=nonnormal, 
            pval=pval, 
            smd=smd
        )
        
        return mytable

    def run_cox_model(self, df, time_col, event_col, covariates):
        """
        Runs Cox Proportional Hazards Model and checks assumptions.
        """
        from lifelines import CoxPHFitter
        
        cph = CoxPHFitter()
        
        # Subset data
        data = df[[time_col, event_col] + covariates].dropna()
        
        # Fit model
        cph.fit(data, duration_col=time_col, event_col=event_col)
        
        # Check Assumptions
        try:
            cph.check_assumptions(data, show_plots=False)
            assumption_status = "✅ Proportional Hazards Assumption Met"
        except:
            assumption_status = "⚠️ Proportional Hazards Assumption Violated"
            
        return cph, assumption_status

    def run_safe_ml(self, df, target_col, feature_cols, model_type='classification'):
        """
        Runs Random Forest with 5-Fold Cross-Validation and SHAP explanation.
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import shap
        
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if model_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'roc_auc'
            metric_name = "AUC"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
            metric_name = "R²"
            
        # 1. Cross-Validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_result = f"{scores.mean():.3f} ± {scores.std():.3f}"
        
        # 2. Train Final Model for SHAP
        model.fit(X, y)
        
        # 3. SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle SHAP for classification (returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Positive class
            
        return model, cv_result, metric_name, shap_values, X

class ResearchCopilot:
    """
    Active AI Assistant that provides context-aware guidance throughout the research workflow.
    """
    
    def __init__(self, df=None, gemini_model=None):
        self.df = df
        self.gemini_model = gemini_model
        self.state = self._detect_state()
    
    def _detect_state(self):
        """Detect current research state based on available data."""
        if self.df is None:
            return "awaiting_data"
        elif self.df is not None:
            return "data_loaded"
        else:
            return "analysis_complete"
    
    def get_research_suggestions(self):
        """
        Use AI to suggest 3 likely research questions based on dataframe columns.
        Returns list of dictionaries with 'question' and 'suggested_test' keys.
        """
        if self.df is None or self.gemini_model is None:
            return []
        
        # Prepare column info
        column_info = []
        for col in self.df.columns[:10]:  # Limit to first 10 columns
            dtype = str(self.df[col].dtype)
            nunique = self.df[col].nunique()
            column_info.append(f"- {col} ({dtype}, {nunique} unique values)")
        
        columns_text = "\n".join(column_info)
        
        prompt = f"""You are a biostatistics research assistant. A medical student has uploaded a dataset with these columns:

{columns_text}

Suggest 3 specific, actionable research questions that would be appropriate for this data. For each question:
1. State the research question
2. Identify which statistical test would be most appropriate
3. Keep it concise (1-2 sentences per question)

Format your response EXACTLY as:
Q1: [question]
Test: [test name]

Q2: [question]
Test: [test name]

Q3: [question]
Test: [test name]"""

        try:
            response = self.gemini_model.generate_content(prompt)
            suggestions = self._parse_suggestions(response.text)
            return suggestions
        except Exception as e:
            return [{"question": f"Error generating suggestions: {str(e)}", "suggested_test": "Manual Selection"}]
    
    def _parse_suggestions(self, response_text):
        """Parse AI response into structured suggestions."""
        suggestions = []
        lines = response_text.strip().split('\n')
        
        current_q = None
        for line in lines:
            line = line.strip()
            if line.startswith('Q'):
                current_q = line.split(':', 1)[1].strip() if ':' in line else line
            elif line.startswith('Test:') and current_q:
                test = line.split(':', 1)[1].strip() if ':' in line else "Manual Selection"
                suggestions.append({"question": current_q, "suggested_test": test})
                current_q = None
        
        # Ensure we return at least something
        if not suggestions:
            suggestions = [
                {"question": "Compare groups across continuous outcome", "suggested_test": "T-Test or ANOVA"},
                {"question": "Test association between categorical variables", "suggested_test": "Chi-Square Test"},
                {"question": "Predict binary outcome from predictors", "suggested_test": "Logistic Regression"}
            ]
        
        return suggestions[:3]  # Return max 3 suggestions
    
    def suggest_test(self, independent_var, dependent_var, df):
        """
        Suggest appropriate statistical test based on variable types.
        Returns (test_name, reason)
        """
        # Use LogicEngine without circular import
        engine = LogicEngine()
        
        ind_type = engine.detect_variable_type(df[independent_var])
        dep_type = engine.detect_variable_type(df[dependent_var])
        
        # Continuous outcome
        if "Continuous" in dep_type or "Ordinal" in dep_type:
            if df[independent_var].nunique() == 2:
                return "Welch's T-Test", "Comparing continuous outcome between 2 groups"
            elif df[independent_var].nunique() > 2:
                return "One-way ANOVA", "Comparing continuous outcome across 3+ groups"
        
        # Binary/Categorical outcome
        elif "Categorical" in dep_type:
            if "Continuous" in ind_type:
                return "Logistic Regression", "Predicting categorical outcome from continuous predictor"
            elif "Categorical" in ind_type:
                return "Chi-Square Test", "Testing association between categorical variables"
        
        return "Correlation or Regression", "Complex relationship - consider correlation or regression analysis"

