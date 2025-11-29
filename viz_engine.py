import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

class VizEngine:
    """
    The Face of the Operation.
    Handles all publication-ready plotting and visualization.
    Enforces NEJM-style aesthetics.
    """
    
    def __init__(self):
        # Set default style settings if needed
        pass

    def create_box_plot(self, df, x, y, color=None, title="", **kwargs):
        """
        Generates a publication-quality box plot.
        """
        fig = px.box(
            df, x=x, y=y, color=color,
            title=title,
            template="simple_white", # Clean, academic look
            points="all", # Show all points for transparency
            **kwargs
        )
        fig.update_layout(
            font_family="Arial",
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16
        )
        return fig

    def create_scatter_plot(self, df, x, y, color=None, title="", **kwargs):
        """
        Generates a publication-quality scatter plot.
        """
        fig = px.scatter(
            df, x=x, y=y, color=color,
            title=title,
            template="simple_white",
            trendline="ols", # Add regression line automatically
            **kwargs
        )
        return fig

    def create_heatmap(self, corr_matrix, title="Correlation Matrix", **kwargs):
        """
        Generates a correlation heatmap.
        """
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title=title,
            template="simple_white",
            **kwargs
        )
        return fig

    def create_bar_plot(self, df, x, y, title="", orientation='v', **kwargs):
        """
        Generates a publication-quality bar plot.
        """
        fig = px.bar(
            df, x=x, y=y,
            title=title,
            template="simple_white",
            orientation=orientation,
            **kwargs
        )
        return fig

    def create_forest_plot(self, df, pooled_or, pooled_lo, pooled_up):
        """
        Generates a forest plot for meta-analysis using Matplotlib.
        Returns the figure object.
        """
        import numpy as np # Ensure numpy is available
        fig, ax = plt.subplots(figsize=(8, len(df) * 1))
        
        # Studies
        y_pos = np.arange(len(df))
        ax.errorbar(df['OR'], y_pos, xerr=[df['OR'] - df['Lower CI'], df['Upper CI'] - df['OR']], 
                    fmt='o', color='black', ecolor='black', capsize=5, label='Studies')
        
        # Pooled
        ax.errorbar(pooled_or, -1, xerr=[[pooled_or - pooled_lo], [pooled_up - pooled_or]], 
                    fmt='D', color='red', ecolor='red', capsize=5, label='Pooled')
        
        # Labels
        ax.set_yticks(np.append(y_pos, -1))
        ax.set_yticklabels(list(df['Study Name']) + ['Pooled'])
        ax.axvline(1, linestyle='--', color='gray')
        ax.set_xlabel("Odds Ratio (95% CI)")
        ax.set_title("Forest Plot")
        
        return fig

    def plot_survival(self, df, time_col, event_col, group_col):
        """
        Generates a NEJM-style Kaplan-Meier survival plot with risk table.
        """
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        from lifelines.plotting import add_at_risk_counts
        
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        
        groups = df[group_col].unique()
        fitters = []
        
        # Fit KMF for each group
        for group in groups:
            mask = df[group_col] == group
            kmf.fit(df[mask][time_col], df[mask][event_col], label=str(group))
            kmf.plot_survival_function(ax=ax, ci_show=True)
            fitters.append(kmf)
            
        # Log-Rank Test
        if len(groups) == 2:
            results = logrank_test(
                df[df[group_col] == groups[0]][time_col],
                df[df[group_col] == groups[1]][time_col],
                event_observed_A=df[df[group_col] == groups[0]][event_col],
                event_observed_B=df[df[group_col] == groups[1]][event_col]
            )
            p_val = results.p_value
            title_suffix = f"(Log-Rank p={p_val:.4f})"
        else:
            title_suffix = ""
        
        # Formatting
        ax.set_title(f"Kaplan-Meier Survival Curve {title_suffix}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.grid(True, alpha=0.3)
        
        # Add Risk Table (NEJM Requirement)
        add_at_risk_counts(*fitters, ax=ax)
        
        return fig

    def plot_shap_summary(self, shap_values, X):
        """
        Generates a SHAP Summary Plot for model interpretation.
        """
        import shap
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False, plot_type="dot")
        plt.tight_layout()
        
        return fig
