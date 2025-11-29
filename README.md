# Rbiostatitics - Medical Biostatistics App

**A zero-code biostatistics platform for medical students with advanced features: auto-switching tests, natural language interface, and AI-powered analysis.**

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1: Push to GitHub
```bash
cd /path/to/Rbiostatitics
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `Rbiostatitics`
5. Main file path: `app.py`
6. Click "Advanced settings" → "Secrets"
   ```toml
   # .streamlit/secrets.toml
   GENAI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   ```
8. Click "Deploy!"

### Step 3: Share with Supervisor
Your app will be live at: `https://[your-app-name].streamlit.app`

---

## ✨ Features

### High-Stakes Research Architecture
- **Variable Setup**: Auto-detect & confirm variable types (prevents errors)
- **Research Question Builder**: Natural language test selection
- **Auto-Switching Tests**: Automatically uses non-parametric tests when assumptions violated
- **Bold P-Values**: Impossible to miss
- **Academic Citations**: Ready for Methods sections

### Statistical Tests (All with Auto-Switching)
- T-Test → Mann-Whitney U (if not normal)
- ANOVA → Kruskal-Wallis (if assumptions violated)
- Chi-Square Test
- Pearson Correlation

### Advanced Features
- **AI Chatbot**: Ask questions in plain English
- **Meta-Analysis**: Forest plots with heterogeneity
- **Machine Learning**: Random Forest (zero-code)
- **Table 1 Generator**: Demographics tables
- **Visualization**: Interactive Plotly charts

### Medical Integrity
- ✅ **Deterministic calculations** (Scipy/Pingouin, never AI)
- ✅ **Code transparency** (view exact code used)
- ✅ **Academic citations** (peer-reviewed library references)
- ✅ **Logic validation** (AI pre-checks test appropriateness)

---

## 📊 Usage Workflow

1. **Upload CSV** → Data Upload page
2. **Confirm variables** → Variable Setup (CRITICAL)
3. **Ask research question** → Research Question Builder
4. **Run test** → App auto-checks assumptions & switches if needed
5. **View bold p-value** → Impossible to misinterpret
6. **Copy Methods text** → Paste in paper

---

## 🔧 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:
```toml
GENAI_API_KEY = "your-key-here"
```

---

## 📚 Technology Stack

- **Backend**: Python 3.11, Streamlit
- **Statistics**: Scipy, Pingouin, Statsmodels
- **AI**: Google Gemini API
- **ML**: Scikit-learn
- **Viz**: Plotly, Matplotlib

---

## 🎓 Academic Use

This tool is designed for **medical students** with **zero coding experience**. It provides a user-friendly interface to industry-standard statistical libraries (same math as SPSS/SAS).

**For your Methods section**: The app auto-generates proper citations referencing the peer-reviewed libraries (Scipy, Pingouin), not the app itself.

---

## 📄 License

MIT License - Free for academic use

---

## 🤝 Citation

If you use this tool in your research, please cite the underlying libraries:
- Vallat, R. (2018). Pingouin: statistics in Python. *Journal of Open Source Software*, 3(31), 1026.
- Virtanen, P., et al. (2020). SciPy 1.0. *Nature Methods*, 17(3), 261-272.
