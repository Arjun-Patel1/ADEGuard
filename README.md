# ADEGuard
Adverse Drug Event (ADE)
# ⚡ ADEGuard: Adverse Event Explorer & Severity Predictor

ADEGuard is an interactive tool for exploring and analyzing **1.2M+ VAERS reports** (Vaccine Adverse Event Reporting System).  
It combines **semantic search, PCA-based visualization, and ML severity prediction** into a single Streamlit dashboard.

---

## 🚀 Features
- **Dataset Explorer** → Preview and filter VAERS reports  
- **PCA Visualization** → 2D projection of SBERT-MiniLM embeddings  
- **Semantic Search** → Find similar reports using transformer embeddings  
- **Severity Prediction** → Logistic Regression with TF-IDF features (ROC-AUC = **0.94**)  
- **Weak Supervision & Age-Aware Clustering** for richer analysis  

---

## 📊 Tech Stack
- **Python, Pandas, NumPy, Scikit-learn, Scipy**
- **SentenceTransformers (SBERT-MiniLM)**
- **Streamlit + Plotly** for interactive UI
- **Joblib** for model persistence

---


## 📦 Installation
```bash
git clone https://github.com/yourusername/ADEGuard.git
cd ADEGuard
pip install -r requirements.txt


streamlit run app/adeguard_app.py
📑 Dataset

VAERS public dataset (1.2M+ reports)

Due to size limits, only a sample (10k rows) is included here.

Full dataset (~278 MB) available at: VAERS Official Site

🔮 Example Prediction

Input: headache
Output: Prediction: Not Serious (probability 0.04)

🏆 Results

ROC-AUC: 0.94 (balanced logistic regression with TF-IDF)

Scaled to 1.2M VAERS reports using sparse matrices and embeddings
```
