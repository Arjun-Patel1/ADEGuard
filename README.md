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

I wasn't able to upload files due to file size limit, so they are uploaded on google drives.
cleaned data- https://drive.google.com/file/d/19XTgUeWQuD0IvSwS75xNe6QJae0xiIEj/view?usp=sharing
target.npy- https://drive.google.com/file/d/1XTS5ZBr5l4KJhADnG5bDeqIEwPdaKaNu/view?usp=sharing
tdidf_vectoriser.pkl- https://drive.google.com/file/d/1CEDaYrqeeJD8o14YSIhGB2918mTZ_-ON/view?usp=sharing
logreg_model.pkl- https://drive.google.com/file/d/1XaM7cGsTN8Tntg-o8cjHn3tRB4S7SCJy/view?usp=sharing
feature_sparse.npz- https://drive.google.com/file/d/1FF33EWX2ku26-usQPtCgr8TziIdR7npE/view?usp=sharing
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

OR
Download csv data file and run notebooks one by pne then run using streamlit.

🔮 Example Prediction

Input: headache
Output: Prediction: Not Serious (probability 0.04)

🏆 Results

ROC-AUC: 0.94 (balanced logistic regression with TF-IDF)

Scaled to 1.2M VAERS reports using sparse matrices and embeddings
```
