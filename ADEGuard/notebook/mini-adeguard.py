import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
import re
import joblib
import os

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, "vaers_data_30k.csv")
    emb_path = os.path.join(base_path, "mini-sbert_minilm_embeddings_split.npy")

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    try:
        embeddings = np.load(emb_path)
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        st.stop()

    return df, embeddings

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    vec_path = os.path.join(base_path, "mini-tfidf_vectorizer.pkl")
    model_path = os.path.join(base_path, "mini-logreg_model.pkl")

    try:
        vectorizer = joblib.load(vec_path)
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    return vectorizer, model

df, embeddings = load_data()
vectorizer, model = load_model()

# Clean vaccine columns for display
df["VAX_TYPE_STR"] = df["VAX_TYPE"].apply(lambda x: ", ".join(eval(x)) if isinstance(x, str) else str(x))
df["VAX_MANU_STR"] = df["VAX_MANU"].apply(lambda x: ", ".join(eval(x)) if isinstance(x, str) else str(x))

# Streamlit page config
st.set_page_config(page_title="ADEGuard Explorer", layout="wide")
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Dataset Preview", "PCA Visualization", "Semantic Search", "Severity Prediction"])

if page == "Home":
    st.title("🚀 ADEGuard Explorer")
    st.markdown("""
    Welcome to **ADEGuard** — your interactive VAERS adverse event explorer.  

    **Features:**
    - Preview dataset  
    - Visualize embeddings  
    - Semantic search  
    - Predict severity of ADEs  
    """)
    col1, col2 = st.columns(2)
    col1.metric("Dataset Rows", df.shape[0])
    col2.metric("Embedding Dimensions", embeddings.shape[1])
    st.info("Use the sidebar to navigate.")

elif page == "Dataset Preview":
    st.title("📊 Dataset Preview")
    rows = st.slider("Rows to display:", 5, 50, 10)
    st.dataframe(df.head(rows), use_container_width=True)

elif page == "PCA Visualization":
    st.title("📉 PCA Projection")
    num_points = st.slider("Points to visualize:", 500, min(embeddings.shape[0], 10000), 2000)
    pca = PCA(n_components=2)
    emb2d = pca.fit_transform(embeddings[:num_points])

    hover_text = df["symptoms_normalized"].head(num_points).apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x
    )
    fig = px.scatter(
        x=emb2d[:, 0],
        y=emb2d[:, 1],
        hover_data={
            "VAERS_ID": df["VAERS_ID"].head(num_points),
            "Age": df["AGE_YRS"].head(num_points),
            "Sex": df["SEX"].head(num_points),
            "Symptoms": hover_text
        },
        labels={"x": "PCA-1", "y": "PCA-2"},
        opacity=0.6,
        color=df["AGE_YRS"].head(num_points).astype(str),
        title="2D PCA of SBERT Embeddings"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Semantic Search":
    st.title("🔍 Semantic Search")
    query = st.text_input("Enter your search query:")
    if query:
        st.info("Finding top matching reports...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = sbert.encode([query], convert_to_numpy=True)
        cos_scores = util.cos_sim(query_emb, embeddings[:200000])[0]

        topk = st.slider("Top K results:", 1, 20, 5)
        top_indices = cos_scores.argsort(descending=True)[:topk]

        results_df = df.iloc[top_indices.numpy()][[
            "VAERS_ID", "AGE_YRS", "SEX", "VAX_TYPE_STR", "VAX_MANU_STR", 
            "VAX_DOSE_SERIES", "symptoms_normalized"
        ]].copy()

        pattern = re.compile("|".join(re.escape(w) for w in query.lower().split()), re.IGNORECASE)
        results_df["Symptoms_Highlight"] = results_df["symptoms_normalized"].apply(
            lambda x: pattern.sub(lambda m: f"**{m.group(0)}**", str(x))
        )
        results_df["Symptoms_Display"] = results_df["Symptoms_Highlight"].apply(
            lambda x: x[:200] + "..." if len(x) > 200 else x
        )

        st.subheader("Top Matching Reports")
        for idx, row in results_df.iterrows():
            with st.expander(f"VAERS_ID: {row['VAERS_ID']} | Age: {row['AGE_YRS']} | Sex: {row['SEX']}"):
                st.markdown(f"""
                **Vaccine Type:** {row['VAX_TYPE_STR']}  
                **Manufacturer:** {row['VAX_MANU_STR']}  
                **Dose Series:** {row['VAX_DOSE_SERIES']}  
                **Symptoms:** {row['Symptoms_Highlight']}
                """)

elif page == "Severity Prediction":
    st.title("⚠️ Predict ADE Severity")
    st.markdown("This uses your trained logistic regression model with TF-IDF features.")

    text_input = st.text_area("Enter symptom text:")
    if st.button("Predict"):
        if text_input.strip():
            X_input = vectorizer.transform([text_input])  # TF-IDF transform
            prob = model.predict_proba(X_input)[0][1]
            pred = "Serious" if prob > 0.5 else "Not Serious"
            st.success(f"Prediction: **{pred}** (probability {prob:.2f})")
        else:
            st.warning("Please enter symptoms to predict.")
