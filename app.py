import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ---- Page Setup ----
st.set_page_config(layout="wide")
st.title("🔎 Sentence Embedding Test")

# ---- User Input ----
st.subheader("📝 Sentences to Analyze")
input_text = st.text_area("Enter one sentence per line:", 
    "The sky is blue.\nI love chocolate.\nThe cat is sleeping.\nThe car is speeding.")

sentences = [s.strip() for s in input_text.strip().split("\n") if s.strip()]
if not sentences:
    st.warning("No sentences detected.")
    st.stop()

st.write("✅ Sentences loaded:", len(sentences))

# ---- Model Loading + Embeddings ----
st.subheader("📐 Generating Embeddings")

# Model selection dropdown
model_name = st.selectbox(
    "Choose an embedding model",
    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-Mpnet-base-v2"]
)

with st.spinner("Downloading the model and encoding the sentences..."):
    model = SentenceTransformer(model_name)
    st.write("✅ Model loaded.")
    embeddings = model.encode(sentences)
    st.write("✅ Embeddings generated.")

# ---- Simple Analysis ----
embeddings = np.array(embeddings)
norms = np.linalg.norm(embeddings, axis=1)
sparsity = np.mean(np.count_nonzero(embeddings, axis=1) / embeddings.shape[1])

col1, col2, col3 = st.columns(3)
col1.metric("🔢 Dimensions", embeddings.shape[1])
col2.metric("📏 Average Norm", f"{norms.mean():.2f}")
col3.metric("⚖️ Average Sparsity", f"{sparsity * 100:.1f}%")

# ---- Projection Selection (3D PCA or 2D t-SNE) ----
projection_type = st.selectbox("Choose the projection type", ["3D PCA", "2D t-SNE"])

# ---- PCA Projection ----
if projection_type == "3D PCA":
    st.subheader("🌀 3D Projection (PCA)")
    with st.spinner("Calculating PCA projection..."):
        pca = PCA(n_components=3)
        projection = pca.fit_transform(embeddings)

    df = pd.DataFrame(projection, columns=["x", "y", "z"])
    df["sentence"] = sentences

    # Optional: Clustering with KMeans
    k = st.slider("Number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k)
    df["cluster"] = kmeans.fit_predict(embeddings)

    st.markdown("<h3 style='text-align: center;'>3D Sentence Embedding Projection</h3>", unsafe_allow_html=True)
    fig_3d = px.scatter_3d(df, x="x", y="y", z="z", text="sentence",
                           hover_name="sentence", color="cluster")
    fig_3d.update_traces(marker=dict(size=6), textposition="top center")
    fig_3d.update_layout(margin=dict(t=50, b=50, l=50, r=50), title="3D Sentence Embedding Projection",
                         title_x=0.5, title_y=0.95)

    st.plotly_chart(fig_3d, use_container_width=True)

# ---- 2D t-SNE Projection ----
elif projection_type == "2D t-SNE":
    st.subheader("🌀 2D Projection (t-SNE)")
    with st.spinner("Calculating t-SNE projection..."):
        # Dynamically adjust perplexity based on number of sentences
        perplexity = min(30, len(sentences) - 1)  # t-SNE perplexity must be less than the number of samples
        tsne = TSNE(n_components=2, perplexity=perplexity)
        tsne_projection = tsne.fit_transform(embeddings)

    df_tsne = pd.DataFrame(tsne_projection, columns=["x", "y"])
    df_tsne["sentence"] = sentences

    # Optional: Clustering with KMeans
    k = st.slider("Number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k)
    df_tsne["cluster"] = kmeans.fit_predict(embeddings)

    st.markdown("<h3 style='text-align: center;'>2D Sentence Embedding (t-SNE)</h3>", unsafe_allow_html=True)
    fig_2d = px.scatter(df_tsne, x="x", y="y", text="sentence", color="cluster")
    fig_2d.update_traces(marker=dict(size=6), textposition="top center")
    fig_2d.update_layout(margin=dict(t=50, b=50, l=50, r=50), title="2D Sentence Embedding (t-SNE)",
                         title_x=0.5, title_y=0.95)

    st.plotly_chart(fig_2d, use_container_width=True)

# ---- Optional: Export Results (CSV) ----
if projection_type == "3D PCA":
    csv = df.to_csv(index=False).encode('utf-8')
elif projection_type == "2D t-SNE":
    csv = df_tsne.to_csv(index=False).encode('utf-8')

st.download_button("📁 Download Data", csv, "projection.csv", "text/csv")
