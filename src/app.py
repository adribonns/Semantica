import numpy as np
import streamlit as st
import plotly.express as px
from utils import load_model, generate_embeddings, pca_projection, tsne_projection, apply_kmeans_clustering, create_projection_dataframe

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
    model = load_model(model_name)
    st.write("✅ Model loaded.")
    embeddings = generate_embeddings(model, sentences)
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
        projection = pca_projection(embeddings)

    df = create_projection_dataframe(projection, sentences, apply_kmeans_clustering(embeddings))
    st.markdown("<h3 style='text-align: center;'>3D Sentence Embedding Projection</h3>", unsafe_allow_html=True)
    fig_3d = px.scatter_3d(df, x="x", y="y", z="z", text="sentence", hover_name="sentence", color="cluster")
    fig_3d.update_traces(marker=dict(size=6), textposition="top center")
    fig_3d.update_layout(margin=dict(t=50, b=50, l=50, r=50), title="3D Sentence Embedding Projection", title_x=0.5, title_y=0.95)
    st.plotly_chart(fig_3d, use_container_width=True)

# ---- 2D t-SNE Projection ----
elif projection_type == "2D t-SNE":
    st.subheader("🌀 2D Projection (t-SNE)")
    with st.spinner("Calculating t-SNE projection..."):
        projection = tsne_projection(embeddings)

    df_tsne = create_projection_dataframe(projection, sentences, apply_kmeans_clustering(embeddings))
    st.markdown("<h3 style='text-align: center;'>2D Sentence Embedding (t-SNE)</h3>", unsafe_allow_html=True)
    fig_2d = px.scatter(df_tsne, x="x", y="y", text="sentence", color="cluster")
    fig_2d.update_traces(marker=dict(size=6), textposition="top center")
    fig_2d.update_layout(margin=dict(t=50, b=50, l=50, r=50), title="2D Sentence Embedding (t-SNE)", title_x=0.5, title_y=0.95)
    st.plotly_chart(fig_2d, use_container_width=True)

# ---- Optional: Export Results (CSV) ----
csv = df.to_csv(index=False).encode('utf-8') if projection_type == "3D PCA" else df_tsne.to_csv(index=False).encode('utf-8')
st.download_button("📁 Download Data", csv, "projection.csv", "text/csv")
