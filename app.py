import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import pandas as pd

# ---- Setup de la page ----
st.set_page_config(layout="wide")
st.title("🔎 Test d'embeddings de phrases")

# ---- Saisie utilisateur ----
st.subheader("📝 Phrases à analyser")
input_text = st.text_area("Entrez une phrase par ligne :", 
    "Le ciel est bleu.\nJ'aime le chocolat.\nLe chat dort.\nLa voiture roule vite.")

phrases = [p.strip() for p in input_text.strip().split("\n") if p.strip()]
if not phrases:
    st.warning("Aucune phrase détectée.")
    st.stop()

st.write("✅ Phrases chargées :", len(phrases))

# ---- Chargement du modèle + embeddings ----
st.subheader("📐 Génération des embeddings")

with st.spinner("Téléchargement du modèle et encodage des phrases..."):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("✅ Modèle chargé.")
    embeddings = model.encode(phrases)
    st.write("✅ Embeddings générés.")

# ---- Analyse simple ----
embeddings = np.array(embeddings)
norms = np.linalg.norm(embeddings, axis=1)
sparsity = np.mean(np.count_nonzero(embeddings, axis=1) / embeddings.shape[1])

col1, col2, col3 = st.columns(3)
col1.metric("🔢 Dimensions", embeddings.shape[1])
col2.metric("📏 Norme moyenne", f"{norms.mean():.2f}")
col3.metric("⚖️ Densité moyenne", f"{sparsity * 100:.1f}%")

# ---- PCA projection ----
st.subheader("🌀 Projection 3D (PCA)")
with st.spinner("Calcul de la projection PCA..."):
    pca = PCA(n_components=3)
    projection = pca.fit_transform(embeddings)

df = pd.DataFrame(projection, columns=["x", "y", "z"])
df["phrase"] = phrases

fig = px.scatter_3d(df, x="x", y="y", z="z", text="phrase",
                    title="Nuage de points 3D des phrases",
                    color=phrases)
fig.update_traces(marker=dict(size=6), textposition="top center")

st.plotly_chart(fig, use_container_width=True)
