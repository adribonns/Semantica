from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Charge un modèle SentenceTransformer à partir du nom fourni.
    """
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(model, sentences):
    """
    Génére les embeddings à partir d'un modèle et de phrases.
    """
    embeddings = model.encode(sentences)
    return embeddings


def pca_projection(embeddings, n_components=3):
    """
    Applique la réduction de dimensions PCA pour obtenir une projection 3D.
    """
    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(embeddings)
    return projection


def tsne_projection(embeddings, n_components=2, perplexity=30):
    """
    Applique la réduction de dimensions t-SNE pour obtenir une projection 2D.
    Ajuste automatiquement la perplexity si nécessaire.
    """
    perplexity = min(perplexity, len(embeddings) - 1)  # Perplexity doit être inférieure au nombre d'échantillons
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    projection = tsne.fit_transform(embeddings)
    return projection


def apply_kmeans_clustering(embeddings, n_clusters=3):
    """
    Applique le clustering KMeans sur les embeddings.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    return clusters


def create_projection_dataframe(projection, sentences, clusters=None):
    """
    Crée un DataFrame pour une projection donnée et les phrases.
    """
    #df = pd.DataFrame(projection, columns=["x", "y", "z" if len(projection[0]) > 2 else "y"])
    if len(projection[0]) > 2:
        df = pd.DataFrame(projection, columns=["x", "y", "z"])
    else:
        df = pd.DataFrame(projection, columns=["x", "y"])
    df["sentence"] = sentences
    if clusters is not None:
        df["cluster"] = clusters
    return df
