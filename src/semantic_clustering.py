from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence-transformer model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(comments, model):
    """
    Convert a list of comments into sentence embeddings.
    """
    return model.encode(comments, show_progress_bar=True)


def cluster_comments(embeddings, n_clusters: int = 5, random_state: int = 42):
    """
    Cluster embeddings using KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    return kmeans, cluster_ids


def add_cluster_labels(df: pd.DataFrame, cluster_ids, cluster_col: str = "semantic_cluster"):
    """
    Add cluster ids to a dataframe.
    """
    df = df.copy()
    df[cluster_col] = cluster_ids
    return df


def map_cluster_names(
    df: pd.DataFrame,
    cluster_name_map: dict,
    cluster_col: str = "semantic_cluster",
    topic_col: str = "semantic_topic"
):
    """
    Map numeric cluster ids to human-readable topic names.
    """
    df = df.copy()
    df[topic_col] = df[cluster_col].map(cluster_name_map)
    return df