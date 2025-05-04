import faiss
import numpy as np
from SQL import SQL
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(query, data, topk=5, threshold=None):
    """
    Compute cosine similarity between two tensors.
    """
    query = query / query.norm(dim=-1, keepdim=True)
    data = data / data.norm(dim=-1, keepdim=True)
    sim = (query @ data.T)
    print(sim.shape)
    most_similar_index = sim.argsort(dim=-1, descending=True)[:, :topk]
    most_similar_values = sim.gather(1, most_similar_index)
    
    # Filter out values below the threshold
    if threshold is not None:
        mask = most_similar_values > threshold
        most_similar_index = most_similar_index[mask]
        most_similar_values = most_similar_values[mask]
        
    most_similar_index = most_similar_index.cpu().numpy().squeeze()
    most_similar_values = most_similar_values.cpu().numpy().squeeze()

    return most_similar_index, most_similar_values


def faiss_similarity(query, data, topk=5, threshold=None):
    """
    Compute cosine similarity using FAISS.
    """
    query = query / query.norm(dim=-1, keepdim=True)
    data = data / data.norm(dim=-1, keepdim=True)
    
    # Convert to numpy arrays
    query_np = query.cpu().numpy()
    data_np = data.cpu().numpy()
    
    # Create FAISS index
    index = faiss.IndexFlatIP(data_np.shape[1])
    index.add(data_np)
    
    # Search for the topk nearest neighbors
    similarities, indices = index.search(query_np, topk)
    
    similarities = similarities.flatten()
    indices = indices.flatten()
    # Filter out values below the threshold
    if threshold is not None:
        mask = similarities > threshold
        indices = indices[mask]
        similarities = similarities[mask]
    return indices, similarities

def bm25_similarity(query, data, topk=5, threshold=0.25):
    """
    Compute BM25 similarity.
    """
    bm25 = BM25Okapi(data)
    scores = bm25.get_scores(query)
    
    # Get topk indices
    topk_indices = np.argsort(scores)[-topk:][::-1]
    topk_scores = scores[topk_indices]
    # Filter out values below the threshold
    if threshold is not None:
        mask = np.abs(topk_scores) > threshold
        topk_indices = topk_indices[mask]
        topk_scores = topk_scores[mask]
    return topk_indices, topk_scores

def tfidf_similarity(query, data, topk=5, threshold=0.25):
    """
    Compute TF-IDF similarity.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    
    # Transform the query
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get topk indices
    topk_indices = np.argsort(cosine_similarities)[-topk:][::-1]
    topk_scores = cosine_similarities[topk_indices]
    
    # Filter out values below the threshold
    if threshold is not None:
        mask = topk_scores > threshold
        topk_indices = topk_indices[mask]
        topk_scores = topk_scores[mask]
        
    return topk_indices, topk_scores


def ivfflat(query, data, topk=5, threshold=None):
    """
    Compute Ivfflat similarity.
    """
    data = data / data.norm(dim=-1, keepdim=True)
    query = query / query.norm(dim=-1, keepdim=True)
    
    # Convert to numpy arrays
    query_np = query.cpu().numpy().squeeze().flatten()
    data_np = data.cpu().numpy().squeeze()
    # Create FAISS index
    sql = SQL()
    sql.connect()
    # add embedding
    sql.add_vectors(data_np.T)
    # Create an IVFFLAT index
    topk_indices, topk_scores = sql.ivfflat(query_np, k=topk)
    topk_scores = 1 - topk_scores
    # Filter out values below the threshold
    if threshold is not None:
        mask = topk_scores > threshold
        topk_indices = topk_indices[mask]
        topk_scores = topk_scores[mask]
        
    return topk_indices, topk_scores

def hnsw(query, data, topk=5, threshold=None):
    """
    Compute HNSW similarity.
    """
    data = data / data.norm(dim=-1, keepdim=True)
    query = query / query.norm(dim=-1, keepdim=True)
    
    # Convert to numpy arrays
    query_np = query.cpu().numpy().squeeze().flatten()
    data_np = data.cpu().numpy().squeeze()
    # Create FAISS index
    sql = SQL()
    sql.connect()
    # add embedding
    sql.add_vectors(data_np.T)
    # Create an HNSW index
    topk_indices, topk_scores = sql.hnsw(query_np, k=topk)
    
    # Filter out values below the threshold
    if threshold is not None:
        mask = topk_scores > threshold
        topk_indices = topk_indices[mask]
        topk_scores = topk_scores[mask]
    return topk_indices, topk_scores

        