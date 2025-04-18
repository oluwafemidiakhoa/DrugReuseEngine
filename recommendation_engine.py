"""
Advanced recommendation engine for the Drug Repurposing Engine
This module provides functions to generate drug repurposing recommendations
based on knowledge graph patterns, similarity metrics, and community structure.
"""

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from community_detection import detect_communities_louvain

@st.cache_data(ttl=3600)
def generate_graph_embeddings(_G, dimensions=64):
    """
    Generate embeddings for graph nodes using node2vec
    
    Parameters:
    - _G: NetworkX graph object
    - dimensions: Number of dimensions for embeddings
    
    Returns:
    - Dictionary mapping node IDs to embeddings
    """
    try:
        from node2vec import Node2Vec
        
        # Create a Node2Vec model
        node2vec = Node2Vec(_G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1)
        
        # Get embeddings for all nodes
        embeddings = {}
        for node in _G.nodes():
            try:
                embeddings[node] = model.wv.get_vector(node)
            except KeyError:
                # If a node doesn't have an embedding, use a zero vector
                embeddings[node] = np.zeros(dimensions)
        
        return embeddings
    
    except ImportError:
        # If node2vec is not available, use a simple alternative
        return generate_simple_embeddings(_G, dimensions)

def generate_simple_embeddings(_G, dimensions=64):
    """
    Generate simple node embeddings using adjacency matrix and SVD
    
    Parameters:
    - _G: NetworkX graph object
    - dimensions: Number of dimensions for embeddings
    
    Returns:
    - Dictionary mapping node IDs to embeddings
    """
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    
    # Get adjacency matrix
    A = nx.to_scipy_sparse_array(_G)
    
    # Apply SVD
    svd = TruncatedSVD(n_components=dimensions, random_state=42)
    embeddings_matrix = svd.fit_transform(A)
    
    # Create a dictionary mapping node IDs to embeddings
    embeddings = {}
    for i, node in enumerate(_G.nodes()):
        embeddings[node] = embeddings_matrix[i]
    
    return embeddings

@st.cache_data(ttl=3600)
def recommend_drugs_for_disease(_G, disease_id, method='graph_structure', top_n=10):
    """
    Recommend drugs for a specific disease
    
    Parameters:
    - _G: NetworkX graph object
    - disease_id: ID of the disease to recommend drugs for
    - method: Method to use for recommendations ('graph_structure', 'embedding', 'path_based')
    - top_n: Number of drugs to recommend
    
    Returns:
    - DataFrame with recommended drugs and scores
    """
    if method == 'graph_structure':
        # Find drugs that have similar connection patterns to drugs known to treat similar diseases
        recommendations = []
        
        # Get all drugs and diseases
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug']
        diseases = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'disease']
        
        # Find diseases similar to the target disease
        similar_diseases = []
        target_neighbors = set(_G.neighbors(disease_id))
        
        for other_disease in diseases:
            if other_disease == disease_id:
                continue
                
            other_neighbors = set(_G.neighbors(other_disease))
            
            # Calculate Jaccard similarity
            union = len(target_neighbors.union(other_neighbors))
            intersection = len(target_neighbors.intersection(other_neighbors))
            
            if union > 0:
                similarity = intersection / union
                similar_diseases.append((other_disease, similarity))
        
        # Sort and get top similar diseases
        similar_diseases.sort(key=lambda x: x[1], reverse=True)
        top_similar_diseases = similar_diseases[:5]
        
        # For each drug, calculate a recommendation score
        for drug_id in drugs:
            # Skip if there's already a direct relationship
            if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                continue
            
            # Calculate features for scoring
            common_neighbors = len(set(_G.neighbors(drug_id)).intersection(target_neighbors))
            
            # Check if the drug treats any similar diseases
            treats_similar = 0
            for similar_disease, similarity in top_similar_diseases:
                if _G.has_edge(drug_id, similar_disease) and _G.get_edge_data(drug_id, similar_disease).get('type') == 'treats':
                    treats_similar += similarity
            
            # Try to find shortest path length
            try:
                path_length = nx.shortest_path_length(_G, drug_id, disease_id)
            except:
                path_length = float('inf')
            
            # Calculate a score
            score = min(0.3, common_neighbors * 0.05)  # Common neighbors boost the score (max 0.3)
            score += treats_similar * 0.5  # Treating similar diseases is a strong signal
            if path_length < float('inf'):
                score += max(0, 0.2 - (path_length * 0.05))  # Short paths boost the score
            
            drug_data = _G.nodes[drug_id]
            
            # Add to recommendations if score is meaningful
            if score > 0:
                recommendations.append({
                    'drug_id': drug_id,
                    'drug_name': drug_data.get('name', drug_id),
                    'common_neighbors': common_neighbors,
                    'treats_similar_diseases': treats_similar > 0,
                    'path_length': path_length if path_length < float('inf') else None,
                    'score': score
                })
        
        # Sort recommendations by score and return top_n
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return pd.DataFrame(recommendations[:top_n])
    
    elif method == 'embedding':
        # Use node embeddings for recommendation
        embeddings = generate_graph_embeddings(_G)
        
        disease_embedding = embeddings.get(disease_id)
        if disease_embedding is None:
            # Fall back to graph structure if embedding fails
            return recommend_drugs_for_disease(_G, disease_id, method='graph_structure', top_n=top_n)
        
        # Get all drugs
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug']
        
        # Calculate cosine similarity with each drug
        similarities = []
        for drug_id in drugs:
            # Skip if there's already a direct relationship
            if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                continue
            
            drug_embedding = embeddings.get(drug_id)
            if drug_embedding is None:
                continue
                
            # Calculate cosine similarity
            sim = cosine_similarity([disease_embedding], [drug_embedding])[0][0]
            
            drug_data = _G.nodes[drug_id]
            similarities.append({
                'drug_id': drug_id,
                'drug_name': drug_data.get('name', drug_id),
                'similarity': sim,
                'score': (sim + 1) / 2  # Transform from [-1, 1] to [0, 1]
            })
        
        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similarities[:top_n])
    
    elif method == 'path_based':
        # Recommend based on meaningful paths in the graph
        recommendations = []
        
        # Get all drugs
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug']
        
        # For each drug, find paths to the disease and calculate scores
        for drug_id in drugs:
            # Skip if there's already a direct relationship
            if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                continue
            
            # Find all simple paths up to length 3
            paths = []
            try:
                for path in nx.all_simple_paths(_G, drug_id, disease_id, cutoff=3):
                    paths.append(path)
            except:
                continue
            
            if not paths:
                continue
                
            # Calculate a score based on paths
            path_scores = []
            for path in paths:
                # Calculate path score based on path length and edge types
                path_length = len(path) - 1
                path_score = 1.0 / path_length
                
                # Check edge types along the path
                for i in range(path_length):
                    edge_data = _G.get_edge_data(path[i], path[i+1])
                    edge_type = edge_data.get('type', '')
                    edge_confidence = edge_data.get('confidence', 0.5)
                    
                    # Boost score for meaningful relationship types
                    if edge_type in ['targets', 'interacts_with', 'regulates', 'expressed_in']:
                        path_score *= 1.2
                    
                    # Adjust by confidence
                    path_score *= edge_confidence
                
                path_scores.append(path_score)
            
            # Use the maximum path score
            max_score = max(path_scores)
            best_path_idx = path_scores.index(max_score)
            best_path = paths[best_path_idx]
            
            drug_data = _G.nodes[drug_id]
            
            recommendations.append({
                'drug_id': drug_id,
                'drug_name': drug_data.get('name', drug_id),
                'best_path_length': len(best_path) - 1,
                'best_path': ' -> '.join([_G.nodes[n].get('name', n) for n in best_path]),
                'score': max_score
            })
        
        # Sort by score and return top_n
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return pd.DataFrame(recommendations[:top_n])
    
    else:
        raise ValueError(f"Unknown method: {method}")

@st.cache_data(ttl=3600)
def recommend_diseases_for_drug(_G, drug_id, method='graph_structure', top_n=10):
    """
    Recommend diseases for a specific drug
    
    Parameters:
    - _G: NetworkX graph object
    - drug_id: ID of the drug to recommend diseases for
    - method: Method to use for recommendations ('graph_structure', 'embedding', 'path_based')
    - top_n: Number of diseases to recommend
    
    Returns:
    - DataFrame with recommended diseases and scores
    """
    if method == 'graph_structure':
        # Find diseases that have similar connection patterns to diseases treated by similar drugs
        recommendations = []
        
        # Get all drugs and diseases
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug']
        diseases = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'disease']
        
        # Find drugs similar to the target drug
        similar_drugs = []
        target_neighbors = set(_G.neighbors(drug_id))
        
        for other_drug in drugs:
            if other_drug == drug_id:
                continue
                
            other_neighbors = set(_G.neighbors(other_drug))
            
            # Calculate Jaccard similarity
            union = len(target_neighbors.union(other_neighbors))
            intersection = len(target_neighbors.intersection(other_neighbors))
            
            if union > 0:
                similarity = intersection / union
                similar_drugs.append((other_drug, similarity))
        
        # Sort and get top similar drugs
        similar_drugs.sort(key=lambda x: x[1], reverse=True)
        top_similar_drugs = similar_drugs[:5]
        
        # For each disease, calculate a recommendation score
        for disease_id in diseases:
            # Skip if there's already a direct relationship
            if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                continue
            
            # Calculate features for scoring
            common_neighbors = len(set(_G.neighbors(disease_id)).intersection(target_neighbors))
            
            # Check if any similar drugs treat this disease
            treated_by_similar = 0
            for similar_drug, similarity in top_similar_drugs:
                if _G.has_edge(similar_drug, disease_id) and _G.get_edge_data(similar_drug, disease_id).get('type') == 'treats':
                    treated_by_similar += similarity
            
            # Try to find shortest path length
            try:
                path_length = nx.shortest_path_length(_G, drug_id, disease_id)
            except:
                path_length = float('inf')
            
            # Calculate a score
            score = min(0.3, common_neighbors * 0.05)  # Common neighbors boost the score (max 0.3)
            score += treated_by_similar * 0.5  # Being treated by similar drugs is a strong signal
            if path_length < float('inf'):
                score += max(0, 0.2 - (path_length * 0.05))  # Short paths boost the score
            
            disease_data = _G.nodes[disease_id]
            
            # Add to recommendations if score is meaningful
            if score > 0:
                recommendations.append({
                    'disease_id': disease_id,
                    'disease_name': disease_data.get('name', disease_id),
                    'common_neighbors': common_neighbors,
                    'treated_by_similar_drugs': treated_by_similar > 0,
                    'path_length': path_length if path_length < float('inf') else None,
                    'score': score
                })
        
        # Sort recommendations by score and return top_n
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return pd.DataFrame(recommendations[:top_n])
    
    elif method == 'embedding':
        # Use node embeddings for recommendation
        embeddings = generate_graph_embeddings(_G)
        
        drug_embedding = embeddings.get(drug_id)
        if drug_embedding is None:
            # Fall back to graph structure if embedding fails
            return recommend_diseases_for_drug(_G, drug_id, method='graph_structure', top_n=top_n)
        
        # Get all diseases
        diseases = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'disease']
        
        # Calculate cosine similarity with each disease
        similarities = []
        for disease_id in diseases:
            # Skip if there's already a direct relationship
            if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                continue
            
            disease_embedding = embeddings.get(disease_id)
            if disease_embedding is None:
                continue
                
            # Calculate cosine similarity
            sim = cosine_similarity([drug_embedding], [disease_embedding])[0][0]
            
            disease_data = _G.nodes[disease_id]
            similarities.append({
                'disease_id': disease_id,
                'disease_name': disease_data.get('name', disease_id),
                'similarity': sim,
                'score': (sim + 1) / 2  # Transform from [-1, 1] to [0, 1]
            })
        
        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similarities[:top_n])
    
    elif method == 'path_based':
        # This is identical to path_based method in recommend_drugs_for_disease but with drug and disease roles swapped
        return recommend_drugs_for_disease(_G, drug_id, method='path_based', top_n=top_n)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def visualize_recommendations(recommendations_df, score_col='score', name_col=None, title='Recommendations'):
    """
    Create a visualization of recommendations
    
    Parameters:
    - recommendations_df: DataFrame with recommendations
    - score_col: Name of the column containing scores
    - name_col: Name of the column containing names to display (if None, uses index)
    - title: Title for the visualization
    
    Returns:
    - Plotly figure object
    """
    if recommendations_df.empty:
        return None
    
    # Create a bar chart
    if name_col:
        fig = px.bar(recommendations_df, 
                    x=name_col, 
                    y=score_col,
                    title=title,
                    labels={score_col: 'Score', name_col: ''},
                    color=score_col,
                    color_continuous_scale='Viridis')
    else:
        fig = px.bar(recommendations_df, 
                    y=score_col,
                    title=title,
                    labels={score_col: 'Score'},
                    color=score_col,
                    color_continuous_scale='Viridis')
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def generate_recommendation_insights(recommendations_df, entity_type='drug'):
    """
    Generate insights about recommendations
    
    Parameters:
    - recommendations_df: DataFrame with recommendations
    - entity_type: Type of entity being recommended ('drug' or 'disease')
    
    Returns:
    - Dictionary with insights
    """
    if recommendations_df.empty:
        return {"narrative": ["No recommendations available."]}
    
    insights = {}
    
    # Generate narrative insights
    narrative = []
    
    # Top recommendations
    top_n = min(3, len(recommendations_df))
    top_names = []
    
    if entity_type == 'drug':
        name_col = 'drug_name'
    else:
        name_col = 'disease_name'
    
    if name_col in recommendations_df.columns:
        for i in range(top_n):
            top_names.append(recommendations_df.iloc[i][name_col])
        
        narrative.append(f"Top {entity_type} recommendations: {', '.join(top_names)}")
    
    # Path-based insights
    if 'path_length' in recommendations_df.columns:
        avg_path_length = recommendations_df['path_length'].mean()
        narrative.append(f"Average path length: {avg_path_length:.2f}")
    
    # Common neighbors insights
    if 'common_neighbors' in recommendations_df.columns:
        avg_common_neighbors = recommendations_df['common_neighbors'].mean()
        narrative.append(f"Average common neighbors: {avg_common_neighbors:.2f}")
    
    # Network-based insights
    if entity_type == 'drug' and 'treated_by_similar_drugs' in recommendations_df.columns:
        similar_drug_count = recommendations_df['treated_by_similar_drugs'].sum()
        if similar_drug_count > 0:
            narrative.append(f"{similar_drug_count} diseases are already treated by similar drugs.")
    
    elif entity_type == 'disease' and 'treats_similar_diseases' in recommendations_df.columns:
        similar_disease_count = recommendations_df['treats_similar_diseases'].sum()
        if similar_disease_count > 0:
            narrative.append(f"{similar_disease_count} drugs already treat similar diseases.")
    
    insights['narrative'] = narrative
    
    return insights