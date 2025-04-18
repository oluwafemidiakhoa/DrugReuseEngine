"""
Community detection and graph analysis algorithms for the Drug Repurposing Engine.
These algorithms help identify clusters and patterns in the knowledge graph that may reveal
new drug repurposing opportunities.
"""

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

@st.cache_data(ttl=3600)
def detect_communities_louvain(_G):
    """
    Detect communities in the knowledge graph using the Louvain method.
    This algorithm finds communities by optimizing modularity in the graph.
    
    Parameters:
    - _G: NetworkX graph object
    
    Returns:
    - Dictionary mapping node IDs to community IDs
    - DataFrame with node data including community assignments
    """
    # Configure logging
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Import community directly - python-louvain package should provide this
        import community
        
        # Debug log what functions are available in the community module
        logger.info(f"Community module functions: {dir(community)}")
        
        # Set community_louvain to the community module
        community_louvain = community
        
        # Convert DiGraph to Graph if needed for the algorithm
        if isinstance(_G, nx.DiGraph):
            G_undirected = _G.to_undirected()
        else:
            G_undirected = _G
        
        if len(G_undirected) == 0:
            logger.warning("Empty graph provided to community detection algorithm")
            return {}, pd.DataFrame()
            
        # Detect communities - use the appropriate function name based on what's available
        if hasattr(community_louvain, 'best_partition'):
            communities = community_louvain.best_partition(G_undirected)
        elif hasattr(community_louvain, 'louvain_communities'):
            communities_list = community_louvain.louvain_communities(G_undirected)
            # Convert communities list to dictionary
            communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    communities[node] = i
        else:
            # Fall back to networkx's greedy modularity communities as a last resort
            communities_list = list(nx.algorithms.community.greedy_modularity_communities(G_undirected))
            communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    communities[node] = i
        
        # Create a DataFrame with node data and community assignments
        node_data = []
        for node_id, community_id in communities.items():
            node_attrs = _G.nodes[node_id]
            node_type = ""
            
            # Determine node type based on its labels if available
            if 'labels' in node_attrs:
                labels = node_attrs['labels']
                if isinstance(labels, list) and len(labels) > 0:
                    node_type = labels[0]
            elif 'type' in node_attrs:
                node_type = node_attrs['type']
            
            # Otherwise, try to infer from attributes
            if not node_type:
                if 'mechanism' in node_attrs:
                    node_type = 'Drug'
                elif 'category' in node_attrs:
                    node_type = 'Disease'
                elif 'symbol' in node_attrs:
                    node_type = 'Gene'
            
            node_data.append({
                'id': node_id,
                'name': node_attrs.get('name', ''),
                'type': node_type,
                'community_id': community_id
            })
        
        return communities, pd.DataFrame(node_data)
    
    except ImportError:
        # Fallback to other community detection method if python-louvain is not available
        st.warning("The python-louvain package is required for Louvain community detection. Using spectral clustering as fallback.")
        return detect_communities_spectral(_G)
    except Exception as e:
        logger.error(f"Error detecting communities using Louvain method: {str(e)}")
        st.error(f"Error detecting communities using Louvain method. Trying spectral clustering.")
        return detect_communities_spectral(_G)

@st.cache_data(ttl=3600)
def detect_communities_spectral(_G, n_clusters=8):
    """
    Detect communities in the knowledge graph using spectral clustering
    
    Parameters:
    - _G: NetworkX graph object
    - n_clusters: Number of clusters to detect (defaults to 8)
    
    Returns:
    - Dictionary mapping node IDs to community IDs
    - DataFrame with node data including community assignments
    """
    # Configure logging
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Convert DiGraph to Graph if needed for the algorithm
        if isinstance(_G, nx.DiGraph):
            G_undirected = _G.to_undirected()
        else:
            G_undirected = _G
        
        if len(G_undirected) == 0:
            logger.warning("Empty graph provided to spectral clustering algorithm")
            return {}, pd.DataFrame()
            
        # Determine a sensible number of clusters if not specified
        if n_clusters is None or n_clusters <= 0:
            # Set n_clusters to be at most 1/3 of the nodes, but at least 2 and at most 12
            suggested_clusters = max(2, min(len(G_undirected) // 3, 12))
            logger.info(f"Setting n_clusters to {suggested_clusters} based on graph size")
            n_clusters = suggested_clusters
            
        # Get the adjacency matrix
        A = nx.adjacency_matrix(G_undirected)
        
        # Spectral clustering
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        affinity='precomputed',
                                        assign_labels='kmeans',
                                        random_state=42)
        
        # Fit and predict
        labels = clustering.fit_predict(A.toarray())
        
        # Create a dictionary mapping node IDs to community IDs
        communities = {}
        for i, node_id in enumerate(G_undirected.nodes()):
            communities[node_id] = int(labels[i])
        
        # Create a DataFrame with node data and community assignments
        node_data = []
        for node_id, community_id in communities.items():
            node_attrs = _G.nodes[node_id]
            node_type = ""
            
            # Determine node type based on its labels if available
            if 'labels' in node_attrs:
                labels = node_attrs['labels']
                if isinstance(labels, list) and len(labels) > 0:
                    node_type = labels[0]
            elif 'type' in node_attrs:
                node_type = node_attrs['type']
            
            # Otherwise, try to infer from attributes
            if not node_type:
                if 'mechanism' in node_attrs:
                    node_type = 'Drug'
                elif 'category' in node_attrs:
                    node_type = 'Disease'
                elif 'symbol' in node_attrs:
                    node_type = 'Gene'
            
            node_data.append({
                'id': node_id,
                'name': node_attrs.get('name', ''),
                'type': node_type,
                'community_id': community_id
            })
        
        return communities, pd.DataFrame(node_data)
    
    except Exception as e:
        logger.error(f"Error detecting communities using spectral clustering: {str(e)}")
        # Return empty data
        return {}, pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_centrality_metrics(_G):
    """
    Calculate various centrality metrics for nodes in the graph
    
    Parameters:
    - _G: NetworkX graph object
    
    Returns:
    - DataFrame with node data and centrality metrics
    """
    # Convert DiGraph to Graph if needed for some algorithms
    if isinstance(_G, nx.DiGraph):
        G_undirected = _G.to_undirected()
    else:
        G_undirected = _G
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(_G)
    closeness_centrality = nx.closeness_centrality(G_undirected)
    
    # Try to calculate eigenvector centrality, but it can fail for some graphs
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected)
    except:
        eigenvector_centrality = {node: 0.0 for node in _G.nodes()}
    
    # Try to calculate betweenness centrality
    try:
        betweenness_centrality = nx.betweenness_centrality(G_undirected)
    except:
        betweenness_centrality = {node: 0.0 for node in _G.nodes()}
    
    # Create a DataFrame with node data and centrality metrics
    node_data = []
    for node_id in _G.nodes():
        node_attrs = _G.nodes[node_id]
        node_data.append({
            'id': node_id,
            'name': node_attrs.get('name', ''),
            'type': node_attrs.get('type', ''),
            'degree_centrality': degree_centrality.get(node_id, 0.0),
            'closeness_centrality': closeness_centrality.get(node_id, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node_id, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node_id, 0.0)
        })
    
    return pd.DataFrame(node_data)

def visualize_communities(_G, communities, node_data_df):
    """
    Create a visualization of communities in the graph
    
    Parameters:
    - _G: NetworkX graph object
    - communities: Dictionary mapping node IDs to community IDs
    - node_data_df: DataFrame with node data
    
    Returns:
    - Plotly figure object
    """
    # Create a graph with position layout
    pos = nx.spring_layout(_G, seed=42)
    
    # Create an edge trace
    edge_x = []
    edge_y = []
    for edge in _G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create a node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in _G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node info
        node_attrs = _G.nodes[node]
        node_name = node_attrs.get('name', node)
        node_type = node_attrs.get('type', '')
        community_id = communities.get(node, 0)
        
        node_text.append(f"Name: {node_name}<br>Type: {node_type}<br>Community: {community_id}")
        node_color.append(community_id)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Community',
                xanchor='left'
            )
        ))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title=dict(
                          text='Community Detection Visualization',
                          font=dict(size=16)
                      ),
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                  )
    
    return fig

def visualize_community_distribution(node_data_df):
    """
    Create a visualization of community distribution by node type
    
    Parameters:
    - node_data_df: DataFrame with node data including community assignments
    
    Returns:
    - Plotly figure object
    """
    # Group by community_id and type, count nodes
    community_counts = node_data_df.groupby(['community_id', 'type']).size().reset_index(name='count')
    
    # Create a grouped bar chart
    fig = px.bar(community_counts, 
                x='community_id', 
                y='count', 
                color='type',
                title='Node Types by Community',
                labels={'community_id': 'Community ID', 'count': 'Number of Nodes', 'type': 'Node Type'})
    
    return fig

def generate_community_insights(node_data_df):
    """
    Generate insights about communities in the graph
    
    Parameters:
    - node_data_df: DataFrame with node data including community assignments
    
    Returns:
    - Dictionary with insights about communities
    """
    insights = {}
    
    # Get number of communities
    num_communities = node_data_df['community_id'].nunique()
    insights['num_communities'] = num_communities
    
    # Get community sizes
    community_sizes = node_data_df.groupby('community_id').size().reset_index(name='size')
    insights['community_sizes'] = community_sizes.to_dict('records')
    
    # Get node type distribution by community
    community_type_dist = node_data_df.groupby(['community_id', 'type']).size().reset_index(name='count')
    community_type_dist = community_type_dist.pivot(index='community_id', columns='type', values='count').fillna(0)
    insights['community_type_distribution'] = community_type_dist.to_dict('index')
    
    # Find the dominant node type in each community
    dominant_types = {}
    for community_id in node_data_df['community_id'].unique():
        community_df = node_data_df[node_data_df['community_id'] == community_id]
        type_counts = community_df['type'].value_counts()
        if len(type_counts) > 0:
            dominant_types[community_id] = type_counts.index[0]
        else:
            dominant_types[community_id] = 'unknown'
    
    insights['dominant_types'] = dominant_types
    
    # Generate narrative insights
    narrative = []
    
    narrative.append(f"The knowledge graph contains {num_communities} distinct communities.")
    
    # Identify the largest community
    largest_community_id = community_sizes.iloc[community_sizes['size'].argmax()]['community_id']
    largest_community_size = community_sizes.iloc[community_sizes['size'].argmax()]['size']
    narrative.append(f"The largest community (ID: {largest_community_id}) contains {largest_community_size} nodes.")
    
    # Identify drug-rich communities
    drug_rich_communities = []
    for community_id, type_dist in insights['community_type_distribution'].items():
        if type_dist.get('drug', 0) > 5:  # Communities with more than 5 drugs
            drug_rich_communities.append((community_id, type_dist.get('drug', 0)))
    
    if drug_rich_communities:
        drug_rich_ids = [str(c[0]) for c in drug_rich_communities]
        narrative.append(f"Communities with high drug concentration: {', '.join(drug_rich_ids)}")
    
    # Identify disease-rich communities
    disease_rich_communities = []
    for community_id, type_dist in insights['community_type_distribution'].items():
        if type_dist.get('disease', 0) > 5:  # Communities with more than 5 diseases
            disease_rich_communities.append((community_id, type_dist.get('disease', 0)))
    
    if disease_rich_communities:
        disease_rich_ids = [str(c[0]) for c in disease_rich_communities]
        narrative.append(f"Communities with high disease concentration: {', '.join(disease_rich_ids)}")
    
    insights['narrative'] = narrative
    
    return insights

def find_similar_drugs(_G, drug_id, method='graph_structure', top_n=10):
    """
    Find drugs similar to a given drug based on different methods
    
    Parameters:
    - _G: NetworkX graph object
    - drug_id: ID of the drug to find similar drugs for
    - method: Method to use for finding similar drugs ('graph_structure', 'common_neighbors', 'community')
    - top_n: Number of similar drugs to return
    
    Returns:
    - DataFrame with similar drugs and similarity scores
    """
    if method == 'graph_structure':
        # Calculate structural similarity based on common neighbors, weighted by edge confidence
        similar_drugs = []
        
        # Get all drugs in the graph
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug' and n != drug_id]
        
        # Get all neighbors of the target drug
        target_neighbors = set(_G.neighbors(drug_id))
        
        # Calculate Jaccard similarity with each other drug
        for other_drug in drugs:
            other_neighbors = set(_G.neighbors(other_drug))
            
            # Calculate Jaccard similarity
            union = len(target_neighbors.union(other_neighbors))
            intersection = len(target_neighbors.intersection(other_neighbors))
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0
            
            drug_data = _G.nodes[other_drug]
            similar_drugs.append({
                'drug_id': other_drug,
                'drug_name': drug_data.get('name', other_drug),
                'similarity': similarity
            })
        
        # Sort by similarity and take top_n
        similar_drugs.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similar_drugs[:top_n])
    
    elif method == 'common_neighbors':
        # Find drugs with common neighbors
        similar_drugs = []
        
        # Get all drugs in the graph
        drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug' and n != drug_id]
        
        # Get all neighbors of the target drug
        target_neighbors = set(_G.neighbors(drug_id))
        
        # Calculate number of common neighbors with each other drug
        for other_drug in drugs:
            other_neighbors = set(_G.neighbors(other_drug))
            
            # Count common neighbors
            common = len(target_neighbors.intersection(other_neighbors))
            
            drug_data = _G.nodes[other_drug]
            similar_drugs.append({
                'drug_id': other_drug,
                'drug_name': drug_data.get('name', other_drug),
                'common_neighbors': common,
                'similarity': common / max(1, len(target_neighbors))  # Normalized by target neighbors
            })
        
        # Sort by common neighbors and take top_n
        similar_drugs.sort(key=lambda x: x['common_neighbors'], reverse=True)
        return pd.DataFrame(similar_drugs[:top_n])
    
    elif method == 'community':
        # First, detect communities
        communities, node_data_df = detect_communities_louvain(_G)
        
        # Find the community of the target drug
        target_community = communities.get(drug_id, -1)
        
        if target_community == -1:
            # If community detection failed, fall back to graph structure method
            return find_similar_drugs(_G, drug_id, method='graph_structure', top_n=top_n)
        
        # Find other drugs in the same community
        similar_drugs = []
        
        for node_id, community_id in communities.items():
            if community_id == target_community and node_id != drug_id:
                node_data = _G.nodes[node_id]
                if node_data.get('type') == 'drug':
                    similar_drugs.append({
                        'drug_id': node_id,
                        'drug_name': node_data.get('name', node_id),
                        'community_id': community_id,
                        'similarity': 1.0  # All drugs in the same community have the same similarity
                    })
        
        # Sort alphabetically by name since they're all in the same community
        similar_drugs.sort(key=lambda x: x['drug_name'])
        return pd.DataFrame(similar_drugs[:top_n])
    
    else:
        raise ValueError(f"Unknown method: {method}")

def recommend_repurposing_candidates(_G, top_n=20):
    """
    Recommend new drug repurposing candidates based on community structure and network analysis.
    
    This advanced recommendation algorithm looks for drugs and diseases that:
    1. Are in the same community but don't have a direct relationship
    2. Have a high number of common neighbors (shared genes, pathways, etc.)
    3. Have similar connection patterns in the knowledge graph
    4. Show strong evidence from multiple analytical approaches
    
    Parameters:
    - _G: NetworkX graph object
    - top_n: Number of candidates to recommend
    
    Returns:
    - DataFrame with recommended repurposing candidates and scores
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # First, detect communities
        logger.info("Detecting communities using Louvain method")
        communities, node_data_df = detect_communities_louvain(_G)
        
        if not communities:
            logger.warning("Louvain community detection failed, falling back to spectral clustering")
            communities, node_data_df = detect_communities_spectral(_G)
            
        if not communities:
            logger.error("All community detection methods failed")
            return pd.DataFrame()
            
        # Find all drugs and diseases - handle different node attribute structures
        logger.info("Finding drugs and diseases in the graph")
        drugs = []
        diseases = []
        
        for node, attrs in _G.nodes(data=True):
            node_type = ""
            
            # Try different ways of determining node type
            if 'type' in attrs:
                node_type = attrs['type'].lower()
            elif 'labels' in attrs:
                labels = attrs['labels']
                if isinstance(labels, list) and len(labels) > 0:
                    node_type = labels[0].lower()
            else:
                # Try to infer from attributes
                if 'mechanism' in attrs or 'targets' in attrs:
                    node_type = 'drug'
                elif 'category' in attrs or 'indication' in attrs:
                    node_type = 'disease'
                elif 'symbol' in attrs or 'protein' in attrs:
                    node_type = 'gene'
            
            if node_type == 'drug':
                drugs.append(node)
            elif node_type == 'disease':
                diseases.append(node)
        
        logger.info(f"Found {len(drugs)} drugs and {len(diseases)} diseases in the graph")
        
        if len(drugs) == 0 or len(diseases) == 0:
            logger.warning("No drugs or diseases found in the graph")
            return pd.DataFrame()
        
        # Check which drug-disease pairs already have relationships
        logger.info("Identifying existing drug-disease relationships")
        existing_relationships = set()
        for drug_id in drugs:
            for disease_id in diseases:
                if _G.has_edge(drug_id, disease_id) or _G.has_edge(disease_id, drug_id):
                    existing_relationships.add((drug_id, disease_id))
        
        logger.info(f"Found {len(existing_relationships)} existing drug-disease relationships")
        
        # Generate candidate pairs
        candidates = []
        
        # Calculate the maximum possible path length for normalization
        max_path_length = 10  # Default
        try:
            if nx.is_connected(_G):
                max_path_length = nx.diameter(_G)
        except:
            # If we can't calculate diameter, stick with default
            pass
        
        for drug_id in drugs:
            drug_community = communities.get(drug_id, -1)
            drug_neighbors = set(_G.neighbors(drug_id))
            
            for disease_id in diseases:
                # Skip if there's already a relationship
                if (drug_id, disease_id) in existing_relationships:
                    continue
                
                disease_community = communities.get(disease_id, -1)
                disease_neighbors = set(_G.neighbors(disease_id))
                
                # Calculate features for scoring
                same_community = (drug_community == disease_community) and (drug_community != -1)
                common_neighbors = len(drug_neighbors.intersection(disease_neighbors))
                
                # Try to find shortest path length
                try:
                    path_length = nx.shortest_path_length(_G, drug_id, disease_id)
                except:
                    path_length = float('inf')
                
                # Calculate community conductance
                # (measure of how well-connected the drug and disease are within their communities)
                conductance = 0.0
                if same_community:
                    community_nodes = [n for n, c in communities.items() if c == drug_community]
                    community_subgraph = _G.subgraph(community_nodes)
                    
                    drug_connections_within = sum(1 for n in community_subgraph.neighbors(drug_id))
                    disease_connections_within = sum(1 for n in community_subgraph.neighbors(disease_id))
                    
                    if drug_connections_within > 0 and disease_connections_within > 0:
                        conductance = (drug_connections_within + disease_connections_within) / (2 * len(community_nodes))
                
                # Calculate a composite score with weights optimized for biological relevance
                score = 0.0
                
                # Being in the same community is a strong signal (40% of the score)
                if same_community:
                    score += 0.4
                
                # Common neighbors boost the score (30% of the score, capped)
                # More emphasis on shared biological mechanisms
                score += min(0.3, common_neighbors * 0.05)
                
                # Short paths boost the score (20% of the score)
                # Normalized by the max path length to make it more meaningful
                if path_length < float('inf'):
                    normalized_path_score = max(0, 0.2 - ((path_length / max_path_length) * 0.2))
                    score += normalized_path_score
                
                # Community conductance boost (10% of the score)
                score += conductance * 0.1
                
                # Get node data - handle potentially missing attributes with fallbacks
                drug_data = _G.nodes[drug_id]
                disease_data = _G.nodes[disease_id]
                
                drug_name = drug_data.get('name', drug_id)
                if not drug_name or drug_name == drug_id:
                    # Try other attribute names that might contain the name
                    for attr in ['title', 'label', 'display_name']:
                        if attr in drug_data and drug_data[attr]:
                            drug_name = drug_data[attr]
                            break
                
                disease_name = disease_data.get('name', disease_id)
                if not disease_name or disease_name == disease_id:
                    # Try other attribute names that might contain the name
                    for attr in ['title', 'label', 'display_name']:
                        if attr in disease_data and disease_data[attr]:
                            disease_name = disease_data[attr]
                            break
                
                # Add to candidates if score is meaningful (>0.2 to focus on strong candidates)
                if score > 0.2:
                    candidates.append({
                        'drug_id': drug_id,
                        'drug_name': drug_name,
                        'disease_id': disease_id,
                        'disease_name': disease_name,
                        'score': round(score, 3),
                        'same_community': same_community,
                        'common_neighbors': common_neighbors,
                        'path_length': path_length if path_length < float('inf') else None,
                        'conductance': round(conductance, 3)
                    })
        
        # Sort by score in descending order
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Generated {len(candidates)} repurposing candidates, returning top {top_n}")
        
        # Return top N candidates
        return pd.DataFrame(candidates[:top_n])
        
    except Exception as e:
        logger.error(f"Error in repurposing candidate recommendation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()