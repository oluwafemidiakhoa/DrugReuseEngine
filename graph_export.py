"""
Graph Export Module for the Neo4j Graph Explorer.
Provides functionality to export graph data to various formats.
"""

import streamlit as st
import pandas as pd
import networkx as nx
import json
import io
import plotly.graph_objects as go
import base64
from PIL import Image
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm

def export_graph_to_json(_G, filename="graph_data.json"):
    """
    Export the graph to a JSON file
    
    Parameters:
    - _G: NetworkX graph object
    - filename: Name of the output file
    
    Returns:
    - JSON string representation of the graph
    """
    data = {
        'nodes': [],
        'edges': []
    }
    
    # Add nodes
    for node_id, attrs in _G.nodes(data=True):
        node_data = {
            'id': str(node_id),
            'type': attrs.get('type', ''),
            'name': attrs.get('name', str(node_id)),
        }
        
        # Add any other attributes
        for key, value in attrs.items():
            if key not in ['id', 'type', 'name']:
                # Convert non-serializable objects to strings
                try:
                    json.dumps({key: value})
                    node_data[key] = value
                except (TypeError, OverflowError):
                    node_data[key] = str(value)
        
        data['nodes'].append(node_data)
    
    # Add edges
    for source, target, attrs in _G.edges(data=True):
        edge_data = {
            'source': str(source),
            'target': str(target),
            'type': attrs.get('type', ''),
        }
        
        # Add any other attributes
        for key, value in attrs.items():
            if key not in ['source', 'target', 'type']:
                # Convert non-serializable objects to strings
                try:
                    json.dumps({key: value})
                    edge_data[key] = value
                except (TypeError, OverflowError):
                    edge_data[key] = str(value)
        
        data['edges'].append(edge_data)
    
    # Convert to JSON
    json_str = json.dumps(data, indent=2)
    
    # Create download link
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return json_str, href

def export_graph_to_csv(_G, prefix="graph_data"):
    """
    Export the graph to CSV files (nodes.csv and edges.csv)
    
    Parameters:
    - _G: NetworkX graph object
    - prefix: Prefix for the output files
    
    Returns:
    - Tuple of (nodes_df, edges_df, nodes_href, edges_href)
    """
    # Create nodes dataframe
    nodes_data = []
    for node_id, attrs in _G.nodes(data=True):
        node_data = {
            'id': node_id,
            'type': attrs.get('type', ''),
            'name': attrs.get('name', str(node_id)),
        }
        
        # Add any other attributes
        for key, value in attrs.items():
            if key not in ['id', 'type', 'name']:
                try:
                    # Check if value is JSON serializable (as a simple check for complex objects)
                    json.dumps({key: value})
                    node_data[key] = value
                except (TypeError, OverflowError):
                    node_data[key] = str(value)
        
        nodes_data.append(node_data)
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Create edges dataframe
    edges_data = []
    for source, target, attrs in _G.edges(data=True):
        edge_data = {
            'source': source,
            'target': target,
            'type': attrs.get('type', ''),
        }
        
        # Add any other attributes
        for key, value in attrs.items():
            if key not in ['source', 'target', 'type']:
                try:
                    json.dumps({key: value})
                    edge_data[key] = value
                except (TypeError, OverflowError):
                    edge_data[key] = str(value)
        
        edges_data.append(edge_data)
    
    edges_df = pd.DataFrame(edges_data)
    
    # Create download links
    nodes_csv = nodes_df.to_csv(index=False)
    edges_csv = edges_df.to_csv(index=False)
    
    nodes_b64 = base64.b64encode(nodes_csv.encode()).decode()
    edges_b64 = base64.b64encode(edges_csv.encode()).decode()
    
    nodes_href = f'<a href="data:file/csv;base64,{nodes_b64}" download="{prefix}_nodes.csv">Download {prefix}_nodes.csv</a>'
    edges_href = f'<a href="data:file/csv;base64,{edges_b64}" download="{prefix}_edges.csv">Download {prefix}_edges.csv</a>'
    
    return nodes_df, edges_df, nodes_href, edges_href

def export_graph_to_graphml(_G, filename="graph_data.graphml"):
    """
    Export the graph to GraphML format
    
    Parameters:
    - _G: NetworkX graph object
    - filename: Name of the output file
    
    Returns:
    - Download link for the GraphML file
    """
    # Convert all attributes to strings (GraphML requires this)
    G_copy = _G.copy()
    for node, attrs in G_copy.nodes(data=True):
        for key, value in list(attrs.items()):
            if not isinstance(value, (str, int, float, bool)) or isinstance(value, bool):
                G_copy.nodes[node][key] = str(value)
    
    for source, target, attrs in G_copy.edges(data=True):
        for key, value in list(attrs.items()):
            if not isinstance(value, (str, int, float, bool)) or isinstance(value, bool):
                G_copy[source][target][key] = str(value)
    
    # Create a temporary file to write the GraphML
    graphml_data = io.BytesIO()
    nx.write_graphml(G_copy, graphml_data)
    
    # Create download link
    b64 = base64.b64encode(graphml_data.getvalue()).decode()
    href = f'<a href="data:application/xml;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_graph_visualization(_G, filename="graph_visualization.png", width=1200, height=800):
    """
    Export a visualization of the graph as an image
    
    Parameters:
    - _G: NetworkX graph object
    - filename: Name of the output file
    - width: Width of the image in pixels
    - height: Height of the image in pixels
    
    Returns:
    - Download link for the image
    """
    # Create a new figure with specified size
    plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Create layout for the graph
    pos = nx.spring_layout(_G, seed=42)
    
    # Define node colors based on node type
    node_colors = []
    for node in _G.nodes():
        node_type = _G.nodes[node].get('type', '')
        if node_type == 'drug':
            node_colors.append('skyblue')
        elif node_type == 'disease':
            node_colors.append('salmon')
        elif node_type == 'gene':
            node_colors.append('lightgreen')
        elif node_type == 'protein':
            node_colors.append('violet')
        else:
            node_colors.append('gray')
    
    # Define edge colors based on relationship type
    edge_colors = []
    for source, target, data in _G.edges(data=True):
        rel_type = data.get('type', '')
        if rel_type == 'treats':
            edge_colors.append('green')
        elif rel_type == 'interacts_with':
            edge_colors.append('blue')
        elif rel_type == 'targets':
            edge_colors.append('red')
        elif rel_type == 'associated_with':
            edge_colors.append('purple')
        else:
            edge_colors.append('gray')
    
    # Draw the graph
    nx.draw_networkx_nodes(_G, pos, node_color=node_colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(_G, pos, edge_color=edge_colors, width=1, alpha=0.5)
    
    # Add labels to nodes (only if not too many)
    if len(_G.nodes()) <= 50:
        labels = {node: _G.nodes[node].get('name', str(node)) for node in _G.nodes()}
        nx.draw_networkx_labels(_G, pos, labels, font_size=8, font_family='sans-serif')
    
    # Create a legend (simplified)
    node_types = ['drug', 'disease', 'gene', 'protein', 'other']
    node_colors = ['skyblue', 'salmon', 'lightgreen', 'violet', 'gray']
    
    plt.figlegend(
        [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in node_colors],
        node_types,
        loc='upper right',
        frameon=True
    )
    
    # Remove axis
    plt.axis('off')
    
    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Create download link
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_plotly_visualization(fig, filename="visualization.html"):
    """
    Export a Plotly figure as an HTML file
    
    Parameters:
    - fig: Plotly figure object
    - filename: Name of the output file
    
    Returns:
    - Download link for the HTML file
    """
    # Convert figure to HTML
    html_str = fig.to_html(include_plotlyjs=True, full_html=True)
    
    # Create download link
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_community_data(_G, communities, filename="community_data.csv"):
    """
    Export community detection results to CSV
    
    Parameters:
    - _G: NetworkX graph object
    - communities: Dictionary mapping node IDs to community IDs
    - filename: Name of the output file
    
    Returns:
    - Download link for the CSV file
    """
    # Create dataframe
    data = []
    for node_id, community_id in communities.items():
        if node_id in _G.nodes():
            node_data = {
                'node_id': node_id,
                'node_name': _G.nodes[node_id].get('name', str(node_id)),
                'node_type': _G.nodes[node_id].get('type', ''),
                'community_id': community_id
            }
            data.append(node_data)
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Create download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return df, href

def export_recommendation_results(recommendations_df, filename="recommendations.csv"):
    """
    Export recommendation results to CSV
    
    Parameters:
    - recommendations_df: DataFrame with recommendation results
    - filename: Name of the output file
    
    Returns:
    - Download link for the CSV file
    """
    # Convert to CSV
    csv = recommendations_df.to_csv(index=False)
    
    # Create download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_cypher_results(results_df, filename="query_results.csv"):
    """
    Export Cypher query results to CSV
    
    Parameters:
    - results_df: DataFrame with query results
    - filename: Name of the output file
    
    Returns:
    - Download link for the CSV file
    """
    # Convert to CSV
    csv = results_df.to_csv(index=False)
    
    # Create download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_time_series_data(metrics_df, filename="time_series.csv"):
    """
    Export time series data to CSV
    
    Parameters:
    - metrics_df: DataFrame with time series metrics
    - filename: Name of the output file
    
    Returns:
    - Download link for the CSV file
    """
    # Convert to CSV
    csv = metrics_df.to_csv(index=False)
    
    # Create download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return href

def export_network_metrics(_G, filename="network_metrics.csv"):
    """
    Export network metrics for all nodes to CSV
    
    Parameters:
    - _G: NetworkX graph object
    - filename: Name of the output file
    
    Returns:
    - Download link for the CSV file
    """
    # Create an undirected copy for some metrics
    G_undirected = _G.to_undirected() if _G.is_directed() else _G
    
    # Calculate metrics
    degree_centrality = nx.degree_centrality(_G)
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G_undirected)
    except:
        betweenness_centrality = {node: 0.0 for node in _G.nodes()}
    
    try:
        closeness_centrality = nx.closeness_centrality(G_undirected)
    except:
        closeness_centrality = {node: 0.0 for node in _G.nodes()}
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected)
    except:
        eigenvector_centrality = {node: 0.0 for node in _G.nodes()}
    
    try:
        clustering = nx.clustering(G_undirected)
    except:
        clustering = {node: 0.0 for node in _G.nodes()}
    
    # Combine metrics into a dataframe
    data = []
    for node in _G.nodes():
        node_data = {
            'node_id': node,
            'node_name': _G.nodes[node].get('name', str(node)),
            'node_type': _G.nodes[node].get('type', ''),
            'degree_centrality': degree_centrality.get(node, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node, 0.0),
            'closeness_centrality': closeness_centrality.get(node, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node, 0.0),
            'clustering_coefficient': clustering.get(node, 0.0)
        }
        data.append(node_data)
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Create download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    return df, href

def create_floating_export_button(download_links, title="Export Data"):
    """
    Create a floating export button with multiple download options
    
    Parameters:
    - download_links: Dictionary of download links with format {name: html}
    - title: Title for the dropdown menu
    
    Returns:
    - HTML string for the floating button
    """
    if not download_links:
        return ""
    
    # Create HTML for the dropdown menu
    dropdown_items = ""
    for name, html in download_links.items():
        dropdown_items += f'<div class="dropdown-item">{html}</div>'
    
    # Create the floating button HTML
    html = f"""
    <style>
    .floating-export {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }}
    .export-button {{
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
    }}
    .export-button:hover {{
        background-color: #45a049;
    }}
    .dropdown-content {{
        display: none;
        position: absolute;
        bottom: 50px;
        right: 0;
        background-color: #f9f9f9;
        min-width: 200px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        padding: 12px 16px;
        z-index: 1;
        border-radius: 5px;
    }}
    .dropdown-item {{
        padding: 8px 0;
        border-bottom: 1px solid #ddd;
    }}
    .dropdown-item:last-child {{
        border-bottom: none;
    }}
    .floating-export:hover .dropdown-content {{
        display: block;
    }}
    </style>
    
    <div class="floating-export">
        <button class="export-button">
            <span>Export Data</span>
            <span style="margin-left: 5px;">⬆️</span>
        </button>
        <div class="dropdown-content">
            <div style="font-weight: bold; margin-bottom: 8px;">{title}</div>
            {dropdown_items}
        </div>
    </div>
    """
    
    return html