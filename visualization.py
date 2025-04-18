import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import time
from plotly.subplots import make_subplots
import math

# Import the create_animated_data_flow function
from animated_flow import create_animated_data_flow

def create_network_graph(G, highlight_nodes=None, highlight_edges=None, layout='spring', node_color_by='type', edge_color_by='type', show_arrows=True):
    """
    Create an enhanced interactive network graph visualization using Plotly
    
    Parameters:
    - G: NetworkX graph object
    - highlight_nodes: List of node IDs to highlight
    - highlight_edges: List of (source, target) tuples to highlight
    - layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'community')
    - node_color_by: Attribute to color nodes by ('type', 'category', 'centrality')
    - edge_color_by: Attribute to color edges by ('type', 'confidence', 'weight')
    - show_arrows: Whether to show directional arrows on edges
    
    Returns:
    - Plotly figure object
    """
    if highlight_nodes is None:
        highlight_nodes = []
    if highlight_edges is None:
        highlight_edges = []
    
    # Calculate node positions based on selected layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'community':
        # First identify communities
        try:
            import community
            partition = community.best_partition(G.to_undirected())
            # Position by community
            pos = nx.spring_layout(G, seed=42)
            # Adjust positions to group by community
            for node, community_id in partition.items():
                angle = 2 * math.pi * community_id / len(set(partition.values()))
                pos[node] = (pos[node][0] + 0.5 * math.cos(angle), 
                             pos[node][1] + 0.5 * math.sin(angle))
        except ImportError:
            # Fall back to spring layout if community detection fails
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate centrality measures if needed
    if node_color_by == 'centrality':
        try:
            centrality = nx.betweenness_centrality(G)
        except:
            centrality = {node: 1.0 for node in G.nodes()}
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_symbols = []
    node_ids = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        
        # Node attributes
        attrs = G.nodes[node]
        node_type = attrs.get('type', 'unknown')
        node_name = attrs.get('name', 'Unknown')
        
        # Enhanced node text with more metadata
        if node_type == 'drug':
            description = attrs.get('description', 'No description')
            mechanism = attrs.get('mechanism', 'Unknown mechanism')
            indication = attrs.get('original_indication', 'Unknown indication')
            chemical_class = attrs.get('chemical_class', 'Unknown class')
            
            text = f"<b>{node_name}</b> (Drug)<br><br>{description}<br><br>Mechanism: {mechanism}<br>Chemical Class: {chemical_class}<br>Original Indication: {indication}"
            
            # Different symbols for different node types
            node_symbols.append('circle')
            
        elif node_type == 'disease':
            description = attrs.get('description', 'No description')
            category = attrs.get('category', 'Unknown category')
            systems = ", ".join(attrs.get('systems', ['unknown']))
            pathways = ", ".join(attrs.get('pathways', ['unknown']))
            
            text = f"<b>{node_name}</b> (Disease)<br><br>{description}<br><br>Category: {category}<br>Body Systems: {systems}<br>Pathways: {pathways}"
            
            node_symbols.append('diamond')
            
        elif node_type == 'gene':
            symbol = attrs.get('symbol', node_name)
            function = attrs.get('function', 'Unknown function')
            
            text = f"<b>{symbol}</b> (Gene)<br><br>Function: {function}"
            
            node_symbols.append('square')
            
        else:  # Other node types
            text = f"<b>{node_name}</b> ({node_type.capitalize()})<br><br>ID: {node}"
            node_symbols.append('triangle-up')
        
        node_text.append(text)
        
        # Node size - based on degree centrality
        degree = G.degree(node)
        if node in highlight_nodes:
            node_size.append(25 + degree)  # Larger for highlighted nodes
        else:
            node_size.append(15 + degree * 0.7)  # Size based on connections
        
        # Node color based on selected attribute
        if node in highlight_nodes:
            # Highlighted nodes get brighter colors
            if node_type == 'drug':
                node_color.append('rgba(255, 65, 54, 1)')  # Bright red for highlighted drugs
            elif node_type == 'disease':
                node_color.append('rgba(50, 168, 82, 1)')  # Bright green for highlighted diseases
            elif node_type == 'gene':
                node_color.append('rgba(66, 135, 245, 1)')  # Bright blue for highlighted genes
            else:
                node_color.append('rgba(255, 185, 15, 1)')  # Bright yellow for other highlighted nodes
        else:
            if node_color_by == 'type':
                if node_type == 'drug':
                    node_color.append('rgba(255, 65, 54, 0.8)')  # Red for drugs
                elif node_type == 'disease':
                    node_color.append('rgba(50, 168, 82, 0.8)')  # Green for diseases
                elif node_type == 'gene':
                    node_color.append('rgba(66, 135, 245, 0.8)')  # Blue for genes
                else:
                    node_color.append('rgba(255, 185, 15, 0.8)')  # Yellow for other nodes
            elif node_color_by == 'category' and 'category' in attrs:
                # Color by category - hash the category string to get a consistent color
                category = attrs.get('category', 'unknown')
                hue = abs(hash(category)) % 360
                node_color.append(f'hsl({hue}, 70%, 60%)')
            elif node_color_by == 'centrality':
                # Color by centrality - blue to red gradient
                value = centrality.get(node, 0.5)
                node_color.append(f'rgba({int(255*value)}, {int(100*(1-value))}, {int(200*(1-value))}, 0.8)')
            else:
                # Default color scheme
                if node_type == 'drug':
                    node_color.append('rgba(255, 65, 54, 0.8)')  # Red for drugs
                elif node_type == 'disease':
                    node_color.append('rgba(50, 168, 82, 0.8)')  # Green for diseases
                elif node_type == 'gene':
                    node_color.append('rgba(66, 135, 245, 0.8)')  # Blue for genes
                else:
                    node_color.append('rgba(255, 185, 15, 0.8)')  # Yellow for other nodes
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []
    edge_color = []
    edge_dash = []
    
    # Arrow shape for directed edges
    arrows = []
    
    for u, v, attrs in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Add line coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge attributes
        rel_type = attrs.get('type', 'unknown')
        confidence = attrs.get('confidence', 0)
        
        # Enhanced edge text with more metadata
        mechanism = attrs.get('mechanism', 'Unknown')
        evidence_count = attrs.get('evidence_count', 0)
        
        edge_text.append(f"<b>Relationship:</b> {rel_type.capitalize()}<br><b>Confidence:</b> {confidence:.2f}<br><b>Mechanism:</b> {mechanism}<br><b>Evidence:</b> {evidence_count} sources")
        
        # Edge width based on confidence and evidence
        if (u, v) in highlight_edges or (v, u) in highlight_edges:
            edge_width.append(4)  # Thicker for highlighted edges
        else:
            evidence_factor = min(3, evidence_count / 2) if evidence_count else 1
            edge_width.append(1 + 2 * confidence * evidence_factor)  # Width based on confidence and evidence
        
        # Edge color based on relationship type with improved semantic colors
        this_edge_color = 'rgba(180, 180, 180, 0.6)'  # Default grey
        
        # Edge dash pattern to indicate relationship certainty
        this_edge_dash = 'solid'  # Default solid line
        
        if (u, v) in highlight_edges or (v, u) in highlight_edges:
            this_edge_color = 'rgba(255, 255, 0, 0.9)'  # Bright yellow for highlighted edges
        else:
            if edge_color_by == 'type':
                # Color by relationship type
                if rel_type.lower() == 'treats':
                    this_edge_color = 'rgba(0, 128, 255, 0.8)'  # Blue for 'treats'
                elif rel_type.lower() == 'potential_treatment' or rel_type.lower() == 'potential':
                    this_edge_color = 'rgba(255, 165, 0, 0.8)'  # Orange for 'potential' relationships
                    this_edge_dash = 'dash'  # Dashed line for potential/hypothetical relationships
                elif rel_type.lower() == 'targets':
                    this_edge_color = 'rgba(255, 0, 255, 0.8)'  # Purple for 'targets'
                elif rel_type.lower() == 'similar_mechanism' or rel_type.lower() == 'similar':
                    this_edge_color = 'rgba(0, 200, 200, 0.8)'  # Teal for similarity relationships
                    this_edge_dash = 'dot'  # Dotted line for similarity relationships
                elif rel_type.lower() == 'associated_with' or rel_type.lower() == 'associated':
                    this_edge_color = 'rgba(100, 200, 100, 0.8)'  # Green for associations
            elif edge_color_by == 'confidence':
                # Color by confidence - gradient from red (low) to green (high)
                r = int(255 * (1 - confidence))
                g = int(255 * confidence)
                b = 100
                this_edge_color = f'rgba({r}, {g}, {b}, 0.8)'
                
                # Show uncertain relationships with dashed lines
                if confidence < 0.5:
                    this_edge_dash = 'dash'
            elif edge_color_by == 'weight':
                # Color by weight if available, otherwise by confidence
                weight = attrs.get('weight', confidence)
                r = int(255 * (1 - weight))
                g = int(255 * weight)
                b = 100
                this_edge_color = f'rgba({r}, {g}, {b}, 0.8)'
        
        # Store the color and dash patterns
        edge_color.append(this_edge_color)
        edge_dash.append(this_edge_dash)
        
        # If showing arrows, add arrow annotations for directed graph visualization
        if show_arrows and isinstance(G, nx.DiGraph):
            # Calculate the arrow position (slightly before the target node)
            # This is a simplification - in a real implementation you'd use proper vector math
            arrow_length = 0.15  # Length of the arrow as a fraction of edge length
            dx, dy = x1 - x0, y1 - y0
            length = (dx**2 + dy**2)**0.5
            
            if length > 0:  # Avoid division by zero
                # Position the arrow near the target node
                ax = x1 - (dx * arrow_length / length)
                ay = y1 - (dy * arrow_length / length)
                
                # Calculate the arrow angle in degrees
                angle = math.degrees(math.atan2(dy, dx))
                
                # Add to arrows list
                arrows.append({
                    'x': ax,
                    'y': ay,
                    'u': dx * 0.1 / length,  # Make the arrow shorter
                    'v': dy * 0.1 / length,
                    'color': this_edge_color,
                    'angle': angle
                })
    
    # Create node trace with improved styling
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            symbol=node_symbols,  # Use different symbols for different node types
            line=dict(width=1.5, color='rgb(50, 50, 50)')
        ),
        ids=node_ids  # Store node IDs for interactivity
    )
    
    # Create edge traces with improved styling - one trace per edge to allow different colors
    edge_traces = []
    
    # Handle empty graphs
    if not edge_x:
        # Create dummy edge trace for empty graph
        edge_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
            hoverinfo='none'
        ))
    else:
        # Group edges by their characteristics to reduce number of traces
        edge_groups = {}
        
        # Process each edge
        edge_idx = 0
        for i in range(0, len(edge_x), 3):  # Process in groups of 3 (start, end, None for each edge)
            if i+2 < len(edge_x):  # Ensure we have all 3 parts
                # Create a key based on edge properties
                key = f"{edge_color[edge_idx]}|{edge_width[edge_idx]}|{edge_dash[edge_idx]}"
                
                # Add to the appropriate group
                if key not in edge_groups:
                    edge_groups[key] = {
                        'x': [],
                        'y': [],
                        'text': [],
                        'color': edge_color[edge_idx],
                        'width': edge_width[edge_idx],
                        'dash': edge_dash[edge_idx]
                    }
                
                # Add this edge's data to its group
                edge_groups[key]['x'].extend([edge_x[i], edge_x[i+1], edge_x[i+2]])
                edge_groups[key]['y'].extend([edge_y[i], edge_y[i+1], edge_y[i+2]])
                edge_groups[key]['text'].append(edge_text[edge_idx])
                
                edge_idx += 1
        
        # Create a trace for each group of edges
        for key, group in edge_groups.items():
            edge_traces.append(go.Scatter(
                x=group['x'],
                y=group['y'],
                mode='lines',
                line=dict(
                    width=group['width'],
                    color=group['color'],
                    dash=group['dash']
                ),
                hoverinfo='text',
                text=group['text'],
                name=f"Edge {key.split('|')[0]}"  # Name based on color for legend
            ))
    
    # Create arrow annotations if needed
    annotations = []
    if show_arrows and arrows:
        for arrow in arrows:
            annotations.append(
                dict(
                    x=arrow['x'],
                    y=arrow['y'],
                    ax=arrow['x'] - arrow['u'] * 20,
                    ay=arrow['y'] - arrow['v'] * 20,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=arrow['color'],
                )
            )
    
    # Create figure with all traces
    # Combine edge traces and node trace
    all_traces = edge_traces + [node_trace]
    
    # Create figure
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title='Drug-Disease Knowledge Graph',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            annotations=annotations,  # Add arrow annotations for directed graphs
            legend=dict(
                title="Node & Edge Types",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)"
            )
        )
    )
    
    return fig

def create_confidence_distribution(candidates):
    """
    Create a histogram of confidence scores
    """
    if not candidates:
        return None
        
    # Extract confidence scores
    scores = [candidate['confidence_score'] for candidate in candidates]
    
    # Create histogram
    fig = px.histogram(
        x=scores,
        nbins=10,
        range_x=[0, 100],
        labels={'x': 'Confidence Score'},
        title='Distribution of Confidence Scores',
        color_discrete_sequence=['rgb(0, 112, 192)']
    )
    
    fig.update_layout(
        xaxis_title='Confidence Score',
        yaxis_title='Count',
        bargap=0.1
    )
    
    return fig

def create_drug_disease_heatmap(relationships):
    """
    Create a heatmap of drug-disease confidence scores
    """
    if not relationships:
        return None
    
    # Get unique drugs and diseases
    drugs = list(set([rel['drug_name'] for rel in relationships]))
    diseases = list(set([rel['disease_name'] for rel in relationships]))
    
    # Create matrix of confidence scores
    confidence_matrix = np.zeros((len(drugs), len(diseases)))
    
    for rel in relationships:
        drug_idx = drugs.index(rel['drug_name'])
        disease_idx = diseases.index(rel['disease_name'])
        confidence_matrix[drug_idx, disease_idx] = rel['confidence']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=confidence_matrix,
        x=diseases,
        y=drugs,
        colorscale='Blues',
        colorbar=dict(title='Confidence')
    ))
    
    fig.update_layout(
        title='Drug-Disease Relationship Confidence Heatmap',
        xaxis_title='Disease',
        yaxis_title='Drug',
        height=600
    )
    
    return fig

def create_path_visualization(G, path, show_mechanism=False):
    """
    Create a visualization of a specific path in the knowledge graph
    """
    if not path or len(path) < 2:
        return None
    
    # Create a subgraph with only the nodes and edges in the path
    subgraph = G.subgraph(path)
    
    # Position nodes in a line
    pos = {}
    for i, node in enumerate(path):
        pos[node] = (i / (len(path) - 1), 0.5)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node attributes
        attrs = G.nodes[node]
        node_type = attrs.get('type', 'unknown')
        node_name = attrs.get('name', 'Unknown')
        
        # Node text
        if show_mechanism and node_type == 'drug':
            mechanism = attrs.get('mechanism', 'Unknown mechanism')
            text = f"<b>{node_name}</b><br><br>Mechanism: {mechanism}"
        else:
            text = f"<b>{node_name}</b>"
        
        node_text.append(text)
        
        # Node color and size
        if node_type == 'drug':
            node_color.append('rgba(255, 65, 54, 0.7)')  # Red for drugs
        else:
            node_color.append('rgba(50, 168, 82, 0.7)')  # Green for diseases
        
        node_size.append(20)  # Larger size for better visibility
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Add line coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge attributes
        edge_data = G.get_edge_data(u, v)
        rel_type = edge_data.get('type', 'unknown')
        confidence = edge_data.get('confidence', 0)
        
        # Edge text
        edge_text.append(f"Type: {rel_type}<br>Confidence: {confidence:.2f}")
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='rgb(50, 50, 50)')
        )
    )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='rgba(100, 100, 100, 0.7)'),
        hoverinfo='text',
        text=edge_text
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Path in Knowledge Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300,
        )
    )
    
    return fig

def create_comparison_chart(candidates, top_n=5):
    """
    Create a bar chart comparing confidence scores for top candidates
    """
    if not candidates or len(candidates) == 0:
        return None
    
    # Sort candidates by confidence score and get top N
    sorted_candidates = sorted(candidates, key=lambda x: x['confidence_score'], reverse=True)
    top_candidates = sorted_candidates[:top_n]
    
    # Prepare data
    labels = [f"{c['drug']} → {c['disease']}" for c in top_candidates]
    scores = [c['confidence_score'] for c in top_candidates]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker_color=['rgba(0, 112, 192, 0.7)' if score < 70 else 'rgba(0, 176, 80, 0.7)' for score in scores]
        )
    ])
    
    fig.update_layout(
        title='Top Repurposing Candidates by Confidence Score',
        xaxis_title='Drug-Disease Pair',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_risk_assessment_visualization(candidate):
    """
    Create a color-coded risk assessment visualization for a repurposing candidate
    
    Parameters:
    - candidate: Dictionary containing candidate information
    
    Returns:
    - Plotly figure object
    """
    # Extract risk factors from candidate
    # Risk factors include: toxicity, interactions, clinical translation barriers, etc.
    risk_factors = [
        {"category": "Safety Profile", "score": calculate_safety_score(candidate), "weight": 0.30},
        {"category": "Drug Interactions", "score": calculate_interaction_score(candidate), "weight": 0.20},
        {"category": "Clinical Translation", "score": calculate_clinical_score(candidate), "weight": 0.25},
        {"category": "Regulatory Hurdles", "score": calculate_regulatory_score(candidate), "weight": 0.15},
        {"category": "Commercial Viability", "score": calculate_commercial_score(candidate), "weight": 0.10}
    ]
    
    # Calculate weighted average for overall risk score
    overall_risk = sum(factor["score"] * factor["weight"] for factor in risk_factors)
    
    # Create radar chart for individual risk factors
    categories = [factor["category"] for factor in risk_factors]
    scores = [factor["score"] for factor in risk_factors]
    
    # Add the first point at the end to close the loop
    categories.append(categories[0])
    scores.append(scores[0])
    
    # Create color scale for the radar chart (green to yellow to red)
    color_scale = [
        [0.0, "rgba(0, 176, 80, 0.7)"],      # Green (low risk)
        [0.5, "rgba(255, 192, 0, 0.7)"],     # Yellow (medium risk)
        [1.0, "rgba(192, 0, 0, 0.7)"]        # Red (high risk)
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor=get_color_for_risk(overall_risk, alpha=0.5),
        line=dict(
            color=get_color_for_risk(overall_risk),
            width=2
        ),
        name=f'Risk Profile (Overall: {overall_risk:.1f})'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["Very Low", "Low", "Moderate", "High", "Very High"]
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                rotation=90,
                direction="clockwise"
            )
        ),
        title=dict(
            text=f"Risk Assessment for {candidate['drug']} → {candidate['disease']}",
            x=0.5
        ),
        showlegend=False,
        height=500,
    )
    
    return fig

def create_risk_assessment_summary(candidates):
    """
    Create a summarized risk assessment visualization for multiple candidates
    
    Parameters:
    - candidates: List of dictionaries containing candidate information
    
    Returns:
    - Plotly figure object
    """
    if not candidates or len(candidates) == 0:
        return None
    
    # Limit to top 10 candidates by confidence score
    top_candidates = sorted(candidates, key=lambda x: x['confidence_score'], reverse=True)[:10]
    
    # Calculate risk scores for each candidate
    data = []
    for candidate in top_candidates:
        safety_score = calculate_safety_score(candidate)
        interaction_score = calculate_interaction_score(candidate)
        clinical_score = calculate_clinical_score(candidate)
        regulatory_score = calculate_regulatory_score(candidate)
        commercial_score = calculate_commercial_score(candidate)
        
        # Calculate weighted overall risk (lower is better)
        overall_risk = (
            safety_score * 0.30 +
            interaction_score * 0.20 +
            clinical_score * 0.25 +
            regulatory_score * 0.15 +
            commercial_score * 0.10
        )
        
        # Calculate final viability score (higher is better, invert the risk)
        viability = candidate['confidence_score'] * (1 - (overall_risk / 100))
        
        data.append({
            "label": f"{candidate['drug']} → {candidate['disease']}",
            "confidence": candidate['confidence_score'],
            "safety": 100 - safety_score,  # Invert for visualization (higher is better)
            "viability": viability,
            "overall_risk": overall_risk
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create bubble chart
    fig = px.scatter(
        df,
        x="confidence",
        y="safety",
        size="viability",
        color="overall_risk",
        color_continuous_scale=[(0, "green"), (0.5, "yellow"), (1, "red")],
        hover_name="label",
        labels={
            "confidence": "Confidence Score",
            "safety": "Safety Score",
            "viability": "Overall Viability",
            "overall_risk": "Risk Level"
        },
        range_x=[0, 100],
        range_y=[0, 100],
        size_max=60,
        title="Risk-Confidence Assessment for Top Candidates"
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Risk Level",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Very Low", "Low", "Moderate", "High", "Very High"]
        ),
        height=600,
        width=800,
        xaxis_title="Confidence Score (higher is better)",
        yaxis_title="Safety Score (higher is better)"
    )
    
    # Add quadrant markers
    fig.add_shape(
        type="rect",
        x0=75, y0=75, x1=100, y1=100,
        line=dict(width=0),
        fillcolor="rgba(0, 176, 80, 0.1)",
        layer="below"
    )
    fig.add_annotation(
        x=87.5, y=87.5,
        text="OPTIMAL",
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=25, y1=25,
        line=dict(width=0),
        fillcolor="rgba(192, 0, 0, 0.1)",
        layer="below"
    )
    fig.add_annotation(
        x=12.5, y=12.5,
        text="HIGH RISK",
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig

# Helper functions for risk assessment

def calculate_safety_score(candidate):
    """Calculate safety risk score (0-100) where higher means more risk"""
    # In a real implementation, this would use actual data about known side effects,
    # toxicity profiles, etc.
    
    # For now, generate a score based on confidence and a random factor
    # Lower confidence candidates get higher risk scores
    base_score = max(0, 100 - candidate['confidence_score'])
    
    # Add some variation based on drug category or other factors
    # Use drug_id modulo to add some determinism
    drug_id_as_int = int(''.join([str(ord(c)) for c in candidate['drug_id']])[-2:])
    variation = drug_id_as_int % 30  # 0-29 variation
    
    # Toxicity increases with evidence count (more known issues reported)
    evidence_factor = min(30, candidate.get('evidence_count', 0)) / 30
    
    return min(100, max(0, base_score + variation - (evidence_factor * 20)))

def calculate_interaction_score(candidate):
    """Calculate drug interaction risk score (0-100) where higher means more risk"""
    # In a real implementation, this would use actual data about known drug interactions
    
    # For now, generate a score based on drug properties
    drug_id_as_int = int(''.join([str(ord(c)) for c in candidate['drug_id']])[-3:])
    base_score = (drug_id_as_int % 60) + 20  # 20-79 range
    
    # Adjust based on confidence
    confidence_factor = (100 - candidate['confidence_score']) / 100
    
    return min(100, max(0, base_score * (0.7 + (confidence_factor * 0.5))))

def calculate_clinical_score(candidate):
    """Calculate clinical translation barrier risk score (0-100) where higher means more risk"""
    # In a real implementation, this would use data about clinical trial requirements,
    # existing trial data, etc.
    
    # For now, generate a score based on disease complexity
    disease_id_as_int = int(''.join([str(ord(c)) for c in candidate['disease_id']])[-2:])
    base_score = (disease_id_as_int % 50) + 25  # 25-74 range
    
    # Modify based on confidence and mechanism clarity
    if candidate.get('mechanism_clarity', 0) > 70:
        base_score -= 20
    
    # Higher confidence reduces clinical risk
    confidence_factor = (100 - candidate['confidence_score']) / 100
    
    return min(100, max(0, base_score * (0.8 + (confidence_factor * 0.4))))

def calculate_regulatory_score(candidate):
    """Calculate regulatory hurdle risk score (0-100) where higher means more risk"""
    # In a real implementation, this would use data about regulatory pathways,
    # approval histories, etc.
    
    # For now, generate a score based on various factors
    base_score = 50  # Start at moderate risk
    
    # Drugs with higher confidence likely have more established profiles
    confidence_factor = (100 - candidate['confidence_score']) / 100
    
    # Use evidence count as a factor (more evidence means less regulatory risk)
    evidence_factor = min(1, candidate.get('evidence_count', 0) / 20)
    
    return min(100, max(0, base_score * (1 + (confidence_factor * 0.5) - (evidence_factor * 0.3))))

def calculate_commercial_score(candidate):
    """Calculate commercial viability risk score (0-100) where higher means more risk"""
    # In a real implementation, this would use market data, patent expiration,
    # competition analysis, etc.
    
    # For now, generate a score based on various factors
    combined_id = candidate['drug_id'] + candidate['disease_id']
    id_as_int = int(''.join([str(ord(c)) for c in combined_id])[-4:])
    base_score = (id_as_int % 70) + 15  # 15-84 range
    
    # Higher confidence candidates generally have better commercial prospects
    confidence_factor = (100 - candidate['confidence_score']) / 200  # Half weight
    
    return min(100, max(0, base_score * (1 - confidence_factor)))

def get_color_for_risk(risk_score, alpha=1.0):
    """Get color for risk score, from green (low) to red (high)"""
    if risk_score < 33:
        # Green for low risk
        return f"rgba(0, 176, 80, {alpha})"
    elif risk_score < 66:
        # Yellow for medium risk
        return f"rgba(255, 192, 0, {alpha})"
    else:
        # Red for high risk
        return f"rgba(192, 0, 0, {alpha})"

def plot_centrality_scores(centrality_df, node_type=None, top_n=10):
    """
    Create a plot of centrality measures for the top N nodes
    """
    if centrality_df is None or len(centrality_df) == 0:
        return None
    
    # Filter by node type if specified
    if node_type:
        filtered_df = centrality_df[centrality_df['type'] == node_type]
    else:
        filtered_df = centrality_df
    
    if len(filtered_df) == 0:
        return None
    
    # Sort by degree centrality and get top N
    sorted_df = filtered_df.sort_values('degree', ascending=False).head(top_n)
    
    # Create bar chart
    fig = px.bar(
        sorted_df,
        x='name',
        y='degree',
        color='type',
        hover_data=['betweenness', 'closeness', 'eigenvector'],
        title=f'Top {top_n} Nodes by Degree Centrality',
        color_discrete_map={'drug': 'rgba(255, 65, 54, 0.7)', 'disease': 'rgba(50, 168, 82, 0.7)'}
    )
    
    fig.update_layout(
        xaxis_title='Node',
        yaxis_title='Degree Centrality',
        legend_title='Node Type'
    )
    
    return fig
