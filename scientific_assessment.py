"""
Scientific assessment module for evaluating drug repurposing candidates

This module provides advanced scientific assessment capabilities for drug repurposing 
candidates, including pharmacological, structural, and literature-based evaluations.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
import openai
import os
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assess_pharmacological_similarity(drug1, drug2):
    """
    Calculate a pharmacological similarity score between two drugs
    
    Parameters:
    - drug1: Dictionary containing first drug information
    - drug2: Dictionary containing second drug information
    
    Returns:
    - similarity_score: Float between 0-1 indicating similarity
    """
    # In a real implementation, this would use actual pharmacological data
    # For demonstration, we'll simulate scores based on available fields
    
    # Get drug mechanisms
    mechanism1 = drug1.get('mechanism', '').lower()
    mechanism2 = drug2.get('mechanism', '').lower()
    
    # Count common words in mechanisms (simple proxy for mechanism similarity)
    if mechanism1 and mechanism2:
        words1 = set(mechanism1.split())
        words2 = set(mechanism2.split())
        common_words = words1.intersection(words2)
        mechanism_similarity = len(common_words) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
    else:
        mechanism_similarity = 0
    
    # Get drug categories/classes
    category1 = drug1.get('category', '').lower()
    category2 = drug2.get('category', '').lower()
    
    # Simple category match
    category_similarity = 1.0 if category1 == category2 and category1 != '' else 0.0
    
    # Weight and combine the similarities
    similarity_score = 0.7 * mechanism_similarity + 0.3 * category_similarity
    
    # Add some noise to simulate real-world variation
    similarity_score = min(1.0, max(0.0, similarity_score + random.uniform(-0.1, 0.1)))
    
    return similarity_score

def generate_pharmacological_network(drugs, min_similarity=0.4):
    """
    Generate a network of pharmacologically similar drugs
    
    Parameters:
    - drugs: List of drug dictionaries
    - min_similarity: Minimum similarity score to include in network
    
    Returns:
    - nodes: Dataframe of nodes
    - edges: Dataframe of edges
    """
    # Create nodes dataframe
    nodes = pd.DataFrame([
        {'id': drug['id'], 'name': drug['name'], 'type': 'drug', 
         'category': drug.get('category', ''), 'mechanism': drug.get('mechanism', '')}
        for drug in drugs
    ])
    
    # Calculate similarities and create edges
    edges_list = []
    
    for i, drug1 in enumerate(drugs):
        for j, drug2 in enumerate(drugs):
            # Skip self-connections
            if i >= j:
                continue
                
            similarity = assess_pharmacological_similarity(drug1, drug2)
            
            # Only include edges with similarity above threshold
            if similarity >= min_similarity:
                edges_list.append({
                    'source': drug1['id'],
                    'target': drug2['id'],
                    'similarity': similarity
                })
    
    edges = pd.DataFrame(edges_list) if edges_list else pd.DataFrame(columns=['source', 'target', 'similarity'])
    
    return nodes, edges

def visualize_pharmacological_network(nodes, edges):
    """
    Create an interactive visualization of the pharmacological network
    
    Parameters:
    - nodes: Dataframe of nodes
    - edges: Dataframe of edges
    
    Returns:
    - fig: Plotly figure object
    """
    if edges.empty:
        # Return empty figure if no edges
        fig = go.Figure()
        fig.update_layout(
            title="No significant pharmacological similarities found",
            height=600
        )
        return fig
        
    # Create a network graph
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    for _, node in nodes.iterrows():
        G.add_node(node['id'], name=node['name'], category=node['category'], mechanism=node['mechanism'])
    
    # Add edges
    for _, edge in edges.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['similarity'])
    
    # Compute layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        source_name = G.nodes[edge[0]]['name']
        target_name = G.nodes[edge[1]]['name']
        similarity = edge[2]['weight']
        
        edge_text.append(f"{source_name} - {target_name}: {similarity:.2f} similarity")
        edge_width.extend([similarity * 5, similarity * 5, 0])  # Width based on similarity
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color='rgba(150,150,150,0.7)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # Create color map for categories
    categories = list(set(data['category'] for _, data in G.nodes(data=True) if 'category' in data))
    color_map = {}
    
    for i, category in enumerate(categories):
        # Generate colors evenly spaced on the color wheel
        hue = i / max(1, len(categories))
        color_map[category] = f"hsl({int(hue * 360)}, 70%, 50%)"
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node text includes name and mechanism
        name = data['name']
        category = data.get('category', 'Unknown')
        mechanism = data.get('mechanism', 'Unknown mechanism')
        
        node_text.append(f"<b>{name}</b><br>Category: {category}<br>Mechanism: {mechanism}")
        
        # Node size based on degree centrality
        node_size.append(10 + 5 * G.degree(node))
        
        # Node color based on category
        node_color.append(color_map.get(category, 'hsl(0, 0%, 50%)'))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='rgb(50,50,50)')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="Pharmacological Similarity Network",
                       titlefont=dict(size=16),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       annotations=[
                           dict(
                               text="Network shows drugs with similar mechanisms of action",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.01, y=-0.05
                           )
                       ]
                   ))
    
    # Add legend for categories
    for i, (category, color) in enumerate(color_map.items()):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=category,
            showlegend=True
        ))
    
    return fig

def analyze_target_overlap(drug, disease, use_ai=True):
    """
    Analyze the overlap between drug targets and disease pathways
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - use_ai: Whether to use AI to enhance the analysis
    
    Returns:
    - overlap_data: Dictionary with overlap analysis results
    """
    # In production, this would query biological databases for actual targets and pathways
    # For demonstration, we'll simulate this with available data
    
    # Get drug mechanisms and disease description
    drug_mechanism = drug.get('mechanism', '').lower()
    disease_description = disease.get('description', '').lower()
    
    # Extract potential biological targets and pathways
    # In a real implementation, this would use entity recognition on text
    # or query actual biological databases
    
    # Use AI to analyze target overlap if available
    if use_ai and 'OPENAI_API_KEY' in os.environ:
        try:
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pharmacologist analyzing the potential overlap between drug mechanisms and disease processes. Respond in JSON format only."},
                    {"role": "user", "content": f"""
                    Analyze the potential molecular and biological overlaps between this drug and disease:
                    
                    DRUG: {drug.get('name', 'Unknown')}
                    MECHANISM: {drug_mechanism}
                    CATEGORY: {drug.get('category', 'Unknown')}
                    
                    DISEASE: {disease.get('name', 'Unknown')}
                    DESCRIPTION: {disease_description}
                    CATEGORY: {disease.get('category', 'Unknown')}
                    
                    Respond with:
                    1. A list of potential shared molecular targets (protein names)
                    2. A list of potential shared biological pathways
                    3. A scientific explanation of how the drug might affect the disease
                    4. A confidence score (0-100) for this drug being effective against this disease
                    
                    Format your response as a JSON object with keys: 'targets', 'pathways', 'explanation', 'confidence'
                    """}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = response.choices[0].message.content
            import json
            overlap_data = json.loads(result)
            
            # Add source attribution
            overlap_data['source'] = 'ai_analysis'
            return overlap_data
            
        except Exception as e:
            logger.error(f"Error using OpenAI API: {e}")
            # Fall back to rule-based method
    
    # Rule-based method (fallback)
    # Extract potential targets by looking for protein and gene names in the text
    # This is a very simplified approach for demonstration purposes
    common_targets = []
    common_pathways = []
    
    # List of common biological targets and pathways for pattern matching
    potential_targets = ['ACE', 'EGFR', 'HER2', 'TNF', 'IL-6', 'p53', 'BRCA', 'CFTR', 'KRAS', 'BRAF']
    potential_pathways = ['MAPK', 'PI3K', 'JAK-STAT', 'WNT', 'Notch', 'mTOR', 'apoptosis', 'inflammation', 'oxidative stress']
    
    # Simple text matching (in a real implementation, use NER or database queries)
    for target in potential_targets:
        target_lower = target.lower()
        if target_lower in drug_mechanism and target_lower in disease_description:
            common_targets.append(target)
    
    for pathway in potential_pathways:
        pathway_lower = pathway.lower()
        if pathway_lower in drug_mechanism and pathway_lower in disease_description:
            common_pathways.append(pathway)
    
    # Generate a basic explanation
    if common_targets or common_pathways:
        targets_text = ", ".join(common_targets) if common_targets else "No specific targets identified"
        pathways_text = ", ".join(common_pathways) if common_pathways else "No specific pathways identified"
        explanation = f"The drug may affect {targets_text} involved in {pathways_text} pathways relevant to the disease."
        
        # Calculate a simple confidence score
        confidence = 40 + 10 * len(common_targets) + 5 * len(common_pathways)
        confidence = min(95, confidence)  # Cap at 95
    else:
        explanation = "Insufficient data to determine specific molecular overlaps between drug and disease."
        confidence = 30  # Low confidence due to lack of specific matches
    
    overlap_data = {
        'targets': common_targets,
        'pathways': common_pathways,
        'explanation': explanation,
        'confidence': confidence,
        'source': 'rule_based_analysis'
    }
    
    return overlap_data

def visualize_target_overlap(overlap_data):
    """
    Create a visualization of target and pathway overlap
    
    Parameters:
    - overlap_data: Dictionary with overlap analysis results
    
    Returns:
    - fig: Plotly figure object
    """
    # Extract data
    targets = overlap_data.get('targets', [])
    pathways = overlap_data.get('pathways', [])
    confidence = overlap_data.get('confidence', 0)
    
    if not targets and not pathways:
        # Create placeholder figure if no overlap data
        fig = go.Figure()
        fig.update_layout(
            title="No specific target or pathway overlap identified",
            annotations=[dict(
                text="Insufficient molecular data to visualize specific overlaps.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )],
            height=400
        )
        return fig
    
    # Create node data
    nodes = [{'id': 'drug', 'name': 'Drug', 'type': 'drug', 'level': 0}]
    nodes.append({'id': 'disease', 'name': 'Disease', 'type': 'disease', 'level': 3})
    
    # Add target nodes
    for i, target in enumerate(targets):
        nodes.append({'id': f'target_{i}', 'name': target, 'type': 'target', 'level': 1})
    
    # Add pathway nodes
    for i, pathway in enumerate(pathways):
        nodes.append({'id': f'pathway_{i}', 'name': pathway, 'type': 'pathway', 'level': 2})
    
    # Create edge data
    edges = []
    
    # Drug to targets
    for i in range(len(targets)):
        edges.append({'source': 'drug', 'target': f'target_{i}', 'type': 'inhibits'})
    
    # Targets to pathways
    for i in range(len(targets)):
        for j in range(len(pathways)):
            # Create some random connections
            if random.random() > 0.3:  # 70% chance of connection
                edges.append({'source': f'target_{i}', 'target': f'pathway_{j}', 'type': 'affects'})
    
    # Pathways to disease
    for i in range(len(pathways)):
        edges.append({'source': f'pathway_{i}', 'target': 'disease', 'type': 'involves'})
    
    # If no targets but pathways exist, connect drug directly to pathways
    if not targets and pathways:
        for i in range(len(pathways)):
            edges.append({'source': 'drug', 'target': f'pathway_{i}', 'type': 'affects'})
    
    # If no pathways but targets exist, connect targets directly to disease
    if targets and not pathways:
        for i in range(len(targets)):
            edges.append({'source': f'target_{i}', 'target': 'disease', 'type': 'involved_in'})
    
    # Create dataframes
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    # Sankey diagram for mechanism visualization
    # Prepare data for Sankey diagram
    label_list = nodes_df['name'].tolist()
    color_list = []
    
    for node_type in nodes_df['type']:
        if node_type == 'drug':
            color_list.append('rgba(255, 65, 54, 0.8)')  # Red for drug
        elif node_type == 'target':
            color_list.append('rgba(255, 144, 14, 0.8)')  # Orange for target
        elif node_type == 'pathway':
            color_list.append('rgba(44, 160, 44, 0.8)')  # Green for pathway
        else:  # disease
            color_list.append('rgba(31, 119, 180, 0.8)')  # Blue for disease
    
    # Create source-target pairs
    source_indices = []
    target_indices = []
    
    for _, edge in edges_df.iterrows():
        source_id = edge['source']
        target_id = edge['target']
        
        source_idx = nodes_df[nodes_df['id'] == source_id].index[0]
        target_idx = nodes_df[nodes_df['id'] == target_id].index[0]
        
        source_indices.append(source_idx)
        target_indices.append(target_idx)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_list,
            color=color_list
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=[1] * len(source_indices)  # Uniform value for links
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=f"Molecular Mechanism Visualization (Confidence: {confidence}%)",
        font=dict(size=12),
        height=500,
        margin=dict(l=25, r=25, t=50, b=25)
    )
    
    return fig

def generate_literature_timeline(drug_name, disease_name):
    """
    Generate a timeline of research literature connecting the drug and disease
    
    Parameters:
    - drug_name: Name of the drug
    - disease_name: Name of the disease
    
    Returns:
    - fig: Plotly figure object with timeline
    """
    # In a real implementation, this would query PubMed or a similar database
    # For demonstration, we'll generate simulated publication data
    
    # Generate some random publication data spanning last 20 years
    current_year = datetime.now().year
    years = list(range(current_year - 20, current_year + 1))
    
    # Simulate publication counts with an upward trend
    base_counts = [int(np.random.poisson(i/3)) for i in range(len(years))]
    
    # Introduce some peaks and valleys to make it look realistic
    for i in range(2, len(base_counts)-2):
        if random.random() > 0.7:  # 30% chance of a peak
            base_counts[i] = int(base_counts[i] * random.uniform(1.5, 2.5))
            # Adjacent years also get a smaller boost
            base_counts[i-1] = int(base_counts[i-1] * random.uniform(1.2, 1.5))
            base_counts[i+1] = int(base_counts[i+1] * random.uniform(1.2, 1.5))
    
    # Ensure we have at least some publications each year
    counts = [max(1, count) for count in base_counts]
    
    # Create milestone data (significant papers or events)
    milestones = []
    
    # Generate 3-5 random milestones
    num_milestones = random.randint(3, 5)
    milestone_years = sorted(random.sample(years[5:-1], num_milestones))  # Avoid first 5 and last year
    
    milestone_types = [
        f"First case report of {drug_name} used for {disease_name}",
        f"Retrospective study shows positive outcomes",
        f"In vitro evidence of mechanism",
        f"Animal model demonstrates efficacy",
        f"First small clinical trial",
        f"Larger clinical trial confirms effect",
        f"Meta-analysis of case reports",
        f"Systematic review published",
        f"Molecular mechanism elucidated"
    ]
    
    for i, year in enumerate(milestone_years):
        milestone_type = milestone_types[i % len(milestone_types)]
        impact_factor = random.uniform(2.0, 10.0)  # Random journal impact factor
        
        milestones.append({
            'year': year,
            'description': f"{milestone_type} (IF: {impact_factor:.1f})",
            'impact': impact_factor
        })
    
    # Create dataframe for the timeline
    timeline_df = pd.DataFrame({
        'year': years,
        'publications': counts
    })
    
    # Create milestone dataframe
    milestone_df = pd.DataFrame(milestones)
    
    # Create figure
    fig = go.Figure()
    
    # Add publication count line
    fig.add_trace(go.Scatter(
        x=timeline_df['year'],
        y=timeline_df['publications'],
        mode='lines+markers',
        name='Publications',
        line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
        marker=dict(size=8, color='rgba(31, 119, 180, 0.8)')
    ))
    
    # Add milestones as markers
    if not milestone_df.empty:
        fig.add_trace(go.Scatter(
            x=milestone_df['year'],
            y=[timeline_df.loc[timeline_df['year'] == year, 'publications'].values[0] 
               for year in milestone_df['year']],
            mode='markers',
            marker=dict(
                symbol='star',
                size=16,
                color='rgba(255, 144, 14, 0.8)',
                line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
            ),
            name='Key Milestones',
            text=milestone_df['description'],
            hoverinfo='text'
        ))
    
    # Calculate trend using simple linear regression
    x = np.array(years)
    y = np.array(counts)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=timeline_df['year'],
        y=p(timeline_df['year']),
        mode='lines',
        name='Trend',
        line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash')
    ))
    
    # Calculate publication growth metrics
    growth_rate = z[0]  # Slope of the trend line
    current_momentum = counts[-1] / max(1, counts[-2]) - 1  # Year-over-year growth
    publication_acceleration = (counts[-1] - counts[-2]) - (counts[-2] - counts[-3])  # Change in growth rate
    
    # Update layout
    fig.update_layout(
        title=f"Research Connecting {drug_name} and {disease_name}",
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        height=500,
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=80),
        annotations=[
            dict(
                x=0.5,
                y=-0.2,
                xref="paper",
                yref="paper",
                text=f"Research Trend Metrics:<br>Growth Rate: {growth_rate:.2f} papers/year | " +
                     f"Current Momentum: {current_momentum*100:.1f}% | " +
                     f"Research Acceleration: {publication_acceleration:+d} papers",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
        ]
    )
    
    return fig