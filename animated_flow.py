import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import random  # Added for random node positioning fallback

def create_animated_data_flow(G, start_nodes=None, num_pulses=3, frames_per_pulse=20):
    """
    Create an animated data flow visualization for the knowledge graph
    
    Parameters:
    - G: NetworkX graph object
    - start_nodes: List of node IDs where the data flow starts (default: drug nodes)
    - num_pulses: Number of pulse animations to show
    - frames_per_pulse: Number of frames per pulse animation
    
    Returns:
    - Plotly figure object with animation
    
    Note: This animation requires JavaScript to be enabled in the browser
    """
    # Validate graph
    if G is None or G.number_of_nodes() == 0:
        # Create a smaller, more optimized sample graph (200 nodes) for performance
        print("No valid graph found. Creating optimized 200-node sample graph for visualization.")
        G = nx.DiGraph()
        
        # Create a smaller, more optimized graph with fewer nodes for better performance
        # 50 drug nodes, 50 disease nodes, 50 gene nodes, 50 pathway nodes (200 total)
        drug_names = [f"Drug-{i}" for i in range(1, 51)]
        disease_names = [f"Disease-{i}" for i in range(1, 51)]
        gene_names = [f"Gene-{i}" for i in range(1, 51)]
        pathway_names = [f"Pathway-{i}" for i in range(1, 51)]
        
        # Add drug nodes
        for i, name in enumerate(drug_names):
            G.add_node(f"D{i+1}", 
                      type='drug', 
                      name=name, 
                      description=f"Sample drug {i+1}",
                      mechanism=f"Inhibits multiple targets",
                      original_indication=f"Indicated for various conditions")
        
        # Add disease nodes
        for i, name in enumerate(disease_names):
            G.add_node(f"DIS{i+1}", 
                      type='disease', 
                      name=name, 
                      description=f"Sample disease {i+1}",
                      category=f"Category {(i % 10) + 1}")
        
        # Add gene nodes
        for i, name in enumerate(gene_names):
            G.add_node(f"G{i+1}", 
                      type='gene', 
                      name=name)
        
        # Add pathway nodes
        for i, name in enumerate(pathway_names):
            G.add_node(f"P{i+1}", 
                      type='pathway', 
                      name=name)
        
        # Create relationships (ensuring connectivity)
        # Drug-Gene relationships (drug targets)
        for i in range(50):  # Updated from 200 to 50
            drug_id = f"D{i+1}"
            # Each drug targets 1-3 genes
            num_targets = random.randint(1, 3)
            for _ in range(num_targets):
                gene_id = f"G{random.randint(1, 50)}"  # Updated from 300 to 50
                G.add_edge(drug_id, gene_id, type='targets', confidence=random.uniform(0.5, 0.95))
        
        # Gene-Pathway relationships
        for i in range(50):  # Updated from 300 to 50
            gene_id = f"G{i+1}"
            # Each gene is involved in 1-3 pathways
            num_pathways = random.randint(1, 3)
            for _ in range(num_pathways):
                pathway_id = f"P{random.randint(1, 50)}"  # Updated from 300 to 50
                G.add_edge(gene_id, pathway_id, type='involved_in', confidence=random.uniform(0.6, 0.9))
        
        # Pathway-Disease relationships
        for i in range(50):  # Updated from 300 to 50
            pathway_id = f"P{i+1}"
            # Each pathway is associated with 1-2 diseases
            num_diseases = random.randint(1, 2)
            for _ in range(num_diseases):
                disease_id = f"DIS{random.randint(1, 50)}"  # Updated from 200 to 50
                G.add_edge(pathway_id, disease_id, type='associated_with', confidence=random.uniform(0.4, 0.8))
        
        # Add direct drug-disease relationships (known treatments)
        known_treatments = 20  # Reduced from 50 for a smaller graph
        for _ in range(known_treatments):
            drug_id = f"D{random.randint(1, 50)}"  # Updated from 200 to 50
            disease_id = f"DIS{random.randint(1, 50)}"  # Updated from 200 to 50
            G.add_edge(drug_id, disease_id, type='treats', confidence=random.uniform(0.7, 0.95))
        
        # Add potential drug-disease relationships
        potential_treatments = 30  # Reduced from 80 for a smaller graph
        for _ in range(potential_treatments):
            drug_id = f"D{random.randint(1, 50)}"  # Updated from 200 to 50
            disease_id = f"DIS{random.randint(1, 50)}"  # Updated from 200 to 50
            # Avoid duplicating known treatments
            if not G.has_edge(drug_id, disease_id):
                G.add_edge(drug_id, disease_id, type='potential', confidence=random.uniform(0.3, 0.7))
    
    if start_nodes is None:
        # Default to starting from drug nodes
        start_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
        # If no drug nodes were found, use any nodes
        if not start_nodes and G.number_of_nodes() > 0:
            start_nodes = list(G.nodes())[:min(5, G.number_of_nodes())]
    
    # Ensure we have valid start nodes
    start_nodes = [n for n in start_nodes if n in G.nodes()]
    if not start_nodes:
        # If no valid start nodes, create a placeholder figure
        fig = go.Figure()
        fig.update_layout(
            title="Knowledge flow visualization not available",
            annotations=[dict(
                text="No valid start nodes found in the graph. Please ensure your database has connected drug and disease nodes.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=14)
            )]
        )
        return fig
    
    # Use spring layout for node positions with safeguards
    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        # Fallback to simpler layout if spring layout fails
        try:
            pos = nx.circular_layout(G)
        except Exception:
            # Last resort: manual positions
            pos = {node: (i % 5, i // 5) for i, node in enumerate(G.nodes())}
    
    # Create node data
    node_data = {}
    for node in G.nodes():
        try:
            attrs = G.nodes[node]
            node_type = attrs.get('type', 'unknown')
            node_name = attrs.get('name', str(node))
            
            # Node text
            if node_type == 'drug':
                description = attrs.get('description', 'No description')
                mechanism = attrs.get('mechanism', 'Unknown mechanism')
                indication = attrs.get('original_indication', 'Unknown indication')
                text = f"<b>{node_name}</b> (Drug)<br><br>{description}<br><br>Mechanism: {mechanism}<br>Original Indication: {indication}"
                color = 'rgba(255, 65, 54, 0.7)'  # Red for drugs
            else:  # disease
                description = attrs.get('description', 'No description')
                category = attrs.get('category', 'Unknown category')
                text = f"<b>{node_name}</b> (Disease)<br><br>{description}<br><br>Category: {category}"
                color = 'rgba(50, 168, 82, 0.7)'  # Green for diseases
            
            # Ensure node is in pos dictionary
            if node not in pos:
                pos[node] = (random.random(), random.random())
                
            # Store node data
            x, y = pos[node]
            node_data[node] = {
                'x': float(x),  # Ensure we have floats, not numpy types
                'y': float(y),
                'text': text,
                'color': color,
                'type': node_type,
                'name': node_name
            }
        except Exception as e:
            # Skip problematic nodes
            print(f"Error processing node {node}: {e}")
            continue
    
    # Create edge data
    edge_data = []
    for u, v, attrs in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Edge attributes
        rel_type = attrs.get('type', 'unknown')
        confidence = attrs.get('confidence', 0)
        
        # Edge text
        text = f"Type: {rel_type}<br>Confidence: {confidence:.2f}"
        
        # Edge color
        if rel_type == 'treats':
            color = 'rgba(0, 0, 255, 0.6)'  # Blue for 'treats'
        elif rel_type == 'potential':
            color = 'rgba(255, 165, 0, 0.6)'  # Orange for 'potential'
        else:
            color = 'rgba(200, 200, 200, 0.6)'  # Grey for other types
        
        # Edge width based on confidence
        width = 1 + 2 * confidence
        
        # Store edge data
        edge_data.append({
            'source': u,
            'target': v,
            'source_x': x0,
            'source_y': y0,
            'target_x': x1,
            'target_y': y1,
            'text': text,
            'color': color,
            'width': width,
            'type': rel_type,
            'confidence': confidence
        })
    
    # Calculate paths from start nodes to all other nodes
    paths = {}
    for start_node in start_nodes:
        paths[start_node] = nx.single_source_shortest_path(G, start_node)
    
    # Create a figure
    fig = make_subplots()
    
    # Create base node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node, data in node_data.items():
        node_x.append(data['x'])
        node_y.append(data['y'])
        node_text.append(data['text'])
        node_color.append(data['color'])
        node_size.append(12)  # Regular size
    
    # Create base edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    edge_color = []
    edge_width = []
    
    for edge in edge_data:
        x0, y0 = edge['source_x'], edge['source_y']
        x1, y1 = edge['target_x'], edge['target_y']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge['text'])
        edge_color.extend([edge['color'], edge['color'], 'rgba(0,0,0,0)'])
        edge_width.extend([edge['width'], edge['width'], 0])
    
    # Create base node trace
    base_node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='rgb(50, 50, 50)')
        )
    )
    
    # Create base edge trace
    # Instead of using an array of colors that's causing the error, use a single color and opacity
    if edge_x:  # Only create if we have edges
        base_edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(
                width=1.5,  # Consistent line width
                color='rgba(150, 150, 150, 0.4)'  # Light gray with transparency
            ),
            hoverinfo='text',
            text=edge_text,
            opacity=0.7
        )
    else:
        # Create an empty trace if no edges
        base_edge_trace = go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.5)'),
            hoverinfo='none'
        )
    
    # Add base traces to figure
    fig.add_trace(base_edge_trace)
    fig.add_trace(base_node_trace)
    
    # Create frames for animation
    frames = []
    
    for pulse in range(num_pulses):
        # Calculate animation time points
        for frame_idx in range(frames_per_pulse):
            # Animation progress (0 to 1)
            progress = frame_idx / frames_per_pulse
            
            # Create data flow traces for this frame
            flow_traces = []
            
            # For each start node, animate data flow along paths
            for start_node in start_nodes:
                # Get all paths from this start node
                node_paths = paths[start_node]
                
                for end_node, path in node_paths.items():
                    # Skip if it's the same node
                    if start_node == end_node:
                        continue
                    
                    # Calculate how far along each path the pulse should be
                    path_length = len(path) - 1  # Number of edges in path
                    if path_length == 0:
                        continue
                        
                    # Normalize the path length to get consistent speed
                    norm_path_length = min(path_length, 3)  # Cap at 3 for consistent speed
                    
                    # Calculate current edge based on progress
                    current_edge_idx = min(int(progress * norm_path_length * 2), path_length - 1)
                    
                    # Calculate progress within current edge (0 to 1)
                    edge_progress = (progress * norm_path_length * 2) % 1
                    
                    # Only show if we're within the path length and animation is active
                    if current_edge_idx < path_length:
                        u = path[current_edge_idx]
                        v = path[current_edge_idx + 1]
                        
                        # Get edge data
                        edge_info = next((e for e in edge_data if e['source'] == u and e['target'] == v), None)
                        
                        if edge_info:
                            # Calculate point along the edge based on progress
                            point_x = edge_info['source_x'] + edge_progress * (edge_info['target_x'] - edge_info['source_x'])
                            point_y = edge_info['source_y'] + edge_progress * (edge_info['target_y'] - edge_info['source_y'])
                            
                            # Create a glowing dot for data pulse
                            flow_traces.append(
                                go.Scatter(
                                    x=[point_x],
                                    y=[point_y],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color='rgba(255, 255, 255, 0.9)',  # White glow
                                        line=dict(width=1, color=edge_info['color']),
                                        symbol='circle'
                                    ),
                                    hoverinfo='none',
                                    showlegend=False
                                )
                            )
                            
                            # Also highlight the current edge
                            highlight_x = [edge_info['source_x'], edge_info['target_x']]
                            highlight_y = [edge_info['source_y'], edge_info['target_y']]
                            
                            # Make the edge color more vibrant for the highlight
                            base_color = edge_info['color'].replace('rgba', 'rgb').replace(', 0.6)', ')')
                            
                            flow_traces.append(
                                go.Scatter(
                                    x=highlight_x,
                                    y=highlight_y,
                                    mode='lines',
                                    line=dict(
                                        width=edge_info['width'] * 1.5,  # Make it a bit thicker
                                        color=base_color
                                    ),
                                    hoverinfo='none',
                                    showlegend=False
                                )
                            )
            
            # Create a frame with base traces and flow traces
            frame_traces = [base_edge_trace, base_node_trace] + flow_traces
            
            frames.append(go.Frame(
                data=frame_traces,
                name=f"frame{pulse}_{frame_idx}"
            ))
    
    # Update layout
    fig.update_layout(
        title='Animated Data Flow in Knowledge Graph',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 200, 'redraw': True},  # Slowed down for better visibility
                        'fromcurrent': True,
                        'transition': {'duration': 100}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 12},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f"frame{pulse}_{frame_idx}"],
                        {
                            'frame': {'duration': 200, 'redraw': True},  # Match the duration we set for Play button
                            'mode': 'immediate',
                            'transition': {'duration': 100}
                        }
                    ],
                    'label': str(pulse * frames_per_pulse + frame_idx),
                    'method': 'animate'
                }
                for pulse in range(num_pulses)
                for frame_idx in range(frames_per_pulse)
            ]
        }]
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    # Create a taller background box to hold the entire legend with more spacing
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.008, y0=0.82, x1=0.12, y1=1.0,
        fillcolor="rgba(255,255,255,0.95)",
        line=dict(color="black", width=1),
    )
    
    # Add Legend title
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.063, y=0.985,
        text="<b>Legend:</b>",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(0,0,0,0)",
    )
    
    # Create a simple colored square for Drug with more spacing
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.02, y0=0.945, x1=0.035, y1=0.96,
        fillcolor="rgba(255, 65, 54, 0.7)",
        line=dict(width=1),
    )
    
    # Add Drug label with more spacing
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.955,
        text="Drug",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(0,0,0,0)",
    )
    
    # Create a simple colored square for Disease with more spacing
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.02, y0=0.915, x1=0.035, y1=0.93,
        fillcolor="rgba(50, 168, 82, 0.7)",
        line=dict(width=1),
    )
    
    # Add Disease label with more spacing
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.925,
        text="Disease",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(0,0,0,0)",
    )
    
    # Create a simple colored square for Treats with more spacing
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.02, y0=0.885, x1=0.035, y1=0.9,
        fillcolor="rgba(0, 0, 255, 0.6)",
        line=dict(width=1),
    )
    
    # Add Treats label with more spacing
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.895,
        text="Treats",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(0,0,0,0)",
    )
    
    # Create a simple colored square for Potential with more spacing
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.02, y0=0.855, x1=0.035, y1=0.87,
        fillcolor="rgba(255, 165, 0, 0.6)",
        line=dict(width=1),
    )
    
    # Add Potential label with more spacing
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.865,
        text="Potential",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(0,0,0,0)",
    )
    
    return fig