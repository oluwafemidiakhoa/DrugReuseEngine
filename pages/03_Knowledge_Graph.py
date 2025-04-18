import streamlit as st
import pandas as pd
import networkx as nx
from utils import get_drug_by_name, get_disease_by_name, initialize_session_state
from visualization import create_network_graph, plot_centrality_scores
from knowledge_graph import find_paths_between, compute_centrality_measures, get_drug_disease_relationships

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph | Drug Repurposing Engine",
    page_icon="🔄",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Knowledge Graph Explorer")
st.write("Explore the relationships between drugs and diseases in our knowledge graph")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Path Analysis", "Network Metrics", "Data Flow", "Data View"])

# Get the knowledge graph
G = st.session_state.graph if "graph" in st.session_state else nx.DiGraph()

with tab1:
    st.header("Knowledge Graph Overview")
    
    # Display graph statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", len(G.nodes))
    
    with col2:
        st.metric("Total Edges", len(G.edges))
    
    with col3:
        drug_nodes = len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug'])
        disease_nodes = len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease'])
        st.metric("Drugs / Diseases", f"{drug_nodes} / {disease_nodes}")
    
    # Display node and edge types
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Edge Types")
        edge_types = {}
        for _, _, attr in G.edges(data=True):
            edge_type = attr.get('type', 'unknown')
            if edge_type in edge_types:
                edge_types[edge_type] += 1
            else:
                edge_types[edge_type] = 1
        
        # Display as a table
        edge_type_df = pd.DataFrame([
            {"Type": key, "Count": value}
            for key, value in edge_types.items()
        ])
        
        st.dataframe(edge_type_df, hide_index=True)
    
    with col2:
        st.subheader("Confidence Distribution")
        # Get edge confidences
        confidences = [attr.get('confidence', 0) for _, _, attr in G.edges(data=True)]
        
        # Create histogram
        import plotly.express as px
        fig = px.histogram(
            x=confidences,
            nbins=10,
            range_x=[0, 1],
            labels={'x': 'Confidence'},
            title='Edge Confidence Distribution',
            color_discrete_sequence=['rgb(0, 112, 192)']
        )
        
        st.plotly_chart(fig, use_container_width=True, key="confidence_histogram")
    
    # Display the full graph
    st.subheader("Full Knowledge Graph")
    fig = create_network_graph(G)
    st.plotly_chart(fig, use_container_width=True, key="full_graph")
    
    # Filter options
    st.subheader("Filter Graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edge_type_filter = st.multiselect(
            "Edge Types",
            options=list(edge_types.keys()),
            default=list(edge_types.keys())
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    # Apply filters
    if edge_type_filter or confidence_threshold > 0:
        # Create a copy of the graph
        filtered_graph = nx.DiGraph()
        
        # Add nodes
        for node, attr in G.nodes(data=True):
            filtered_graph.add_node(node, **attr)
        
        # Add filtered edges
        for source, target, attr in G.edges(data=True):
            if attr.get('type') in edge_type_filter and attr.get('confidence', 0) >= confidence_threshold:
                filtered_graph.add_edge(source, target, **attr)
        
        # Display filtered graph
        st.subheader("Filtered Knowledge Graph")
        
        # Only display if there are edges
        if filtered_graph.edges:
            fig = create_network_graph(filtered_graph)
            st.plotly_chart(fig, use_container_width=True, key="filtered_graph")
        else:
            st.warning("No edges match the filter criteria")

with tab2:
    st.header("Path Analysis")
    st.write("Analyze paths between drugs and diseases in the knowledge graph")
    
    # Path analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        path_types = st.radio(
            "Path Direction",
            options=["Drug → Disease", "Disease → Drug"],
            index=0
        )
    
    with col2:
        max_path_length = st.slider(
            "Maximum Path Length",
            min_value=1,
            max_value=5,
            value=3
        )
    
    # Search for specific paths
    st.subheader("Search for Paths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if path_types == "Drug → Disease":
            source_type = "drug"
            target_type = "disease"
            source_label = "Drug"
            target_label = "Disease"
        else:
            source_type = "disease"
            target_type = "drug"
            source_label = "Disease"
            target_label = "Drug"
        
        # Get all nodes of the source type
        source_nodes = [G.nodes[n]['name'] for n in G.nodes() if G.nodes[n].get('type') == source_type]
        source_node = st.selectbox(f"Select {source_label}", options=source_nodes)
    
    with col2:
        # Get all nodes of the target type
        target_nodes = [G.nodes[n]['name'] for n in G.nodes() if G.nodes[n].get('type') == target_type]
        target_node = st.selectbox(f"Select {target_label}", options=target_nodes)
    
    # Find paths
    if st.button("Find Paths", type="primary"):
        # Get node IDs
        source_id = None
        target_id = None
        
        for node, attr in G.nodes(data=True):
            if attr.get('type') == source_type and attr.get('name') == source_node:
                source_id = node
            elif attr.get('type') == target_type and attr.get('name') == target_node:
                target_id = node
        
        if source_id and target_id:
            # Try to find paths
            try:
                paths = list(nx.all_simple_paths(G, source_id, target_id, cutoff=max_path_length))
                
                if paths:
                    st.success(f"Found {len(paths)} path(s) from {source_node} to {target_node}")
                    
                    # Display each path
                    for i, path in enumerate(paths):
                        # Get node names
                        node_names = [G.nodes[node]['name'] for node in path]
                        
                        # Get edge types
                        edge_types = []
                        edge_confidences = []
                        for j in range(len(path)-1):
                            edge_data = G.get_edge_data(path[j], path[j+1])
                            edge_types.append(edge_data['type'])
                            edge_confidences.append(edge_data['confidence'])
                        
                        # Calculate average confidence
                        avg_confidence = sum(edge_confidences) / len(edge_confidences) if edge_confidences else 0
                        
                        # Create path string
                        path_str = " → ".join(node_names)
                        
                        # Display path
                        with st.expander(f"Path {i+1}: {path_str} (Avg. Confidence: {avg_confidence:.2f})"):
                            # Display path details
                            for j in range(len(path)-1):
                                st.write(f"{node_names[j]} --({edge_types[j]}, {edge_confidences[j]:.2f})--> {node_names[j+1]}")
                            
                            # Visualize path
                            from visualization import create_path_visualization
                            path_fig = create_path_visualization(G, path, show_mechanism=True)
                            st.plotly_chart(path_fig, use_container_width=True, key=f"path_viz_{i}")
                else:
                    st.warning(f"No paths found from {source_node} to {target_node} within {max_path_length} steps")
            except nx.NetworkXNoPath:
                st.warning(f"No paths found from {source_node} to {target_node}")
        else:
            st.error("Could not find the selected nodes in the graph")
    
    # Find all paths
    st.subheader("All Paths Analysis")
    
    if st.button("Find All Paths"):
        with st.spinner("Finding all paths..."):
            # Find all paths between drugs and diseases
            if path_types == "Drug → Disease":
                paths = find_paths_between(G, "drug", "disease", max_length=max_path_length)
            else:
                paths = find_paths_between(G, "disease", "drug", max_length=max_path_length)
            
            if paths:
                st.success(f"Found {len(paths)} path(s)")
                
                # Display top paths by confidence
                st.subheader("Top Paths by Confidence")
                
                # Create DataFrame
                paths_df = pd.DataFrame([
                    {
                        "Path": " → ".join(p['node_names']),
                        "Length": len(p['path']) - 1,
                        "Avg. Confidence": f"{p['avg_confidence']:.2f}",
                        "Details": str(p['path'])
                    }
                    for p in paths[:20]  # Limit to top 20
                ])
                
                st.dataframe(paths_df, hide_index=True)
            else:
                st.warning(f"No paths found between {source_type}s and {target_type}s within {max_path_length} steps")

with tab3:
    st.header("Network Metrics")
    st.write("Analyze the centrality and importance of nodes in the knowledge graph")
    
    # Calculate centrality measures
    if st.button("Calculate Network Metrics"):
        with st.spinner("Calculating network metrics..."):
            centrality_df = compute_centrality_measures(G)
            
            if not centrality_df.empty:
                # Display top nodes by degree centrality
                st.subheader("Top Nodes by Degree Centrality")
                
                # Filter options
                node_type_filter = st.radio(
                    "Node Type",
                    options=["All", "Drugs", "Diseases"],
                    index=0,
                    horizontal=True
                )
                
                # Apply filter
                if node_type_filter == "Drugs":
                    filtered_df = centrality_df[centrality_df['type'] == 'drug']
                elif node_type_filter == "Diseases":
                    filtered_df = centrality_df[centrality_df['type'] == 'disease']
                else:
                    filtered_df = centrality_df
                
                # Sort by degree centrality
                sorted_df = filtered_df.sort_values('degree', ascending=False)
                
                # Display top 10
                st.dataframe(
                    sorted_df[['name', 'type', 'degree', 'betweenness', 'closeness', 'eigenvector']].head(10),
                    hide_index=True
                )
                
                # Plot centrality scores
                st.subheader("Centrality Scores Visualization")
                
                node_type_for_plot = None
                if node_type_filter == "Drugs":
                    node_type_for_plot = "drug"
                elif node_type_filter == "Diseases":
                    node_type_for_plot = "disease"
                
                fig = plot_centrality_scores(centrality_df, node_type=node_type_for_plot)
                st.plotly_chart(fig, use_container_width=True, key="centrality_scores")
                
                # Network density
                st.subheader("Network Density")
                
                density = nx.density(G)
                st.metric("Graph Density", f"{density:.4f}")
                st.write("Density measures how interconnected the graph is (0=sparse, 1=fully connected)")
                
                # Clustering coefficient
                try:
                    avg_clustering = nx.average_clustering(G.to_undirected())
                    st.metric("Average Clustering Coefficient", f"{avg_clustering:.4f}")
                    st.write("Clustering coefficient measures how nodes tend to cluster together")
                except Exception as e:
                    st.warning(f"Could not calculate clustering coefficient: {str(e)}")

with tab4:
    st.header("Animated Data Flow")
    st.write("Visualize how information flows through the knowledge graph")
    
    # Options for the animation
    st.subheader("Animation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_node_type = st.radio(
            "Start From",
            options=["Drugs", "Diseases", "Selected Node"],
            index=0
        )
    
    with col2:
        num_pulses = st.slider(
            "Number of Pulses",
            min_value=1,
            max_value=5,
            value=3
        )
    
    with col3:
        animation_speed = st.slider(
            "Animation Speed",
            min_value=10,
            max_value=50,
            value=20,
            help="Higher values = faster animation"
        )
    
    # Select specific start nodes if "Selected Node" is chosen
    start_nodes = None
    if start_node_type == "Selected Node":
        # Get all node names
        all_nodes = [G.nodes[n]['name'] for n in G.nodes()]
        selected_node = st.selectbox("Select Start Node", options=all_nodes)
        
        # Find the node ID
        for node, attr in G.nodes(data=True):
            if attr.get('name') == selected_node:
                start_nodes = [node]
                break
    elif start_node_type == "Drugs":
        start_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    else:  # Diseases
        start_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease']
    
    # Generate and display the animation
    if st.button("Generate Animation", type="primary"):
        with st.spinner("Generating animated data flow..."):
            # Import the function
            from visualization import create_animated_data_flow
            
            # Create the animation
            fig = create_animated_data_flow(
                G, 
                start_nodes=start_nodes,
                num_pulses=num_pulses,
                frames_per_pulse=animation_speed
            )
            
            # Display the animation
            st.plotly_chart(fig, use_container_width=True, key="animated_data_flow")
            
            st.info("""
            👆 Use the play button to start the animation. 
            The colored dots show how information flows through the knowledge graph.
            This helps visualize how drugs might influence diseases through various pathways.
            """)

with tab5:
    st.header("Data View")
    st.write("View the raw data in the knowledge graph")
    
    # Create tabs for different data views
    data_tab1, data_tab2, data_tab3 = st.tabs(["Drugs", "Diseases", "Relationships"])
    
    with data_tab1:
        # Display drugs
        drugs = st.session_state.drugs
        
        if drugs:
            # Convert to DataFrame
            drugs_df = pd.DataFrame([
                {
                    "ID": drug['id'],
                    "Name": drug['name'],
                    "Description": drug['description'],
                    "Original Indication": drug['original_indication'],
                    "Mechanism": drug.get('mechanism', 'Unknown')
                }
                for drug in drugs
            ])
            
            st.dataframe(drugs_df, hide_index=True)
        else:
            st.info("No drugs in the database")
    
    with data_tab2:
        # Display diseases
        diseases = st.session_state.diseases
        
        if diseases:
            # Convert to DataFrame
            diseases_df = pd.DataFrame([
                {
                    "ID": disease['id'],
                    "Name": disease['name'],
                    "Description": disease['description'],
                    "Category": disease['category']
                }
                for disease in diseases
            ])
            
            st.dataframe(diseases_df, hide_index=True)
        else:
            st.info("No diseases in the database")
    
    with data_tab3:
        # Display relationships
        relationships = get_drug_disease_relationships(G)
        
        if relationships:
            # Convert to DataFrame
            relationships_df = pd.DataFrame([
                {
                    "Drug": rel['drug_name'],
                    "Disease": rel['disease_name'],
                    "Relationship Type": rel['type'],
                    "Confidence": f"{rel['confidence']:.2f}"
                }
                for rel in relationships
            ])
            
            # Add filter
            rel_type_filter = st.multiselect(
                "Filter by Relationship Type",
                options=relationships_df["Relationship Type"].unique(),
                default=relationships_df["Relationship Type"].unique()
            )
            
            if rel_type_filter:
                filtered_df = relationships_df[relationships_df["Relationship Type"].isin(rel_type_filter)]
                st.dataframe(filtered_df, hide_index=True)
            else:
                st.dataframe(relationships_df, hide_index=True)
        else:
            st.info("No relationships in the database")
    
    # Add relationship form
    st.subheader("Add Relationship")
    st.write("Add a new relationship between a drug and a disease")
    
    with st.form("add_relationship_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.selectbox("Drug", options=[drug['name'] for drug in st.session_state.drugs])
        
        with col2:
            disease_name = st.selectbox("Disease", options=[disease['name'] for disease in st.session_state.diseases])
        
        col1, col2 = st.columns(2)
        
        with col1:
            relationship_type = st.selectbox("Relationship Type", options=["treats", "potential"])
        
        with col2:
            confidence = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Submit button
        submitted = st.form_submit_button("Add Relationship")
        
        if submitted:
            # Get drug and disease IDs
            drug = get_drug_by_name(drug_name)
            disease = get_disease_by_name(disease_name)
            
            if drug and disease:
                # Create relationship
                new_relationship = {
                    "source": drug['id'],
                    "target": disease['id'],
                    "type": relationship_type,
                    "confidence": confidence
                }
                
                # Add to database
                from utils import add_relationship
                message = add_relationship(new_relationship)
                
                st.success(message)
            else:
                st.error("Could not find the selected drug or disease")
