import streamlit as st
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import neo4j_utils
from utils import initialize_session_state
import time
from datetime import datetime

# Import the new modules we created
import community_detection
import recommendation_engine
import time_analysis
import cypher_query
import graph_export

# Set page configuration
st.set_page_config(
    page_title="Graph Explorer | Drug Repurposing Engine",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Advanced Pharmaceutical Knowledge Explorer")

# Check if Neo4j connection is available and automatically initialize demo data if needed
neo4j_available = neo4j_utils.is_connected()
if neo4j_available:
    # Get graph statistics
    stats = neo4j_utils.get_graph_statistics()
    
    # Auto-initialize demo data if graph is empty
    if stats.get('total_nodes', 0) < 5:
        with st.spinner("Initializing pharmaceutical knowledge base..."):
            if neo4j_utils.initialize_demo_data():
                st.success("Pharmaceutical knowledge graph initialized successfully!")
            else:
                st.warning("Note: The knowledge graph is empty. Use the 'Initialize Demo Data' button in the Graph Overview tab to populate it.")
st.write("Explore and analyze the knowledge graph using the power of Neo4j graph database with advanced features for community detection, recommendations, time-based analysis, and custom queries.")

# Check if Neo4j is available
neo4j_available = neo4j_utils.NEO4J_AVAILABLE

if not neo4j_available:
    st.warning("""
    Neo4j graph database is not connected. This page requires a Neo4j connection to work properly.
    
    Please provide the Neo4j connection details in the Settings page or through environment variables:
    - NEO4J_URI
    - NEO4J_USERNAME
    - NEO4J_PASSWORD
    """)
    
    # Allow connecting to Neo4j
    st.subheader("Connect to Neo4j")
    
    with st.form("neo4j_connection_form"):
        # Use the AuraDB values from the .env file instead of localhost defaults
        uri = st.text_input("Neo4j URI", value="neo4j+s://9615a24a.databases.neo4j.io")
        username = st.text_input("Username", value="neo4j")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Connect")
        
        if submitted:
            # Set environment variables
            import os
            os.environ["NEO4J_URI"] = uri
            os.environ["NEO4J_USERNAME"] = username
            os.environ["NEO4J_PASSWORD"] = password
            
            # Attempt to initialize Neo4j
            if neo4j_utils.initialize_neo4j():
                st.success("Successfully connected to Neo4j!")
                st.rerun()
            else:
                st.error("Failed to connect to Neo4j. Please check your connection details.")

# Create tabs for different functionalities
if neo4j_available:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Graph Overview", 
        "Migrate Data", 
        "Path Analysis",
        "Community Detection",
        "Recommendation Engine",
        "Time-based Analysis",
        "Cypher Query Interface",
        "Export"
    ])
    
    with tab1:
        st.header("Graph Database Overview")
        
        # Get graph statistics
        stats = neo4j_utils.get_graph_statistics()
        
        if stats:
            # Create columns for statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Node Statistics")
                st.metric("Total Nodes", stats.get('total_nodes', 0))
                st.metric("Drugs", stats.get('drug_count', 0))
                st.metric("Diseases", stats.get('disease_count', 0))
                st.metric("Genes", stats.get('gene_count', 0))
            
            with col2:
                st.subheader("Relationship Statistics")
                st.metric("Total Relationships", stats.get('total_relationships', 0))
                st.metric("Treatments", stats.get('treats_count', 0))
                st.metric("Targets", stats.get('targets_count', 0))
                st.metric("Disease Associations", stats.get('associated_with_count', 0))
            
            with col3:
                st.subheader("Repurposing Statistics")
                st.metric("Potential Treatments", stats.get('potential_treatment_count', 0))
                
                # Add option to generate demo data if the graph is empty
                if stats.get('total_nodes', 0) < 5:
                    if st.button("Initialize Demo Data"):
                        with st.spinner("Initializing demo data..."):
                            if neo4j_utils.initialize_demo_data():
                                st.success("Demo data initialized successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to initialize demo data.")
        
        # Create a search section
        st.subheader("Search the Graph")
        
        search_query = st.text_input("Search for drugs, diseases, genes, etc.", placeholder="e.g., Aspirin, Diabetes, BRCA1")
        
        if search_query:
            with st.spinner("Searching..."):
                search_results = neo4j_utils.search_entities(search_query)
                
                if search_results:
                    # Create tabs for different entity types
                    entity_tabs = st.tabs(["Drugs", "Diseases", "Genes", "Proteins", "Pathways"])
                    
                    # Drugs tab
                    with entity_tabs[0]:
                        drugs = search_results.get('drugs', [])
                        if drugs:
                            df = pd.DataFrame(drugs)
                            st.dataframe(df, use_container_width=True)
                            
                            # Select a drug for detailed view
                            if len(drugs) > 0:
                                selected_drug = st.selectbox(
                                    "Select a drug for details",
                                    options=[(d['id'], d['name']) for d in drugs],
                                    format_func=lambda x: x[1]
                                )
                                
                                if selected_drug:
                                    drug_id = selected_drug[0]
                                    st.subheader(f"Neighbors of {selected_drug[1]}")
                                    
                                    # Get neighbors
                                    neighbors = neo4j_utils.get_neighbors(drug_id)
                                    
                                    if neighbors:
                                        # Split neighbors by type
                                        disease_neighbors = [n for n in neighbors if 'Disease' in n['labels']]
                                        gene_neighbors = [n for n in neighbors if 'Gene' in n['labels']]
                                        other_neighbors = [n for n in neighbors if 'Disease' not in n['labels'] and 'Gene' not in n['labels']]
                                        
                                        # Display split by relationship type
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Treats Diseases:**")
                                            treats = [n for n in disease_neighbors if n['relationship']['type'] == 'TREATS']
                                            if treats:
                                                for n in treats:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                            
                                            st.write("**Potential Treatments:**")
                                            potential = [n for n in disease_neighbors if n['relationship']['type'] == 'POTENTIAL_TREATMENT']
                                            if potential:
                                                for n in potential:
                                                    confidence = n['relationship']['properties'].get('confidence_score', 0)
                                                    st.write(f"• {n['name']} (Confidence: {confidence})")
                                            else:
                                                st.write("*None found*")
                                        
                                        with col2:
                                            st.write("**Targets Genes:**")
                                            targets = [n for n in gene_neighbors if n['relationship']['type'] == 'TARGETS']
                                            if targets:
                                                for n in targets:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                    else:
                                        st.info("No neighbors found for this drug.")
                        else:
                            st.info("No drugs found matching your search.")
                    
                    # Diseases tab
                    with entity_tabs[1]:
                        diseases = search_results.get('diseases', [])
                        if diseases:
                            df = pd.DataFrame(diseases)
                            st.dataframe(df, use_container_width=True)
                            
                            # Select a disease for detailed view
                            if len(diseases) > 0:
                                selected_disease = st.selectbox(
                                    "Select a disease for details",
                                    options=[(d['id'], d['name']) for d in diseases],
                                    format_func=lambda x: x[1]
                                )
                                
                                if selected_disease:
                                    disease_id = selected_disease[0]
                                    st.subheader(f"Neighbors of {selected_disease[1]}")
                                    
                                    # Get neighbors
                                    neighbors = neo4j_utils.get_neighbors(disease_id)
                                    
                                    if neighbors:
                                        # Split neighbors by type
                                        drug_neighbors = [n for n in neighbors if 'Drug' in n['labels']]
                                        gene_neighbors = [n for n in neighbors if 'Gene' in n['labels']]
                                        
                                        # Display split by relationship type
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Treated By Drugs:**")
                                            treated_by = [n for n in drug_neighbors if n['relationship']['type'] == 'TREATS']
                                            if treated_by:
                                                for n in treated_by:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                            
                                            st.write("**Potential Treatments:**")
                                            potential = [n for n in drug_neighbors if n['relationship']['type'] == 'POTENTIAL_TREATMENT']
                                            if potential:
                                                for n in potential:
                                                    confidence = n['relationship']['properties'].get('confidence_score', 0)
                                                    st.write(f"• {n['name']} (Confidence: {confidence})")
                                            else:
                                                st.write("*None found*")
                                        
                                        with col2:
                                            st.write("**Associated Genes:**")
                                            associated = [n for n in gene_neighbors if n['relationship']['type'] == 'ASSOCIATED_WITH']
                                            if associated:
                                                for n in associated:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                    else:
                                        st.info("No neighbors found for this disease.")
                        else:
                            st.info("No diseases found matching your search.")
                    
                    # Genes tab
                    with entity_tabs[2]:
                        genes = search_results.get('genes', [])
                        if genes:
                            df = pd.DataFrame(genes)
                            st.dataframe(df, use_container_width=True)
                            
                            # Select a gene for detailed view
                            if len(genes) > 0:
                                selected_gene = st.selectbox(
                                    "Select a gene for details",
                                    options=[(g['id'], g.get('symbol', g['name'])) for g in genes],
                                    format_func=lambda x: x[1]
                                )
                                
                                if selected_gene:
                                    gene_id = selected_gene[0]
                                    st.subheader(f"Neighbors of {selected_gene[1]}")
                                    
                                    # Get neighbors
                                    neighbors = neo4j_utils.get_neighbors(gene_id)
                                    
                                    if neighbors:
                                        # Split neighbors by type
                                        drug_neighbors = [n for n in neighbors if 'Drug' in n['labels']]
                                        disease_neighbors = [n for n in neighbors if 'Disease' in n['labels']]
                                        
                                        # Display split by relationship type
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Targeted By Drugs:**")
                                            targeted_by = [n for n in drug_neighbors if n['relationship']['type'] == 'TARGETS']
                                            if targeted_by:
                                                for n in targeted_by:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                        
                                        with col2:
                                            st.write("**Associated Diseases:**")
                                            associated = [n for n in disease_neighbors if n['relationship']['type'] == 'ASSOCIATED_WITH']
                                            if associated:
                                                for n in associated:
                                                    st.write(f"• {n['name']} ({n['id']})")
                                            else:
                                                st.write("*None found*")
                                    else:
                                        st.info("No neighbors found for this gene.")
                        else:
                            st.info("No genes found matching your search.")
                    
                    # Proteins tab
                    with entity_tabs[3]:
                        proteins = search_results.get('proteins', [])
                        if proteins:
                            df = pd.DataFrame(proteins)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No proteins found matching your search.")
                    
                    # Pathways tab
                    with entity_tabs[4]:
                        pathways = search_results.get('pathways', [])
                        if pathways:
                            df = pd.DataFrame(pathways)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No pathways found matching your search.")
                else:
                    st.warning("No results found matching your search.")
    
    with tab2:
        st.header("Migrate Existing Data to Neo4j")
        
        st.write("""
        This section allows you to migrate your existing data from the application's state or
        PostgreSQL database into the Neo4j graph database. This will create a more powerful
        knowledge graph representation of your data.
        """)
        
        # Add migration options
        migration_source = st.radio(
            "Migration Source",
            options=["Application State", "PostgreSQL Database"]
        )
        
        # Button to start migration
        if st.button("Start Migration"):
            with st.spinner("Migrating data to Neo4j..."):
                if migration_source == "Application State":
                    # Get data from session state
                    drugs = st.session_state.drugs
                    diseases = st.session_state.diseases
                    
                    # Convert NetworkX relationships to format for Neo4j
                    nx_relationships = []
                    if 'graph' in st.session_state and isinstance(st.session_state.graph, nx.Graph):
                        G = st.session_state.graph
                        
                        for u, v, data in G.edges(data=True):
                            rel = {
                                'source': u,
                                'target': v,
                                'type': data.get('type', 'UNKNOWN').upper(),
                                'confidence': data.get('confidence', 0.5)
                            }
                            nx_relationships.append(rel)
                    
                    # Migrate data to Neo4j
                    success = neo4j_utils.migrate_from_postgres(drugs, diseases, nx_relationships)
                    
                    if success:
                        st.success("Data migration completed successfully!")
                    else:
                        st.error("Data migration failed. Please check the logs for errors.")
                else:
                    # Use PostgreSQL data
                    from db_utils import get_drugs, get_diseases, get_drug_disease_relationships
                    
                    # Get data from PostgreSQL
                    drugs = get_drugs(limit=10000)
                    diseases = get_diseases(limit=10000)
                    db_relationships = get_drug_disease_relationships()
                    
                    # Convert DB relationship format to Neo4j format
                    relationships = []
                    for rel in db_relationships:
                        neo4j_rel = {
                            'source': rel['source_id'],
                            'target': rel['target_id'],
                            'type': rel['relationship_type'].upper(),
                            'confidence': float(rel['confidence']),
                            'evidence_count': rel['evidence_count']
                        }
                        relationships.append(neo4j_rel)
                    
                    # Migrate data to Neo4j
                    success = neo4j_utils.migrate_from_postgres(drugs, diseases, relationships)
                    
                    if success:
                        st.success("Data migration from PostgreSQL completed successfully!")
                    else:
                        st.error("Data migration failed. Please check the logs for errors.")
    
    with tab3:
        st.header("Path Analysis")
        
        st.write("""
        Analyze paths between drugs and diseases to understand potential mechanisms of action,
        discover new drug repurposing opportunities, and visualize the biological relationships.
        """)
        
        # Create a form for path analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Drug selection
            st.subheader("Select Drug")
            drug_name = st.text_input("Drug Name", placeholder="e.g., Aspirin")
            
            if drug_name:
                # Search for matching drugs
                search_results = neo4j_utils.search_entities(drug_name, entity_types=["Drug"])
                drugs = search_results.get('drugs', [])
                
                if drugs:
                    selected_drug = st.selectbox(
                        "Select Drug",
                        options=[(d['id'], d['name']) for d in drugs],
                        format_func=lambda x: x[1]
                    )
                    
                    if selected_drug:
                        st.session_state.selected_drug_id = selected_drug[0]
                        st.session_state.selected_drug_name = selected_drug[1]
                else:
                    st.warning(f"No drugs found matching '{drug_name}'")
        
        with col2:
            # Disease selection
            st.subheader("Select Disease")
            disease_name = st.text_input("Disease Name", placeholder="e.g., Alzheimer's Disease")
            
            if disease_name:
                # Search for matching diseases
                search_results = neo4j_utils.search_entities(disease_name, entity_types=["Disease"])
                diseases = search_results.get('diseases', [])
                
                if diseases:
                    selected_disease = st.selectbox(
                        "Select Disease",
                        options=[(d['id'], d['name']) for d in diseases],
                        format_func=lambda x: x[1]
                    )
                    
                    if selected_disease:
                        st.session_state.selected_disease_id = selected_disease[0]
                        st.session_state.selected_disease_name = selected_disease[1]
                else:
                    st.warning(f"No diseases found matching '{disease_name}'")
        
        # Path options
        st.subheader("Path Options")
        max_path_length = st.slider("Maximum Path Length", min_value=1, max_value=6, value=3)
        
        # Find paths button
        if st.button("Find Paths"):
            if 'selected_drug_id' in st.session_state and 'selected_disease_id' in st.session_state:
                with st.spinner("Finding paths..."):
                    # Get paths between drug and disease
                    paths = neo4j_utils.get_drug_disease_paths(
                        st.session_state.selected_drug_id,
                        st.session_state.selected_disease_id,
                        max_length=max_path_length
                    )
                    
                    if paths:
                        st.success(f"Found {len(paths)} paths between {st.session_state.selected_drug_name} and {st.session_state.selected_disease_name}")
                        
                        # Store paths in session state
                        st.session_state.drug_disease_paths = paths
                    else:
                        st.warning(f"No paths found between {st.session_state.selected_drug_name} and {st.session_state.selected_disease_name} with maximum length {max_path_length}")
            else:
                st.warning("Please select both a drug and a disease to find paths")
        
        # Display paths if available
        if 'drug_disease_paths' in st.session_state and st.session_state.drug_disease_paths:
            paths = st.session_state.drug_disease_paths
            
            # Display paths as a list
            for i, path in enumerate(paths):
                with st.expander(f"Path {i+1} (Length: {path['length']})"):
                    # Display nodes in the path
                    nodes = path['nodes']
                    relationships = path['relationships']
                    
                    # Create a visual representation of the path
                    path_str = " → ".join([f"{node['name']} ({node['labels'][0]})" for node in nodes])
                    st.write(path_str)
                    
                    # Create a simple path visualization
                    G = nx.DiGraph()
                    
                    # Add nodes
                    for node in nodes:
                        G.add_node(node['id'], name=node['name'], type=node['labels'][0])
                    
                    # Add edges (assuming relationships are in order)
                    for i in range(len(nodes) - 1):
                        relationship = relationships[i]
                        G.add_edge(
                            nodes[i]['id'], 
                            nodes[i+1]['id'], 
                            type=relationship['type']
                        )
                    
                    # Create a Plotly figure for the path
                    pos = nx.spring_layout(G)
                    
                    # Create edge trace
                    edge_x = []
                    edge_y = []
                    edge_text = []
                    
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_text.append(edge[2]['type'])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Create node trace
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    
                    for node in G.nodes(data=True):
                        x, y = pos[node[0]]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(f"{node[1]['name']} ({node[1]['type']})")
                        
                        # Color by node type
                        if node[1]['type'] == 'Drug':
                            node_color.append('#FF5733')  # Orange for drugs
                        elif node[1]['type'] == 'Disease':
                            node_color.append('#33A1FF')  # Blue for diseases
                        elif node[1]['type'] == 'Gene':
                            node_color.append('#33FF57')  # Green for genes
                        else:
                            node_color.append('#A233FF')  # Purple for others
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition='top center',
                        marker=dict(
                            showscale=False,
                            color=node_color,
                            size=15,
                            line_width=2
                        )
                    )
                    
                    # Create figure
                    fig = go.Figure(data=[edge_trace, node_trace],
                                   layout=go.Layout(
                                       showlegend=False,
                                       hovermode='closest',
                                       margin=dict(b=20, l=5, r=5, t=40),
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       plot_bgcolor='rgba(255,255,255,1)',
                                       title=f"Path from {nodes[0]['name']} to {nodes[-1]['name']}",
                                       height=300
                                   ))
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Community Detection")
        st.write("Identify clusters or communities within the knowledge graph to find groups of drugs, diseases, or genes that are closely related.")
        
        # Add note about optimization
        st.info("For optimal performance, community detection is limited to 200 nodes. This provides a clear visualization while maintaining good performance.")
        
        # Add cache key to enable caching
        from caching import get_session_cache_key
        cache_key = get_session_cache_key(['community_detection'])
        
        # Add a timestamp field for cache busting when user wants fresh data
        if 'last_community_refresh' not in st.session_state:
            st.session_state.last_community_refresh = datetime.now().isoformat()
        
        # Get the graph from Neo4j with reduced limit for better performance
        nx_graph = neo4j_utils.get_graph_for_visualization(limit=200)
        
        if nx_graph and nx_graph.number_of_nodes() > 0:
            # Show graph stats
            st.write(f"Graph loaded with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")
            
            # Add community detection algorithm options
            algo_type = st.selectbox(
                "Select Community Detection Algorithm",
                options=["Louvain Method", "Spectral Clustering", "Connected Components"]
            )
            
            # Customize algorithm parameters
            if algo_type == "Spectral Clustering":
                n_clusters = st.slider("Number of Clusters", 2, 15, 8)
            else:
                n_clusters = None
            
            # Add refresh button
            col1, col2 = st.columns([3, 1])
            with col1:
                detect_button = st.button("Detect Communities")
            with col2:
                if st.button("🔄 Refresh Data"):
                    st.session_state.last_community_refresh = datetime.now().isoformat()
                    st.rerun()
            
            if detect_button:
                with st.spinner("Detecting communities in the knowledge graph..."):
                    # Run community detection
                    if algo_type == "Louvain Method":
                        communities, node_data_df = community_detection.detect_communities_louvain(nx_graph)
                    elif algo_type == "Spectral Clustering":
                        communities, node_data_df = community_detection.detect_communities_spectral(nx_graph, n_clusters=n_clusters)
                    else:
                        # For connected components, use NetworkX's connected components
                        communities = {}
                        for i, component in enumerate(nx.connected_components(nx_graph.to_undirected())):
                            for node in component:
                                communities[node] = i
                        
                        # Create node data dataframe
                        node_data = []
                        for node_id, community_id in communities.items():
                            node_attrs = nx_graph.nodes[node_id]
                            node_data.append({
                                'id': node_id,
                                'name': node_attrs.get('name', ''),
                                'type': node_attrs.get('type', ''),
                                'community_id': community_id
                            })
                        node_data_df = pd.DataFrame(node_data)
                    
                    # Display community visualization
                    st.subheader("Community Visualization")
                    fig = community_detection.visualize_communities(nx_graph, communities, node_data_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display community distribution
                    st.subheader("Community Distribution")
                    fig2 = community_detection.visualize_community_distribution(node_data_df)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display insights
                    st.subheader("Community Insights")
                    insights = community_detection.generate_community_insights(node_data_df)
                    
                    for insight in insights.get('narrative', []):
                        st.write(f"• {insight}")
                    
                    # Display community data table
                    st.subheader("Community Data")
                    st.dataframe(node_data_df, use_container_width=True)
                    
                    # Add export functionality
                    st.subheader("Export Results")
                    export_format = st.selectbox(
                        "Export Format",
                        options=["CSV", "JSON"],
                        key="community_export_format"
                    )
                    
                    if export_format == "CSV":
                        _, href = graph_export.export_community_data(nx_graph, communities, "community_data.csv")
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        json_str, href = graph_export.export_graph_to_json(nx_graph, "graph_with_communities.json")
                        st.markdown(href, unsafe_allow_html=True)
            
            # Add related analysis options
            with st.expander("Centrality Analysis"):
                st.write("Calculate centrality metrics for nodes in the graph to identify important entities")
                
                if st.button("Run Centrality Analysis"):
                    with st.spinner("Calculating centrality metrics..."):
                        centrality_df = community_detection.calculate_centrality_metrics(nx_graph)
                        
                        # Display top nodes by centrality
                        st.subheader("Top Nodes by Centrality")
                        
                        # Get top 10 nodes by different centrality metrics
                        centrality_metrics = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality']
                        
                        for metric in centrality_metrics:
                            sorted_df = centrality_df.sort_values(metric, ascending=False).head(10)
                            
                            fig = px.bar(
                                sorted_df, 
                                x='name', 
                                y=metric,
                                color='type',
                                title=f"Top Nodes by {metric.replace('_', ' ').title()}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display centrality data table
                        st.subheader("Centrality Metrics")
                        st.dataframe(centrality_df, use_container_width=True)
                        
                        # Add export functionality
                        metrics_df, href = graph_export.export_network_metrics(nx_graph, "network_metrics.csv")
                        st.markdown(href, unsafe_allow_html=True)
            
            # Add drug similarity analysis
            with st.expander("Drug Similarity Analysis"):
                st.write("Find drugs that are similar based on their position in the knowledge graph")
                
                # Get drugs
                drugs = [n for n, attr in nx_graph.nodes(data=True) if attr.get('type') == 'drug']
                
                if drugs:
                    drug_nodes = [(drug_id, nx_graph.nodes[drug_id].get('name', drug_id)) for drug_id in drugs]
                    
                    selected_drug = st.selectbox(
                        "Select a drug",
                        options=drug_nodes,
                        format_func=lambda x: x[1]
                    )
                    
                    similarity_method = st.selectbox(
                        "Similarity Method",
                        options=["Graph Structure", "Common Neighbors", "Community"]
                    )
                    
                    if st.button("Find Similar Drugs"):
                        with st.spinner("Finding similar drugs..."):
                            method_map = {
                                "Graph Structure": "graph_structure",
                                "Common Neighbors": "common_neighbors",
                                "Community": "community"
                            }
                            
                            similar_drugs = community_detection.find_similar_drugs(
                                nx_graph, 
                                selected_drug[0], 
                                method=method_map[similarity_method]
                            )
                            
                            if not similar_drugs.empty:
                                st.subheader(f"Drugs Similar to {selected_drug[1]}")
                                st.dataframe(similar_drugs, use_container_width=True)
                                
                                # Visualize similarity
                                fig = px.bar(
                                    similar_drugs,
                                    x='drug_name',
                                    y='similarity',
                                    title=f"Drugs Similar to {selected_drug[1]}",
                                    color='similarity',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add export functionality
                                href = graph_export.export_recommendation_results(similar_drugs, f"similar_drugs_to_{selected_drug[0]}.csv")
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.info("No similar drugs found.")
                else:
                    st.info("No drug nodes found in the graph.")
        else:
            st.info("No graph data available for community detection. Please migrate some data to Neo4j first.")
    
    with tab5:
        st.header("Recommendation Engine")
        st.write("Advanced recommendation engine that suggests new drug repurposing candidates based on graph patterns, similarity metrics, and community structure.")
        
        # Get the graph from Neo4j
        nx_graph = neo4j_utils.get_graph_for_visualization(limit=1000)
        
        if nx_graph and nx_graph.number_of_nodes() > 0:
            # Create tabs for different recommendation types
            adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Drug Repurposing Candidates", "Find Similar Drugs", "Diseases for Drug"])
            
            # Button to calculate centrality
            if st.button("Calculate Centrality Measures"):
                with st.spinner("Calculating centrality measures..."):
                    centrality_results = neo4j_utils.calculate_centrality_measures()
                    
                    if centrality_results:
                        # Store in session state
                        st.session_state.centrality_results = centrality_results
                        st.success("Centrality measures calculated successfully!")
                    else:
                        st.warning("Could not calculate centrality measures. The graph may be empty or not properly configured.")
            
            # Display centrality results if available
            if 'centrality_results' in st.session_state and st.session_state.centrality_results:
                results = st.session_state.centrality_results
                
                # Create columns for different node types
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Most Central Drugs**")
                    drug_results = results.get('drugs', [])
                    if drug_results:
                        for drug in drug_results:
                            st.write(f"• {drug['name']} (Score: {drug['degree_centrality']:.2f})")
                    else:
                        st.write("*No drug centrality data available*")
                
                with col2:
                    st.write("**Most Central Diseases**")
                    disease_results = results.get('diseases', [])
                    if disease_results:
                        for disease in disease_results:
                            st.write(f"• {disease['name']} (Score: {disease['degree_centrality']:.2f})")
                    else:
                        st.write("*No disease centrality data available*")
                
                with col3:
                    st.write("**Most Central Genes**")
                    gene_results = results.get('genes', [])
                    if gene_results:
                        for gene in gene_results:
                            st.write(f"• {gene['name']} (Score: {gene['degree_centrality']:.2f})")
                    else:
                        st.write("*No gene centrality data available*")
            
            with adv_tab2:
                st.subheader("Find Similar Drugs")
                
                st.write("""
                Find drugs similar to a selected drug based on shared targets, pathways, or mechanisms.
                This can help identify potential drug repurposing candidates.
                """)
                
                # Drug selection
                drug_name = st.text_input("Drug Name for Similarity Search", placeholder="e.g., Aspirin")
                
                if drug_name:
                    # Search for matching drugs
                    search_results = neo4j_utils.search_entities(drug_name, entity_types=["Drug"])
                    drugs = search_results.get('drugs', [])
                    
                    if drugs:
                        selected_drug = st.selectbox(
                            "Select Drug for Similarity Analysis",
                            options=[(d['id'], d['name']) for d in drugs],
                            format_func=lambda x: x[1],
                            key="similar_drug_select"
                        )
                        
                        if selected_drug:
                            drug_id = selected_drug[0]
                            # Button to find similar drugs
                            if st.button("Find Similar Drugs"):
                                with st.spinner("Finding similar drugs..."):
                                    similar_drugs = neo4j_utils.find_similar_drugs(drug_id)
                                    
                                    if similar_drugs:
                                        # Store in session state
                                        st.session_state.similar_drugs = similar_drugs
                                        st.success(f"Found {len(similar_drugs)} drugs similar to {selected_drug[1]}")
                                    else:
                                        st.warning(f"No drugs found similar to {selected_drug[1]}")
                    else:
                        st.warning(f"No drugs found matching '{drug_name}'")
                
                # Display similar drugs if available
                if 'similar_drugs' in st.session_state and st.session_state.similar_drugs:
                    similar_drugs = st.session_state.similar_drugs
                    
                    # Convert to DataFrame for display
                    df = pd.DataFrame([
                        {
                            "Drug Name": drug['drug_name'],
                            "Similarity Score": drug['similarity_score'],
                            "Shared Entities": len(drug['shared_entities']),
                            "Details": ", ".join(drug['shared_entities'][:3]) + ("..." if len(drug['shared_entities']) > 3 else "")
                        }
                        for drug in similar_drugs
                    ])
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Create a visualziation of similarity
                    if len(similar_drugs) > 0:
                        # Create a bar chart of similarity scores
                        import plotly.express as px
                        
                        fig = px.bar(
                            df, 
                            x="Drug Name", 
                            y="Similarity Score",
                            title="Drug Similarity Scores",
                            color="Similarity Score",
                            hover_data=["Shared Entities"]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with adv_tab3:
                st.subheader("Identify Repurposing Opportunities")
                
                st.write("""
                Identify potential drug repurposing opportunities by analyzing the graph structure.
                This looks for drugs that target disease-associated genes but aren't currently
                indicated for those diseases.
                """)
                
                # Disease selection
                disease_name = st.text_input("Disease Name for Repurposing Search", placeholder="e.g., Alzheimer's Disease")
                
                if disease_name:
                    # Search for matching diseases
                    search_results = neo4j_utils.search_entities(disease_name, entity_types=["Disease"])
                    diseases = search_results.get('diseases', [])
                    
                    if diseases:
                        selected_disease = st.selectbox(
                            "Select Disease for Repurposing Analysis",
                            options=[(d['id'], d['name']) for d in diseases],
                            format_func=lambda x: x[1],
                            key="repurposing_disease_select"
                        )
                        
                        if selected_disease:
                            disease_id = selected_disease[0]
                            
                            # Confidence threshold
                            min_confidence = st.slider(
                                "Minimum Confidence Score", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=0.5,
                                step=0.1
                            )
                            
                            # Button to find repurposing opportunities
                            if st.button("Find Repurposing Opportunities"):
                                with st.spinner("Finding repurposing opportunities..."):
                                    opportunities = neo4j_utils.find_repurposing_opportunities(
                                        disease_id, 
                                        min_confidence=min_confidence
                                    )
                                    
                                    if opportunities:
                                        # Store in session state
                                        st.session_state.repurposing_opportunities = opportunities
                                        st.success(f"Found {len(opportunities)} potential repurposing opportunities for {selected_disease[1]}")
                                    else:
                                        st.warning(f"No repurposing opportunities found for {selected_disease[1]}")
                    else:
                        st.warning(f"No diseases found matching '{disease_name}'")
                
                # Display repurposing opportunities if available
                if 'repurposing_opportunities' in st.session_state and st.session_state.repurposing_opportunities:
                    opportunities = st.session_state.repurposing_opportunities
                    
                    # Convert to DataFrame for display
                    df = pd.DataFrame([
                        {
                            "Drug": opp['drug_name'],
                            "Disease": opp['disease_name'],
                            "Confidence": f"{opp['confidence_score']:.2f}",
                            "Common Targets": opp['common_targets'],
                            "Target Genes": ", ".join(opp['target_genes'][:3]) + ("..." if len(opp['target_genes']) > 3 else "")
                        }
                        for opp in opportunities
                    ])
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Create a visualization of repurposing opportunities
                    if len(opportunities) > 0:
                        # Create a bar chart of confidence scores
                        import plotly.express as px
                        
                        fig = px.bar(
                            opportunities, 
                            x="drug_name", 
                            y="confidence_score",
                            title="Drug Repurposing Confidence Scores",
                            color="confidence_score",
                            hover_data=["common_targets"],
                            labels={"drug_name": "Drug", "confidence_score": "Confidence Score"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to add selected opportunity to the database
                    st.subheader("Add Repurposing Candidate to Database")
                    
                    # Select a drug from the opportunities
                    selected_opportunity = st.selectbox(
                        "Select an opportunity to add as repurposing candidate",
                        options=[(i, f"{opp['drug_name']} for {opp['disease_name']}") for i, opp in enumerate(opportunities)],
                        format_func=lambda x: x[1]
                    )
                    
                    if selected_opportunity:
                        opp_index = selected_opportunity[0]
                        opp = opportunities[opp_index]
                        
                        # Button to add to database
                        if st.button("Add to Repurposing Candidates", key="add_neo4j_candidate_btn"):
                            # Add to both Neo4j and PostgreSQL/session state
                            
                            # First, add to Neo4j
                            neo4j_success = neo4j_utils.create_repurposing_candidate({
                                'drug_id': opp['drug_id'],
                                'disease_id': opp['disease_id'],
                                'confidence_score': opp['confidence_score'],
                                'mechanism': f"Targets {len(opp['target_genes'])} disease-associated genes",
                                'source': 'Neo4j Graph Analysis',
                                'evidence': json.dumps({'target_genes': opp['target_genes']})
                            })
                            
                            # Then, add to PostgreSQL/session state
                            from utils import add_repurposing_candidate
                            
                            candidate_data = {
                                'drug_id': opp['drug_id'],
                                'drug_name': opp['drug_name'],
                                'disease_id': opp['disease_id'],
                                'disease_name': opp['disease_name'],
                                'confidence': int(opp['confidence_score'] * 100),
                                'mechanism': f"Targets {len(opp['target_genes'])} disease-associated genes",
                                'source': 'Neo4j Graph Analysis',
                                'evidence': json.dumps({'target_genes': opp['target_genes']})
                            }
                            
                            candidate_id, message = add_repurposing_candidate(candidate_data)
                            
                            if neo4j_success and "successfully" in message:
                                st.success(f"Repurposing candidate added successfully to both Neo4j and the database (ID: {candidate_id})")
                            elif neo4j_success:
                                st.warning(f"Repurposing candidate added to Neo4j but not to the database: {message}")
                            elif "successfully" in message:
                                st.warning(f"Repurposing candidate added to the database but not to Neo4j (ID: {candidate_id})")
                            else:
                                st.error(f"Failed to add repurposing candidate: {message}")

    with tab6:
        st.header("Time-based Analysis")
        
        st.write("""
        Analyze how the knowledge graph evolves over time, tracking the addition of new drugs, diseases, 
        and relationships. Identify trends and emerging patterns in the data.
        """)
        
        # Create tabs for different time-based analyses
        time_tab1, time_tab2 = st.tabs(["Growth Analysis", "Trend Detection"])
        
        with time_tab1:
            st.subheader("Graph Growth Over Time")
            
            # Get graph growth stats from Neo4j, or use demo data if not available
            if neo4j_available:
                try:
                    # Try to get real data from time_analysis module if available
                    from time_analysis import get_graph_growth_over_time
                    growth_data = get_graph_growth_over_time()
                    if not growth_data:
                        raise Exception("No time-based data available")
                except Exception as e:
                    # Generate demo data
                    import datetime
                    today = datetime.datetime.now()
                    dates = [(today - datetime.timedelta(days=30*i)).strftime('%Y-%m') for i in range(6)]
                    dates.reverse()
                    
                    growth_data = pd.DataFrame({
                        'month': dates,
                        'drugs': [12, 18, 23, 32, 45, 56],
                        'diseases': [15, 22, 28, 35, 42, 48],
                        'relationships': [23, 45, 78, 110, 156, 203]
                    })
            else:
                # Generate demo data
                import datetime
                today = datetime.datetime.now()
                dates = [(today - datetime.timedelta(days=30*i)).strftime('%Y-%m') for i in range(6)]
                dates.reverse()
                
                growth_data = pd.DataFrame({
                    'month': dates,
                    'drugs': [12, 18, 23, 32, 45, 56],
                    'diseases': [15, 22, 28, 35, 42, 48],
                    'relationships': [23, 45, 78, 110, 156, 203]
                })
            
            # Create growth visualization
            fig = px.line(
                growth_data,
                x='month',
                y=['drugs', 'diseases', 'relationships'],
                title="Knowledge Graph Growth Over Time",
                labels={'value': 'Count', 'variable': 'Entity Type', 'month': 'Month'},
                markers=True,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Growth Data")
            st.dataframe(growth_data, use_container_width=True)
            
        with time_tab2:
            st.subheader("Emerging Trends")
            
            # Get trending entities from Neo4j, or use demo data if not available
            if neo4j_available:
                try:
                    # Try to get real data from time_analysis module if available
                    from time_analysis import get_trending_entities
                    trending_data = get_trending_entities()
                    if not trending_data:
                        raise Exception("No trending data available")
                except Exception as e:
                    # Generate demo data
                    trending_data = pd.DataFrame({
                        'entity': ['Metformin', 'GLP-1 Receptor', 'Alzheimer\'s Disease', 'SGLT2 Inhibitors', 'Cancer Immunotherapy'],
                        'type': ['Drug', 'Target', 'Disease', 'Drug Class', 'Therapy'],
                        'trend_score': [0.92, 0.87, 0.82, 0.76, 0.71],
                        'new_connections': [15, 12, 10, 8, 7]
                    })
            else:
                # Generate demo data
                trending_data = pd.DataFrame({
                    'entity': ['Metformin', 'GLP-1 Receptor', 'Alzheimer\'s Disease', 'SGLT2 Inhibitors', 'Cancer Immunotherapy'],
                    'type': ['Drug', 'Target', 'Disease', 'Drug Class', 'Therapy'],
                    'trend_score': [0.92, 0.87, 0.82, 0.76, 0.71],
                    'new_connections': [15, 12, 10, 8, 7]
                })
            
            # Create trending visualization
            fig = px.bar(
                trending_data,
                x='entity',
                y='trend_score',
                color='type',
                title="Trending Biomedical Entities",
                labels={'entity': 'Entity', 'trend_score': 'Trend Score', 'type': 'Entity Type'},
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with details
            st.subheader("Trending Entities")
            st.dataframe(trending_data, use_container_width=True)
            
            st.markdown("""
            **What is Trend Score?**
            
            The trend score indicates how rapidly an entity is gaining new connections in the knowledge graph.
            Entities with high trend scores are experiencing increased research interest and may represent
            emerging areas of focus in drug repurposing research.
            """)

    with tab7:
        st.header("Cypher Query Interface")
        
        st.write("""
        Advanced users can directly query the Neo4j graph database using Cypher, the graph query language.
        This interface allows for custom, complex queries beyond what the UI provides.
        """)
        
        # Create columns for the query interface
        query_col1, query_col2 = st.columns([2, 1])
        
        with query_col1:
            st.subheader("Cypher Query Editor")
            
            # Query templates
            query_templates = {
                "Get all drugs": "MATCH (d:Drug) RETURN d.id AS id, d.name AS name LIMIT 10",
                "Get all diseases": "MATCH (d:Disease) RETURN d.id AS id, d.name AS name LIMIT 10",
                "Find treatments for a disease": "MATCH (d:Drug)-[r:TREATS]->(dis:Disease) WHERE dis.name CONTAINS 'DISEASE_NAME' RETURN d.name AS drug, dis.name AS disease",
                "Find common targets": "MATCH (d1:Drug)-[:TARGETS]->(g:Gene)<-[:TARGETS]-(d2:Drug) WHERE d1.id <> d2.id RETURN d1.name AS drug1, d2.name AS drug2, g.name AS gene, COUNT(g) AS common_targets ORDER BY common_targets DESC LIMIT 20",
                "Find potential treatments": "MATCH (d:Drug)-[:TARGETS]->(g:Gene)<-[:ASSOCIATED_WITH]-(dis:Disease) WHERE NOT (d)-[:TREATS]->(dis) RETURN d.name AS drug, dis.name AS disease, COLLECT(g.name) AS genes, COUNT(g) AS gene_count ORDER BY gene_count DESC LIMIT 20",
                "Community Analysis": "MATCH (n) WITH n.community_id AS community, COUNT(*) AS size RETURN community, size ORDER BY size DESC LIMIT 10"
            }
            
            # Template selector
            selected_template = st.selectbox(
                "Select a query template",
                options=list(query_templates.keys()),
                key="cypher_template"
            )
            
            # Query text area
            cypher_query = st.text_area(
                "Cypher Query",
                value=query_templates[selected_template],
                height=200,
                key="cypher_query"
            )
            
            # Query parameters
            st.subheader("Query Parameters (JSON)")
            query_params = st.text_area(
                "Parameters",
                value="{}",
                height=100,
                key="cypher_params"
            )
            
            # Execute button
            if st.button("Execute Query", key="execute_cypher"):
                if not cypher_query.strip():
                    st.error("Please enter a query.")
                else:
                    with st.spinner("Executing query..."):
                        try:
                            # Parse parameters if provided
                            params = {}
                            if query_params.strip():
                                params = json.loads(query_params)
                            
                            # Import cypher_query module if available
                            from cypher_query import execute_cypher_query, format_cypher_query
                            
                            # Format query for better readability
                            formatted_query = format_cypher_query(cypher_query)
                            
                            # Execute query
                            success, result, message = execute_cypher_query(formatted_query, params)
                            
                            if success:
                                st.session_state.cypher_result = result
                                st.success(f"Query executed successfully. {len(result)} records returned.")
                            else:
                                st.error(f"Query failed: {message}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
        with query_col2:
            st.subheader("Query Help")
            
            st.markdown("""
            **Cypher Query Language**
            
            Cypher is Neo4j's graph query language that allows for querying and updating the graph database.
            
            **Basic Patterns:**
            - `(n)` - A node
            - `[r]` - A relationship
            - `(a)-[r]->(b)` - A directed relationship
            
            **Common Clauses:**
            - `MATCH` - Pattern matching
            - `WHERE` - Filter conditions
            - `RETURN` - Return results
            - `ORDER BY` - Sort results
            - `LIMIT` - Limit results
            
            **Example:**
            ```
            MATCH (d:Drug)-[r:TREATS]->(dis:Disease)
            WHERE d.name CONTAINS 'metformin'
            RETURN d.name AS drug, dis.name AS disease
            ```
            """)
            
            # Add query validation
            if cypher_query.strip():
                try:
                    # Import cypher_query if available 
                    from cypher_query import validate_cypher_query
                    is_valid, validation_message = validate_cypher_query(cypher_query)
                    
                    if is_valid:
                        st.success("Query syntax valid")
                    else:
                        st.warning(f"Query validation: {validation_message}")
                except:
                    # Skip validation if module not available
                    pass
        
        # Display query results if available
        if 'cypher_result' in st.session_state and st.session_state.cypher_result:
            st.subheader("Query Results")
            
            results = st.session_state.cypher_result
            
            # Convert to DataFrame for display
            try:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Add CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results (CSV)",
                    csv,
                    "cypher_results.csv",
                    "text/csv",
                    key="download_cypher_csv"
                )
                
                # Visualization toggle
                if st.checkbox("Visualize Results", key="viz_cypher_results"):
                    try:
                        # Import visualization module
                        from cypher_query import visualize_query_results, create_graph_visualization
                        
                        # Create visualization based on result structure
                        if len(df.columns) >= 2:
                            fig = create_graph_visualization(df)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Results cannot be visualized as a graph. Need at least 2 columns.")
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
            except Exception as e:
                # Display as JSON if DataFrame conversion fails
                st.json(results)

    with tab8:
        st.header("Graph Export")
        
        st.write("Export the knowledge graph or parts of it for external use and analysis.")
        
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "JSON", "GraphML", "GEXF (Gephi)"],
            key="export_format"
        )
        
        export_scope = st.radio(
            "Export Scope",
            options=["Complete Graph", "Current Visualization", "Query Results"],
            key="export_scope"
        )
        
        if st.button("Export Graph", key="export_graph_btn"):
            with st.spinner("Preparing export..."):
                try:
                    # Import graph_export module if available
                    from graph_export import export_graph
                    
                    # Get the right graph based on scope
                    if export_scope == "Complete Graph":
                        graph_to_export = neo4j_utils.get_graph_for_visualization(limit=5000)
                    elif export_scope == "Current Visualization":
                        graph_to_export = neo4j_utils.get_graph_for_visualization()
                    else:  # Query Results
                        if 'cypher_result' in st.session_state and st.session_state.cypher_result:
                            # Convert query results to graph
                            from cypher_query import create_graph_from_results
                            graph_to_export = create_graph_from_results(st.session_state.cypher_result)
                        else:
                            st.error("No query results available to export.")
                            graph_to_export = None
                    
                    if graph_to_export:
                        # Map format to file extension
                        format_map = {
                            "CSV": "csv",
                            "JSON": "json",
                            "GraphML": "graphml",
                            "GEXF (Gephi)": "gexf"
                        }
                        
                        extension = format_map[export_format]
                        filename = f"drug_repurposing_graph.{extension}"
                        
                        # Export the graph
                        if export_format == "CSV":
                            node_csv, edge_csv = export_graph(graph_to_export, format="csv")
                            
                            # Provide download buttons for both files
                            st.download_button(
                                "Download Nodes (CSV)",
                                node_csv,
                                f"drug_repurposing_nodes.csv",
                                "text/csv",
                                key="download_nodes_csv"
                            )
                            
                            st.download_button(
                                "Download Edges (CSV)",
                                edge_csv,
                                f"drug_repurposing_edges.csv",
                                "text/csv",
                                key="download_edges_csv"
                            )
                        else:
                            graph_data = export_graph(graph_to_export, format=extension.lower())
                            
                            # Determine MIME type based on format
                            mime_map = {
                                "json": "application/json",
                                "graphml": "application/xml",
                                "gexf": "application/xml"
                            }
                            
                            mime_type = mime_map.get(extension.lower(), "application/octet-stream")
                            
                            # Provide download button
                            st.download_button(
                                f"Download Graph ({export_format})",
                                graph_data,
                                filename,
                                mime_type,
                                key=f"download_graph_{extension}"
                            )
                        
                        st.success(f"Graph exported successfully as {export_format}.")
                except Exception as e:
                    st.error(f"Export error: {str(e)}")

# Include documentation in the sidebar
st.sidebar.title("Neo4j Graph Explorer")

st.sidebar.markdown("""
### About Neo4j Integration

This integration adds powerful graph database capabilities to the Drug Repurposing Engine:

- **Enhanced Knowledge Graph**: Store and query complex relationships between drugs, diseases, genes, proteins, and pathways.
- **Path Discovery**: Find biological pathways between drugs and diseases to understand mechanisms of action.
- **Advanced Analytics**: Calculate centrality measures, find similar drugs, and identify repurposing opportunities.
- **Scalability**: Neo4j is designed to handle billions of nodes and relationships, allowing the system to scale with growing data.
""")

st.sidebar.markdown("""
### Graph Database Benefits

- **Superior Performance**: Query complex relationships in milliseconds, even with billions of connections.
- **Pattern Recognition**: Discover non-obvious connections between biological entities.
- **Visualization**: Natively visualize complex networks of biological entities and their relationships.
- **Inference**: Leverage graph algorithms to infer new knowledge from existing data.
""")

# Status indicator for Neo4j connection
st.sidebar.markdown("---")
st.sidebar.subheader("Neo4j Connection Status")

if neo4j_available:
    st.sidebar.markdown("✅ **Connected to Neo4j**")
else:
    st.sidebar.markdown("❌ **Not connected to Neo4j**")
    st.sidebar.markdown("Provide connection details in the form above to connect.")