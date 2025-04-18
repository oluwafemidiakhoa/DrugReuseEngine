import streamlit as st
import pandas as pd
import numpy as np
from utils import get_disease_by_name, get_pubmed_data, get_disease_by_id, initialize_session_state
from data_ingestion import fetch_disease_info
from knowledge_graph import get_drug_disease_relationships, find_paths_between
from visualization import create_network_graph
from ai_analysis import analyze_repurposing_candidate

# Set page configuration
st.set_page_config(
    page_title="Disease Search | Drug Repurposing Engine",
    page_icon="🧬",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Advanced Disease Search & Analysis")
st.write("Search for diseases and explore their characteristics, known treatments, and repurposing opportunities")

# Advanced search options
with st.expander("Advanced Search Options", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        search_by_category = st.checkbox("Search by Disease Category")
        if search_by_category:
            disease_categories = st.multiselect(
                "Disease Categories",
                options=["Metabolic", "Neurological", "Cardiovascular", "Autoimmune", 
                         "Infectious", "Cancer", "Respiratory", "Other"],
                default=[]
            )
    
    with col2:
        search_by_pathway = st.checkbox("Search by Biological Pathway")
        if search_by_pathway:
            pathway_keywords = st.multiselect(
                "Biological Pathways",
                options=["inflammatory", "immune", "metabolic", "signaling", "apoptosis",
                        "cell cycle", "oxidative stress", "hormonal", "neuronal", "vascular"],
                default=[]
            )

# Search input
search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
with search_col1:
    search_query = st.text_input("Search for a disease:", "")

with search_col2:
    search_button = st.button("Search", type="primary")
    
with search_col3:
    search_type = st.selectbox(
        "Match Type",
        options=["Contains", "Exact Match"],
        index=0
    )

if search_query or search_button:
    # Perform disease search with the advanced options
    if search_query:
        if search_type == "Exact Match":
            disease = get_disease_by_name(search_query, exact_match=True)
        else:
            disease = get_disease_by_name(search_query, exact_match=False)
            
        # If advanced search is enabled but no direct match was found, search all diseases
        if not disease and (search_by_category or search_by_pathway):
            # Get all diseases for advanced filtering
            all_diseases = st.session_state.diseases
            matching_diseases = []
            
            for disease_item in all_diseases:
                match = True
                
                # Apply category filter if enabled
                if search_by_category and disease_categories:
                    disease_category = disease_item.get('category', '').lower()
                    if not any(category.lower() in disease_category for category in disease_categories):
                        match = False
                
                # Apply pathway filter if enabled
                if search_by_pathway and pathway_keywords:
                    # Get description to search for pathway terms
                    description = disease_item.get('description', '').lower()
                    if not any(pathway.lower() in description for pathway in pathway_keywords):
                        match = False
                
                # Add to matches if all filters pass
                if match:
                    matching_diseases.append(disease_item)
            
            # Display matching diseases for user selection
            if matching_diseases:
                st.success(f"Found {len(matching_diseases)} diseases matching your advanced criteria")
                
                # Display as a selectable table
                disease_df = pd.DataFrame([
                    {
                        "Name": d['name'],
                        "Category": d['category'],
                        "Description": d['description'][:100] + "..." if len(d['description']) > 100 else d['description'],
                        "ID": d['id']
                    }
                    for d in matching_diseases
                ])
                
                # Show the table
                st.dataframe(disease_df, hide_index=True)
                
                # Allow selecting a disease
                selected_disease_id = st.selectbox(
                    "Select a disease to view details:",
                    options=[(d['id'], d['name']) for d in matching_diseases],
                    format_func=lambda x: x[1]
                )
                
                if selected_disease_id:
                    disease = next((d for d in matching_diseases if d['id'] == selected_disease_id[0]), None)
            else:
                st.warning("No diseases found matching your advanced criteria.")
    
    if not disease:
        st.info(f"Disease '{search_query}' not found in the database.")
        
        # Option to add disease information from external API
        st.write("Would you like to add this disease to the database?")
        
        if st.button("Add from MeSH database"):
            with st.spinner("Fetching disease information..."):
                disease_info = fetch_disease_info(search_query)
                
                if disease_info:
                    # Format as a disease record
                    new_disease = {
                        "name": disease_info["name"],
                        "description": disease_info["description"],
                        "category": "Unknown"
                    }
                    
                    # Display disease information
                    st.success(f"Found disease information for {disease_info['name']}")
                    
                    # Edit fields
                    with st.form("add_disease_form"):
                        new_disease["name"] = st.text_input("Disease Name", new_disease["name"])
                        new_disease["description"] = st.text_area("Description", new_disease["description"])
                        new_disease["category"] = st.selectbox("Category", 
                            ["Metabolic", "Neurological", "Cardiovascular", "Autoimmune", 
                             "Infectious", "Cancer", "Respiratory", "Other"],
                            index=7)
                        
                        # Submit button
                        submitted = st.form_submit_button("Add to Database")
                        
                        if submitted:
                            from utils import add_disease
                            disease_id, message = add_disease(new_disease)
                            
                            st.success(message)
                            
                            # Update disease variable to continue with display
                            disease = get_disease_by_id(disease_id)
                else:
                    st.error("Could not find disease information in MeSH database.")
        
        # Manual entry option
        with st.expander("Or add disease manually"):
            with st.form("manual_disease_form"):
                new_disease = {
                    "name": search_query,
                    "description": "",
                    "category": "Other"
                }
                
                new_disease["name"] = st.text_input("Disease Name", new_disease["name"])
                new_disease["description"] = st.text_area("Description", new_disease["description"])
                new_disease["category"] = st.selectbox("Category", 
                    ["Metabolic", "Neurological", "Cardiovascular", "Autoimmune", 
                     "Infectious", "Cancer", "Respiratory", "Other"],
                    index=7)
                
                # Submit button
                submitted = st.form_submit_button("Add to Database")
                
                if submitted:
                    from utils import add_disease
                    disease_id, message = add_disease(new_disease)
                    
                    st.success(message)
                    
                    # Update disease variable to continue with display
                    disease = get_disease_by_id(disease_id)
    
    # If we now have a disease (either found or added), display its information
    if disease:
        st.header(disease['name'])
        
        # Create tabs for different views of disease data
        disease_tabs = st.tabs(["Overview", "Detailed Analysis", "Treatment Options", "Literature"])
        
        # Get drug-disease relationships
        relationships = get_drug_disease_relationships(st.session_state.graph, disease_id=disease['id'])
        
        with disease_tabs[0]:  # Overview Tab
            # Display disease information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Disease Details")
                st.write(f"**Description:** {disease['description']}")
                st.write(f"**Category:** {disease['category']}")
                
                # Extract and show pathway keywords if available
                G = st.session_state.graph
                pathways = []
                try:
                    pathways = G.nodes[disease['id']].get('pathways', [])
                except:
                    # Fallback if not in graph or older structure
                    from knowledge_graph import extract_disease_pathways
                    pathways = extract_disease_pathways(disease.get('description', ''))
                
                if pathways:
                    st.write("**Biological Pathways:**")
                    for pathway in pathways:
                        st.markdown(f"<span style='background-color:#f3e6ff; padding:3px 8px; border-radius:10px; margin-right:5px;'>{pathway}</span>", unsafe_allow_html=True)
                
                # Extract and show body systems if available
                systems = []
                try:
                    systems = G.nodes[disease['id']].get('systems', [])
                except:
                    # Fallback if not in graph or older structure
                    from knowledge_graph import infer_body_systems
                    systems = infer_body_systems(disease)
                
                if systems and systems[0] != 'unknown':
                    st.write("**Affected Body Systems:**")
                    for system in systems:
                        st.markdown(f"<span style='background-color:#e6fff3; padding:3px 8px; border-radius:10px; margin-right:5px;'>{system}</span>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Known Treatments")
                
                if relationships:
                    # Filter to only 'treats' relationships
                    treats_relationships = [r for r in relationships if r['type'] == 'treats']
                    
                    if treats_relationships:
                        for rel in treats_relationships:
                            st.write(f"• **{rel['drug_name']}** (Confidence: {rel['confidence']:.2f})")
                    else:
                        st.write("No known treatments in the database.")
                else:
                    st.write("No treatments found for this disease.")
                
                # Show similar diseases based on pathways
                st.subheader("Similar Diseases")
                
                similar_diseases = []
                try:
                    # Look for similar_pathway edges
                    for source, target, attr in G.out_edges(disease['id'], data=True):
                        if attr['type'] == 'similar_pathway' and G.nodes[target]['type'] == 'disease':
                            similar_diseases.append((target, G.nodes[target]['name'], attr['confidence']))
                    for source, target, attr in G.in_edges(disease['id'], data=True):
                        if attr['type'] == 'similar_pathway' and G.nodes[source]['type'] == 'disease':
                            similar_diseases.append((source, G.nodes[source]['name'], attr['confidence']))
                except:
                    # Fallback for older graph structure
                    similar_diseases = []
                
                if similar_diseases:
                    for disease_id, disease_name, confidence in similar_diseases:
                        st.write(f"• **{disease_name}** (Similarity: {confidence:.2f})")
                else:
                    st.write("No similar diseases found based on pathways.")
                    
        with disease_tabs[1]:  # Detailed Analysis Tab
            st.subheader("Pathophysiology Analysis")
            
            # Extract key biological terms from description
            description = disease.get('description', 'No description available')
            
            # Show word cloud or key terms visualization
            if description and description != 'No description available':
                try:
                    # Generate word frequency for visualization
                    from collections import Counter
                    import re
                    
                    # Clean and tokenize text
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
                    
                    # Remove common stopwords
                    stopwords = ['the', 'and', 'for', 'with', 'that', 'this', 'its', 'has', 'are', 'disease']
                    words = [word for word in words if word not in stopwords]
                    
                    word_freq = Counter(words).most_common(10)
                    
                    # Create horizontal bar chart
                    if word_freq:
                        import plotly.express as px
                        
                        df = pd.DataFrame(word_freq, columns=['Term', 'Frequency'])
                        fig = px.bar(df, x='Frequency', y='Term', orientation='h',
                                    title="Key Terms in Disease Description",
                                    color='Frequency',
                                    color_continuous_scale='Viridis')
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True, key="disease_term_freq_chart")
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
            else:
                st.info("Detailed description information not available.")
            
            # Graph-based analysis
            st.subheader("Network Position Analysis")
            
            # Analyze disease's position in the knowledge network
            try:
                # Calculate various centrality measures for this disease
                disease_id = disease['id']
                degree = len(list(G.edges(disease_id)))
                
                # Calculate clustering coefficient
                import networkx as nx
                try:
                    clustering = nx.clustering(G.to_undirected(), disease_id)
                except:
                    clustering = 0
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Network Connections", degree)
                    st.write("Number of direct connections to other entities")
                
                with col2:
                    st.metric("Clustering Coefficient", f"{clustering:.4f}")
                    st.write("Measure of how interconnected the disease's neighborhood is")
                
                # Calculate other metrics if they exist in the network
                try:
                    from knowledge_graph import compute_centrality_measures
                    centrality_df = compute_centrality_measures(G)
                    disease_centrality = centrality_df[centrality_df['id'] == disease_id]
                    
                    if not disease_centrality.empty:
                        st.subheader("Centrality Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Degree Centrality", f"{disease_centrality['degree'].values[0]:.4f}")
                        with col2:
                            st.metric("Betweenness Centrality", f"{disease_centrality['betweenness'].values[0]:.4f}")
                        with col3:
                            st.metric("Eigenvector Centrality", f"{disease_centrality['eigenvector'].values[0]:.4f}")
                            
                        st.write("Higher values indicate the disease is more central in the knowledge graph")
                except Exception as e:
                    st.write("Could not compute advanced network metrics.")
            except Exception as e:
                st.write("Network position analysis not available.")
                
        with disease_tabs[2]:  # Treatment Options Tab
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Current Treatment Options")
                if relationships:
                    # Filter to only 'treats' relationships
                    treats_relationships = [r for r in relationships if r['type'] == 'treats']
                    
                    if treats_relationships:
                        # Convert to DataFrame for better display
                        treats_df = pd.DataFrame([
                            {
                                "Drug": r['drug_name'],
                                "Confidence": f"{r['confidence']:.2f}",
                                "Mechanism": G.nodes[r['drug_id']].get('mechanism', 'Unknown')[:100] + "..." 
                                    if len(G.nodes[r['drug_id']].get('mechanism', 'Unknown')) > 100 
                                    else G.nodes[r['drug_id']].get('mechanism', 'Unknown')
                            }
                            for r in treats_relationships
                        ])
                        
                        st.dataframe(treats_df, hide_index=True)
                    else:
                        st.info("No known treatments in the database.")
                else:
                    st.info("No treatments found for this disease.")
                    
                # Show potential drug repurposing candidates
                st.subheader("Repurposing Candidates")
                
                if relationships:
                    # Filter to only 'potential' relationships
                    potential_relationships = [r for r in relationships if r['type'] == 'potential']
                    
                    if potential_relationships:
                        # Convert to DataFrame for display
                        potential_df = pd.DataFrame([
                            {
                                "Drug": r['drug_name'],
                                "Confidence": f"{r['confidence']:.2f}",
                                "Original Indication": G.nodes[r['drug_id']].get('original_indication', 'Unknown'),
                                "View Details": f"{r['drug_id']}"
                            }
                            for r in potential_relationships
                        ])
                        
                        st.dataframe(potential_df, hide_index=True)
                        
                        # Allow selecting a candidate for detailed analysis
                        if len(potential_relationships) > 0:
                            st.write("Select a candidate for detailed mechanism analysis:")
                            
                            selected_candidate = st.selectbox(
                                "Choose a repurposing candidate",
                                options=[(r['drug_id'], r['drug_name']) for r in potential_relationships],
                                format_func=lambda x: x[1]
                            )
                            
                            if selected_candidate:
                                selected_drug_id = selected_candidate[0]
                                selected_drug = None
                                
                                # Get the drug object
                                try:
                                    selected_drug = {
                                        'id': selected_drug_id,
                                        'name': G.nodes[selected_drug_id]['name'],
                                        'description': G.nodes[selected_drug_id].get('description', ''),
                                        'mechanism': G.nodes[selected_drug_id].get('mechanism', 'Unknown'),
                                        'original_indication': G.nodes[selected_drug_id].get('original_indication', 'Unknown')
                                    }
                                except:
                                    st.error("Could not retrieve drug details.")
                                
                                if selected_drug:
                                    # Generate mechanistic explanation
                                    st.subheader(f"Mechanistic Analysis: {selected_drug['name']} for {disease['name']}")
                                    
                                    with st.spinner("Analyzing potential mechanism..."):
                                        mechanism_explanation = "Mechanistic explanation not available."
                                        try:
                                            # Import AI analysis functions
                                            from ai_analysis import generate_mechanistic_explanation
                                            mechanism_explanation = generate_mechanistic_explanation(selected_drug, disease)
                                        except Exception as e:
                                            st.error(f"Error generating mechanism: {str(e)}")
                                        
                                        st.write(mechanism_explanation)
                    else:
                        st.write("No potential repurposing candidates found for this disease.")
                else:
                    st.write("No relationships found for this disease.")
            
            with col2:
                st.subheader("Treatment Statistics")
                
                # Calculate and show treatment statistics
                if relationships:
                    treats_count = len([r for r in relationships if r['type'] == 'treats'])
                    potential_count = len([r for r in relationships if r['type'] == 'potential'])
                    similar_pathway_count = len([r for r in relationships if r['type'] == 'similar_pathway'])
                    
                    import plotly.express as px
                    
                    # Create pie chart of relationship types
                    rel_counts = {
                        'Established Treatments': treats_count,
                        'Potential Repurposing': potential_count,
                        'Similar Pathway': similar_pathway_count
                    }
                    
                    # Filter out zero counts
                    rel_counts = {k: v for k, v in rel_counts.items() if v > 0}
                    
                    if rel_counts:
                        fig = px.pie(
                            values=list(rel_counts.values()),
                            names=list(rel_counts.keys()),
                            title="Treatment Relationship Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="treatment_relations_pie")
                    else:
                        st.info("No relationship data available for statistics.")
                else:
                    st.info("No relationship data available.")
                
        with disease_tabs[3]:  # Literature Tab
            st.subheader("Recent Literature")
            
            if st.button("Search PubMed for Literature", key="literature_tab_button"):
                with st.spinner("Searching PubMed database..."):
                    articles, article_relationships = get_pubmed_data(disease['name'])
                    
                    if articles:
                        st.success(f"Found {len(articles)} articles related to {disease['name']}")
                        
                        # Display articles in a table
                        articles_df = pd.DataFrame([
                            {
                                "Title": article['title'],
                                "Year": article['year'],
                                "PMID": article['pmid']
                            }
                            for article in articles
                        ])
                        
                        st.dataframe(articles_df, hide_index=True)
                        
                        # Display potential relationships
                        if article_relationships:
                            st.subheader("Extracted Relationships")
                            st.write(f"Found {len(article_relationships)} potential relationships")
                            
                            # Group by type (drug-disease, target-disease, etc.)
                            relationship_types = {}
                            for rel in article_relationships:
                                rel_type = rel.get('type', 'unknown')
                                if rel_type in relationship_types:
                                    relationship_types[rel_type].append(rel)
                                else:
                                    relationship_types[rel_type] = [rel]
                            
                            # Display relationships by type
                            for rel_type, rels in relationship_types.items():
                                with st.expander(f"{rel_type.title()} Relationships ({len(rels)})"):
                                    for i, rel in enumerate(rels[:5]):  # Limit to 5 per type
                                        st.markdown(f"**From:** {rel['title']}")
                                        st.write(rel['text'])
                                        st.write(f"Source: PMID {rel['source']} ({rel['year']})")
                                        if i < len(rels) - 1:
                                            st.divider()
                                    
                                    if len(rels) > 5:
                                        st.info(f"{len(rels) - 5} more relationships not shown")
                        else:
                            st.info("No relationships extracted from the literature.")
                    else:
                        st.warning("No recent literature found for this disease.")
            else:
                st.write("Click the button above to search PubMed for literature related to this disease.")
        
        # Display potential repurposing candidates
        st.subheader("Potential Repurposing Candidates")
        
        if relationships:
            # Filter to only 'potential' relationships
            potential_relationships = [r for r in relationships if r['type'] == 'potential']
            
            if potential_relationships:
                # Convert to DataFrame for display
                potential_df = pd.DataFrame([
                    {
                        "Drug": r['drug_name'],
                        "Confidence": f"{r['confidence']:.2f}",
                        "View Details": f"{r['drug_id']}"
                    }
                    for r in potential_relationships
                ])
                
                st.dataframe(potential_df, hide_index=True)
            else:
                st.write("No potential repurposing candidates found for this disease.")
        else:
            st.write("No repurposing candidates found for this disease.")
        
        # Network visualization
        st.subheader("Network Visualization")
        
        # Create network graph
        G = st.session_state.graph
        highlight_nodes = [disease['id']]
        highlight_edges = []
        
        # Add related drug nodes to highlight
        for rel in relationships:
            highlight_nodes.append(rel['drug_id'])
            highlight_edges.append((rel['drug_id'], disease['id']))
        
        fig = create_network_graph(G, highlight_nodes=highlight_nodes, highlight_edges=highlight_edges)
        st.plotly_chart(fig, use_container_width=True, key="network_visualization")
        
        # Get PubMed data
        st.subheader("Recent Literature")
        
        if st.button("Search PubMed for Literature", key="main_literature_button"):
            articles, article_relationships = get_pubmed_data(disease['name'])
            
            if articles:
                st.write(f"Found {len(articles)} articles related to {disease['name']}")
                
                # Display articles in a table
                articles_df = pd.DataFrame([
                    {
                        "Title": article['title'],
                        "Year": article['year'],
                        "PMID": article['pmid']
                    }
                    for article in articles
                ])
                
                st.dataframe(articles_df, hide_index=True)
                
                # Display potential relationships
                if article_relationships:
                    st.subheader("Extracted Relationships")
                    st.write(f"Found {len(article_relationships)} potential relationships")
                    
                    for i, rel in enumerate(article_relationships[:10]):  # Limit to 10
                        with st.expander(f"From: {rel['title']}"):
                            st.write(rel['text'])
                            st.write(f"Source: PMID {rel['source']} ({rel['year']})")
                else:
                    st.write("No relationships extracted from the literature.")
            else:
                st.write("No recent literature found for this disease.")
