import streamlit as st
import pandas as pd
import numpy as np
from utils import get_drug_by_name, get_pubmed_data, get_drug_by_id, initialize_session_state
from data_ingestion import fetch_drug_info
from knowledge_graph import get_drug_disease_relationships, find_paths_between
from visualization import create_network_graph
from ai_analysis import analyze_repurposing_candidate
import pubchem_api
import requests

# Set page configuration
st.set_page_config(
    page_title="Drug Search | Drug Repurposing Engine",
    page_icon="💊",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Advanced Drug Search & Analysis")
st.write("Search for drugs and explore their therapeutic applications, mechanisms, and repurposing opportunities")

# Advanced search options
with st.expander("Advanced Search Options", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        search_mechanism = st.checkbox("Search by Mechanism of Action")
        if search_mechanism:
            mechanism_keywords = st.multiselect(
                "Mechanism Keywords",
                options=["inhibitor", "agonist", "antagonist", "blocker", "modulator", 
                         "enzyme", "receptor", "channel", "transporter", "reuptake"],
                default=[]
            )
    
    with col2:
        search_indication = st.checkbox("Search by Original Indication")
        if search_indication:
            indication_category = st.multiselect(
                "Original Indication Category",
                options=["Metabolic", "Neurological", "Cardiovascular", "Autoimmune", 
                         "Infectious", "Cancer", "Respiratory", "Other"],
                default=[]
            )

# Search input
search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
with search_col1:
    search_query = st.text_input("Search for a drug:", "")

with search_col2:
    search_button = st.button("Search", type="primary")
    
with search_col3:
    search_type = st.selectbox(
        "Match Type",
        options=["Contains", "Exact Match"],
        index=0
    )

if search_query or search_button:
    # Perform drug search with the advanced options
    if search_query:
        if search_type == "Exact Match":
            drug = get_drug_by_name(search_query, exact_match=True)
        else:
            drug = get_drug_by_name(search_query, exact_match=False)
            
        # If advanced search is enabled but no direct match was found, search all drugs
        if not drug and (search_mechanism or search_indication):
            # Get all drugs for advanced filtering
            all_drugs = st.session_state.drugs
            matching_drugs = []
            
            for drug_item in all_drugs:
                match = True
                
                # Apply mechanism filter if enabled
                if search_mechanism and mechanism_keywords:
                    mechanism_text = drug_item.get('mechanism', '').lower()
                    if not any(keyword.lower() in mechanism_text for keyword in mechanism_keywords):
                        match = False
                
                # Apply indication filter if enabled
                if search_indication and indication_category:
                    indication_text = drug_item.get('original_indication', '').lower()
                    if not any(category.lower() in indication_text for category in indication_category):
                        match = False
                
                # Add to matches if all filters pass
                if match:
                    matching_drugs.append(drug_item)
            
            # Display matching drugs for user selection
            if matching_drugs:
                st.success(f"Found {len(matching_drugs)} drugs matching your advanced criteria")
                
                # Display as a selectable table
                drug_df = pd.DataFrame([
                    {
                        "Name": d['name'],
                        "Original Indication": d['original_indication'],
                        "Mechanism": d.get('mechanism', 'Unknown'),
                        "ID": d['id']
                    }
                    for d in matching_drugs
                ])
                
                # Show the table
                st.dataframe(drug_df, hide_index=True)
                
                # Allow selecting a drug
                selected_drug_id = st.selectbox(
                    "Select a drug to view details:",
                    options=[(d['id'], d['name']) for d in matching_drugs],
                    format_func=lambda x: x[1]
                )
                
                if selected_drug_id:
                    drug = next((d for d in matching_drugs if d['id'] == selected_drug_id[0]), None)
            else:
                st.warning("No drugs found matching your advanced criteria.")
    
    # Check if drug variable exists and has a value
    if 'drug' not in locals() or drug is None:
        st.info(f"Drug '{search_query}' not found in the database.")
        
        # Option to add drug information from external API
        st.write("Would you like to add this drug to the database?")
        
        if st.button("Add from RxNorm database"):
            with st.spinner("Fetching drug information..."):
                drug_info = fetch_drug_info(search_query)
                
                if drug_info:
                    # Format as a drug record
                    new_drug = {
                        "name": drug_info["name"],
                        "description": f"Drug with RxCUI {drug_info['rxcui']}",
                        "original_indication": "Unknown",
                        "mechanism": "Unknown"
                    }
                    
                    # Display drug information
                    st.success(f"Found drug information for {drug_info['name']}")
                    
                    # Edit fields
                    with st.form("add_drug_form"):
                        new_drug["name"] = st.text_input("Drug Name", new_drug["name"])
                        new_drug["description"] = st.text_area("Description", new_drug["description"])
                        new_drug["original_indication"] = st.text_input("Original Indication", new_drug["original_indication"])
                        new_drug["mechanism"] = st.text_area("Mechanism of Action", new_drug["mechanism"])
                        
                        # Submit button
                        submitted = st.form_submit_button("Add to Database")
                        
                        if submitted:
                            from utils import add_drug
                            drug_id, message = add_drug(new_drug)
                            
                            st.success(message)
                            
                            # Update drug variable to continue with display
                            drug = get_drug_by_id(drug_id)
                else:
                    st.error("Could not find drug information in RxNorm database.")
        
        # Manual entry option
        with st.expander("Or add drug manually"):
            with st.form("manual_drug_form"):
                new_drug = {
                    "name": search_query,
                    "description": "",
                    "original_indication": "",
                    "mechanism": ""
                }
                
                new_drug["name"] = st.text_input("Drug Name", new_drug["name"])
                new_drug["description"] = st.text_area("Description", new_drug["description"])
                new_drug["original_indication"] = st.text_input("Original Indication", new_drug["original_indication"])
                new_drug["mechanism"] = st.text_area("Mechanism of Action", new_drug["mechanism"])
                
                # Submit button
                submitted = st.form_submit_button("Add to Database")
                
                if submitted:
                    from utils import add_drug
                    drug_id, message = add_drug(new_drug)
                    
                    st.success(message)
                    
                    # Update drug variable to continue with display
                    drug = get_drug_by_id(drug_id)
    
    # If we now have a drug (either found or added), display its information
    if drug:
        st.header(drug['name'])
        
        # Create tabs for different views of drug data
        drug_tabs = st.tabs(["Overview", "Detailed Analysis", "Molecular Properties", "Clinical Data"])
        
        # Get drug-disease relationships
        relationships = get_drug_disease_relationships(st.session_state.graph, drug_id=drug['id'])
        
        with drug_tabs[0]:  # Overview Tab
            # Display drug information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Drug Details")
                st.write(f"**Description:** {drug['description']}")
                st.write(f"**Original Indication:** {drug['original_indication']}")
                st.write(f"**Mechanism of Action:** {drug['mechanism']}")
                
                # Extract and show mechanism keywords if available
                mechanism_keywords = []
                if 'mechanism_keywords' in drug:
                    mechanism_keywords = drug['mechanism_keywords']
                else:
                    # Extract using the same function as the knowledge graph
                    from knowledge_graph import extract_mechanism_keywords
                    mechanism_keywords = extract_mechanism_keywords(drug.get('mechanism', ''))
                
                if mechanism_keywords:
                    st.write("**Mechanism Keywords:**")
                    for keyword in mechanism_keywords:
                        st.markdown(f"<span style='background-color:#e6f3ff; padding:3px 8px; border-radius:10px; margin-right:5px;'>{keyword}</span>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Known Therapeutic Applications")
                
                if relationships:
                    # Filter to only 'treats' relationships
                    treats_relationships = [r for r in relationships if r['type'] == 'treats']
                    
                    if treats_relationships:
                        for rel in treats_relationships:
                            st.write(f"• **{rel['disease_name']}** (Confidence: {rel['confidence']:.2f})")
                    else:
                        st.write("No known therapeutic applications in the database.")
                else:
                    st.write("No relationships found for this drug.")
                    
                # Show similar drugs based on mechanism
                st.subheader("Similar Drugs")
                G = st.session_state.graph
                
                similar_drugs = []
                try:
                    # Look for similar_mechanism edges
                    for source, target, attr in G.out_edges(drug['id'], data=True):
                        if attr['type'] == 'similar_mechanism' and G.nodes[target]['type'] == 'drug':
                            similar_drugs.append((target, G.nodes[target]['name'], attr['confidence']))
                    for source, target, attr in G.in_edges(drug['id'], data=True):
                        if attr['type'] == 'similar_mechanism' and G.nodes[source]['type'] == 'drug':
                            similar_drugs.append((source, G.nodes[source]['name'], attr['confidence']))
                except:
                    # Fallback for older graph structure
                    similar_drugs = []
                
                if similar_drugs:
                    for drug_id, drug_name, confidence in similar_drugs:
                        st.write(f"• **{drug_name}** (Similarity: {confidence:.2f})")
                else:
                    st.write("No similar drugs found based on mechanism.")
        
        with drug_tabs[1]:  # Detailed Analysis Tab
            st.subheader("Mechanism of Action Analysis")
            
            # Extract key biological targets from mechanism
            mechanism = drug.get('mechanism', 'Unknown mechanism of action')
            
            # Show word cloud or key terms visualization
            if mechanism and mechanism != 'Unknown' and mechanism != 'Unknown mechanism of action':
                try:
                    # Generate word frequency for visualization
                    from collections import Counter
                    import re
                    
                    # Clean and tokenize text
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', mechanism.lower())
                    
                    # Remove common stopwords
                    stopwords = ['the', 'and', 'for', 'with', 'that', 'this', 'its', 'has', 'are', 'drug']
                    words = [word for word in words if word not in stopwords]
                    
                    word_freq = Counter(words).most_common(10)
                    
                    # Create horizontal bar chart
                    if word_freq:
                        import plotly.express as px
                        
                        df = pd.DataFrame(word_freq, columns=['Term', 'Frequency'])
                        fig = px.bar(df, x='Frequency', y='Term', orientation='h',
                                    title="Key Terms in Mechanism of Action",
                                    color='Frequency',
                                    color_continuous_scale='Viridis')
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
            else:
                st.info("Detailed mechanism information not available.")
                
            # Pathway analysis
            st.subheader("Pathway Analysis")
            st.write("Predicted biological pathways affected by this drug:")
            
            # Extract pathways from treated diseases
            disease_pathways = {}
            if relationships:
                treats_relationships = [r for r in relationships if r['type'] == 'treats']
                for rel in treats_relationships:
                    disease_id = rel['disease_id']
                    # Get pathways if they exist in the graph
                    try:
                        pathways = G.nodes[disease_id].get('pathways', [])
                        if pathways:
                            for pathway in pathways:
                                if pathway in disease_pathways:
                                    disease_pathways[pathway] += 1
                                else:
                                    disease_pathways[pathway] = 1
                    except:
                        pass
            
            if disease_pathways:
                for pathway, count in sorted(disease_pathways.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• **{pathway}** (Found in {count} {'diseases' if count > 1 else 'disease'})")
            else:
                st.write("No specific pathway information available.")
        
        with drug_tabs[2]:  # Molecular Properties Tab
            st.subheader("Molecular Properties from PubChem")
            
            # Fetch data from PubChem if not already in session state
            pubchem_data_key = f"pubchem_data_{drug['id']}"
            
            if pubchem_data_key not in st.session_state:
                fetch_pubchem = st.button("Fetch data from PubChem")
                if fetch_pubchem:
                    with st.spinner("Fetching data from PubChem..."):
                        pubchem_data = pubchem_api.search_compound_by_name(drug['name'])
                        if pubchem_data:
                            st.session_state[pubchem_data_key] = pubchem_data
                            st.success("Successfully retrieved data from PubChem")
                            st.experimental_rerun()
                        else:
                            st.error("Could not find this drug in PubChem")
            else:
                pubchem_data = st.session_state[pubchem_data_key]
                
                # Display molecular structure
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Molecular Structure")
                    if 'cid' in pubchem_data:
                        image_url = pubchem_api.get_compound_image_url(int(pubchem_data['cid']))
                        try:
                            response = requests.get(image_url)
                            if response.status_code == 200:
                                st.image(response.content, caption=f"Structure of {drug['name']}")
                            else:
                                st.warning("Could not load compound structure image")
                        except Exception as e:
                            st.warning(f"Could not load compound structure image: {str(e)}")
                    
                    # Basic information
                    st.subheader("Chemical Information")
                    info_table = []
                    
                    if 'molecular_formula' in pubchem_data:
                        info_table.append(["Molecular Formula", pubchem_data['molecular_formula']])
                    if 'molecular_weight' in pubchem_data:
                        info_table.append(["Molecular Weight", f"{pubchem_data['molecular_weight']} g/mol"])
                    if 'iupac_name' in pubchem_data:
                        info_table.append(["IUPAC Name", pubchem_data['iupac_name']])
                    
                    if info_table:
                        st.table(pd.DataFrame(info_table, columns=["Property", "Value"]))
                    
                    # Provide link to PubChem
                    if 'cid' in pubchem_data:
                        st.markdown(f"[View full details on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_data['cid']})")
                
                with col2:
                    # Create tabs for different types of information
                    info_tabs = st.tabs(["Chemical Properties", "Pharmacology", "Drug Interactions"])
                    
                    with info_tabs[0]:
                        st.subheader("Chemical Properties")
                        if 'chemical_properties' in pubchem_data:
                            props = pubchem_data['chemical_properties']
                            prop_table = []
                            
                            for key, value in props.items():
                                prop_table.append([key, value])
                            
                            if prop_table:
                                st.table(pd.DataFrame(prop_table, columns=["Property", "Value"]))
                        else:
                            st.info("No chemical property information available")
                        
                        # Display SMILES and InChI if available
                        if 'smiles' in pubchem_data:
                            st.text_area("SMILES Notation", pubchem_data['smiles'], height=80)
                        if 'inchi' in pubchem_data:
                            st.text_area("InChI", pubchem_data['inchi'], height=80)
                    
                    with info_tabs[1]:
                        st.subheader("Pharmacological Information")
                        
                        # Display mechanism of action
                        if 'mechanism_of_action' in pubchem_data:
                            st.markdown("### Mechanism of Action")
                            st.write(pubchem_data['mechanism_of_action'])
                        
                        # Display pharmacology information
                        if 'pharmacology' in pubchem_data:
                            st.markdown("### Pharmacology")
                            st.write(pubchem_data['pharmacology'])
                        
                        # Display therapeutic uses
                        if 'therapeutic_uses' in pubchem_data:
                            st.markdown("### Therapeutic Uses")
                            st.write(pubchem_data['therapeutic_uses'])
                        
                        # If none of these are available
                        if 'mechanism_of_action' not in pubchem_data and 'pharmacology' not in pubchem_data and 'therapeutic_uses' not in pubchem_data:
                            st.info("No pharmacological information available in PubChem")
                    
                    with info_tabs[2]:
                        st.subheader("Drug Interactions")
                        
                        if 'drug_interactions' in pubchem_data and pubchem_data['drug_interactions']:
                            interactions = pubchem_data['drug_interactions']
                            
                            # Display each interaction
                            for i, interaction in enumerate(interactions):
                                with st.expander(f"Interaction {i+1}", expanded=i==0):
                                    st.write(interaction.get('description', 'No description available'))
                                    st.caption(f"Source: {interaction.get('source', 'PubChem')}")
                        else:
                            st.info("No drug interaction information available in PubChem")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Pharmacokinetic Properties")
                st.write("• **Absorption:** Data not available")
                st.write("• **Distribution:** Data not available")
                st.write("• **Metabolism:** Data not available")
                st.write("• **Excretion:** Data not available")
            
            with col2:
                st.subheader("Chemical Properties")
                st.write("• **Molecular Weight:** Data not available")
                st.write("• **LogP:** Data not available")
                st.write("• **Hydrogen Bond Donors:** Data not available")
                st.write("• **Hydrogen Bond Acceptors:** Data not available")
        
        with drug_tabs[3]:  # Clinical Data Tab
            st.info("This tab would contain clinical trial information, adverse effects, and dosing guidelines. This is a placeholder for future implementation.")
            
            # Placeholder for clinical trials
            st.subheader("Clinical Trials")
            st.write("Clinical trial information would appear here.")
            
            # Placeholder for adverse effects
            st.subheader("Adverse Effects")
            st.write("Adverse effect information would appear here.")
        
        # Display potential repurposing candidates
        st.subheader("Potential Repurposing Candidates")
        
        if relationships:
            # Filter to only 'potential' relationships
            potential_relationships = [r for r in relationships if r['type'] == 'potential']
            
            if potential_relationships:
                # Convert to DataFrame for display
                potential_df = pd.DataFrame([
                    {
                        "Disease": r['disease_name'],
                        "Confidence": f"{r['confidence']:.2f}",
                        "View Details": f"{r['disease_id']}"
                    }
                    for r in potential_relationships
                ])
                
                st.dataframe(potential_df, hide_index=True)
            else:
                st.write("No potential repurposing candidates found for this drug.")
        else:
            st.write("No relationships found for this drug.")
        
        # Network visualization
        st.subheader("Network Visualization")
        
        # Create network graph
        G = st.session_state.graph
        highlight_nodes = [drug['id']]
        highlight_edges = []
        
        # Add related disease nodes to highlight
        for rel in relationships:
            highlight_nodes.append(rel['disease_id'])
            highlight_edges.append((drug['id'], rel['disease_id']))
        
        fig = create_network_graph(G, highlight_nodes=highlight_nodes, highlight_edges=highlight_edges)
        st.plotly_chart(fig, use_container_width=True)
        
        # Get PubMed data
        st.subheader("Recent Literature")
        
        if st.button("Search PubMed for Literature"):
            articles, article_relationships = get_pubmed_data(drug['name'])
            
            if articles:
                st.write(f"Found {len(articles)} articles related to {drug['name']}")
                
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
                st.write("No recent literature found for this drug.")
