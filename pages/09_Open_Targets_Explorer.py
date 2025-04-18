import streamlit as st
import pandas as pd
import plotly.express as px
import open_targets_api
import json
from utils import initialize_session_state
import networkx as nx

# Set page configuration
st.set_page_config(
    page_title="Open Targets Explorer | Drug Repurposing Engine",
    page_icon="🎯",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Open Targets Explorer")
st.write("Explore drug-target-disease associations and evidence-based scoring from the Open Targets Platform")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Disease Associations", "Drug Targets", "Repurposing Opportunities"])

with tab1:
    st.header("Explore Disease Associations")
    
    # Search input
    disease_query = st.text_input("Enter disease name", placeholder="e.g., Alzheimer's disease, Diabetes, Cancer")
    
    # Search button
    if st.button("Search Open Targets", key="search_disease_btn"):
        if disease_query:
            with st.spinner(f"Searching Open Targets for '{disease_query}'..."):
                disease_results = open_targets_api.search_disease(disease_query)
                
                if disease_results:
                    st.session_state.disease_results = disease_results
                    st.success(f"Found {len(disease_results)} diseases matching '{disease_query}'")
                else:
                    st.error(f"No diseases found for: {disease_query}")
                    st.session_state.disease_results = None
        else:
            st.warning("Please enter a disease name")
    
    # Display disease results if available
    if 'disease_results' in st.session_state and st.session_state.disease_results:
        # Display as selectable list
        disease_options = [(d['id'], d['name']) for d in st.session_state.disease_results]
        selected_disease = st.selectbox(
            "Select a disease to explore",
            options=disease_options,
            format_func=lambda x: x[1]
        )
        
        if selected_disease:
            disease_id = selected_disease[0]
            disease_name = selected_disease[1]
            
            # Fetch associated targets
            with st.spinner(f"Fetching targets associated with {disease_name}..."):
                associated_targets = open_targets_api.get_associations(disease_id=disease_id, limit=20)
                
                if associated_targets:
                    st.success(f"Found {len(associated_targets)} targets associated with {disease_name}")
                    
                    # Display targets in a table
                    target_df = pd.DataFrame([
                        {
                            "Target": t['target_symbol'],
                            "Target Name": t.get('target_name', 'Unknown'),
                            "Association Score": f"{t['association_score']:.3f}",
                            "Target ID": t['target_id']
                        }
                        for t in associated_targets
                    ])
                    
                    st.dataframe(target_df, use_container_width=True, hide_index=True)
                    
                    # Create association score visualization
                    if len(associated_targets) > 0:
                        # Create bar chart of association scores
                        fig = px.bar(
                            associated_targets, 
                            x=[t['target_symbol'] for t in associated_targets], 
                            y=[t['association_score'] for t in associated_targets],
                            labels={"x": "Target", "y": "Association Score"},
                            title=f"Target Association Scores for {disease_name}",
                            color=[t['association_score'] for t in associated_targets],
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Evidence score breakdown for the top target
                        if associated_targets[0]['evidence_scores']:
                            st.subheader("Evidence Score Breakdown")
                            st.write(f"Detailed evidence scores for {associated_targets[0]['target_symbol']}")
                            
                            # Create dataframe for evidence scores
                            evidence_types = {
                                "literature": "Literature",
                                "rna_expression": "RNA Expression",
                                "genetic_association": "Genetic Association",
                                "somatic_mutation": "Somatic Mutation",
                                "known_drug": "Known Drug",
                                "animal_model": "Animal Model",
                                "affected_pathway": "Affected Pathway"
                            }
                            
                            evidence_data = []
                            for key, score in associated_targets[0]['evidence_scores'].items():
                                if key in evidence_types:
                                    evidence_data.append({
                                        "Evidence Type": evidence_types.get(key, key),
                                        "Score": score
                                    })
                            
                            evidence_df = pd.DataFrame(evidence_data)
                            
                            # Create a radar chart for evidence scores
                            fig = px.line_polar(
                                evidence_df, 
                                r="Score", 
                                theta="Evidence Type", 
                                line_close=True,
                                range_r=[0, 1],
                                title=f"Evidence Profile for {associated_targets[0]['target_symbol']}"
                            )
                            fig.update_traces(fill='toself')
                            st.plotly_chart(fig)
                            
                            # Create a table for evidence scores
                            st.dataframe(evidence_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No targets associated with {disease_name} found")

with tab2:
    st.header("Explore Drug Targets")
    
    # Search input
    drug_query = st.text_input("Enter drug name", placeholder="e.g., Aspirin, Metformin, Ibuprofen")
    
    # Search button
    if st.button("Search Open Targets", key="search_drug_btn"):
        if drug_query:
            with st.spinner(f"Searching Open Targets for '{drug_query}'..."):
                drug_results = open_targets_api.search_drugs(drug_query)
                
                if drug_results:
                    st.session_state.drug_results = drug_results
                    st.success(f"Found {len(drug_results)} drugs matching '{drug_query}'")
                else:
                    st.error(f"No drugs found for: {drug_query}")
                    st.session_state.drug_results = None
        else:
            st.warning("Please enter a drug name")
    
    # Display drug results if available
    if 'drug_results' in st.session_state and st.session_state.drug_results:
        # Display as selectable list
        drug_options = [(d['id'], d['name']) for d in st.session_state.drug_results]
        selected_drug = st.selectbox(
            "Select a drug to explore",
            options=drug_options,
            format_func=lambda x: x[1]
        )
        
        if selected_drug:
            drug_id = selected_drug[0]
            drug_name = selected_drug[1]
            
            # Fetch drug details
            with st.spinner(f"Fetching details for {drug_name}..."):
                drug_details = open_targets_api.get_drug_details(drug_id)
                drug_targets = open_targets_api.get_drug_targets(drug_id)
                
                if drug_details:
                    # Display basic information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Drug Information")
                        
                        st.write(f"**Name:** {drug_name}")
                        
                        if 'synonyms' in drug_details and drug_details['synonyms']:
                            st.write(f"**Synonyms:** {', '.join(drug_details['synonyms'][:5])}")
                        
                        if 'trade_names' in drug_details and drug_details['trade_names']:
                            st.write(f"**Trade Names:** {', '.join(drug_details['trade_names'])}")
                        
                        if 'year_of_approval' in drug_details and drug_details['year_of_approval']:
                            st.write(f"**Year of First Approval:** {drug_details['year_of_approval']}")
                        
                        if 'max_phase' in drug_details:
                            phase_descriptions = {
                                0: "Research/Preclinical",
                                1: "Phase I Clinical Trial",
                                2: "Phase II Clinical Trial",
                                3: "Phase III Clinical Trial",
                                4: "Approved"
                            }
                            phase = drug_details['max_phase']
                            phase_desc = phase_descriptions.get(phase, f"Phase {phase}")
                            st.write(f"**Maximum Clinical Trial Phase:** {phase_desc}")
                        
                        if 'drug_type' in drug_details and drug_details['drug_type']:
                            st.write(f"**Drug Type:** {drug_details['drug_type']}")
                        
                        if 'black_box_warning' in drug_details:
                            warning = "Yes" if drug_details['black_box_warning'] else "No"
                            st.write(f"**Black Box Warning:** {warning}")
                    
                    with col2:
                        st.subheader("Mechanisms & Indications")
                        
                        # Display mechanisms of action
                        if 'mechanisms_of_action' in drug_details and drug_details['mechanisms_of_action']:
                            st.write("**Mechanisms of Action:**")
                            for mechanism in drug_details['mechanisms_of_action']:
                                st.markdown(f"• {mechanism.get('mechanism', 'Unknown mechanism')}")
                                if 'targets' in mechanism and mechanism['targets']:
                                    targets_text = ", ".join([t.get('name', t.get('id', 'Unknown')) for t in mechanism['targets']])
                                    st.markdown(f"  - Targets: {targets_text}")
                        
                        # Display indications
                        if 'indications' in drug_details and drug_details['indications']:
                            st.write("**Approved Indications:**")
                            for indication in drug_details['indications']:
                                phase = indication.get('phase', 0)
                                if phase == 4:  # Approved
                                    st.markdown(f"• {indication.get('disease_name', 'Unknown disease')}")
                    
                    # Display detailed description if available
                    if 'description' in drug_details and drug_details['description']:
                        st.subheader("Description")
                        st.write(drug_details['description'])
                    
                    # Display targets
                    if drug_targets:
                        st.subheader("Drug Targets")
                        
                        # Create a dataframe for targets
                        target_df = pd.DataFrame([
                            {
                                "Target": t.get('name', 'Unknown'),
                                "ID": t.get('id', '-'),
                                "Mechanism": t.get('mechanism', 'Unknown')
                            }
                            for t in drug_targets
                        ])
                        
                        st.dataframe(target_df, use_container_width=True, hide_index=True)
                        
                        # Create a network visualization of drug-target interactions
                        G = nx.Graph()
                        
                        # Add drug node
                        G.add_node(drug_id, name=drug_name, type='drug')
                        
                        # Add target nodes and edges
                        for target in drug_targets:
                            target_id = target.get('id')
                            if target_id:
                                G.add_node(target_id, name=target.get('name', 'Unknown'), type='target')
                                G.add_edge(drug_id, target_id, mechanism=target.get('mechanism', 'Unknown'))
                        
                        # Create positions for nodes
                        pos = nx.spring_layout(G)
                        
                        # Create edge trace
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        edge_trace = {
                            'x': edge_x,
                            'y': edge_y,
                            'mode': 'lines',
                            'line': {'width': 1, 'color': '#888'},
                            'hoverinfo': 'none'
                        }
                        
                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []
                        node_color = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(G.nodes[node]['name'])
                            if G.nodes[node]['type'] == 'drug':
                                node_color.append('#FF5733')  # Drug color
                            else:
                                node_color.append('#33A1FF')  # Target color
                        
                        node_trace = {
                            'x': node_x,
                            'y': node_y,
                            'mode': 'markers+text',
                            'text': node_text,
                            'marker': {
                                'color': node_color,
                                'size': 15
                            },
                            'textposition': 'top center',
                            'textfont': {'size': 10}
                        }
                        
                        # Create figure
                        fig_data = [edge_trace, node_trace]
                        
                        fig_layout = {
                            'title': f"Drug-Target Network for {drug_name}",
                            'showlegend': False,
                            'hovermode': 'closest',
                            'margin': {'b': 20, 'l': 5, 'r': 5, 't': 40},
                            'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False}
                        }
                        
                        fig = {'data': fig_data, 'layout': fig_layout}
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No details found for {drug_name}")

with tab3:
    st.header("Drug Repurposing Opportunities")
    st.write("Discover potential drug repurposing opportunities based on target-disease associations")
    
    # Search disease for repurposing
    repurpose_disease = st.text_input("Enter disease name for repurposing", placeholder="e.g., Alzheimer's disease, COVID-19")
    
    # Search button
    if st.button("Find Repurposing Opportunities", key="repurposing_btn"):
        if repurpose_disease:
            with st.spinner(f"Searching for repurposing opportunities for '{repurpose_disease}'..."):
                # First, search for the disease
                disease_results = open_targets_api.search_disease(repurpose_disease)
                
                if disease_results:
                    # Use the first (most relevant) disease
                    disease = disease_results[0]
                    
                    # Find repurposing opportunities
                    opportunities = open_targets_api.find_repurposing_opportunities(disease['id'])
                    
                    if opportunities:
                        st.session_state.repurposing_opportunities = opportunities
                        st.session_state.repurposing_disease = disease
                        st.success(f"Found {len(opportunities)} potential repurposing opportunities for {disease['name']}")
                    else:
                        st.warning(f"No repurposing opportunities found for {disease['name']}")
                        st.session_state.repurposing_opportunities = None
                else:
                    st.error(f"No diseases found for: {repurpose_disease}")
                    st.session_state.repurposing_opportunities = None
        else:
            st.warning("Please enter a disease name")
    
    # Display repurposing opportunities if available
    if 'repurposing_opportunities' in st.session_state and st.session_state.repurposing_opportunities:
        opportunities = st.session_state.repurposing_opportunities
        disease = st.session_state.repurposing_disease
        
        st.subheader(f"Repurposing Opportunities for {disease['name']}")
        
        # Create a dataframe for display
        df = pd.DataFrame([
            {
                "Drug": opp['drug_name'],
                "Target": opp['target_symbol'],
                "Mechanism": opp.get('mechanism_of_action', 'Unknown'),
                "Score": f"{opp['association_score']:.3f}",
                "Phase": opp.get('max_phase', 0),
                "Current Indication": opp['current_indication'].get('name', 'Unknown'),
                "Drug ID": opp['drug_id'],
                "Target ID": opp['target_id']
            }
            for opp in opportunities
        ])
        
        # Add a calculated repurposing score (simplified example)
        df['Repurposing Score'] = df['Score'].apply(lambda x: float(x) * 100)
        
        # Sort by repurposing score
        df = df.sort_values('Repurposing Score', ascending=False)
        
        # Display as interactive table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Create visualization of top opportunities
        top_opportunities = df.head(10)
        
        fig = px.bar(
            top_opportunities, 
            x='Drug', 
            y='Repurposing Score',
            color='Repurposing Score',
            color_continuous_scale='Viridis',
            hover_data=['Target', 'Mechanism', 'Current Indication'],
            title=f"Top 10 Repurposing Opportunities for {disease['name']}"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to add selected opportunity to the database
        st.subheader("Add Repurposing Candidate to Database")
        
        # Select a drug from the opportunities
        drug_options = [(opp['drug_id'], opp['drug_name']) for opp in opportunities]
        selected_drug = st.selectbox(
            "Select a drug to add as repurposing candidate",
            options=drug_options,
            format_func=lambda x: x[1]
        )
        
        if selected_drug:
            # Find the selected opportunity
            opportunity = next((opp for opp in opportunities if opp['drug_id'] == selected_drug[0]), None)
            
            if opportunity:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Drug:** {opportunity['drug_name']}")
                    st.write(f"**Target:** {opportunity['target_symbol']}")
                    st.write(f"**Mechanism:** {opportunity.get('mechanism_of_action', 'Unknown')}")
                
                with col2:
                    st.write(f"**Disease:** {disease['name']}")
                    st.write(f"**Association Score:** {opportunity['association_score']:.3f}")
                    st.write(f"**Current Indication:** {opportunity['current_indication'].get('name', 'Unknown')}")
                
                # Button to add to database
                if st.button("Add to Repurposing Candidates"):
                    from utils import add_repurposing_candidate
                    
                    # Prepare candidate data
                    candidate_data = {
                        'drug_id': opportunity['drug_id'],
                        'drug_name': opportunity['drug_name'],
                        'disease_id': disease['id'],
                        'disease_name': disease['name'],
                        'confidence': int(opportunity['association_score'] * 100),
                        'source': 'Open Targets API',
                        'mechanism': opportunity.get('mechanism_of_action', 'Unknown'),
                        'evidence': json.dumps({
                            'target': opportunity['target_symbol'],
                            'association_score': opportunity['association_score'],
                            'current_indication': opportunity['current_indication'].get('name', 'Unknown')
                        })
                    }
                    
                    # Add to database
                    candidate_id, message = add_repurposing_candidate(candidate_data)
                    
                    if "successfully" in message:
                        st.success(f"{message} (ID: {candidate_id})")
                    else:
                        st.warning(message)

# Add resource links in sidebar
st.sidebar.header("Open Targets Resources")
st.sidebar.markdown("[Open Targets Platform](https://platform.opentargets.org/)")
st.sidebar.markdown("[API Documentation](https://platform-api.opentargets.io/api/v4/graphql/schema)")
st.sidebar.markdown("[Open Targets Publications](https://www.opentargets.org/publications)")

# Add explanation
st.sidebar.header("About Open Targets")
st.sidebar.write("""
Open Targets is a public-private partnership that uses human genetics and genomics data 
for systematic identification and prioritization of drug targets.

The platform integrates data from various sources to help researchers identify and 
prioritize potential therapeutic targets for drug discovery.
""")

# Add feature explanation
st.sidebar.header("Features")
st.sidebar.write("""
- **Disease Associations**: Explore targets associated with specific diseases
- **Drug Targets**: Discover what targets are affected by specific drugs
- **Repurposing Opportunities**: Find potential new indications for existing drugs
""")

# Add citation information
st.sidebar.header("Citation")
st.sidebar.markdown("""
Ochoa, D., et al. (2021). Open Targets Platform: supporting systematic drug-target identification and prioritisation. *Nucleic Acids Research*, 49(D1), D1302-D1310.
""")