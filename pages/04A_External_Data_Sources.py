"""
External Data Sources Explorer

This page allows users to search for drugs across external data sources like ChEMBL and OpenFDA,
compare the information available from each source, and import selected drugs into the
Drug Repurposing Engine for further analysis.
"""

import streamlit as st
import pandas as pd
import json
import time
import sys
from typing import Dict, List, Any

# Configure page settings
st.set_page_config(
    page_title="External Data Sources | Drug Repurposing Engine",
    page_icon="🌐",
    layout="wide"
)

# Apply custom styling
from utils import get_custom_theme
st.markdown(get_custom_theme(), unsafe_allow_html=True)

# Import our integration module
from external_data_integration import (
    search_drug_across_sources,
    get_enriched_drug_data,
    format_consolidated_drug_data,
    add_external_drug_to_database,
    test_data_sources
)

# Page header with prominent styling
st.markdown("""
<div style="background-color:#4A6EB0; padding:15px; border-radius:10px; margin-bottom:20px;">
    <h1 style="color:white; text-align:center;">🌐 External Data Sources Explorer</h1>
    <p style="color:white; text-align:center; font-size:1.2em;">
        Search across ChEMBL and OpenFDA for comprehensive drug information
    </p>
</div>
""", unsafe_allow_html=True)

# Introduction
st.write("""
Search for drugs across multiple biomedical databases and import them into the Drug Repurposing Engine.
Current data sources include ChEMBL and OpenFDA, providing extensive information on drugs, their
mechanisms, indications, and more.
""")

# Show connection status for data sources
with st.expander("Data Source Status", expanded=False):
    # Add a refresh button
    if st.button("Test Connections"):
        status = test_data_sources()
        
        # ChEMBL status
        st.subheader("ChEMBL Database")
        if status['chembl']['status'] == 'OK':
            st.success(f"✅ Connected - {status['chembl']['message']} in {status['chembl']['response_time']}")
        elif status['chembl']['status'] == 'WARNING':
            st.warning(f"⚠️ {status['chembl']['message']} in {status['chembl']['response_time']}")
        else:
            st.error(f"❌ Connection error: {status['chembl']['message']}")
        
        # OpenFDA status
        st.subheader("OpenFDA Database")
        if status['openfda']['status'] == 'OK':
            st.success(f"✅ Connected - {status['openfda']['message']} in {status['openfda']['response_time']}")
        elif status['openfda']['status'] == 'WARNING':
            st.warning(f"⚠️ {status['openfda']['message']} in {status['openfda']['response_time']}")
        else:
            st.error(f"❌ Connection error: {status['openfda']['message']}")
    else:
        st.info("Click 'Test Connections' to check connectivity with external data sources.")

# Search interface
st.subheader("Search for Drugs")
col1, col2 = st.columns([3, 1])

with col1:
    drug_query = st.text_input("Enter drug name to search across databases:", 
                             value="",
                             placeholder="e.g., aspirin, ibuprofen, acetaminophen")

with col2:
    search_btn = st.button("Search", type="primary", use_container_width=True)
    
# Source selection
col1, col2 = st.columns(2)
with col1:
    include_chembl = st.checkbox("Include ChEMBL Database", value=True)
with col2:
    include_openfda = st.checkbox("Include OpenFDA Database", value=True)

# Number of results
col1, col2 = st.columns(2)
with col1:
    chembl_limit = st.slider("Max ChEMBL results", 1, 20, 5)
with col2:
    openfda_limit = st.slider("Max OpenFDA results", 1, 20, 5)

# Search when button is clicked
if search_btn and drug_query:
    with st.spinner(f"Searching for '{drug_query}' across databases..."):
        # Set session state for search
        if 'drug_search_results' not in st.session_state:
            st.session_state.drug_search_results = {}
            
        # Perform search
        results = search_drug_across_sources(
            drug_query,
            chembl_limit=chembl_limit,
            openfda_limit=openfda_limit,
            include_chembl=include_chembl,
            include_openfda=include_openfda
        )
        
        # Store in session state
        st.session_state.drug_search_results = results
        
        # Count total results
        total_results = sum(len(drugs) for drugs in results.values())
        
        # Show data source status
        api_errors = []
        if include_chembl and not results.get('chembl'):
            api_errors.append({
                "source": "ChEMBL",
                "error": "API error or no results returned. The ChEMBL server may be experiencing issues."
            })
        
        if include_openfda and not results.get('openfda'):
            api_errors.append({
                "source": "OpenFDA",
                "error": "API error or no results returned. The OpenFDA server may be experiencing issues."
            })
        
        # Show user-friendly errors if needed
        if api_errors:
            with st.expander("API Connection Details", expanded=True):
                st.warning("Some external data services returned errors:")
                for error in api_errors:
                    st.markdown(f"**{error['source']}**: {error['error']}")
                st.markdown("""
                **What to do:**
                - Try again in a few minutes
                - Try a different search term
                - Check if you're trying to search for a very new or uncommon drug that may not be in these databases
                """)
        
        # Process results
        if total_results > 0:
            st.success(f"Found {total_results} results for '{drug_query}'")
            
            # Process and display the results
            chembl_data = results.get('chembl', [])
            openfda_data = results.get('openfda', [])
            
            # Show data source summary
            st.info(f"Results: ChEMBL ({len(chembl_data)}), OpenFDA ({len(openfda_data)})")
            
            # Consolidate results
            consolidated_drugs = format_consolidated_drug_data(chembl_data, openfda_data)
            
            # Store consolidated results in session state
            st.session_state.consolidated_drugs = consolidated_drugs
            
            # Display consolidated results
            st.subheader("Results")
            
            # Create a table of results
            result_data = []
            for drug in consolidated_drugs:
                result_data.append({
                    "Name": drug['name'],
                    "Sources": ", ".join(drug['sources']),
                    "Description": drug.get('description', '')[:100] + "..." if drug.get('description', '') else "N/A",
                    "ID": drug['id']
                })
            
            df = pd.DataFrame(result_data)
            st.dataframe(df, use_container_width=True)
            
            # Allow selecting a drug for detailed view
            drug_ids = [None] + [drug['id'] for drug in consolidated_drugs]
            selected_drug_id = st.selectbox("Select a drug for detailed information:", 
                                          options=drug_ids,
                                          format_func=lambda x: "Select a drug..." if x is None else next((drug['name'] for drug in consolidated_drugs if drug['id'] == x), x))
            
            if selected_drug_id is not None:
                # Find the selected drug
                selected_drug = next((drug for drug in consolidated_drugs if drug['id'] == selected_drug_id), None)
                
                if selected_drug:
                    # Display detailed information about the selected drug - safely handle potentially missing name
                    drug_name = selected_drug.get('name', 'Selected Drug')
                    st.subheader(f"{drug_name} Details")
                    
                    # Basic information
                    col1, col2 = st.columns(2)
                    with col1:
                        # Safe handling of potential None values
                        drug_name = selected_drug.get('name', 'Unknown')
                        st.markdown(f"**Name:** {drug_name}")
                        
                        # Safely join sources list
                        sources = selected_drug.get('sources', [])
                        if isinstance(sources, list) and sources:
                            st.markdown(f"**Sources:** {', '.join(sources)}")
                        else:
                            st.markdown("**Sources:** Unknown")
                            
                        # Only show brand name if it exists and is not empty
                        brand_name = selected_drug.get('brand_name')
                        if brand_name and isinstance(brand_name, str) and brand_name.strip():
                            st.markdown(f"**Brand Name:** {brand_name}")
                    
                    with col2:
                        # Add to database button
                        if st.button("➕ Add to Database", type="primary", key=f"add_{selected_drug_id}"):
                            success = add_external_drug_to_database(selected_drug)
                            drug_name = selected_drug.get('name', 'Selected drug')
                            if success:
                                st.success(f"Successfully added {drug_name} to the database!")
                            else:
                                st.error(f"Failed to add {drug_name} to the database.")
                    
                    # Description
                    st.markdown("**Description:**")
                    st.markdown(selected_drug.get('description', 'No description available.'))
                    
                    # Mechanism of action
                    if selected_drug.get('mechanism'):
                        st.markdown("**Mechanism of Action:**")
                        st.markdown(selected_drug['mechanism'])
                    
                    # Categories/Classifications
                    if selected_drug.get('categories'):
                        st.markdown("**Drug Classes:**")
                        for category in selected_drug['categories']:
                            st.markdown(f"- {category}")
                    
                    # SMILES structure if available
                    if selected_drug.get('smiles'):
                        st.markdown("**Chemical Structure (SMILES):**")
                        st.code(selected_drug['smiles'])
                    
                    # Additional data from sources
                    st.subheader("Source-Specific Data")
                    
                    tabs = []
                    # Safely handle potential missing sources list
                    sources = selected_drug.get('sources', [])
                    if isinstance(sources, list):
                        if 'ChEMBL' in sources:
                            tabs.append("ChEMBL")
                        if 'OpenFDA' in sources:
                            tabs.append("OpenFDA")
                    
                    if tabs:
                        selected_tab = st.radio("Select Source:", tabs, horizontal=True)
                        
                        # Safely handle potential missing ids dictionary
                        ids = selected_drug.get('ids', {})
                        if isinstance(ids, dict) and selected_tab == "ChEMBL" and ids.get('chembl'):
                            chembl_id = ids.get('chembl')
                            
                            with st.spinner("Loading detailed ChEMBL data..."):
                                enriched_data = get_enriched_drug_data(chembl_id, 'chembl')
                                
                                if enriched_data:
                                    # Show mechanisms
                                    if enriched_data.get('mechanisms'):
                                        st.markdown("**Mechanisms of Action:**")
                                        for mech in enriched_data['mechanisms']:
                                            st.markdown(f"- **Action:** {mech.get('action_type', 'Unknown')}")
                                            st.markdown(f"  **Target:** {mech.get('target_name', 'Unknown')}")
                                            if mech.get('mechanism_of_action'):
                                                st.markdown(f"  **Mechanism:** {mech['mechanism_of_action']}")
                                    
                                    # Show indications
                                    if enriched_data.get('indications'):
                                        st.markdown("**Known Indications:**")
                                        for ind in enriched_data['indications']:
                                            st.markdown(f"- **Disease:** {ind.get('mesh_heading', 'Unknown')}")
                                            if ind.get('efo_term'):
                                                st.markdown(f"  **EFO Term:** {ind['efo_term']}")
                                            if ind.get('max_phase_for_ind'):
                                                phases = {
                                                    0: "Not active",
                                                    1: "Phase I",
                                                    2: "Phase II",
                                                    3: "Phase III",
                                                    4: "Approved"
                                                }
                                                phase = phases.get(ind['max_phase_for_ind'], "Unknown")
                                                st.markdown(f"  **Development Phase:** {phase}")
                                else:
                                    st.info("No detailed ChEMBL data available.")
                        
                        elif selected_tab == "OpenFDA" and ids.get('openfda'):
                            openfda_id = ids.get('openfda')
                            
                            with st.spinner("Loading detailed OpenFDA data..."):
                                enriched_data = get_enriched_drug_data(openfda_id, 'openfda')
                                
                                if enriched_data:
                                    # Show interactions
                                    if enriched_data.get('interactions'):
                                        st.markdown("**Drug Interactions:**")
                                        for interaction in enriched_data['interactions']:
                                            st.markdown(f"- {interaction}")
                                    
                                    # Show adverse events (simplified)
                                    if enriched_data.get('adverse_events'):
                                        st.markdown("**Reported Adverse Events (Sample):**")
                                        for event in enriched_data['adverse_events'][:5]:  # Limit to first 5
                                            if event.get('patient') and event['patient'].get('reaction'):
                                                for reaction in event['patient']['reaction']:
                                                    st.markdown(f"- {reaction.get('reactionmeddrapt', 'Unknown reaction')}")
                                else:
                                    st.info("No detailed OpenFDA data available.")
        else:
            st.warning(f"No results found for '{drug_query}'. Try another search term or select different data sources.")

# If no search has been performed yet, show some example drugs that could be searched
if not search_btn or not drug_query:
    st.info("""
    ### Example Drugs to Search
    
    Try searching for these common drugs:
    - Aspirin (acetylsalicylic acid)
    - Ibuprofen
    - Metformin
    - Atorvastatin
    - Amoxicillin
    
    Or search for a specific drug of interest.
    """)

# Additional information about the databases
with st.expander("About External Data Sources"):
    st.markdown("""
    ### ChEMBL Database
    
    The ChEMBL database is a manually curated database of bioactive molecules with drug-like properties.
    It brings together chemical, bioactivity and genomic data to aid the translation of genomic information 
    into effective new drugs. ChEMBL provides:
    
    - Detailed molecular information
    - Binding affinities and targets
    - Mechanisms of action
    - Known disease indications
    
    ### OpenFDA Database
    
    OpenFDA is a research project that provides open access to FDA data, including:
    
    - Drug product information
    - Adverse events reported to FDA
    - Drug labeling and packaging information
    - Drug interactions and warnings
    
    This information is valuable for understanding the real-world use and effects of medications.
    """)