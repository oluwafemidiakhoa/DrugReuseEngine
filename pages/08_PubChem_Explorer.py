import streamlit as st
import pandas as pd
import pubchem_api
from utils import initialize_session_state
import plotly.express as px
import base64
import io
import requests
from PIL import Image
import json

# Set page configuration
st.set_page_config(
    page_title="PubChem Explorer | Drug Repurposing Engine",
    page_icon="🧪",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("PubChem Explorer")
st.write("Explore chemical compounds, bioactivity data, and drug-disease relationships from PubChem")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Compound Search", "Similar Compounds", "Disease-Drug Relationships"])

with tab1:
    st.header("Search for Chemical Compounds")
    
    # Search input
    search_query = st.text_input("Enter drug or compound name", placeholder="e.g., Aspirin, Metformin, Ibuprofen")
    
    # Search button
    if st.button("Search PubChem", key="search_pubchem_btn"):
        if search_query:
            with st.spinner(f"Searching PubChem for '{search_query}'..."):
                compound_data = pubchem_api.search_compound_by_name(search_query)
                
                if compound_data:
                    st.session_state.pubchem_result = compound_data
                    st.success(f"Found compound: {search_query}")
                else:
                    st.error(f"No compounds found for: {search_query}")
                    st.session_state.pubchem_result = None
        else:
            st.warning("Please enter a search query")
    
    # Display compound information if available
    if 'pubchem_result' in st.session_state and st.session_state.pubchem_result:
        compound = st.session_state.pubchem_result
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display compound image
            if 'cid' in compound:
                image_url = pubchem_api.get_compound_image_url(compound['cid'])
                try:
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        st.image(response.content, caption=f"Structure of {search_query}")
                    else:
                        st.warning("Could not load compound structure image")
                except Exception as e:
                    st.warning(f"Could not load compound structure image: {str(e)}")
            
            # Basic information
            st.subheader("Basic Information")
            info_table = []
            
            if 'cid' in compound:
                info_table.append(["PubChem CID", str(compound['cid'])])
            if 'molecular_formula' in compound:
                info_table.append(["Molecular Formula", str(compound['molecular_formula'])])
            if 'molecular_weight' in compound:
                if compound['molecular_weight'] is not None:
                    info_table.append(["Molecular Weight", f"{str(compound['molecular_weight'])} g/mol"])
                else:
                    info_table.append(["Molecular Weight", "None g/mol"])
            if 'iupac_name' in compound:
                info_table.append(["IUPAC Name", str(compound['iupac_name'])])
            
            if info_table:
                # Convert all values to strings to avoid type conversion issues
                df = pd.DataFrame(info_table, columns=["Property", "Value"])
                df["Property"] = df["Property"].astype(str)
                df["Value"] = df["Value"].astype(str)
                st.table(df)
            
            # Provide link to PubChem
            if 'cid' in compound:
                st.markdown(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{compound['cid']})")
        
        with col2:
            # Create tabs for different types of information
            info_tabs = st.tabs(["Chemical Properties", "Pharmacology", "Bioactivity", "Drug Interactions"])
            
            with info_tabs[0]:
                st.subheader("Chemical Properties")
                if 'chemical_properties' in compound and compound['chemical_properties']:
                    props = compound['chemical_properties']
                    prop_table = []
                    
                    for key, value in props.items():
                        # Convert all values to strings to prevent Arrow conversion errors
                        prop_table.append([str(key), str(value) if value is not None else ""])
                    
                    if prop_table:
                        df = pd.DataFrame(prop_table, columns=["Property", "Value"])
                        df["Property"] = df["Property"].astype(str)
                        df["Value"] = df["Value"].astype(str)
                        st.table(df)
                else:
                    st.info("No chemical property information available")
                
                # Display SMILES and InChI if available
                if 'smiles' in compound:
                    st.text_area("SMILES Notation", compound['smiles'], height=80)
                if 'inchi' in compound:
                    st.text_area("InChI", compound['inchi'], height=80)
            
            with info_tabs[1]:
                st.subheader("Pharmacological Information")
                
                # Display mechanism of action
                if 'mechanism_of_action' in compound:
                    st.markdown("### Mechanism of Action")
                    st.write(compound['mechanism_of_action'])
                
                # Display pharmacology information
                if 'pharmacology' in compound:
                    st.markdown("### Pharmacology")
                    st.write(compound['pharmacology'])
                
                # Display therapeutic uses
                if 'therapeutic_uses' in compound:
                    st.markdown("### Therapeutic Uses")
                    st.write(compound['therapeutic_uses'])
                
                # If none of these are available
                if 'mechanism_of_action' not in compound and 'pharmacology' not in compound and 'therapeutic_uses' not in compound:
                    st.info("No pharmacological information available")
            
            with info_tabs[2]:
                st.subheader("Bioactivity Data")
                
                if 'bioactivity' in compound and compound['bioactivity']:
                    bioactivities = compound['bioactivity']
                    
                    # Create dataframe for display
                    if isinstance(bioactivities, list) and bioactivities:
                        # Extract key information for table
                        bioactivity_table = []
                        for bio in bioactivities:
                            row = {}
                            # Add important fields if they exist, ensuring all values are strings
                            for field in ['Target', 'ActivityOutcome', 'ActivityValue', 'AssayName', 'Organism']:
                                # Convert to string to avoid Arrow conversion errors
                                value = bio.get(field, "-")
                                row[field] = str(value) if value is not None else "-"
                            bioactivity_table.append(row)
                        
                        # Display as dataframe
                        if bioactivity_table:
                            df = pd.DataFrame(bioactivity_table)
                            # Ensure all columns are string type
                            for col in df.columns:
                                df[col] = df[col].astype(str)
                            st.dataframe(df)
                    else:
                        st.info("Bioactivity data is not in the expected format")
                else:
                    st.info("No bioactivity information available")
            
            with info_tabs[3]:
                st.subheader("Drug Interactions")
                
                if 'drug_interactions' in compound and compound['drug_interactions']:
                    interactions = compound['drug_interactions']
                    
                    # Display each interaction
                    for i, interaction in enumerate(interactions):
                        with st.expander(f"Interaction {i+1}", expanded=i==0):
                            st.write(interaction.get('description', 'No description available'))
                            st.caption(f"Source: {interaction.get('source', 'PubChem')}")
                else:
                    st.info("No drug interaction information available")
            
            # Display synonyms if available
            if 'synonyms' in compound and compound['synonyms']:
                st.subheader("Alternative Names")
                synonyms = compound['synonyms']
                synonyms_text = ", ".join(synonyms)
                st.write(synonyms_text)

with tab2:
    st.header("Find Similar Compounds")
    
    # Input options based on either compound name or SMILES
    input_type = st.radio("Input Type", ["Compound Name", "SMILES Notation"])
    
    if input_type == "Compound Name":
        similar_query = st.text_input("Enter compound name", placeholder="e.g., Aspirin", key="similar_compound_name")
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
        
        if st.button("Find Similar Compounds", key="find_similar_btn"):
            if similar_query:
                with st.spinner(f"Finding compounds similar to '{similar_query}'..."):
                    # First get the compound data to extract SMILES
                    compound_data = pubchem_api.search_compound_by_name(similar_query)
                    
                    if compound_data and 'smiles' in compound_data:
                        # Then find similar compounds using the SMILES
                        similar_compounds = pubchem_api.get_similar_compounds(compound_data['smiles'], similarity_threshold)
                        
                        if similar_compounds:
                            st.session_state.similar_compounds = similar_compounds
                            st.success(f"Found {len(similar_compounds)} compounds similar to {similar_query}")
                        else:
                            st.warning(f"No similar compounds found for {similar_query}")
                            st.session_state.similar_compounds = None
                    else:
                        st.error(f"Could not retrieve SMILES notation for {similar_query}")
                        st.session_state.similar_compounds = None
            else:
                st.warning("Please enter a compound name")
    
    else:  # SMILES Notation
        smiles_input = st.text_area("Enter SMILES Notation", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O for Aspirin")
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01, key="similarity_slider_smiles")
        
        if st.button("Find Similar Compounds", key="find_similar_smiles_btn"):
            if smiles_input:
                with st.spinner("Finding similar compounds..."):
                    similar_compounds = pubchem_api.get_similar_compounds(smiles_input, similarity_threshold)
                    
                    if similar_compounds:
                        st.session_state.similar_compounds = similar_compounds
                        st.success(f"Found {len(similar_compounds)} similar compounds")
                    else:
                        st.warning("No similar compounds found")
                        st.session_state.similar_compounds = None
            else:
                st.warning("Please enter a SMILES notation")
    
    # Display similar compounds if available
    if 'similar_compounds' in st.session_state and st.session_state.similar_compounds:
        similar = st.session_state.similar_compounds
        
        # Display as a grid of cards
        cols = st.columns(3)
        for i, compound in enumerate(similar):
            with cols[i % 3]:
                with st.container():
                    st.subheader(f"Compound {i+1}")
                    
                    # Display image
                    if 'cid' in compound:
                        image_url = pubchem_api.get_compound_image_url(compound['cid'])
                        try:
                            response = requests.get(image_url)
                            if response.status_code == 200:
                                st.image(response.content, width=150)
                            else:
                                st.warning("Could not load structure image")
                        except Exception as e:
                            st.warning("Could not load structure image")
                    
                    # Basic information
                    st.write(f"**CID:** {compound['cid']}")
                    st.write(f"**Similarity:** {compound.get('similarity', '-'):.2f}")
                    
                    # Link to PubChem
                    st.markdown(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{compound['cid']})")
                    
                    # Button to explore this compound
                    if st.button("Explore This Compound", key=f"explore_similar_{i}"):
                        # Get full compound data
                        with st.spinner("Loading compound details..."):
                            full_data = pubchem_api.get_compound_details(compound['cid'])
                            if full_data:
                                # Add CID to the data
                                full_data['cid'] = compound['cid']
                                # Store in session state
                                st.session_state.pubchem_result = full_data
                                # Switch to the compound search tab
                                st.experimental_set_query_params(tab="compound_search")
                                st.experimental_rerun()

with tab3:
    st.header("Explore Disease-Drug Relationships")
    
    # Disease search input
    disease_name = st.text_input("Enter disease name", placeholder="e.g., Diabetes, Alzheimer, Cancer")
    
    # Search button
    if st.button("Search Disease-Drug Relationships", key="search_disease_btn"):
        if disease_name:
            with st.spinner(f"Searching for drugs related to '{disease_name}'..."):
                related_compounds = pubchem_api.search_compounds_by_disease(disease_name)
                disease_genes = pubchem_api.get_disease_gene_relationships(disease_name)
                
                # Store results in session state
                st.session_state.disease_compounds = related_compounds
                st.session_state.disease_genes = disease_genes
                
                if related_compounds:
                    st.success(f"Found {len(related_compounds)} compounds related to {disease_name}")
                else:
                    st.warning(f"No compounds found related to {disease_name}")
                    
                if disease_genes:
                    st.success(f"Found {len(disease_genes)} genes associated with {disease_name}")
                else:
                    st.warning(f"No genes found associated with {disease_name}")
        else:
            st.warning("Please enter a disease name")
    
    # Display related compounds if available
    if 'disease_compounds' in st.session_state and st.session_state.disease_compounds:
        compounds = st.session_state.disease_compounds
        
        st.subheader(f"Compounds Related to {disease_name}")
        
        # Display as a table
        compound_table = []
        for compound in compounds:
            row = {
                "CID": compound.get('cid', '-'),
                "Name": compound.get('name', 'Unknown'),
                "PubChem URL": f"[View](https://pubchem.ncbi.nlm.nih.gov/compound/{compound['cid']})",
            }
            compound_table.append(row)
        
        if compound_table:
            df = pd.DataFrame(compound_table)
            st.dataframe(df, use_container_width=True)
    
    # Display associated genes if available
    if 'disease_genes' in st.session_state and st.session_state.disease_genes:
        genes = st.session_state.disease_genes
        
        st.subheader(f"Genes Associated with {disease_name}")
        
        # Display as a table
        gene_table = []
        for gene in genes:
            row = {
                "Gene ID": gene.get('gene_id', '-'),
                "Symbol": gene.get('symbol', '-'),
                "Name": gene.get('name', '-'),
                "Source": gene.get('source', 'PubChem')
            }
            gene_table.append(row)
        
        if gene_table:
            df = pd.DataFrame(gene_table)
            st.dataframe(df, use_container_width=True)

# Add option to add a compound to the database
st.sidebar.header("Add to Database")
st.sidebar.write("Add the current compound to the Drug Repurposing Engine database")

if 'pubchem_result' in st.session_state and st.session_state.pubchem_result:
    compound = st.session_state.pubchem_result
    
    # Display basic info in sidebar
    if 'cid' in compound:
        st.sidebar.write(f"**PubChem CID:** {compound['cid']}")
    
    # Form to add to database
    with st.sidebar.form("add_to_database_form"):
        # Extract some suggested values from PubChem data
        suggested_name = compound.get('synonyms', [''])[0] if 'synonyms' in compound and compound['synonyms'] else ""
        suggested_description = compound.get('pharmacology', '') if 'pharmacology' in compound else ""
        suggested_mechanism = compound.get('mechanism_of_action', '') if 'mechanism_of_action' in compound else ""
        suggested_indication = compound.get('therapeutic_uses', '') if 'therapeutic_uses' in compound else ""
        
        # Form inputs
        st.write("### Add as Drug")
        drug_name = st.text_input("Drug Name", value=suggested_name)
        drug_description = st.text_area("Description", value=suggested_description, height=100)
        mechanism_of_action = st.text_area("Mechanism of Action", value=suggested_mechanism, height=100)
        original_indication = st.text_input("Original Indication", value=suggested_indication)
        
        # Submit button
        submitted = st.form_submit_button("Add to Database")
        
        if submitted:
            if drug_name:
                # Prepare drug data
                from utils import add_drug
                
                drug_data = {
                    'name': drug_name,
                    'description': drug_description,
                    'mechanism': mechanism_of_action,
                    'original_indication': original_indication,
                    'pubchem_data': json.dumps(compound)  # Store the PubChem data as JSON
                }
                
                # Add to database
                drug_id, message = add_drug(drug_data)
                
                if "successfully" in message:
                    st.sidebar.success(f"{message} (ID: {drug_id})")
                else:
                    st.sidebar.warning(message)
            else:
                st.sidebar.warning("Please enter a drug name")

# Include a link to the OpenFDA API for additional information
st.sidebar.markdown("---")
st.sidebar.subheader("Additional Resources")
st.sidebar.markdown("[OpenFDA Drug Information](https://open.fda.gov/apis/drug/)")
st.sidebar.markdown("[ChEMBL Database](https://www.ebi.ac.uk/chembl/)")
st.sidebar.markdown("[DrugBank](https://go.drugbank.com/)")