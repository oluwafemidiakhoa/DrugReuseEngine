"""
Page for exporting research findings to various formats (PDF, CSV, etc.)
"""
import streamlit as st
import pandas as pd
from export_utils import (
    generate_pdf_download_link,
    generate_csv_download_link,
    create_repurposing_candidates_pdf,
    create_drug_disease_pdf,
    candidates_to_dataframe,
    relationships_to_dataframe
)
from utils import search_candidates, get_drug_by_id, get_disease_by_id, initialize_session_state

def display_export_options():
    """Display options for exporting data"""
    st.title("Export Research Findings")
    
    st.markdown("""
    Export your research findings to various formats for further analysis, publication, or sharing with colleagues.
    """)
    
    # Display tabs for different export options
    tab1, tab2, tab3 = st.tabs(["Repurposing Candidates", "Drug-Disease Relationships", "Custom Export"])
    
    # Tab 1: Export repurposing candidates
    with tab1:
        st.subheader("Export Repurposing Candidates")
        
        # Filters for candidates
        col1, col2, col3 = st.columns(3)
        with col1:
            drug_filter = st.text_input("Filter by Drug Name", "")
        with col2:
            disease_filter = st.text_input("Filter by Disease Name", "")
        with col3:
            min_confidence = st.slider("Minimum Confidence Score", 0, 100, 0)
        
        # Get candidates based on filters
        candidates = search_candidates(drug_filter, disease_filter, min_confidence)
        
        # Show number of candidates found
        st.write(f"Found {len(candidates)} candidates matching your criteria.")
        
        # Show sample of candidates
        if candidates:
            st.write("Sample of candidates (up to 5):")
            df_sample = candidates_to_dataframe(candidates[:5])
            st.dataframe(df_sample)
            
            # Show export buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to PDF
                if st.button("Export to PDF", key="export_candidates_pdf"):
                    st.session_state.candidates_pdf_data = create_repurposing_candidates_pdf(candidates)
                    st.session_state.show_candidates_pdf_link = True
            
            with col2:
                # Export to CSV
                if st.button("Export to CSV", key="export_candidates_csv"):
                    st.session_state.candidates_df = candidates_to_dataframe(candidates)
                    st.session_state.show_candidates_csv_link = True
            
            # Show download links if generated
            st.write("")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.get('show_candidates_pdf_link', False):
                    pdf_link = generate_pdf_download_link(
                        st.session_state.candidates_pdf_data,
                        filename=f"drug_repurposing_candidates_{min_confidence}pct.pdf",
                        text="Download PDF Report"
                    )
                    st.markdown(pdf_link, unsafe_allow_html=True)
            
            with col2:
                if st.session_state.get('show_candidates_csv_link', False):
                    csv_link = generate_csv_download_link(
                        st.session_state.candidates_df,
                        filename=f"drug_repurposing_candidates_{min_confidence}pct.csv",
                        text="Download CSV Data"
                    )
                    st.markdown(csv_link, unsafe_allow_html=True)
        else:
            st.info("No candidates found matching your criteria. Try adjusting the filters.")
    
    # Tab 2: Export drug-disease relationships
    with tab2:
        st.subheader("Export Drug-Disease Relationships")
        
        # Select drug and disease
        col1, col2 = st.columns(2)
        
        with col1:
            drug_id = st.selectbox(
                "Select Drug",
                options=[d['id'] for d in st.session_state.drugs],
                format_func=lambda x: get_drug_by_id(x)['name'] if get_drug_by_id(x) else x
            )
        
        with col2:
            disease_id = st.selectbox(
                "Select Disease",
                options=[d['id'] for d in st.session_state.diseases],
                format_func=lambda x: get_disease_by_id(x)['name'] if get_disease_by_id(x) else x
            )
        
        # Get drug and disease details
        drug = get_drug_by_id(drug_id)
        disease = get_disease_by_id(disease_id)
        
        # Get relationships between this drug and disease
        relationships = [r for r in st.session_state.relationships 
                        if r['source'] == drug_id and r['target'] == disease_id]
        
        # Get candidates for this drug and disease
        candidates = [c for c in st.session_state.candidates 
                     if c.get('drug_id') == drug_id and c.get('disease_id') == disease_id]
        
        # Display info about the relationship
        if relationships:
            st.write(f"Found {len(relationships)} relationships between {drug['name']} and {disease['name']}.")
            
            # Display the relationships
            for rel in relationships:
                st.write(f"- Type: {rel.get('type', '').upper()}, Confidence: {rel.get('confidence', 0)*100:.1f}%, Evidence: {rel.get('evidence_count', 0)}")
        else:
            st.info(f"No direct relationships found between {drug['name']} and {disease['name']}.")
        
        # Show export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to PDF
            if st.button("Export to PDF", key="export_relationship_pdf"):
                st.session_state.relationship_pdf_data = create_drug_disease_pdf(
                    drug, 
                    disease, 
                    relationships, 
                    candidates
                )
                st.session_state.show_relationship_pdf_link = True
        
        with col2:
            # Export to CSV
            if st.button("Export to CSV", key="export_relationship_csv"):
                st.session_state.relationship_df = relationships_to_dataframe(
                    relationships, 
                    [drug], 
                    [disease]
                )
                st.session_state.show_relationship_csv_link = True
        
        # Show download links if generated
        st.write("")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('show_relationship_pdf_link', False):
                pdf_link = generate_pdf_download_link(
                    st.session_state.relationship_pdf_data,
                    filename=f"{drug['name']}_{disease['name']}_report.pdf",
                    text="Download PDF Report"
                )
                st.markdown(pdf_link, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.get('show_relationship_csv_link', False):
                csv_link = generate_csv_download_link(
                    st.session_state.relationship_df,
                    filename=f"{drug['name']}_{disease['name']}_relationships.csv",
                    text="Download CSV Data"
                )
                st.markdown(csv_link, unsafe_allow_html=True)
    
    # Tab 3: Custom export
    with tab3:
        st.subheader("Custom Export")
        
        # Select what to export
        export_type = st.selectbox(
            "Select data to export",
            options=["All Drugs", "All Diseases", "All Relationships", "Knowledge Graph Statistics"]
        )
        
        # Export button
        if st.button("Generate Export", key="custom_export"):
            if export_type == "All Drugs":
                st.session_state.custom_df = pd.DataFrame([{
                    'ID': d['id'],
                    'Name': d['name'],
                    'Description': d['description'],
                    'Original Indication': d.get('original_indication', ''),
                    'Mechanism': d.get('mechanism', '')
                } for d in st.session_state.drugs])
                st.session_state.custom_filename = "all_drugs"
            
            elif export_type == "All Diseases":
                st.session_state.custom_df = pd.DataFrame([{
                    'ID': d['id'],
                    'Name': d['name'],
                    'Description': d['description'],
                    'Category': d.get('category', '')
                } for d in st.session_state.diseases])
                st.session_state.custom_filename = "all_diseases"
            
            elif export_type == "All Relationships":
                st.session_state.custom_df = relationships_to_dataframe(
                    st.session_state.relationships,
                    st.session_state.drugs,
                    st.session_state.diseases
                )
                st.session_state.custom_filename = "all_relationships"
            
            elif export_type == "Knowledge Graph Statistics":
                # Compute some basic statistics
                stats = {
                    'Metric': [
                        'Number of Drugs',
                        'Number of Diseases',
                        'Number of Relationships',
                        'Number of Repurposing Candidates',
                        'Average Confidence Score'
                    ],
                    'Value': [
                        len(st.session_state.drugs),
                        len(st.session_state.diseases),
                        len(st.session_state.relationships),
                        len(st.session_state.candidates),
                        sum(c['confidence_score'] for c in st.session_state.candidates) / 
                            len(st.session_state.candidates) if st.session_state.candidates else 0
                    ]
                }
                st.session_state.custom_df = pd.DataFrame(stats)
                st.session_state.custom_filename = "kg_statistics"
            
            st.session_state.show_custom_csv_link = True
        
        # Show download link if generated
        if st.session_state.get('show_custom_csv_link', False):
            st.write("Preview:")
            st.dataframe(st.session_state.custom_df)
            
            csv_link = generate_csv_download_link(
                st.session_state.custom_df,
                filename=f"{st.session_state.custom_filename}.csv",
                text=f"Download {export_type} as CSV"
            )
            st.markdown(csv_link, unsafe_allow_html=True)

# Main function to run the page
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Export Results | Drug Repurposing Engine",
        page_icon="📊",
        layout="wide",
    )
    
    # Initialize session state variables
    initialize_session_state()
    
    # Initialize necessary session state variables if they don't exist
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
        
    if 'filtered_candidates' not in st.session_state:
        st.session_state.filtered_candidates = []
    
    if 'show_candidates_pdf_link' not in st.session_state:
        st.session_state.show_candidates_pdf_link = False
    
    if 'show_candidates_csv_link' not in st.session_state:
        st.session_state.show_candidates_csv_link = False
    
    if 'show_relationship_pdf_link' not in st.session_state:
        st.session_state.show_relationship_pdf_link = False
    
    if 'show_relationship_csv_link' not in st.session_state:
        st.session_state.show_relationship_csv_link = False
    
    if 'show_custom_csv_link' not in st.session_state:
        st.session_state.show_custom_csv_link = False
    
    # Display the page content
    display_export_options()