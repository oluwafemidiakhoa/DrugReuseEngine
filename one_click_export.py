"""
One-click export functionality for research findings in the Drug Repurposing Engine.
This module provides a floating export button that appears on all pages and allows
users to easily export the current view to common academic formats (PDF, CSV).
"""
import streamlit as st
import pandas as pd
import base64
import io
from datetime import datetime
from export_utils import (
    generate_pdf_download_link,
    generate_csv_download_link,
    create_repurposing_candidates_pdf,
    create_drug_disease_pdf,
    candidates_to_dataframe,
    relationships_to_dataframe
)

def get_current_context():
    """
    Determine the current view context to decide what data to export.
    
    Returns:
        dict: Context information including page type, data objects, etc.
    """
    # Determine the current page name from session state
    if 'current_page' not in st.session_state:
        # Default to Home if not set
        current_page = 'Home'
    else:
        current_page = st.session_state.current_page
    
    context = {'page': current_page}
    
    # Context for Repurposing Candidates page
    if current_page == 'Repurposing_Candidates':
        if 'filtered_candidates' in st.session_state:
            context['type'] = 'candidates_list'
            context['candidates'] = st.session_state.filtered_candidates
        else:
            context['type'] = 'all_candidates'
            context['candidates'] = st.session_state.get('candidates', [])
            
        # Selected candidate detail
        if 'selected_candidate' in st.session_state and st.session_state.selected_candidate:
            context['type'] = 'candidate_detail'
            context['candidate'] = st.session_state.selected_candidate
            
            # Get related drug and disease
            drug_id = context['candidate'].get('drug_id', '')
            disease_id = context['candidate'].get('disease_id', '')
            
            if drug_id and 'drugs' in st.session_state:
                context['drug'] = next((d for d in st.session_state.drugs if d.get('id') == drug_id), None)
            
            if disease_id and 'diseases' in st.session_state:
                context['disease'] = next((d for d in st.session_state.diseases if d.get('id') == disease_id), None)
    
    # Context for Drug Search page
    elif current_page == 'Drug_Search':
        if 'selected_drug' in st.session_state and st.session_state.selected_drug:
            context['type'] = 'drug_detail'
            context['drug'] = st.session_state.selected_drug
            
            # Get relationships for this drug
            if 'relationships' in st.session_state:
                context['relationships'] = [
                    r for r in st.session_state.relationships 
                    if r.get('source') == context['drug'].get('id')
                ]
        else:
            context['type'] = 'drugs_list'
            context['drugs'] = st.session_state.get('drugs', [])
    
    # Context for Disease Search page
    elif current_page == 'Disease_Search':
        if 'selected_disease' in st.session_state and st.session_state.selected_disease:
            context['type'] = 'disease_detail'
            context['disease'] = st.session_state.selected_disease
            
            # Get relationships for this disease
            if 'relationships' in st.session_state:
                context['relationships'] = [
                    r for r in st.session_state.relationships 
                    if r.get('target') == context['disease'].get('id')
                ]
        else:
            context['type'] = 'diseases_list'
            context['diseases'] = st.session_state.get('diseases', [])
    
    # Context for Knowledge Graph page
    elif current_page == 'Knowledge_Graph':
        context['type'] = 'knowledge_graph'
        context['graph'] = st.session_state.get('graph', None)
        
    # Context for Neo4j Graph Explorer page
    elif current_page == 'Graph_Explorer':
        context['type'] = 'neo4j_graph'
        # Any neo4j specific data would be captured here
        
    # Default context for Home page and others
    else:
        context['type'] = 'general'
        
    return context

def export_to_format(format_type, context=None):
    """
    Export the current view to the specified format.
    
    Args:
        format_type (str): 'pdf' or 'csv'
        context (dict, optional): Context information. If None, it will be determined.
        
    Returns:
        bool: True if export was successful, False otherwise
    """
    if context is None:
        context = get_current_context()
    
    # Record the format for user feedback
    st.session_state.export_format = format_type
    
    try:
        # Handle export based on context type
        if context['type'] == 'candidate_detail' and 'candidate' in context:
            candidate = context['candidate']
            drug = context.get('drug')
            disease = context.get('disease')
            
            if drug and disease:
                # Get relationships between this drug and disease
                relationships = []
                if 'relationships' in st.session_state:
                    relationships = [r for r in st.session_state.relationships 
                                   if r.get('source') == drug.get('id') and r.get('target') == disease.get('id')]
                
                if format_type == 'pdf':
                    # Generate PDF for detailed candidate view
                    pdf_data = create_drug_disease_pdf(
                        drug, 
                        disease, 
                        relationships, 
                        [candidate]
                    )
                    filename = f"{candidate.get('drug', 'drug')}_{candidate.get('disease', 'disease')}_analysis.pdf"
                    pdf_link = generate_pdf_download_link(pdf_data, filename, "Download PDF Report")
                    st.markdown(pdf_link, unsafe_allow_html=True)
                    
                elif format_type == 'csv':
                    # Generate CSV for detailed candidate view
                    df = candidates_to_dataframe([candidate])
                    filename = f"{candidate.get('drug', 'drug')}_{candidate.get('disease', 'disease')}_analysis.csv"
                    csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                    st.markdown(csv_link, unsafe_allow_html=True)
                    
                return True
        
        elif context['type'] in ['candidates_list', 'all_candidates'] and 'candidates' in context:
            candidates = context['candidates']
            
            if candidates:
                if format_type == 'pdf':
                    # Generate PDF for candidates list
                    pdf_data = create_repurposing_candidates_pdf(candidates)
                    filename = "drug_repurposing_candidates.pdf"
                    pdf_link = generate_pdf_download_link(pdf_data, filename, "Download PDF Report")
                    st.markdown(pdf_link, unsafe_allow_html=True)
                    
                elif format_type == 'csv':
                    # Generate CSV for candidates list
                    df = candidates_to_dataframe(candidates)
                    filename = "drug_repurposing_candidates.csv"
                    csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                    st.markdown(csv_link, unsafe_allow_html=True)
                    
                return True
        
        elif context['type'] == 'drug_detail' and 'drug' in context:
            drug = context['drug']
            relationships = context.get('relationships', [])
            
            # Get all candidates for this drug
            candidates = []
            if 'candidates' in st.session_state:
                candidates = [c for c in st.session_state.candidates 
                             if c.get('drug_id') == drug.get('id')]
            
            if format_type == 'pdf':
                # Generate PDF for drug detail view
                # If no specific disease, create a summary report for all related diseases
                placeholder_disease = {
                    "id": "", 
                    "name": "All Related Diseases", 
                    "description": "Summary of all diseases related to this drug"
                }
                pdf_data = create_drug_disease_pdf(drug, placeholder_disease, relationships, candidates)
                filename = f"{drug.get('name', 'drug')}_analysis.pdf"
                pdf_link = generate_pdf_download_link(pdf_data, filename, "Download PDF Report")
                st.markdown(pdf_link, unsafe_allow_html=True)
                
            elif format_type == 'csv':
                # Generate CSV for drug detail view
                data = [{
                    'ID': drug.get('id', ''),
                    'Name': drug.get('name', ''),
                    'Description': drug.get('description', ''),
                    'Original Indication': drug.get('original_indication', ''),
                    'Mechanism': drug.get('mechanism', ''),
                    'Related Diseases Count': len({r.get('target', '') for r in relationships}),
                    'Related Candidates Count': len(candidates)
                }]
                df = pd.DataFrame(data)
                filename = f"{drug.get('name', 'drug')}_details.csv"
                csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                st.markdown(csv_link, unsafe_allow_html=True)
                
            return True
        
        elif context['type'] == 'disease_detail' and 'disease' in context:
            disease = context['disease']
            relationships = context.get('relationships', [])
            
            # Get all candidates for this disease
            candidates = []
            if 'candidates' in st.session_state:
                candidates = [c for c in st.session_state.candidates 
                             if c.get('disease_id') == disease.get('id')]
            
            if format_type == 'pdf':
                # Generate PDF for disease detail view
                # If no specific drug, create a summary report for all related drugs
                placeholder_drug = {
                    "id": "", 
                    "name": "All Related Drugs", 
                    "description": "Summary of all drugs related to this disease"
                }
                pdf_data = create_drug_disease_pdf(placeholder_drug, disease, relationships, candidates)
                filename = f"{disease.get('name', 'disease')}_analysis.pdf"
                pdf_link = generate_pdf_download_link(pdf_data, filename, "Download PDF Report")
                st.markdown(pdf_link, unsafe_allow_html=True)
                
            elif format_type == 'csv':
                # Generate CSV for disease detail view
                data = [{
                    'ID': disease.get('id', ''),
                    'Name': disease.get('name', ''),
                    'Description': disease.get('description', ''),
                    'Category': disease.get('category', ''),
                    'Related Drugs Count': len({r.get('source', '') for r in relationships}),
                    'Related Candidates Count': len(candidates)
                }]
                df = pd.DataFrame(data)
                filename = f"{disease.get('name', 'disease')}_details.csv"
                csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                st.markdown(csv_link, unsafe_allow_html=True)
                
            return True
        
        elif context['type'] == 'drugs_list' and 'drugs' in context:
            drugs = context['drugs']
            
            if drugs:
                if format_type == 'csv':
                    # Generate CSV for drugs list
                    data = [{
                        'ID': drug.get('id', ''),
                        'Name': drug.get('name', ''),
                        'Description': drug.get('description', '')[:100] + '...' if len(drug.get('description', '')) > 100 else drug.get('description', ''),
                        'Original Indication': drug.get('original_indication', ''),
                        'Mechanism': drug.get('mechanism', '')
                    } for drug in drugs]
                    
                    df = pd.DataFrame(data)
                    filename = "drugs_list.csv"
                    csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                    st.markdown(csv_link, unsafe_allow_html=True)
                    
                    return True
        
        elif context['type'] == 'diseases_list' and 'diseases' in context:
            diseases = context['diseases']
            
            if diseases:
                if format_type == 'csv':
                    # Generate CSV for diseases list
                    data = [{
                        'ID': disease.get('id', ''),
                        'Name': disease.get('name', ''),
                        'Description': disease.get('description', '')[:100] + '...' if len(disease.get('description', '')) > 100 else disease.get('description', ''),
                        'Category': disease.get('category', '')
                    } for disease in diseases]
                    
                    df = pd.DataFrame(data)
                    filename = "diseases_list.csv"
                    csv_link = generate_csv_download_link(df, filename, "Download CSV Data")
                    st.markdown(csv_link, unsafe_allow_html=True)
                    
                    return True
        
        # If we reached here, no export was performed
        st.warning(f"Export to {format_type.upper()} is not supported for the current view.")
        return False
    
    except Exception as e:
        st.error(f"Error exporting to {format_type.upper()}: {str(e)}")
        return False

def display_one_click_export_buttons():
    """
    Display one-click export buttons for the current view.
    
    Returns:
        dict: Context information for the current view
    """
    # Get current context
    context = get_current_context()
    
    # Create a clean UI with side-by-side buttons
    st.write("### Export Current View")
    st.write("Export the current data to standard academic formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export to PDF", key="pdf_export_btn", help="Export current view to PDF format"):
            if export_to_format('pdf', context):
                st.success("PDF export successful! Click the link above to download.")
            
    with col2:
        if st.button("📊 Export to CSV", key="csv_export_btn", help="Export current view to CSV format"):
            if export_to_format('csv', context):
                st.success("CSV export successful! Click the link above to download.")
    
    return context

def add_floating_export_button():
    """
    Add a floating export button that follows the user across all pages.
    This provides convenient one-click export functionality from anywhere in the app.
    """
    # Add custom CSS for the floating button
    st.markdown("""
    <style>
    .floating-export-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }
    
    .floating-export-button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #0068C9;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .floating-export-button:hover {
        background-color: #004A8F;
        transform: scale(1.05);
    }
    
    .floating-export-icon {
        font-size: 24px;
    }
    
    .floating-export-menu {
        position: absolute;
        bottom: 65px;
        right: 0;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        padding: 10px;
        display: none;
    }
    
    .floating-export-menu.show {
        display: block;
    }
    
    .floating-export-menu-item {
        display: block;
        padding: 8px 16px;
        text-decoration: none;
        color: #333;
        transition: background-color 0.2s;
        border-radius: 4px;
        margin-bottom: 4px;
        cursor: pointer;
    }
    
    .floating-export-menu-item:hover {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add HTML for the floating button
    st.markdown("""
    <div class="floating-export-container">
        <div class="floating-export-button" onclick="toggleExportMenu()">
            <div class="floating-export-icon">📥</div>
        </div>
        <div id="exportMenu" class="floating-export-menu">
            <div class="floating-export-menu-item" onclick="exportToPDF()">Export to PDF</div>
            <div class="floating-export-menu-item" onclick="exportToCSV()">Export to CSV</div>
        </div>
    </div>
    
    <script>
    function toggleExportMenu() {
        const menu = document.getElementById('exportMenu');
        menu.classList.toggle('show');
    }
    
    function exportToPDF() {
        // Communicate with Streamlit via the session state
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: {'format': 'pdf', 'export': true},
            dataType: 'json'
        }, '*');
    }
    
    function exportToCSV() {
        // Communicate with Streamlit via the session state
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: {'format': 'csv', 'export': true},
            dataType: 'json'
        }, '*');
    }
    
    // Close the menu when clicking outside
    document.addEventListener('click', function(event) {
        const container = document.querySelector('.floating-export-container');
        const menu = document.getElementById('exportMenu');
        
        if (!container.contains(event.target)) {
            menu.classList.remove('show');
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Return the current context
    return get_current_context()