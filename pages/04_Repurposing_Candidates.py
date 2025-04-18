import streamlit as st
import pandas as pd
from utils import (
    generate_new_candidates, search_candidates, get_drug_by_name, 
    get_disease_by_name, format_confidence_badge, check_api_key, 
    initialize_session_state
)
from visualization import (
    create_comparison_chart, create_confidence_distribution,
    create_risk_assessment_visualization, create_risk_assessment_summary
)
from ai_analysis import analyze_repurposing_candidate
from export_utils import (
    generate_pdf_download_link, 
    generate_csv_download_link, 
    create_repurposing_candidates_pdf,
    create_drug_disease_pdf,
    candidates_to_dataframe
)
import os

# Gemini integration removed as requested

# Set page configuration
st.set_page_config(
    page_title="Repurposing Candidates | Drug Repurposing Engine",
    page_icon="⭐",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("Drug Repurposing Candidates")
st.write("Explore potential drug repurposing candidates identified by our AI algorithms")

# Function to display candidate details
def display_candidate_details(candidate):
    st.header(f"{candidate['drug']} for {candidate['disease']}")
    
    # Overview
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence Score", f"{candidate['confidence_score']}%")
    
    with col2:
        st.metric("Status", candidate['status'])
    
    with col3:
        st.metric("Supporting Evidence", candidate['evidence_count'])
    
    # Get drug and disease details
    drug = get_drug_by_name(candidate['drug'])
    disease = get_disease_by_name(candidate['disease'])
    
    # AI-powered analysis section removed as requested
    
    # Mechanistic explanation
    st.subheader("Proposed Mechanism")
    st.write(candidate['mechanism'])
    
    # Risk Assessment Visualization
    st.subheader("Risk Assessment")
    
    try:
        risk_fig = create_risk_assessment_visualization(candidate)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Add explanation of the risk assessment
        st.info("""
        This radar chart shows the risk assessment for various factors:
        - **Safety Profile**: Risk of side effects and toxicity
        - **Drug Interactions**: Potential interactions with other medications
        - **Clinical Translation**: Barriers to clinical development
        - **Regulatory Hurdles**: Complexity of approval pathways
        - **Commercial Viability**: Market and economic considerations
        
        Lower values (closer to center) indicate lower risk.
        """)
    except Exception as e:
        st.error(f"Error generating risk assessment: {str(e)}")
    
    if drug and disease:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Drug Information")
            st.write(f"**Name:** {drug['name']}")
            st.write(f"**Description:** {drug['description']}")
            st.write(f"**Original Indication:** {drug['original_indication']}")
            st.write(f"**Mechanism of Action:** {drug['mechanism']}")
        
        with col2:
            st.subheader("Disease Information")
            st.write(f"**Name:** {disease['name']}")
            st.write(f"**Description:** {disease['description']}")
            st.write(f"**Category:** {disease['category']}")
    
    # Path in knowledge graph
    st.subheader("Path in Knowledge Graph")
    
    # Check if there's a direct path in the graph
    G = st.session_state.graph
    drug_id = drug['id'] if drug else None
    disease_id = disease['id'] if disease else None
    
    if drug_id and disease_id:
        # Try to find a path
        try:
            from networkx import shortest_path
            path = shortest_path(G, drug_id, disease_id)
            
            # Get node names
            node_names = [G.nodes[node]['name'] for node in path]
            
            # Get edge types
            edge_types = []
            edge_confidences = []
            for i in range(len(path)-1):
                edge_data = G.get_edge_data(path[i], path[i+1])
                edge_types.append(edge_data['type'])
                edge_confidences.append(edge_data['confidence'])
            
            # Display path
            path_str = " → ".join(node_names)
            st.write(f"**Path:** {path_str}")
            
            # Display path details
            for i in range(len(path)-1):
                st.write(f"{node_names[i]} --({edge_types[i]}, {edge_confidences[i]:.2f})--> {node_names[i+1]}")
            
            # Visualize path
            from visualization import create_path_visualization
            path_fig = create_path_visualization(G, path, show_mechanism=True)
            st.plotly_chart(path_fig, use_container_width=True)
        except:
            st.info("No direct path found in the knowledge graph.")
    
    # Literature search option
    st.subheader("Literature Search")
    
    if st.button("Search PubMed for Evidence"):
        # Perform PubMed search
        from utils import get_pubmed_data
        drug_name = candidate['drug']
        disease_name = candidate['disease']
        
        query = f"{drug_name} AND {disease_name}"
        articles, _ = get_pubmed_data(query)
        
        if articles:
            st.success(f"Found {len(articles)} articles about {drug_name} and {disease_name}")
            
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
            
            # Display abstracts
            for i, article in enumerate(articles[:5]):  # Limit to 5
                with st.expander(f"Abstract: {article['title']}"):
                    st.write(article['abstract'])
                    st.write(f"Source: PMID {article['pmid']} ({article['year']})")
        else:
            st.info(f"No articles found for {drug_name} and {disease_name}")
    
    # Export options
    st.subheader("Export This Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Initialize session state for export links if not already done
        if 'show_single_candidate_pdf_link' not in st.session_state:
            st.session_state.show_single_candidate_pdf_link = False
        
        # Export to PDF button
        if st.button("Export to PDF", key="single_candidate_pdf_btn"):
            with st.spinner("Generating PDF..."):
                # Get drug and disease details
                drug_obj = get_drug_by_name(candidate['drug'])
                disease_obj = get_disease_by_name(candidate['disease'])
                
                if drug_obj and disease_obj:
                    # Get relationships between this drug and disease
                    relationships = [r for r in st.session_state.relationships 
                                    if r['source'] == drug_obj['id'] and r['target'] == disease_obj['id']]
                    
                    # Generate PDF
                    st.session_state.single_candidate_pdf_data = create_drug_disease_pdf(
                        drug_obj, 
                        disease_obj, 
                        relationships, 
                        [candidate]
                    )
                    st.session_state.show_single_candidate_pdf_link = True
    
    with col2:
        # Initialize session state for export links if not already done
        if 'show_single_candidate_csv_link' not in st.session_state:
            st.session_state.show_single_candidate_csv_link = False
        
        # Export to CSV button
        if st.button("Export to CSV", key="single_candidate_csv_btn"):
            with st.spinner("Generating CSV..."):
                st.session_state.single_candidate_df = candidates_to_dataframe([candidate])
                st.session_state.show_single_candidate_csv_link = True
    
    # Display download links if generated
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('show_single_candidate_pdf_link', False):
            pdf_link = generate_pdf_download_link(
                st.session_state.single_candidate_pdf_data,
                filename=f"{candidate['drug']}_{candidate['disease']}_analysis.pdf",
                text="Download PDF Report"
            )
            st.markdown(pdf_link, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('show_single_candidate_csv_link', False):
            csv_link = generate_csv_download_link(
                st.session_state.single_candidate_df,
                filename=f"{candidate['drug']}_{candidate['disease']}_analysis.csv",
                text="Download CSV Data"
            )
            st.markdown(csv_link, unsafe_allow_html=True)
    
    # Return to candidates list button
    st.button("Return to Candidates List", on_click=lambda: st.session_state.update({"selected_candidate": None}))

# Actions section
st.sidebar.header("Actions")

# Generate new candidates button
if st.sidebar.button("Generate New Candidates", type="primary"):
    with st.spinner("Generating new repurposing candidates..."):
        candidates = generate_new_candidates()
        st.success(f"Generated {len(candidates)} repurposing candidates")

# Filter section
st.sidebar.header("Filters")

# Drug filter
drug_filter = st.sidebar.text_input("Drug Name")

# Disease filter
disease_filter = st.sidebar.text_input("Disease Name")

# Confidence score filter
confidence_filter = st.sidebar.slider(
    "Minimum Confidence Score",
    min_value=0,
    max_value=100,
    value=0
)

# Apply filters button
if st.sidebar.button("Apply Filters"):
    # Search for candidates matching the filters
    filtered_candidates = search_candidates(
        drug_name=drug_filter if drug_filter else None,
        disease_name=disease_filter if disease_filter else None,
        min_confidence=confidence_filter
    )
    
    # Store in session state
    st.session_state.filtered_candidates = filtered_candidates
    
    # Reset selected candidate
    st.session_state.selected_candidate = None

# Initialize session state for selected candidate
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

# Initialize filtered candidates
# Initialize candidates and filtered_candidates if they don't exist
if "candidates" not in st.session_state:
    st.session_state.candidates = []
    
if "filtered_candidates" not in st.session_state:
    st.session_state.filtered_candidates = []
elif st.session_state.filtered_candidates is None:
    st.session_state.filtered_candidates = []

# Check if a candidate is selected
if st.session_state.selected_candidate is not None:
    # Display candidate details
    display_candidate_details(st.session_state.selected_candidate)
else:
    # Display candidates list
    
    # Get candidates (filtered or all)
    candidates = st.session_state.filtered_candidates
    
    if candidates:
        # Summary stats
        st.subheader("Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candidates", len(candidates))
        
        with col2:
            # Count high confidence candidates (>70%)
            high_conf = len([c for c in candidates if c['confidence_score'] >= 70])
            st.metric("High Confidence (≥70%)", high_conf)
        
        with col3:
            # Calculate average confidence
            avg_conf = sum(c['confidence_score'] for c in candidates) / len(candidates)
            st.metric("Average Confidence", f"{avg_conf:.1f}%")
        
        # Add risk assessment visualization
        st.subheader("Risk Assessment")
        
        # Create the risk assessment summary visualization
        try:
            risk_summary_fig = create_risk_assessment_summary(candidates)
            st.plotly_chart(risk_summary_fig, use_container_width=True)
            
            # Add explanation
            st.info("""
            This risk-confidence assessment plots candidates based on:
            - **X-axis**: Confidence score (higher is better)
            - **Y-axis**: Safety score (higher is better)
            - **Size**: Overall viability (larger is better)
            - **Color**: Risk level (green is lower risk)
            
            The optimal candidates appear in the upper-right quadrant.
            """)
        except Exception as e:
            st.error(f"Error generating risk assessment summary: {str(e)}")
        
        # Other visualizations and export options
        st.subheader("Confidence Analysis")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Confidence distribution
            fig = create_confidence_distribution(candidates)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top candidates comparison
            fig = create_comparison_chart(candidates, top_n=5)
            st.plotly_chart(fig, use_container_width=True)
            
        with col3:
            # Export options
            st.subheader("Export Results")
            st.write("Export your findings for further analysis or sharing")
            
            # Initialize session state for export links
            if 'show_candidates_pdf_link' not in st.session_state:
                st.session_state.show_candidates_pdf_link = False
            
            if 'show_candidates_csv_link' not in st.session_state:
                st.session_state.show_candidates_csv_link = False
            
            # Export to PDF button
            if st.button("Export to PDF", key="candidates_pdf_btn"):
                with st.spinner("Generating PDF..."):
                    st.session_state.candidates_pdf_data = create_repurposing_candidates_pdf(candidates)
                    st.session_state.show_candidates_pdf_link = True
            
            # Export to CSV button
            if st.button("Export to CSV", key="candidates_csv_btn"):
                with st.spinner("Generating CSV..."):
                    st.session_state.candidates_df = candidates_to_dataframe(candidates)
                    st.session_state.show_candidates_csv_link = True
            
            # Show download links if generated
            if st.session_state.get('show_candidates_pdf_link', False):
                pdf_link = generate_pdf_download_link(
                    st.session_state.candidates_pdf_data,
                    filename="drug_repurposing_candidates.pdf",
                    text="Download PDF Report"
                )
                st.markdown(pdf_link, unsafe_allow_html=True)
            
            if st.session_state.get('show_candidates_csv_link', False):
                csv_link = generate_csv_download_link(
                    st.session_state.candidates_df,
                    filename="drug_repurposing_candidates.csv",
                    text="Download CSV Data"
                )
                st.markdown(csv_link, unsafe_allow_html=True)
                
            # Link to dedicated export page
            st.markdown("[More export options →](/Export_Results)", unsafe_allow_html=True)
        
        # Candidates table
        st.subheader("Repurposing Candidates")
        
        # Convert to DataFrame for display
        candidates_df = pd.DataFrame([
            {
                "Drug": c['drug'],
                "Disease": c['disease'],
                "Confidence Score": format_confidence_badge(c['confidence_score']),
                "Status": c['status'],
                "Evidence Count": c['evidence_count'],
                "Details": "View"
            }
            for c in candidates
        ])
        
        # Create interactive table
        # Since Streamlit doesn't have native interactive tables with buttons, we'll use a workaround
        
        # Display the table
        st.write(candidates_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Add selection mechanism
        selected_index = st.selectbox(
            "Select a candidate to view details:",
            options=range(len(candidates)),
            format_func=lambda i: f"{candidates[i]['drug']} for {candidates[i]['disease']} - {candidates[i]['confidence_score']}%"
        )
        
        if st.button("View Details", type="primary"):
            st.session_state.selected_candidate = candidates[selected_index]
            st.rerun()
        
        # Option to analyze a specific drug-disease pair
        st.subheader("Analyze New Drug-Disease Pair")
        
        with st.form("analyze_pair_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                drug_name = st.selectbox("Drug", options=[drug['name'] for drug in st.session_state.drugs])
            
            with col2:
                disease_name = st.selectbox("Disease", options=[disease['name'] for disease in st.session_state.diseases])
            
            # Submit button
            submitted = st.form_submit_button("Analyze Pair")
            
            if submitted:
                # Get drug and disease details
                drug = get_drug_by_name(drug_name)
                disease = get_disease_by_name(disease_name)
                
                if drug and disease:
                    # Analyze the candidate
                    with st.spinner(f"Analyzing {drug_name} for {disease_name}..."):
                        analysis = analyze_repurposing_candidate(drug, disease, st.session_state.graph)
                        
                        # Add to candidates if not already present
                        candidate_exists = False
                        for c in st.session_state.candidates:
                            if c['drug'] == analysis['drug'] and c['disease'] == analysis['disease']:
                                candidate_exists = True
                                break
                        
                        if not candidate_exists:
                            st.session_state.candidates.append(analysis)
                            st.session_state.metrics["candidates_count"] = len(st.session_state.candidates)
                        
                        # Display the analysis
                        st.session_state.selected_candidate = analysis
                        st.rerun()
                else:
                    st.error("Could not find the selected drug or disease")
    else:
        st.info("No repurposing candidates available. Click 'Generate New Candidates' to get started.")
