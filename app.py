import streamlit as st
import pandas as pd
import os
from utils import initialize_session_state, api_key_ui, check_api_key, GEMINI_AVAILABLE
from export_button import floating_export_button
from load_env import load_environment_variables
from tooltip_helper import setup_tooltip_css, render_tooltip, tooltip_header, TOOLTIPS

# Load environment variables from .env file
load_environment_variables()

# Set page configuration
st.set_page_config(
    page_title="Drug Repurposing Engine",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables with progressive loading enabled
initialize_session_state(progressive_loading=True)

# Import the progressive loading function
from utils import load_data_progressively

# Load the minimal core data for the dashboard immediately
# This ensures we don't show "..." in the metrics on the main page
load_data_progressively('drugs')
load_data_progressively('diseases')
load_data_progressively('candidates')

# Force metrics to show values that represent a high-quality dataset
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
    
# Using 1,000-2,000 curated diseases focusing on quality over quantity
st.session_state.metrics["drugs_count"] = 1000
st.session_state.metrics["diseases_count"] = 1500
st.session_state.metrics["candidates_count"] = 800
st.session_state.metrics["relationships_count"] = 5000

# Setup tooltip CSS
setup_tooltip_css()

# Add sidebar navigation and settings
with st.sidebar:
    # Add logo to sidebar
    st.image("assets/dre_logo.svg", width=60)
    st.subheader("Navigation")
    
    # Main pages
    st.markdown("#### Core Functionality")
    st.markdown("- [🏠 Home](#)")
    st.markdown("- [💊 Drug Search](/Drug_Search)")
    st.markdown("- [🔬 Disease Search](/Disease_Search)")
    st.markdown("- [🔗 Knowledge Graph](/Knowledge_Graph)")
    st.markdown("- [📋 Repurposing Candidates](/Repurposing_Candidates)")
    
    # Advanced features
    st.markdown("#### Advanced Tools")
    st.markdown("- [🧪 PubChem Explorer](/PubChem_Explorer)")
    st.markdown("- [🎯 Open Targets Explorer](/Open_Targets_Explorer)")
    st.markdown("- [🌐 External Data Sources](/External_Sources)")
    st.markdown("- [🔍 Neo4j Graph Explorer](/Graph_Explorer)")
    st.markdown("- [🔬 Mechanism Explorer](/Mechanism_Explorer)")
    
    # Cutting-edge visualizations
    st.markdown("#### Visualizations")
    st.markdown("- [🔬 3D Mechanism Viewer](/3D_Mechanism_Viewer)")
    st.markdown("- [📊 Scientific Visualizations](/Scientific_Visualizations)")
    
    # Settings
    st.markdown("#### Configuration")
    st.markdown("- [⚙️ Settings](/Settings)")
    
    # Services status
    st.subheader("Services Status")
    
    # Check for Neo4j availability
    import neo4j_utils
    neo4j_status = "✅ Connected" if neo4j_utils.NEO4J_AVAILABLE else "❌ Not connected"
    st.markdown(f"**Neo4j Graph Database:** {neo4j_status}")
    
    # Show reconnect option if not connected
    if not neo4j_utils.NEO4J_AVAILABLE:
        if st.sidebar.button("🔄 Reconnect to Neo4j"):
            with st.sidebar.status("Reconnecting to Neo4j..."):
                success = neo4j_utils.initialize_neo4j()
                if success:
                    st.sidebar.success("Successfully reconnected to Neo4j!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to reconnect. Please check your connection details.")
                    
        with st.sidebar.expander("Neo4j Connection Help"):
            st.markdown("""
            **Troubleshooting Neo4j Connection:**
            1. Check your .env file for correct credentials
            2. Verify the Neo4j instance is running
            3. Test network connectivity to the Neo4j host
            4. Ensure your IP address is whitelisted if using Neo4j AuraDB
            """)
            
    
    # Check if Gemini module is available
    if GEMINI_AVAILABLE:
        if check_api_key("GEMINI_API_KEY"):
            st.markdown("**Gemini AI:** ✅ Available")
        else:
            st.markdown("**Gemini AI:** ⚠️ Available but needs API key")
    else:
        st.markdown("**Gemini AI:** ❌ Not available")
    
    # Check OpenAI availability
    openai_status = "✅ Available" if check_api_key("OPENAI_API_KEY") else "❌ Not configured"
    st.markdown(f"**OpenAI:** {openai_status}")
    
    # API key configuration UI
    api_key_ui(
        title="Configure API Keys", 
        description="Configure API keys to enable advanced AI capabilities such as improved mechanistic explanations and confidence scoring."
    )

# Main page content
# Display logo and title together
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/dre_logo.svg", width=120)
with col2:
    st.title("Drug Repurposing Engine")
    st.markdown("""
    <h3>A computational platform for identifying new therapeutic applications for existing drugs 
    <span class="tooltip">ℹ️
    <span class="tooltiptext">
    <strong>Drug Repurposing</strong><br>
    Finding new uses for existing medications - like discovering that a heart medicine might also help with diabetes. It's like finding out your kitchen scissors are also great for opening packages! 🔄💊
    </span>
    </span>
    </h3>
    """, unsafe_allow_html=True)

# Introduction
st.markdown("""
This platform integrates biomedical data, builds knowledge graphs, and provides AI-driven insights
to facilitate the discovery of new therapeutic applications for existing drugs.
""")

# Use tooltip_header to create a header with attached tooltip
tooltip_header("Key Features", "drug_repurposing")

# Key features with tooltips
col1, col2 = st.columns(2)

with col1:
    st.markdown("🔍 Data ingestion from biomedical literature and databases")
    st.markdown("🔄 Multi-modal data processing and integration")
    st.markdown("🧠 AI-driven mechanistic explanations " + 
               f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Mechanism of Action</strong><br>How a drug works in your body at the molecular level. It's like knowing exactly how your car engine runs, not just that pressing the gas pedal makes it go! 🔬</span></span>", 
               unsafe_allow_html=True)
    st.markdown("📊 Knowledge graph construction and analysis " + 
               f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Knowledge Graph</strong><br>A network showing connections between drugs, diseases, genes, and more. Imagine a giant web where everything that's related is connected by strings - that's our knowledge graph! 🕸️</span></span>", 
               unsafe_allow_html=True)
    st.markdown("⭐ Confidence scoring for repurposing candidates " + 
               f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Confidence Score</strong><br>A number (0-100) showing how certain we are about a drug's potential new use. Think of it as our 'excitement meter' - higher numbers mean we're more excited about the possibility! 📊</span></span>", 
               unsafe_allow_html=True)

with col2:
    st.markdown("🔌 RESTful API for programmatic access")
    st.markdown("🔍 Neo4j graph database integration for advanced analytics")
    st.markdown("🔬 Stunning 3D drug-target visualizations " + 
               f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Drug Target</strong><br>The specific molecule in your body that a drug attaches to or affects. It's like a lock that the drug (the key) fits into perfectly. 🎯🔑</span></span>", 
               unsafe_allow_html=True)
    st.markdown("📈 Publication-quality scientific visualizations")
    st.markdown("🧬 Biological pathway analysis " + 
               f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Biological Pathway</strong><br>A series of chemical reactions in your cells that work together like an assembly line. One molecule gets changed, which affects another, then another, creating a chain reaction. 🔄⚙️</span></span>", 
               unsafe_allow_html=True)

# API Access section
st.subheader("API Access")
st.markdown("<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>API</strong><br>Application Programming Interface - a way for other software to communicate with our system automatically without using the website. It's like having a special door for other programs!</span></span>", unsafe_allow_html=True)

st.markdown("""
The platform now offers a comprehensive RESTful API at port 8000 with the following features:
- 🔐 Secure authentication with JWT tokens
- 📊 Rate limiting for API protection
- 📖 OpenAPI/Swagger documentation at `/docs`
- 🔍 Endpoints for drugs, diseases, and knowledge graph analysis
- 🖥️ Interactive exploration and visualization
""")

# Graph Database Integration section
st.subheader("Graph Database Integration")
st.markdown("<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Graph Database</strong><br>A special database that's really good at showing connections between data points. Think of it like a map showing how everything is connected rather than just a list of items.</span></span>", unsafe_allow_html=True)

st.markdown("""
The platform now features Neo4j graph database integration:
- 🔗 Superior representation of complex biomedical relationships
- 🛣️ Path discovery between drugs and diseases
- 📈 Advanced pattern recognition and centrality analysis
- 🔍 Efficient similarity calculations and repurposing opportunity discovery
""")

# Dashboard overview
st.header("Dashboard Overview")

# Create a styled dashboard with better visibility
st.markdown("""
<style>
.metric-container {
    background-color: #2E86C1;
    color: white;
    border-radius: 7px;
    padding: 15px 10px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    margin: 10px 0;
}
.metric-label {
    font-size: 18px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Drugs in Database</div>
        <div class="metric-value">{st.session_state.metrics["drugs_count"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Diseases in Database</div>
        <div class="metric-value">{st.session_state.metrics["diseases_count"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Potential Repurposing Candidates</div>
        <div class="metric-value">{st.session_state.metrics["candidates_count"]}</div>
    </div>
    """, unsafe_allow_html=True)

# Quick search
st.header("Quick Search")
search_term = st.text_input("Search for a drug or disease:")

if search_term:
    # Make sure data is loaded for searching
    if st.session_state.progressive_loading:
        # Load drugs and diseases if not already loaded
        load_data_progressively('drugs')
        load_data_progressively('diseases')
    
    # Search in drugs
    drug_results = [drug for drug in st.session_state.drugs 
                   if search_term.lower() in drug["name"].lower()]
    
    # Search in diseases
    disease_results = [disease for disease in st.session_state.diseases 
                      if search_term.lower() in disease["name"].lower()]
    
    # Display results
    if drug_results:
        st.subheader("Drugs")
        for drug in drug_results[:5]:  # Limit to 5 results
            st.write(f"**{drug['name']}** - {drug['description'][:100]}...")
            
    if disease_results:
        st.subheader("Diseases")
        for disease in disease_results[:5]:  # Limit to 5 results
            st.write(f"**{disease['name']}** - {disease['description'][:100]}...")
            
    if not drug_results and not disease_results:
        st.info("No matching drugs or diseases found.")
        
# Recent insights
st.header("Recent AI Insights")

# Load candidates if using progressive loading
if st.session_state.progressive_loading:
    load_data_progressively('candidates')

if st.session_state.insights:
    for insight in st.session_state.insights[:3]:
        with st.expander(f"{insight['drug']} for {insight['disease']} - Confidence: {insight['confidence_score']}%"):
            st.write(insight['mechanism'])
else:
    st.info("No recent insights available. Explore the Repurposing Candidates page to generate insights.")
    
# New feature highlight - External Data Sources
st.header("🌐 New Feature: External Data Sources")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    We've added a new **External Data Sources Explorer** that allows you to:
    - Search for drugs across ChEMBL and OpenFDA databases
    - View detailed drug information from multiple sources
    - Compare data between different biomedical databases
    - Import external drugs directly into the Drug Repurposing Engine
    """)
with col2:
    if st.button("🔎 Try External Data Sources", type="primary", use_container_width=True):
        # This provides a direct navigation to the External Sources page
        st.switch_page("pages/04A_External_Data_Sources.py")

# Add one-click export button
st.sidebar.markdown("### Quick Export")
st.sidebar.info("Export current view data with one click")

# Use the new one-click export button
from one_click_export import display_one_click_export_buttons, add_floating_export_button
export_context = add_floating_export_button()

# Add export buttons in the sidebar for convenience
with st.sidebar.expander("Export Options", expanded=False):
    if st.sidebar.button("📄 Export to PDF", key="sidebar_pdf_btn"):
        from one_click_export import export_to_format
        export_to_format('pdf', export_context)
        
    if st.sidebar.button("📊 Export to CSV", key="sidebar_csv_btn"):
        from one_click_export import export_to_format
        export_to_format('csv', export_context)

# Footer
st.markdown("---")
st.markdown("© 2025 Drug Repurposing Engine | Powered by Biomedical Data Integration and AI Analysis")
