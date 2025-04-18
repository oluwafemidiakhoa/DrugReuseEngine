import streamlit as st
import pandas as pd
import os
import threading
import time
from utils import initialize_session_state, api_key_ui, check_api_key
from export_button import floating_export_button
from load_env import load_environment_variables
from tooltip_helper import setup_tooltip_css, render_tooltip, tooltip_header

# Load environment variables from .env file or Streamlit Cloud secrets
load_environment_variables()

# Set page configuration
st.set_page_config(
    page_title="Drug Repurposing Engine",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Start API server in a separate thread for Streamlit Cloud deployment
def run_api():
    import uvicorn
    # Use port 8000 for API in Streamlit Cloud
    os.environ["API_PORT"] = "8000"
    try:
        import run_api
        uvicorn.run("run_api:app", host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting API server: {e}")

# Start API server if not already running
if 'api_server_started' not in st.session_state:
    try:
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        st.session_state.api_server_started = True
        print("API server started in background thread")
    except Exception as e:
        print(f"Failed to start API server: {e}")
        st.session_state.api_server_started = False

# Initialize session state variables
initialize_session_state()

# Setup tooltip CSS
setup_tooltip_css()

# Sidebar with API key input and settings
with st.sidebar:
    st.image("generated-icon.png", width=100)
    st.title("Drug Repurposing Engine")
    
    st.markdown("---")
    
    # API key input section
    api_key_ui()
    
    st.markdown("---")
    
    st.markdown("### Navigation")
    st.markdown("""
    - [Home](#drug-repurposing-engine)
    - [Drug Search](/Drug_Search)
    - [Disease Search](/Disease_Search)
    - [Knowledge Graph](/Knowledge_Graph)
    - [Repurposing Candidates](/Repurposing_Candidates)
    - [Global Impact](/Global_Impact)
    - [Settings](/Settings)
    - [How It Works](/How_It_Works)
    """)
    
    st.markdown("---")
    
    # Display API test results
    if hasattr(st.session_state, 'api_test_results'):
        st.markdown("### API Status")
        for api, status in st.session_state.api_test_results.items():
            if status:
                st.success(f"{api} API: ✓")
            else:
                st.error(f"{api} API: ✗")

# Main content area starts here
st.title("💊 Drug Repurposing Engine")
st.subheader("AI-Powered Drug Repurposing Platform")

# Show API server status
if st.session_state.get('api_server_started', False):
    st.success("API Server running on port 8000")
else:
    st.warning("API Server failed to start. Some features may be limited.")

# Import and run the rest of the original app.py code
try:
    from app import display_main_interface
    display_main_interface()
except Exception as e:
    # If function doesn't exist, we'll use the remaining code from app.py
    # Copy the rest of your original app.py content here

    # Verify API keys are available
    api_key_valid = check_api_key()
    
    if not api_key_valid:
        st.warning("⚠️ Please enter valid API keys in the sidebar to enable all features.")
        
    # Introduction section with tooltips
    st.markdown("""
    Welcome to the Drug Repurposing Engine, a cutting-edge computational platform leveraging 
    AI and network analysis to revolutionize pharmaceutical discovery through drug repurposing.
    """)
    
    # Key Features section with tooltips
    st.header("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("🧬 PubMed literature mining and knowledge extraction")
        st.markdown("🔬 Drug-disease similarity analysis across multiple dimensions")
        st.markdown("🧠 AI-powered mechanistic explanations " + 
                   f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Mechanistic Explanation</strong><br>A detailed description of exactly how a drug might work for a new disease at the molecular level. It's like explaining step-by-step how the key (drug) can unlock a different door (disease) than it was designed for. 🔑🚪</span></span>", 
                   unsafe_allow_html=True)
        st.markdown("🧩 Knowledge graph construction and analysis " + 
                   f"<span class=\"tooltip\">ℹ️<span class=\"tooltiptext\"><strong>Knowledge Graph</strong><br>A network connecting drugs, diseases, genes, and proteins based on known relationships. Think of it as a map showing how everything in biology is connected, like a web connecting all the dots. 🕸️</span></span>", 
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