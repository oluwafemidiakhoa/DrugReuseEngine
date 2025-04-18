import streamlit as st
import os
import sys
sys.path.append(".")
import neo4j_utils

st.set_page_config(page_title="Settings - Drug Repurposing Engine", page_icon="💊")

st.title("Settings")
st.markdown("""
Configure API keys and database connections for the Drug Repurposing Engine.
These settings allow you to customize the functionality and enhance the capabilities of the platform.
""")

# Function to update environment variable
def update_env_var(key, value):
    if value:
        os.environ[key] = value
        return True
    return False

# Horizontal separator
st.divider()

# OpenAI settings - Primary AI model
st.subheader("🧠 OpenAI Settings (Primary AI)")
st.markdown("""
[OpenAI](https://openai.com/) provides state-of-the-art AI models for advanced biomedical analysis.
The platform uses GPT-4o for generating mechanistic explanations and analyzing drug-disease relationships.
""")

current_openai_key = os.environ.get("OPENAI_API_KEY", "")
openai_key_placeholder = "•" * len(current_openai_key) if current_openai_key else ""

openai_key = st.text_input(
    "OpenAI API Key", 
    value=openai_key_placeholder,
    type="password",
    help="Enter your OpenAI API key. Get a key at https://platform.openai.com/api-keys",
    key="openai_key_input"
)

if st.button("Save OpenAI API Key", key="save_openai"):
    if openai_key and openai_key != openai_key_placeholder:
        if update_env_var("OPENAI_API_KEY", openai_key):
            st.success("OpenAI API key saved successfully!")
            # Force a session state update
            st.session_state.openai_key_set = True
    elif openai_key == "":
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            st.success("OpenAI API key removed!")
            st.session_state.openai_key_set = False

# Horizontal separator
st.divider()

# Horizontal separator
st.divider()

# Hugging Face settings
st.subheader("🤗 Hugging Face Settings")
st.markdown("""
[Hugging Face](https://huggingface.co/) provides access to state-of-the-art open-source models for natural language processing.
The platform uses Hugging Face models as a fallback when OpenAI is not available.
""")

current_hf_key = os.environ.get("HUGGINGFACE_API_KEY", "")
hf_key_placeholder = "•" * len(current_hf_key) if current_hf_key else ""

hf_key = st.text_input(
    "Hugging Face API Key", 
    value=hf_key_placeholder,
    type="password",
    help="Enter your Hugging Face API key. Get a key at https://huggingface.co/settings/tokens",
    key="hf_key_input"
)

if st.button("Save Hugging Face API Key", key="save_hf"):
    if hf_key and hf_key != hf_key_placeholder:
        if update_env_var("HUGGINGFACE_API_KEY", hf_key):
            st.success("Hugging Face API key saved successfully!")
            # Force a session state update
            st.session_state.hf_key_set = True
    elif hf_key == "":
        if "HUGGINGFACE_API_KEY" in os.environ:
            del os.environ["HUGGINGFACE_API_KEY"]
            st.success("Hugging Face API key removed!")
            st.session_state.hf_key_set = False

# Horizontal separator
st.divider()

# Status information
st.subheader("Current Status")

# Check API key status
openai_status = "Available ✅" if os.environ.get("OPENAI_API_KEY") else "Not configured ❌" 
gemini_status = "Available ✅" if os.environ.get("GEMINI_API_KEY") else "Not configured ❌"
hf_status = "Available ✅" if os.environ.get("HUGGINGFACE_API_KEY") else "Not configured ❌"

# Check Neo4j status
neo4j_status = "Available ✅" if neo4j_utils.NEO4J_AVAILABLE else "Not configured ❌"

# Create a table to display status of all integrated services
status_data = {
    "Service": ["OpenAI", "Hugging Face", "Neo4j Graph Database"],
    "Status": [openai_status, hf_status, neo4j_status],
    "Role": ["Primary AI", "Fallback AI", "Graph Database"]
}

st.table(status_data)

# Information about fallback behavior
st.info("""
**AI Model Priority**:  
The system will try to use OpenAI first for the best results. If OpenAI is not available or encounters an error, 
the system will automatically fall back to using Hugging Face models. If no AI model is available, 
the system will use traditional rule-based methods for analysis.
""")

# Add a note about API key security
st.warning("""
**API Key Security**:  
API keys are stored only in the server's environment variables for the current session and are never saved to disk. 
They will need to be re-entered if the server restarts.
""")

# Horizontal separator
st.divider()

# Neo4j database settings
st.subheader("🔌 Neo4j Graph Database")
st.markdown("""
[Neo4j](https://neo4j.com/) is a powerful graph database that provides enhanced capabilities for analyzing complex relationships in the biomedical domain. With Neo4j integration, the Drug Repurposing Engine can perform advanced graph analytics, path discovery, and similarity calculations.
""")

# Import additional utils for Neo4j connection
from utils import save_neo4j_config

# Get current configuration
current_uri = os.environ.get("NEO4J_URI", "neo4j+s://9615a24a.databases.neo4j.io")
current_username = os.environ.get("NEO4J_USERNAME", "neo4j")
current_password = os.environ.get("NEO4J_PASSWORD", "")
password_placeholder = "•" * len(current_password) if current_password else ""

# Connection status - use the already imported neo4j_utils from the top of the file
neo4j_connected = neo4j_utils.NEO4J_AVAILABLE
status_color = "green" if neo4j_connected else "red"
status_icon = "✅" if neo4j_connected else "❌"
st.markdown(f"<span style='color:{status_color}'>{status_icon} **Connection Status:** {'Connected' if neo4j_connected else 'Not Connected'}</span>", unsafe_allow_html=True)

# Connection form
with st.form("neo4j_connection_form"):
    st.write("Neo4j Connection Configuration")
    
    neo4j_uri = st.text_input("Neo4j URI", value=current_uri)
    neo4j_username = st.text_input("Username", value=current_username)
    neo4j_password = st.text_input("Password", value=password_placeholder, type="password")
    
    submitted = st.form_submit_button("Connect to Neo4j")
    
    if submitted:
        # Don't update if the password field contains only placeholder dots
        if neo4j_password != password_placeholder:
            # Save the configuration
            if save_neo4j_config(neo4j_uri, neo4j_username, neo4j_password):
                # Initialize Neo4j connection
                if neo4j_utils.initialize_neo4j():
                    st.success("Successfully connected to Neo4j!")
                    st.rerun()  # Refresh to show updated status
                else:
                    st.error("Failed to connect to Neo4j. Please check your connection details.")

# Information about Neo4j
st.info("""
**Neo4j Integration Benefits**:  
- **Superior Knowledge Graph**: Store and query complex relationships between drugs, diseases, genes, and proteins.
- **Path Discovery**: Find biological pathways between drugs and diseases to understand mechanisms of action.
- **Advanced Analytics**: Calculate centrality measures, find similar drugs, and identify repurposing opportunities.
- **Visual Exploration**: Visualize complex networks with interactive graph visualizations.
""")