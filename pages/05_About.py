import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="About | Drug Repurposing Engine",
    page_icon="ℹ️",
    layout="wide",
)

st.title("About the Drug Repurposing Engine")

# Overview section
st.header("Overview")
st.write("""
The Drug Repurposing Engine is a comprehensive computational platform designed to accelerate the discovery of new 
therapeutic applications for existing drugs. By leveraging multiple data sources, advanced analytics, and machine 
learning techniques, the system identifies promising drug-disease pairs and provides mechanistic explanations 
for their potential efficacy.
""")

# Features section
st.header("Key Features")
features = {
    "🔍 Automated data ingestion": "Collects and processes data from biomedical literature and databases",
    "🔄 Multi-modal data integration": "Combines information from various sources for comprehensive analysis",
    "🧠 AI-driven insights": "Generates mechanistic explanations and confidence scores",
    "📊 Knowledge graph analysis": "Maps complex relationships between drugs, diseases, and biological entities",
    "⭐ Confidence scoring": "Provides quantitative assessment of repurposing candidates",
    "🖥️ Interactive exploration": "Enables researchers to explore and visualize results"
}

for icon, feature in features.items():
    st.write(f"**{icon}** - {feature}")

# System architecture
st.header("System Architecture")
st.write("""
The Drug Repurposing Engine consists of several interconnected modules:

1. **Data Ingestion**: Collects and normalizes data from various sources
2. **Data Processing**: Analyzes and integrates multi-modal data
3. **AI Insights**: Generates mechanistic explanations and confidence scores
4. **User Interface**: Provides interactive exploration and visualization
5. **API**: Enables programmatic access to the engine's functionality
""")

# Module descriptions
with st.expander("Module Descriptions"):
    st.subheader("Data Ingestion")
    st.write("""
    The data ingestion module collects and normalizes data from various biomedical sources:
    
    - **PubMed Miner**: Extracts drug-disease relationships from scientific literature
    - **RxNorm Integration**: Normalizes drug names and retrieves drug information
    - **UMLS Mapping**: Maps terms to standardized medical concepts
    """)
    
    st.subheader("Data Processing")
    st.write("""
    The data processing module analyzes and integrates multi-modal data:
    
    - **Text Analysis**: Extracts relationships and insights from textual data
    - **Gene Expression**: Analyzes differential gene expression patterns
    - **Knowledge Graph**: Builds and analyzes a comprehensive knowledge graph
    """)
    
    st.subheader("AI Insights")
    st.write("""
    The AI insights module generates mechanistic explanations and confidence scores:
    
    - **Mechanistic Explanation**: Provides detailed explanations of potential mechanisms
    - **Confidence Scoring**: Calculates confidence scores for repurposing candidates
    """)
    
    st.subheader("User Interface")
    st.write("""
    The user interface module provides interactive exploration and visualization:
    
    - **Dashboard**: Displays summary statistics and key insights
    - **Query Interface**: Enables searching and filtering of repurposing candidates
    - **Visualizations**: Creates interactive visualizations of data and results
    """)

# How it works
st.header("How It Works")
st.write("""
The Drug Repurposing Engine follows a systematic approach to identify and evaluate repurposing candidates:

1. **Data Collection**: Gathers information from biomedical literature, databases, and other sources
2. **Knowledge Graph Construction**: Builds a comprehensive graph of relationships between drugs, diseases, and biological entities
3. **Candidate Identification**: Uses graph analysis and AI algorithms to identify potential repurposing candidates
4. **Mechanism Prediction**: Generates plausible mechanistic explanations for how drugs might treat diseases
5. **Confidence Scoring**: Calculates confidence scores based on multiple factors, including literature evidence, network structure, and mechanism plausibility
6. **Presentation and Exploration**: Presents results through an interactive interface for researchers to explore and evaluate
""")

# Benefits
st.header("Benefits")
col1, col2 = st.columns(2)

with col1:
    st.subheader("For Researchers")
    st.write("""
    - Accelerates drug discovery process
    - Provides mechanistic insights
    - Integrates diverse data sources
    - Enables hypothesis generation
    - Facilitates literature discovery
    """)

with col2:
    st.subheader("For Healthcare")
    st.write("""
    - Reduces drug development costs
    - Shortens time to market
    - Expands treatment options
    - Addresses unmet medical needs
    - Improves patient outcomes
    """)

# Technology stack
st.header("Technology Stack")
st.write("""
The Drug Repurposing Engine is built using the following technologies:

- **Python**: Core programming language
- **NetworkX**: Knowledge graph construction and analysis
- **NLTK/spaCy**: Natural language processing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
""")

# Limitations
st.header("Limitations and Future Work")
st.write("""
While the Drug Repurposing Engine provides valuable insights, it has some limitations:

- **Data Completeness**: The system is limited by the availability and quality of input data
- **Biological Complexity**: Some biological mechanisms are too complex to be fully captured
- **Validation Requirements**: Computational predictions require experimental validation
- **Regulatory Considerations**: Regulatory requirements for repurposed drugs are not addressed

Future work will focus on:

- Expanding data sources to include genomic, proteomic, and clinical data
- Enhancing AI models to better capture complex biological mechanisms
- Implementing validation frameworks to assess the quality of predictions
- Incorporating regulatory considerations into the evaluation process
""")

# References
st.header("References")
references = [
    "Pushpakom, S., Iorio, F., Eyers, P. A., Escott, K. J., Hopper, S., Wells, A., ... & Pirmohamed, M. (2019). Drug repurposing: progress, challenges and recommendations. Nature Reviews Drug Discovery, 18(1), 41-58.",
    "Aliper, A., Plis, S., Artemov, A., Ulloa, A., Mamoshina, P., & Zhavoronkov, A. (2016). Deep learning applications for predicting pharmacological properties of drugs and drug repurposing using transcriptomic data. Molecular Pharmaceutics, 13(7), 2524-2530.",
    "Yao, X., Hao, H., Li, Y., & Li, S. (2011). Modularity-based credible prediction of disease genes and detection of disease subtypes on the phenotype-gene heterogeneous network. BMC Systems Biology, 5(1), 1-11.",
    "Chen, B., & Altman, R. B. (2017). Drug repurposing in the era of deep learning. Journal of Proteomics & Bioinformatics, 10(9), 1-5."
]

for ref in references:
    st.write(f"- {ref}")

# Footer
st.markdown("---")
st.markdown("© 2023 Drug Repurposing Engine | Powered by Biomedical Data Integration and AI Analysis")
