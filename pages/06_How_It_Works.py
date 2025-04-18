import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import time
from utils import initialize_session_state

# Set page configuration
st.set_page_config(
    page_title="How It Works | Drug Repurposing Engine",
    page_icon="🔄",
    layout="wide",
)

# Initialize session state variables
initialize_session_state()

st.title("How It Works")
st.write("Learn about the data flow and scoring system in the Drug Repurposing Engine")

tab1, tab2, tab3, tab4 = st.tabs(["Data Flow", "Confidence Scoring", "Knowledge Graph", "AI Integration"])

with tab1:
    st.header("Data Flow")
    st.write("The Drug Repurposing Engine processes data through the following flow:")
    
    # Create the data flow visualization
    steps = [
        {
            "title": "1. Data Collection",
            "description": "PubMed articles, drug databases, and gene expression datasets are collected through the data ingestion modules.",
            "icon": "📚",
            "color": "#4287f5"  # Blue
        },
        {
            "title": "2. Data Normalization",
            "description": "Drug names, disease terms, and gene identifiers are normalized using RxNorm and UMLS mapping.",
            "icon": "🔄",
            "color": "#42c5f5"  # Light Blue
        },
        {
            "title": "3. Text Analysis",
            "description": "Natural language processing techniques extract drug-disease relationships and potential mechanisms from scientific literature.",
            "icon": "📝",
            "color": "#42f5a7"  # Teal
        },
        {
            "title": "4. Gene Expression Analysis",
            "description": "Differential gene expression analysis identifies gene signatures associated with diseases and drug responses.",
            "icon": "🧬",
            "color": "#42f55a"  # Green
        },
        {
            "title": "5. Knowledge Graph Construction",
            "description": "A comprehensive knowledge graph is built, connecting drugs, diseases, genes, proteins, and biological pathways.",
            "icon": "🕸️",
            "color": "#d3f542"  # Yellow
        },
        {
            "title": "6. Mechanistic Explanation Generation",
            "description": "AI models analyze the knowledge graph and other data to generate detailed explanations of potential mechanisms.",
            "icon": "🤖",
            "color": "#f5a742"  # Orange
        },
        {
            "title": "7. Confidence Score Calculation",
            "description": "Multiple lines of evidence are integrated to calculate confidence scores for repurposing candidates.",
            "icon": "📊",
            "color": "#f55a42"  # Red
        },
        {
            "title": "8. Result Presentation",
            "description": "Results are presented through the user interface and made available through the API.",
            "icon": "📈",
            "color": "#f542a7"  # Pink
        }
    ]
    
    # Display the steps as a diagram
    def create_data_flow_diagram():
        # Create figure
        fig = go.Figure()
        
        num_steps = len(steps)
        
        # Add nodes
        for i, step in enumerate(steps):
            x = i / (num_steps - 1) if num_steps > 1 else 0.5
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[0.5],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=40,
                    color=step["color"],
                    line=dict(color='white', width=2)
                ),
                text=[step["icon"]],
                textposition="middle center",
                textfont=dict(size=20),
                hoverinfo='text',
                hovertext=f"{step['title']}: {step['description']}",
                name=step['title']
            ))
            
            # Add step title below
            fig.add_trace(go.Scatter(
                x=[x],
                y=[0.35],
                mode='text',
                text=[step["title"].split(". ")[1] if ". " in step["title"] else step["title"]],
                textposition="top center",
                textfont=dict(size=12, color='black'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add arrows between nodes
            if i < num_steps - 1:
                x_start = i / (num_steps - 1)
                x_end = (i + 1) / (num_steps - 1)
                
                fig.add_trace(go.Scatter(
                    x=[x_start + 0.04, x_end - 0.04],
                    y=[0.5, 0.5],
                    mode='lines',
                    line=dict(
                        color='gray',
                        width=2,
                        dash='solid'
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add arrowhead
                fig.add_trace(go.Scatter(
                    x=[x_end - 0.04],
                    y=[0.5],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-right',
                        size=10,
                        color='gray'
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
                
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-0.05, 1.05]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[0, 1]
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig
    
    # Display the diagram
    st.plotly_chart(create_data_flow_diagram(), use_container_width=True)
    
    # Display detailed step information
    selected_step = st.selectbox("Select a step to learn more", options=[step["title"] for step in steps])
    
    # Find the selected step
    step_info = next((step for step in steps if step["title"] == selected_step), None)
    
    if step_info:
        st.subheader(step_info["title"])
        st.markdown(f"**{step_info['description']}**")
        
        # Add more detailed information for each step
        if "1. Data Collection" in selected_step:
            st.write("""
            The data collection step involves gathering information from multiple sources:
            
            - **PubMed Articles**: Scientific articles are retrieved based on relevant keywords and MeSH terms.
            - **Drug Databases**: Information on approved drugs is collected from sources like DrugBank, RxNorm, and ChEMBL.
            - **Disease Databases**: Disease information is collected from sources like MeSH, UMLS, and OMIM.
            - **Gene Expression Databases**: Transcriptomic data is collected from sources like GEO, ArrayExpress, and LINCS.
            
            This step establishes the foundational data for all subsequent analyses.
            """)
            
            # Create a sample visualization of data sources
            sources = ["PubMed", "DrugBank", "RxNorm", "ChEMBL", "MeSH", "UMLS", "OMIM", "GEO", "ArrayExpress", "LINCS"]
            counts = [1243, 892, 765, 612, 543, 498, 412, 356, 289, 254]  # Sample counts
            
            source_df = pd.DataFrame({"Source": sources, "Records": counts})
            
            st.bar_chart(source_df.set_index("Source"))
            
        elif "2. Data Normalization" in selected_step:
            st.write("""
            Data normalization ensures consistency across different data sources:
            
            - **Drug Normalization**: Maps different names and identifiers for the same drug to a canonical representation.
            - **Disease Normalization**: Maps different disease terms to standard ontologies like MeSH and ICD.
            - **Gene/Protein Normalization**: Maps gene symbols and protein names to standard identifiers like Entrez Gene IDs and UniProt.
            
            Normalization reduces redundancy and enables integration of heterogeneous data sources.
            """)
            
            # Show a before/after example
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before Normalization")
                st.markdown("""
                - Drug: "Acetaminophen", "Paracetamol", "APAP", "N-acetyl-p-aminophenol"
                - Disease: "Diabetes", "Diabetes Mellitus", "T2DM", "Type 2 Diabetes"
                - Gene: "TNF", "TNFA", "Tumor Necrosis Factor", "TNF-alpha"
                """)
                
            with col2:
                st.subheader("After Normalization")
                st.markdown("""
                - Drug: "Acetaminophen" (RxNorm: 161)
                - Disease: "Diabetes Mellitus, Type 2" (MeSH: D003924)
                - Gene: "TNF" (Entrez Gene: 7124)
                """)
            
        elif "3. Text Analysis" in selected_step:
            st.write("""
            Text analysis extracts structured information from unstructured text:
            
            - **Named Entity Recognition**: Identifies mentions of drugs, diseases, genes, proteins, and other biomedical entities.
            - **Relationship Extraction**: Identifies relationships between entities, such as drug-disease treatments, drug-gene interactions, etc.
            - **Sentiment Analysis**: Analyzes the context of relationships to determine their nature (positive, negative, speculative).
            
            This step transforms unstructured literature into structured data that can be incorporated into the knowledge graph.
            """)
            
            # Show a simple text analysis example
            st.subheader("Example Text Analysis")
            
            example_text = "Metformin inhibits hepatic gluconeogenesis and improves insulin sensitivity, making it effective for treating type 2 diabetes."
            
            st.text_area("Original Text", value=example_text, height=100, disabled=True)
            
            entities = {
                "Metformin": "DRUG",
                "hepatic gluconeogenesis": "BIOLOGICAL_PROCESS",
                "insulin sensitivity": "BIOLOGICAL_PROCESS",
                "type 2 diabetes": "DISEASE"
            }
            
            relationships = [
                {"subject": "Metformin", "predicate": "INHIBITS", "object": "hepatic gluconeogenesis"},
                {"subject": "Metformin", "predicate": "IMPROVES", "object": "insulin sensitivity"},
                {"subject": "Metformin", "predicate": "TREATS", "object": "type 2 diabetes"}
            ]
            
            # Display entities
            st.subheader("Extracted Entities")
            entities_df = pd.DataFrame([{"Entity": k, "Type": v} for k, v in entities.items()])
            st.dataframe(entities_df, hide_index=True)
            
            # Display relationships
            st.subheader("Extracted Relationships")
            st.dataframe(pd.DataFrame(relationships), hide_index=True)
            
        elif "4. Gene Expression Analysis" in selected_step:
            st.write("""
            Gene expression analysis identifies patterns in transcriptomic data:
            
            - **Differential Expression Analysis**: Identifies genes that are up- or down-regulated in disease states or in response to drugs.
            - **Pathway Enrichment Analysis**: Identifies biological pathways that are enriched in differentially expressed genes.
            - **Gene Set Enrichment Analysis**: Identifies sets of genes that show coordinated expression changes.
            
            This step provides molecular-level insights into disease mechanisms and drug effects.
            """)
            
            # Show a simplified gene expression heatmap
            st.subheader("Sample Gene Expression Heatmap")
            
            # Create sample data
            genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
            conditions = ["Disease", "Treatment", "Control"]
            
            # Sample expression values
            expression_data = [
                [1.5, -0.8, 0.2],
                [-1.2, 0.9, 0.1],
                [0.8, 1.2, -0.3],
                [-0.6, -1.5, 0.4],
                [1.9, -0.7, -0.2]
            ]
            
            # Create a heatmap
            fig = go.Figure(data=go.Heatmap(
                z=expression_data,
                x=conditions,
                y=genes,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="Log2 Fold Change")
            ))
            
            fig.update_layout(
                title="Gene Expression Changes",
                xaxis_title="Condition",
                yaxis_title="Gene",
                height=300,
                width=500
            )
            
            st.plotly_chart(fig)
            
        elif "5. Knowledge Graph Construction" in selected_step:
            st.write("""
            Knowledge graph construction integrates data from all previous steps:
            
            - **Node Creation**: Creates nodes for drugs, diseases, genes, proteins, pathways, etc.
            - **Edge Creation**: Creates edges representing relationships between nodes.
            - **Property Assignment**: Assigns properties to nodes and edges, such as confidence scores, evidence sources, etc.
            
            The knowledge graph provides a holistic view of biomedical knowledge, enabling complex queries and inferences.
            """)
            
            # Create a simplified knowledge graph visualization
            st.subheader("Sample Knowledge Graph")
            
            # Create a simple graph
            G = nx.DiGraph()
            
            # Add nodes
            G.add_node("Drug1", type="drug", name="Metformin")
            G.add_node("Drug2", type="drug", name="Aspirin")
            G.add_node("Disease1", type="disease", name="Type 2 Diabetes")
            G.add_node("Disease2", type="disease", name="Cardiovascular Disease")
            G.add_node("Gene1", type="gene", name="TNF")
            G.add_node("Gene2", type="gene", name="IL6")
            G.add_node("Pathway1", type="pathway", name="Insulin Signaling")
            
            # Add edges
            G.add_edge("Drug1", "Disease1", type="TREATS", confidence=0.9)
            G.add_edge("Drug1", "Gene1", type="INHIBITS", confidence=0.7)
            G.add_edge("Gene1", "Disease2", type="ASSOCIATED_WITH", confidence=0.8)
            G.add_edge("Drug2", "Gene2", type="INHIBITS", confidence=0.85)
            G.add_edge("Gene2", "Disease2", type="ASSOCIATED_WITH", confidence=0.75)
            G.add_edge("Gene1", "Pathway1", type="PART_OF", confidence=0.95)
            G.add_edge("Drug1", "Disease2", type="POTENTIAL", confidence=0.6)
            
            # Create positions
            pos = {
                "Drug1": [0, 0.5],
                "Drug2": [0, -0.5],
                "Disease1": [2, 0.5],
                "Disease2": [2, -0.5],
                "Gene1": [1, 0.3],
                "Gene2": [1, -0.3],
                "Pathway1": [1, 0.8]
            }
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node, position in pos.items():
                node_x.append(position[0])
                node_y.append(position[1])
                
                node_type = G.nodes[node]["type"]
                node_name = G.nodes[node]["name"]
                
                node_text.append(f"{node_name} ({node_type})")
                
                if node_type == "drug":
                    node_color.append("rgba(255, 65, 54, 0.8)")  # Red for drugs
                elif node_type == "disease":
                    node_color.append("rgba(50, 168, 82, 0.8)")  # Green for diseases
                elif node_type == "gene":
                    node_color.append("rgba(66, 135, 245, 0.8)")  # Blue for genes
                else:
                    node_color.append("rgba(178, 102, 255, 0.8)")  # Purple for pathways
            
            # Create edge traces
            edge_traces = []
            
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                edge_type = data["type"]
                confidence = data["confidence"]
                
                # Define color based on type
                if edge_type == "TREATS":
                    color = "rgba(0, 0, 255, 0.6)"  # Blue
                elif edge_type == "POTENTIAL":
                    color = "rgba(255, 165, 0, 0.6)"  # Orange
                elif edge_type == "INHIBITS":
                    color = "rgba(255, 0, 0, 0.6)"  # Red
                elif edge_type == "ASSOCIATED_WITH":
                    color = "rgba(128, 128, 128, 0.6)"  # Gray
                else:
                    color = "rgba(200, 200, 200, 0.6)"  # Light gray
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=1 + 3 * confidence,
                        color=color
                    ),
                    hoverinfo="text",
                    text=f"{edge_type} (Confidence: {confidence})",
                    showlegend=False
                )
                
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=20,
                    color=node_color,
                    line=dict(width=1, color="white")
                ),
                showlegend=False
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace])
            
            fig.update_layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif "6. Mechanistic Explanation Generation" in selected_step:
            st.write("""
            Mechanistic explanation generation applies AI to propose biological mechanisms:
            
            - **Path Analysis**: Analyzes paths in the knowledge graph to identify potential mechanisms of action.
            - **Literature-Based Reasoning**: Uses information extracted from scientific literature to support or refute mechanisms.
            - **Causal Reasoning**: Applies causal inference to generate hypotheses about drug mechanisms.
            
            This step provides interpretable explanations for why a drug might be effective for a disease.
            """)
            
            # Show an example mechanism
            st.subheader("Example Mechanistic Explanation")
            
            st.markdown("""
            **Drug**: Metformin
            
            **Repurposed Disease**: Cancer
            
            **Mechanistic Explanation**:
            
            Metformin may inhibit cancer cell growth through multiple pathways:
            
            1. **AMPK Activation**: Metformin activates AMP-activated protein kinase (AMPK), which inhibits the mammalian target of rapamycin (mTOR) signaling pathway. mTOR is a key regulator of cell growth and proliferation, and its inhibition can suppress cancer cell growth.
            
            2. **Reduced Insulin/IGF-1 Signaling**: Metformin reduces insulin and insulin-like growth factor 1 (IGF-1) levels, which are growth factors that can promote cancer cell proliferation and survival.
            
            3. **Cell Cycle Arrest**: Through AMPK activation, metformin can induce cell cycle arrest by upregulating p53 and p27 expression.
            
            4. **Inflammation Reduction**: Metformin reduces inflammation by inhibiting NF-κB signaling, which can contribute to cancer development and progression.
            
            These mechanisms are supported by multiple lines of evidence from the knowledge graph and scientific literature.
            """)
            
        elif "7. Confidence Score Calculation" in selected_step:
            st.write("""
            Confidence score calculation integrates multiple sources of evidence:
            
            - **Literature Evidence**: Number and quality of scientific articles supporting the repurposing hypothesis.
            - **Network Evidence**: Strength and number of paths connecting the drug and disease in the knowledge graph.
            - **Mechanistic Evidence**: Clarity and plausibility of the proposed mechanisms of action.
            - **Experimental Evidence**: Support from experimental data, such as gene expression or proteomics.
            
            The final score provides a quantitative measure of confidence in the repurposing candidate.
            """)
            
            # Create a visualization of confidence score components
            st.subheader("Confidence Score Components")
            
            # Sample data
            components = ["Literature Evidence", "Network Evidence", "Mechanistic Evidence", "Experimental Evidence", "Overall Score"]
            scores = [0.85, 0.72, 0.91, 0.65, 0.78]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"]
            
            fig = go.Figure()
            
            for i, (component, score, color) in enumerate(zip(components, scores, colors)):
                fig.add_trace(go.Bar(
                    x=[component],
                    y=[score],
                    name=component,
                    marker_color=color,
                    text=[f"{score:.2f}"],
                    textposition="auto"
                ))
            
            fig.update_layout(
                title="Confidence Score Components for Metformin → Cancer",
                yaxis=dict(
                    title="Score",
                    range=[0, 1]
                ),
                height=400,
                barmode="group"
            )
            
            st.plotly_chart(fig)
            
        elif "8. Result Presentation" in selected_step:
            st.write("""
            Result presentation delivers insights to users:
            
            - **Interactive Visualizations**: Knowledge graph visualizations, confidence score charts, etc.
            - **Tabular Data**: Ranked lists of repurposing candidates, evidence summaries, etc.
            - **Detailed Reports**: In-depth analysis of specific repurposing candidates.
            - **API Access**: Programmatic access to all results and underlying data.
            
            This step ensures that insights are actionable and accessible to researchers.
            """)
            
            # Show a sample results dashboard
            st.subheader("Sample Results Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("https://docs.streamlit.io/images/brand/streamlit-mark-color.png", width=100)
                st.markdown("#### Interactive Web Interface")
                st.markdown("""
                - Knowledge graph explorer
                - Repurposing candidate search
                - Detailed analysis views
                - Customizable visualizations
                """)
            
            with col2:
                st.subheader("API Access")
                st.code("""
# Example API request
import requests

response = requests.get(
    "https://api.drugrepurposingengine.com/candidates",
    params={
        "drug": "metformin",
        "min_confidence": 0.7
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

results = response.json()
                """, language="python")
    
    # Add a call-to-action to try the system
    st.success("**Want to explore the Drug Repurposing Engine?** Navigate to the Knowledge Graph page to see the interactive graph, or use the Drug Search page to find repurposing candidates.")

with tab2:
    st.header("Confidence Scoring")
    st.write("Confidence scores are calculated based on multiple factors:")
    
    # Create confidence score components
    components = [
        {
            "name": "Evidence Score",
            "description": "Based on the quantity and quality of supporting evidence",
            "subcomponents": [
                {"name": "Publication Count", "weight": 0.3},
                {"name": "Evidence Quality", "weight": 0.4},
                {"name": "Source Diversity", "weight": 0.3}
            ]
        },
        {
            "name": "Mechanism Score",
            "description": "Based on the clarity and plausibility of the proposed mechanism",
            "subcomponents": [
                {"name": "Pathway Support", "weight": 0.4},
                {"name": "Mechanistic Clarity", "weight": 0.3},
                {"name": "Biological Plausibility", "weight": 0.3}
            ]
        },
        {
            "name": "Novelty Score",
            "description": "Based on the novelty of the repurposing candidate",
            "subcomponents": [
                {"name": "Patent Status", "weight": 0.2},
                {"name": "Clinical Trial Status", "weight": 0.4},
                {"name": "Research Interest", "weight": 0.4}
            ]
        }
    ]
    
    # Display the components
    for component in components:
        st.subheader(component["name"])
        st.write(component["description"])
        
        # Create a table of subcomponents
        subcomponents_df = pd.DataFrame(component["subcomponents"])
        
        # Format the weight column
        subcomponents_df["weight"] = subcomponents_df["weight"].apply(lambda x: f"{x:.1f}")
        
        # Rename columns
        subcomponents_df.columns = ["Subcomponent", "Weight"]
        
        st.dataframe(subcomponents_df, hide_index=True)
    
    st.write("The overall confidence score is a weighted combination of these factors, with values ranging from 0.0 to 1.0.")
    
    # Create a visualization of the scoring formula
    st.subheader("Confidence Score Formula")
    
    formula = r'''
    \begin{align}
    \text{Confidence Score} = 0.4 \times \text{Evidence Score} + 0.4 \times \text{Mechanism Score} + 0.2 \times \text{Novelty Score}
    \end{align}
    '''
    
    st.latex(formula)
    
    # Add a sample confidence distribution
    st.subheader("Sample Confidence Score Distribution")
    
    # Sample data
    import numpy as np
    
    # Generate random scores weighted toward the middle-high range
    np.random.seed(42)  # For reproducibility
    scores = np.clip(np.random.beta(2, 1.5, 100) * 100, 0, 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color="#1f77b4",
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Confidence Scores Across All Repurposing Candidates",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=400
    )
    
    # Add vertical lines for confidence thresholds
    fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="High Confidence")
    fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Medium Confidence")
    fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Low Confidence")
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Knowledge Graph")
    st.write("The knowledge graph represents entities (drugs, diseases, genes, proteins, pathways) as nodes and relationships between them as edges.")
    
    # Create a visualization of entity types
    entity_types = [
        {"name": "Drug", "description": "Pharmaceutical compounds", "color": "rgba(255, 65, 54, 0.7)", "examples": ["Metformin", "Aspirin", "Ibuprofen"]},
        {"name": "Disease", "description": "Medical conditions", "color": "rgba(50, 168, 82, 0.7)", "examples": ["Type 2 Diabetes", "Hypertension", "Cancer"]},
        {"name": "Gene", "description": "Genetic elements", "color": "rgba(66, 135, 245, 0.7)", "examples": ["TNF", "IL6", "EGFR"]},
        {"name": "Protein", "description": "Functional molecules", "color": "rgba(255, 165, 0, 0.7)", "examples": ["TNF-alpha", "Insulin", "p53"]},
        {"name": "Pathway", "description": "Biological processes", "color": "rgba(178, 102, 255, 0.7)", "examples": ["Insulin Signaling", "Inflammation", "Apoptosis"]}
    ]
    
    # Display entity types
    st.subheader("Entity Types (Nodes)")
    
    entity_cols = st.columns(len(entity_types))
    
    for i, entity_type in enumerate(entity_types):
        with entity_cols[i]:
            st.markdown(f"#### {entity_type['name']}")
            st.markdown(f"<div style='width:30px;height:30px;background-color:{entity_type['color']};border-radius:50%;'></div>", unsafe_allow_html=True)
            st.write(entity_type["description"])
            st.write("Examples: " + ", ".join(entity_type["examples"]))
    
    # Create a visualization of relationship types
    relationship_types = [
        {"name": "TREATS", "description": "Drug treats disease", "color": "rgba(0, 0, 255, 0.6)", "example": "Metformin TREATS Type 2 Diabetes"},
        {"name": "CAUSES", "description": "Entity causes disease", "color": "rgba(255, 0, 0, 0.6)", "example": "TNF CAUSES Inflammation"},
        {"name": "ASSOCIATED_WITH", "description": "Entity is associated with disease", "color": "rgba(128, 128, 128, 0.6)", "example": "IL6 ASSOCIATED_WITH Cancer"},
        {"name": "INTERACTS_WITH", "description": "Drug interacts with protein", "color": "rgba(0, 255, 0, 0.6)", "example": "Aspirin INTERACTS_WITH COX-2"},
        {"name": "PART_OF", "description": "Entity is part of pathway", "color": "rgba(255, 165, 0, 0.6)", "example": "EGFR PART_OF MAPK Signaling"},
        {"name": "REGULATES", "description": "Entity regulates another entity", "color": "rgba(0, 255, 255, 0.6)", "example": "p53 REGULATES Apoptosis"}
    ]
    
    # Display relationship types
    st.subheader("Relationship Types (Edges)")
    
    for i, rel_type in enumerate(relationship_types):
        st.markdown(f"""
        <div style='display:flex;align-items:center;margin-bottom:10px;'>
            <div style='width:100px;'>
                <div style='height:3px;background-color:{rel_type['color']};'></div>
            </div>
            <div style='margin-left:10px;'>
                <b>{rel_type['name']}</b>: {rel_type['description']} (e.g., {rel_type['example']})
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a sample subgraph visualization
    st.subheader("Sample Knowledge Graph Subgraph")
    
    # Create a simple graph
    G = nx.DiGraph()
    
    # Add nodes (with more detail than before)
    G.add_node("Drug1", type="drug", name="Metformin")
    G.add_node("Drug2", type="drug", name="Aspirin")
    G.add_node("Disease1", type="disease", name="Type 2 Diabetes")
    G.add_node("Disease2", type="disease", name="Cardiovascular Disease")
    G.add_node("Gene1", type="gene", name="TNF")
    G.add_node("Gene2", type="gene", name="IL6")
    G.add_node("Protein1", type="protein", name="AMPK")
    G.add_node("Protein2", type="protein", name="COX-2")
    G.add_node("Pathway1", type="pathway", name="Insulin Signaling")
    G.add_node("Pathway2", type="pathway", name="Inflammation")
    
    # Add edges (with more types)
    G.add_edge("Drug1", "Disease1", type="TREATS", confidence=0.9)
    G.add_edge("Drug2", "Disease2", type="TREATS", confidence=0.85)
    G.add_edge("Drug1", "Protein1", type="INTERACTS_WITH", confidence=0.8)
    G.add_edge("Drug2", "Protein2", type="INTERACTS_WITH", confidence=0.9)
    G.add_edge("Protein1", "Pathway1", type="PART_OF", confidence=0.95)
    G.add_edge("Protein2", "Pathway2", type="PART_OF", confidence=0.9)
    G.add_edge("Gene1", "Pathway2", type="PART_OF", confidence=0.85)
    G.add_edge("Gene2", "Pathway2", type="PART_OF", confidence=0.8)
    G.add_edge("Pathway2", "Disease2", type="CAUSES", confidence=0.75)
    G.add_edge("Protein1", "Pathway2", type="REGULATES", confidence=0.7)
    G.add_edge("Drug1", "Disease2", type="POTENTIAL", confidence=0.6)
    G.add_edge("Gene1", "Disease2", type="ASSOCIATED_WITH", confidence=0.8)
    G.add_edge("Gene2", "Disease2", type="ASSOCIATED_WITH", confidence=0.75)
    
    # Create positions (force-directed layout)
    pos = nx.spring_layout(G, seed=42)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])
        
        node_type = G.nodes[node]["type"]
        node_name = G.nodes[node]["name"]
        
        node_text.append(f"{node_name} ({node_type})")
        
        if node_type == "drug":
            node_color.append("rgba(255, 65, 54, 0.8)")  # Red for drugs
            node_size.append(15)
        elif node_type == "disease":
            node_color.append("rgba(50, 168, 82, 0.8)")  # Green for diseases
            node_size.append(15)
        elif node_type == "gene":
            node_color.append("rgba(66, 135, 245, 0.8)")  # Blue for genes
            node_size.append(12)
        elif node_type == "protein":
            node_color.append("rgba(255, 165, 0, 0.8)")  # Orange for proteins
            node_size.append(12)
        else:
            node_color.append("rgba(178, 102, 255, 0.8)")  # Purple for pathways
            node_size.append(15)
    
    # Create edge traces
    edge_traces = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_type = data["type"]
        confidence = data["confidence"]
        
        # Define color based on type
        if edge_type == "TREATS":
            color = "rgba(0, 0, 255, 0.6)"  # Blue
        elif edge_type == "POTENTIAL":
            color = "rgba(255, 165, 0, 0.6)"  # Orange
        elif edge_type == "INTERACTS_WITH":
            color = "rgba(0, 255, 0, 0.6)"  # Green
        elif edge_type == "CAUSES":
            color = "rgba(255, 0, 0, 0.6)"  # Red
        elif edge_type == "ASSOCIATED_WITH":
            color = "rgba(128, 128, 128, 0.6)"  # Gray
        elif edge_type == "PART_OF":
            color = "rgba(255, 165, 0, 0.6)"  # Orange
        elif edge_type == "REGULATES":
            color = "rgba(0, 255, 255, 0.6)"  # Cyan
        else:
            color = "rgba(200, 200, 200, 0.6)"  # Light gray
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=1 + 3 * confidence,
                color=color
            ),
            hoverinfo="text",
            text=f"{G.nodes[u]['name']} {edge_type} {G.nodes[v]['name']} (Confidence: {confidence})",
            showlegend=False
        )
        
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white")
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Knowledge Graph Sample (Metformin & Aspirin Subgraph)",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a description of how the knowledge graph is used
    st.subheader("Applications of the Knowledge Graph")
    
    st.write("""
    The knowledge graph enables various types of analysis:
    
    - **Path Finding**: Identifying paths between drugs and diseases to discover potential repurposing opportunities.
    - **Network Analysis**: Analyzing graph properties to identify important nodes and relationships.
    - **Similarity Search**: Finding drugs or diseases with similar network structures.
    - **Causal Inference**: Inferring causal relationships between entities.
    
    These analyses power the drug repurposing recommendations and mechanistic explanations.
    """)
    
with tab4:
    st.header("AI Integration")
    st.write("The Drug Repurposing Engine leverages advanced AI capabilities to enhance analysis and insights.")
    
    # AI integration explanation
    st.markdown("""
    ### OpenAI Integration
    
    The platform integrates with OpenAI's advanced models to provide enhanced analysis capabilities:
    
    - **Enhanced Mechanistic Explanations**: Generates detailed, scientifically-grounded explanations of how drugs might treat diseases
    - **Advanced Confidence Scoring**: Provides more nuanced confidence scores by considering complex biological interactions
    - **Literature Analysis**: Extracts deeper insights from scientific literature by understanding context and relationships
    - **Research Direction Suggestions**: Recommends promising research directions based on current knowledge
    
    The OpenAI integration is completely optional and configurable by the user. When not available, the system falls back to Hugging Face models and then to traditional algorithms to ensure continuous functionality.
    """)
    
    # Show comparison between traditional and AI-enhanced analysis
    st.subheader("Traditional vs. AI-Enhanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Traditional Analysis")
        st.markdown("""
        - Rule-based confidence scoring
        - Template-based mechanistic explanations
        - Keyword-based literature analysis
        - Statistical pattern recognition
        - Fixed scoring weights
        """)
    
    with col2:
        st.markdown("#### AI-Enhanced Analysis")
        st.markdown("""
        - Nuanced, context-aware confidence scoring
        - Detailed mechanistic explanations with citations
        - Semantic understanding of literature
        - Advanced pattern recognition in complex data
        - Dynamic scoring based on evidence quality
        """)
    
    # Show API Key Management
    st.subheader("API Key Management")
    st.markdown("""
    Users can provide their own API keys to enable AI-enhanced functionalities:
    
    1. **OpenAI API Key**: Enables advanced analysis capabilities using GPT-4o models
    2. **Hugging Face API Key**: Provides alternative AI capabilities when OpenAI is not available
    
    Keys are securely stored and never shared. The platform respects usage limits and provides transparent information about API usage.
    
    To configure API keys, visit the Settings page, where you'll find dedicated UI controls for managing your keys.
    """)
    
    # Example of enhanced analysis
    st.subheader("Example Enhanced Analysis")
    
    st.info("""
    **Standard Mechanism Explanation:**
    
    Metformin may treat Alzheimer's disease by improving insulin sensitivity in the brain, reducing inflammation, and inhibiting amyloid formation.
    
    **OpenAI-Enhanced Explanation:**
    
    Metformin shows potential for treating Alzheimer's disease through multiple mechanisms:
    
    1. **Enhanced neuronal insulin signaling**: Metformin activates AMPK, which increases insulin receptor sensitivity in neurons, potentially reversing the brain insulin resistance observed in Alzheimer's disease.
    
    2. **Reduction of neuroinflammation**: By inhibiting NF-κB signaling pathway, metformin decreases the production of pro-inflammatory cytokines (IL-1β, TNF-α, IL-6) by activated microglia, reducing neurotoxic inflammation common in Alzheimer's pathology.
    
    3. **Inhibition of amyloid pathology**: Metformin may reduce amyloid precursor protein (APP) expression and promote non-amyloidogenic APP processing through AMPK activation, resulting in decreased Aβ production.
    
    4. **Protection against tau hyperphosphorylation**: Through inhibition of GSK-3β, metformin may reduce pathological tau phosphorylation, potentially preventing neurofibrillary tangle formation.
    
    5. **Enhanced mitochondrial function**: By improving mitochondrial biogenesis via PGC-1α activation, metformin may address the mitochondrial dysfunction observed in Alzheimer's disease neurons.
    
    These mechanisms are supported by studies including Chen et al. (2021) in Journal of Neurochemistry and Ou et al. (2018) in European Journal of Pharmacology, though further clinical validation is needed.
    """)