
import streamlit as st
import plotly.graph_objects as go
import sys
from utils import initialize_session_state
sys.path.append('.')
from ai_analysis import analyze_repurposing_candidate
from db_utils import get_drug_by_name, get_disease_by_name

st.set_page_config(page_title="Mechanism Explorer", page_icon="🔬", layout="wide")
initialize_session_state()

st.title("Interactive Mechanism Explorer")

def create_mechanism_diagram(analysis_report):
    """
    Creates a mechanism diagram visualization based on the analysis report
    
    Parameters:
    - analysis_report: The report returned by analyze_repurposing_candidate function
    
    Returns:
    - Plotly figure object with the visualization
    """
    # Create entities and interactions based on the mechanism description
    drug_name = analysis_report.get('drug', '')
    disease_name = analysis_report.get('disease', '')
    mechanism = analysis_report.get('mechanism', 'Unknown mechanism')
    
    # Create a simple diagram with the drug, disease, and pathway
    entities = [
        {'name': drug_name, 'x': 1, 'y': 2, 'color': '#4287f5'},  # Blue for drug
        {'name': 'Molecular\nTarget', 'x': 3, 'y': 2, 'color': '#42f5b3'},  # Green for target
        {'name': disease_name, 'x': 5, 'y': 2, 'color': '#f54242'}  # Red for disease
    ]
    
    interactions = [
        {'start_x': 1, 'start_y': 2, 'end_x': 3, 'end_y': 2, 'type': 'Targets'},
        {'start_x': 3, 'start_y': 2, 'end_x': 5, 'end_y': 2, 'type': 'Affects'}
    ]
    
    # Create the figure
    fig = go.Figure()
    
    # Add nodes for biological entities
    for entity in entities:
        fig.add_trace(go.Scatter(
            x=[entity['x']], y=[entity['y']],
            mode='markers+text',
            name=entity['name'],
            text=[entity['name']],
            marker=dict(size=30, color=entity['color'])
        ))
    
    # Add arrows for interactions
    for interaction in interactions:
        fig.add_annotation(
            x=interaction['start_x'], y=interaction['start_y'],
            xref="x", yref="y",
            axref="x", ayref="y",
            ax=interaction['end_x'], ay=interaction['end_y'],
            text=interaction['type'],
            showarrow=True,
            arrowhead=2
        )
    
    # Update layout
    fig.update_layout(
        title=f"Mechanism of action: {drug_name} for {disease_name}",
        autosize=True,
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    
    return fig

# Main interface
st.markdown("""
<div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
    <p>The Mechanism Explorer helps you understand how drugs might work for treating diseases through visual interactive diagrams.</p>
    <p>Select a drug and disease combination, then click "Explore Mechanism" to generate a visualization of the potential mechanism of action.</p>
    <p><a href="/3D_Mechanism_Viewer" target="_self" style="color: #1E88E5; font-weight: bold;">Try our new 3D Mechanism Viewer for stunning animated visualizations!</a></p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_drug = st.selectbox("Select Drug", options=[d['name'] for d in st.session_state.drugs])
with col2:
    selected_disease = st.selectbox("Select Disease", options=[d['name'] for d in st.session_state.diseases])

if st.button("Explore Mechanism", type="primary"):
    with st.spinner("Analyzing mechanism of action..."):
        # Get the drug and disease objects
        drug = get_drug_by_name(selected_drug)
        disease = get_disease_by_name(selected_disease)
        
        # Generate the analysis report
        analysis_report = analyze_repurposing_candidate(drug, disease)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Mechanism Diagram", "Detailed Analysis"])
        
        with tab1:
            # Display the diagram
            st.plotly_chart(create_mechanism_diagram(analysis_report), use_container_width=True)
        
        with tab2:
            # Display detailed analysis
            st.markdown(f"## Analysis for {drug['name']} in {disease['name']}")
            
            # Confidence score with colored badge
            confidence = analysis_report.get('confidence_score', 0)
            confidence_color = "red"
            if confidence > 70:
                confidence_color = "green"
            elif confidence > 40:
                confidence_color = "orange"
                
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-weight: bold; margin-right: 10px;">Confidence Score:</div>
                <div style="background-color: {confidence_color}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold;">
                    {confidence}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mechanism of action
            st.markdown("### Mechanism of Action")
            st.write(analysis_report.get('mechanism', 'No mechanism data available'))
            
            # Key insights if available
            if 'key_insights' in analysis_report:
                st.markdown("### Key Insights")
                insights = analysis_report['key_insights']
                if isinstance(insights, list):
                    for insight in insights:
                        st.markdown(f"• {insight}")
                else:
                    st.write(insights)
            
            # Potential advantages if available
            if 'potential_advantages' in analysis_report:
                st.markdown("### Potential Advantages")
                advantages = analysis_report['potential_advantages']
                if isinstance(advantages, list):
                    for advantage in advantages:
                        st.markdown(f"• {advantage}")
                else:
                    st.write(advantages)
            
            # Challenges if available
            if 'challenges' in analysis_report:
                st.markdown("### Challenges")
                challenges = analysis_report['challenges']
                if isinstance(challenges, list):
                    for challenge in challenges:
                        st.markdown(f"• {challenge}")
                else:
                    st.write(challenges)
            
            # Research directions if available
            if 'research_directions' in analysis_report:
                st.markdown("### Future Research Directions")
                directions = analysis_report['research_directions']
                if isinstance(directions, list):
                    for direction in directions:
                        st.markdown(f"• {direction}")
                else:
                    st.write(directions)
            
            # Evidence count
            if 'evidence_count' in analysis_report:
                st.info(f"This analysis is based on {analysis_report['evidence_count']} pieces of evidence from the literature.")
            
            # AI enhanced badge
            if analysis_report.get('ai_enhanced', False):
                st.markdown("""
                <div style="display: inline-block; background-color: #8B5CF6; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; margin-top: 20px;">
                    <span style="margin-right: 5px;">🧠</span> AI-Enhanced Analysis
                </div>
                """, unsafe_allow_html=True)
