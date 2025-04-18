"""
Tooltip Helper Module for the Drug Repurposing Engine

This module provides contextual help tooltips with engaging and user-friendly explanations
for complex biomedical concepts in the application.
"""

import streamlit as st

# Dictionary of tooltip explanations for various concepts
# Format: concept_key: {"title": "Concept Name", "text": "Explanation text"}
TOOLTIPS = {
    # General concepts
    "drug_repurposing": {
        "title": "Drug Repurposing",
        "text": "Finding new uses for existing medications - like discovering that a heart medicine might also help with diabetes. It's like finding out your kitchen scissors are also great for opening packages! 🔄💊"
    },
    "confidence_score": {
        "title": "Confidence Score",
        "text": "A number (0-100) showing how certain we are about a drug's potential new use. Think of it as our 'excitement meter' - higher numbers mean we're more excited about the possibility! 📊"
    },
    
    # Drug-related tooltips
    "mechanism_of_action": {
        "title": "Mechanism of Action",
        "text": "How a drug works in your body at the molecular level. It's like knowing exactly how your car engine runs, not just that pressing the gas pedal makes it go! 🔬"
    },
    "drug_target": {
        "title": "Drug Target",
        "text": "The specific molecule in your body that a drug attaches to or affects. It's like a lock that the drug (the key) fits into perfectly. 🎯🔑"
    },
    "bioavailability": {
        "title": "Bioavailability",
        "text": "The percentage of a drug that actually reaches its target in your body. Not all medicine you take makes it to where it needs to go - like sending 10 letters but only 7 reaching their destination! 📬"
    },
    
    # Disease-related tooltips
    "indication": {
        "title": "Indication",
        "text": "A disease or condition that a drug is officially approved to treat. It's the 'You Are Here' marker on the drug's map of uses! 🗺️"
    },
    "comorbidity": {
        "title": "Comorbidity",
        "text": "When a person has two or more diseases at the same time. It's like trying to fix multiple home appliances that broke on the same day - complicated! 🏥"
    },
    
    # Biological process tooltips
    "gene_expression": {
        "title": "Gene Expression",
        "text": "When your genes become activated to produce proteins. Think of genes as recipes in a cookbook - gene expression is when the chef actually decides to make that recipe! 👨‍🍳📖"
    },
    "pathway": {
        "title": "Biological Pathway",
        "text": "A series of chemical reactions in your cells that work together like an assembly line. One molecule gets changed, which affects another, then another, creating a chain reaction. 🔄⚙️"
    },
    
    # Data science tooltips
    "knowledge_graph": {
        "title": "Knowledge Graph",
        "text": "A network showing connections between drugs, diseases, genes, and more. Imagine a giant web where everything that's related is connected by strings - that's our knowledge graph! 🕸️"
    },
    "ai_model": {
        "title": "AI Model",
        "text": "A computer program that learns patterns from data to make predictions about drug repurposing. Like a student who studies thousands of successful recipes to predict which ingredient substitutions will taste good! 🤖"
    },
    
    # External data sources tooltips
    "chembl": {
        "title": "ChEMBL Database",
        "text": "A huge public database containing information about drug-like molecules and their activities. It's like a massive library specifically for chemical compounds! 📚🧪"
    },
    "openfda": {
        "title": "OpenFDA",
        "text": "An open-source platform providing access to FDA data about drugs, including adverse events and recalls. It's like having access to the FDA's filing cabinets! 🏛️📁"
    },
    "pubmed": {
        "title": "PubMed",
        "text": "A free search engine for biomedical literature. If scientists have written about it, you'll probably find it here - like Google, but specifically for medical research! 🔍📄"
    },
    
    # Technical terms
    "smiles": {
        "title": "SMILES Notation",
        "text": "A way to represent chemical structures using text strings. It's like writing down directions instead of drawing a map - more compact but requires translation! 🧬📝"
    },
    "adverse_event": {
        "title": "Adverse Event",
        "text": "An unwanted side effect from a medication. The unexpected plot twist in your treatment story that nobody wanted! 😧💊"
    },
    
    # Items from the main interface
    "knowledge_graph_construction": {
        "title": "Knowledge Graph Construction",
        "text": "Building connections between drugs, diseases, and genes like a detective connecting evidence with pins and string on a board. We're making a map of how everything in medicine relates to each other! 🕸️🔍"
    },
    "biological_pathway_analysis": {
        "title": "Biological Pathway Analysis",
        "text": "Studying the step-by-step chemical reactions in cells to understand how drugs affect your body. Like tracking a package's journey through multiple shipping facilities to understand delivery delays! 🧬🔄"
    },
    "confidence_scoring": {
        "title": "Confidence Scoring for Repurposing",
        "text": "Rating how likely a drug is to treat a new disease on a scale of 0-100. Just like movie ratings help you decide what to watch, our confidence scores help scientists decide which drugs to test next! ⭐📊"
    },
    "scientific_visualization": {
        "title": "Scientific Visualization",
        "text": "Creating publication-quality charts and graphs that help scientists understand complex data at a glance. Like turning a spreadsheet of numbers into a beautiful picture that tells a story! 📊🖼️"
    }
}

def get_tooltip(key):
    """
    Get tooltip content for a specific key.
    
    Parameters:
    - key: The dictionary key for the tooltip content
    
    Returns:
    - Dictionary with title and text, or None if key not found
    """
    return TOOLTIPS.get(key)

def render_tooltip(key, icon="ℹ️"):
    """
    Render a tooltip using Streamlit.
    
    Parameters:
    - key: The dictionary key for the tooltip content
    - icon: The icon to display for the tooltip (default: ℹ️)
    
    Returns:
    - The tooltip HTML (or plain text fallback if key not found)
    """
    tooltip_data = get_tooltip(key)
    
    if tooltip_data:
        return st.markdown(f"""
        <span class="tooltip">
            {icon}
            <span class="tooltiptext">
                <strong>{tooltip_data['title']}</strong><br>
                {tooltip_data['text']}
            </span>
        </span>
        """, unsafe_allow_html=True)
    else:
        return st.text(f"{icon} No tooltip available for '{key}'")

def setup_tooltip_css():
    """
    Add the required CSS for tooltips to the Streamlit page.
    Call this function once at the top of your app.
    """
    st.markdown("""
    <style>
    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #f1f9ff;
        color: #0a4069;
        text-align: left;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #4a9ce0;
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        
        /* Position the tooltip text */
        position: absolute;
        z-index: 1;
        top: -5px;
        left: 105%;
        
        /* Fade in transition */
        opacity: 0;
        transition: opacity 0.3s;
    }

    /* Show the tooltip on hover */
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Add arrow to tooltip */
    .tooltip .tooltiptext::after {
        content: " ";
        position: absolute;
        top: 15px;
        right: 100%; /* To the left of the tooltip */
        margin-top: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: transparent #4a9ce0 transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)

def tooltip_header(title, tooltip_key, level=2):
    """
    Create a header with an attached tooltip.
    
    Parameters:
    - title: The header text
    - tooltip_key: The key for the tooltip content
    - level: Header level (1-6)
    
    Returns:
    - Renders header with tooltip
    """
    tooltip_data = get_tooltip(tooltip_key)
    
    if tooltip_data:
        st.markdown(f"""
        <h{level}>
            {title} 
            <span class="tooltip">
                ℹ️
                <span class="tooltiptext">
                    <strong>{tooltip_data['title']}</strong><br>
                    {tooltip_data['text']}
                </span>
            </span>
        </h{level}>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<h{level}>{title}</h{level}>", unsafe_allow_html=True)