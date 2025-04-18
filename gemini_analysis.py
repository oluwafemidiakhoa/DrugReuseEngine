"""
Gemini AI integration for the Drug Repurposing Engine.

This module provides functions to leverage Google's Gemini AI capabilities for:
1. Generating mechanistic explanations for drug-disease interactions
2. Analyzing drug repurposing candidates with advanced reasoning
3. Extracting insights from biomedical literature
"""

import os
import re
import json
import streamlit as st

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def get_gemini_model():
    """Get the configured Gemini model if available"""
    if not GEMINI_AVAILABLE:
        return None
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Use the newest model available (Gemini 1.5 Pro)
    # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
    return genai.GenerativeModel('gemini-pro')

def generate_mechanistic_explanation(drug, disease, include_molecular_details=True):
    """
    Generate a detailed mechanistic explanation for how a drug might treat a disease
    using Gemini's advanced reasoning capabilities.
    
    Parameters:
    - drug: Dictionary containing drug information (name, mechanism, etc.)
    - disease: Dictionary containing disease information (name, description, etc.)
    - include_molecular_details: Whether to include molecular pathway details
    
    Returns:
    - Dictionary containing the explanation and supporting details
    """
    model = get_gemini_model()
    if not model:
        return None
    
    # Create a prompt that includes all relevant information
    prompt = f"""
    As a biomedical expert, generate a detailed mechanistic explanation for how {drug['name']} might be
    repurposed to treat {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug['description']}
    - Mechanism of action: {drug.get('mechanism', 'Unknown')}
    - Original indication: {drug.get('original_indication', 'Unknown')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease['description']}
    - Category: {disease.get('category', 'Unknown')}
    
    {"Include molecular pathway details and potential biological targets." if include_molecular_details else "Focus on high-level mechanisms without detailed molecular pathways."}
    
    Format your response as a clear, scientifically plausible explanation that connects the drug's mechanisms to
    the disease pathophysiology. Cite specific pathways or targets where appropriate.
    
    Only include information that is scientifically plausible based on known mechanisms, not speculative claims.
    """
    
    # Get the response from Gemini
    try:
        response = model.generate_content(prompt)
        explanation = response.text.strip()
        
        # Extract the main explanation while removing any disclaimers or meta-text
        explanation = re.sub(r'^(Note|Disclaimer|As a|I will).*?\n', '', explanation, flags=re.IGNORECASE | re.MULTILINE)
        explanation = explanation.strip()
        
        return {
            "explanation": explanation,
            "includes_molecular_details": include_molecular_details
        }
    except Exception as e:
        st.error(f"Error generating explanation with Gemini: {str(e)}")
        return None

def analyze_repurposing_candidate(drug, disease, literature_count=0):
    """
    Analyze a drug repurposing candidate using Gemini AI and generate a detailed assessment.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - literature_count: Number of literature references found (for context)
    
    Returns:
    - Dictionary with the analysis results
    """
    model = get_gemini_model()
    if not model:
        return None
    
    # Create a prompt that includes all relevant information
    prompt = f"""
    As a biomedical expert specializing in drug repurposing, analyze the potential of repurposing
    {drug['name']} for treating {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug['description']}
    - Mechanism of action: {drug.get('mechanism', 'Unknown')}
    - Original indication: {drug.get('original_indication', 'Unknown')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease['description']}
    - Category: {disease.get('category', 'Unknown')}
    
    Context: There are approximately {literature_count} published articles that mention both this drug and disease.
    
    Provide a comprehensive analysis including:
    1. Key insights about the potential repurposing opportunity (2-4 bullet points)
    2. Potential advantages of using this drug for this indication (2-3 bullet points)
    3. Challenges and limitations (2-3 bullet points)
    4. Suggested future research directions (2-3 bullet points)
    
    Format your analysis as JSON with the following structure:
    {{
      "key_insights": ["Insight 1", "Insight 2", ...],
      "potential_advantages": ["Advantage 1", "Advantage 2", ...],
      "challenges": ["Challenge 1", "Challenge 2", ...],
      "research_directions": ["Direction 1", "Direction 2", ...]
    }}
    
    Ensure all points are scientifically sound and based on the known mechanisms of the drug and pathophysiology of the disease.
    """
    
    # Get the response from Gemini
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        
        # Extract JSON part from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
        if json_match:
            analysis_json = json_match.group(1)
        else:
            # If no JSON block, try to parse the whole response
            analysis_json = analysis_text
        
        # Parse the JSON
        result = json.loads(analysis_json)
        return result
        
    except Exception as e:
        st.error(f"Error analyzing candidate with Gemini: {str(e)}")
        return None

def extract_insights_from_literature(abstracts, drug_name, disease_name):
    """
    Extract key insights from scientific literature abstracts related to a drug-disease pair.
    
    Parameters:
    - abstracts: List of text abstracts from scientific literature
    - drug_name: Name of the drug
    - disease_name: Name of the disease
    
    Returns:
    - Dictionary with key insights and evidence
    """
    model = get_gemini_model()
    if not model or not abstracts:
        return None
    
    # Prepare the abstracts text (limit to first 5 for context length)
    abstracts_text = "\n\n".join([f"Abstract {i+1}: {abstract}" for i, abstract in enumerate(abstracts[:5])])
    
    # Create a prompt
    prompt = f"""
    As a biomedical research expert, analyze the following scientific abstracts related to 
    the use of {drug_name} for treating {disease_name}.
    
    ABSTRACTS:
    {abstracts_text}
    
    Extract and summarize:
    1. Key findings about the relationship between {drug_name} and {disease_name}
    2. Evidence supporting potential efficacy
    3. Reported mechanisms of action
    4. Safety considerations or adverse effects
    5. Research gaps or limitations
    
    Format your analysis as JSON with the following structure:
    {{
      "key_findings": ["Finding 1", "Finding 2", ...],
      "efficacy_evidence": ["Evidence 1", "Evidence 2", ...],
      "mechanisms": ["Mechanism 1", "Mechanism 2", ...],
      "safety_considerations": ["Consideration 1", "Consideration 2", ...],
      "research_gaps": ["Gap 1", "Gap 2", ...]
    }}
    
    Ensure all points are directly supported by information in the abstracts.
    """
    
    # Get the response from Gemini
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        
        # Extract JSON part from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
        if json_match:
            analysis_json = json_match.group(1)
        else:
            # If no JSON block, try to parse the whole response
            analysis_json = analysis_text
        
        # Parse the JSON
        result = json.loads(analysis_json)
        result["abstract_count"] = len(abstracts)
        return result
        
    except Exception as e:
        st.error(f"Error extracting insights with Gemini: {str(e)}")
        return None

def calculate_gemini_confidence_score(drug, disease, publications=None, mechanism_details=None):
    """
    Calculate a comprehensive confidence score for a drug-disease pair using Gemini AI.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - publications: List of publications (optional)
    - mechanism_details: Mechanism details (optional)
    
    Returns:
    - Integer confidence score from 0-100
    """
    model = get_gemini_model()
    if not model:
        return None
    
    # Count publications mentioning both drug and disease
    publication_count = 0
    if publications:
        for pub in publications:
            if (drug['name'].lower() in pub.get('title', '').lower() or 
                drug['name'].lower() in pub.get('abstract', '').lower()) and (
                disease['name'].lower() in pub.get('title', '').lower() or 
                disease['name'].lower() in pub.get('abstract', '').lower()):
                publication_count += 1
    
    # Create a prompt
    prompt = f"""
    As a biomedical expert specializing in drug repurposing, calculate a confidence score (0-100) 
    for repurposing {drug['name']} to treat {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug['description']}
    - Mechanism of action: {drug.get('mechanism', 'Unknown')}
    - Original indication: {drug.get('original_indication', 'Unknown')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease['description']}
    - Category: {disease.get('category', 'Unknown')}
    
    Context:
    - Number of relevant publications: {publication_count}
    - Mechanistic details: {mechanism_details['explanation'] if mechanism_details and 'explanation' in mechanism_details else 'Not available'}
    
    Consider the following factors in your assessment:
    1. Strength of mechanistic plausibility
    2. Evidence from literature
    3. Similarity between original indication and target disease
    4. Known side effects and safety profile
    5. Clinical investigation status
    
    Provide your final confidence score (0-100) as a single number, where:
    - 0-20: Minimal evidence or significant contraindications
    - 21-40: Early research stage with limited evidence
    - 41-60: Promising theoretical basis needing further investigation
    - 61-80: Strong evidence from multiple sources
    - 81-100: Overwhelming evidence with clinical validation
    
    Your response should be only a single integer between 0 and 100.
    """
    
    # Get the response from Gemini
    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
        # Extract just the numeric score
        score_match = re.search(r'(\d+)', score_text)
        if score_match:
            score = int(score_match.group(1))
            # Ensure the score is in range
            score = max(0, min(100, score))
            return score
        else:
            # Default fallback score if parsing fails
            return 50
            
    except Exception as e:
        st.error(f"Error calculating confidence score with Gemini: {str(e)}")
        return None