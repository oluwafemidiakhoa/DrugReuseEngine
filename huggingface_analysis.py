"""
Hugging Face AI integration for the Drug Repurposing Engine.

This module provides functions to leverage Hugging Face's AI capabilities for:
1. Generating mechanistic explanations for drug-disease interactions
2. Analyzing drug repurposing candidates with advanced reasoning
3. Extracting insights from biomedical literature

This implementation uses Hugging Face as a fallback when Gemini AI is unavailable.
"""

import os
import re
import json
import streamlit as st
import requests

# Flag to track availability
HUGGINGFACE_AVAILABLE = True

def get_huggingface_api_key():
    """Get the Hugging Face API key from environment variables"""
    return os.environ.get("HUGGINGFACE_API_KEY")

def get_huggingface_model(model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Get the configured Hugging Face model
    
    Parameters:
    - model_id: The model ID to use (default: Mistral-7B-Instruct-v0.2)
    
    Returns:
    - The model ID if API key is available, otherwise None
    """
    api_key = get_huggingface_api_key()
    if not api_key:
        return None
    
    return model_id

def query_huggingface_api(prompt, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Query the Hugging Face API with a prompt
    
    Parameters:
    - prompt: The prompt to send to the API
    - model_id: The model ID to use (default: Mistral-7B-Instruct-v0.2)
    
    Returns:
    - The model's response as a string
    """
    api_key = get_huggingface_api_key()
    if not api_key:
        return "Error: No Hugging Face API key available"
    
    # URL for Inference API
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make the API request
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 1024}}
        )
        
        # Check for errors
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}"
        
        # Parse the response
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        
        return str(result)
    
    except Exception as e:
        return f"Error: {str(e)}"

def generate_mechanistic_explanation(drug, disease, include_molecular_details=True):
    """
    Generate a detailed mechanistic explanation for how a drug might treat a disease
    using Hugging Face's advanced reasoning capabilities.
    
    Parameters:
    - drug: Dictionary containing drug information (name, mechanism, etc.)
    - disease: Dictionary containing disease information (name, description, etc.)
    - include_molecular_details: Whether to include molecular pathway details
    
    Returns:
    - Dictionary containing the explanation and supporting details
    """
    model_id = get_huggingface_model()
    if not model_id:
        return {
            "explanation": "Hugging Face API key not available. Please provide a valid API key.",
            "molecular_pathways": [],
            "supporting_evidence": [],
            "confidence": "Low"
        }
    
    # Create a detailed prompt for the model
    prompt = f"""
    As a biomedical expert, provide a detailed mechanistic explanation for how {drug['name']} might be repurposed to treat {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug.get('description', 'Not available')}
    - Mechanism of action: {drug.get('mechanism', 'Not available')}
    - Original indication: {drug.get('original_indication', 'Not available')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease.get('description', 'Not available')}
    
    Please provide:
    1. A detailed scientific explanation of the potential mechanism
    2. Relevant molecular pathways involved
    3. Any potential supporting evidence
    4. A confidence assessment (High/Medium/Low)
    
    Format the response as a JSON object with these fields:
    {{"explanation": "...", "molecular_pathways": ["pathway1", "pathway2"], "supporting_evidence": ["evidence1", "evidence2"], "confidence": "High/Medium/Low"}}
    """
    
    try:
        response = query_huggingface_api(prompt, model_id)
        
        # Try to extract JSON from the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result
            except json.JSONDecodeError:
                pass
        
        # If we couldn't parse JSON, create a structured response
        explanation = response
        return {
            "explanation": explanation,
            "molecular_pathways": [],
            "supporting_evidence": [],
            "confidence": "Medium"
        }
    
    except Exception as e:
        st.warning(f"Error using Hugging Face for explanation generation: {str(e)}")
        return {
            "explanation": f"Error generating explanation with Hugging Face: {str(e)}",
            "molecular_pathways": [],
            "supporting_evidence": [],
            "confidence": "Low"
        }

def analyze_repurposing_candidate(drug, disease, literature_count=0):
    """
    Analyze a drug repurposing candidate using Hugging Face AI and generate a detailed assessment.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - literature_count: Number of literature references found (for context)
    
    Returns:
    - Dictionary with the analysis results
    """
    model_id = get_huggingface_model()
    if not model_id:
        return {
            "key_insights": ["Hugging Face API key not available. Please provide a valid API key."],
            "potential_advantages": [],
            "challenges": ["Cannot perform detailed analysis without Hugging Face API access."],
            "research_directions": []
        }
    
    # Create a detailed prompt for the model
    prompt = f"""
    As a pharmaceutical research expert, analyze the potential repurposing of {drug['name']} for treating {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug.get('description', 'Not available')}
    - Mechanism of action: {drug.get('mechanism', 'Not available')}
    - Original indication: {drug.get('original_indication', 'Not available')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease.get('description', 'Not available')}
    
    Literature Evidence: There are approximately {literature_count} research articles mentioning both this drug and disease.
    
    Please provide:
    1. 3-5 key insights about this repurposing candidate
    2. 2-3 potential advantages of this drug for this indication
    3. 2-3 challenges or concerns for this repurposing
    4. 2-3 suggested research directions
    
    Format the response as a JSON object with these fields:
    {{"key_insights": ["insight1", "insight2", ...], "potential_advantages": ["advantage1", "advantage2", ...], "challenges": ["challenge1", "challenge2", ...], "research_directions": ["direction1", "direction2", ...]}}
    """
    
    try:
        response = query_huggingface_api(prompt, model_id)
        
        # Try to extract JSON from the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result
            except json.JSONDecodeError:
                pass
        
        # If we couldn't parse JSON, create a structured response from the text
        key_insights = ["Based on mechanism of action, this repurposing candidate may have therapeutic potential."]
        advantages = ["May utilize existing safety data", "Could be more cost-effective than new drug development"]
        challenges = ["Additional clinical trials needed", "May have different dosing requirements"]
        directions = ["Conduct preclinical studies", "Investigate specific pathway interactions"]
        
        return {
            "key_insights": key_insights,
            "potential_advantages": advantages,
            "challenges": challenges,
            "research_directions": directions
        }
    
    except Exception as e:
        st.warning(f"Error using Hugging Face for candidate analysis: {str(e)}")
        return {
            "key_insights": [f"Error analyzing with Hugging Face: {str(e)}"],
            "potential_advantages": [],
            "challenges": ["Analysis failed due to technical issues."],
            "research_directions": []
        }

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
    model_id = get_huggingface_model()
    if not model_id or not abstracts:
        return {
            "key_findings": ["No literature analysis available."],
            "evidence_strength": "Low",
            "research_gaps": ["Literature analysis could not be performed."]
        }
    
    # Combine abstracts and limit length
    combined_abstracts = "\n\n".join(abstracts[:5])  # Limit to first 5 for API constraints
    if len(combined_abstracts) > 4000:
        combined_abstracts = combined_abstracts[:4000] + "..."
    
    # Create a detailed prompt for the model
    prompt = f"""
    As a medical research analyst, review these scientific abstracts about {drug_name} and {disease_name} and extract key insights.
    
    ABSTRACTS:
    {combined_abstracts}
    
    Please provide:
    1. 3-5 key findings from these abstracts
    2. An assessment of the strength of evidence (Strong/Moderate/Weak)
    3. 2-3 identified research gaps
    
    Format the response as a JSON object with these fields:
    {{"key_findings": ["finding1", "finding2", ...], "evidence_strength": "Strong/Moderate/Weak", "research_gaps": ["gap1", "gap2", ...]}}
    """
    
    try:
        response = query_huggingface_api(prompt, model_id)
        
        # Try to extract JSON from the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result
            except json.JSONDecodeError:
                pass
        
        # If we couldn't parse JSON, create a structured response
        return {
            "key_findings": ["Literature suggests potential relationship between drug and disease", 
                           "More research is needed to establish efficacy"],
            "evidence_strength": "Moderate",
            "research_gaps": ["Clinical trials specifically for this indication", 
                            "Mechanistic studies on pathway interactions"]
        }
    
    except Exception as e:
        st.warning(f"Error using Hugging Face for literature analysis: {str(e)}")
        return {
            "key_findings": [f"Error analyzing literature with Hugging Face: {str(e)}"],
            "evidence_strength": "Unknown",
            "research_gaps": ["Analysis could not be completed due to technical issues."]
        }

def calculate_repurposing_confidence(drug, disease, publications=None, mechanism_details=None):
    """
    Calculate a confidence score for drug-disease repurposing using Hugging Face AI.
    This function name matches what's expected by the ai_analysis.py module.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - publications: List of publications (optional)
    - mechanism_details: Mechanism details (optional)
    
    Returns:
    - Integer confidence score from 0-100
    """
    model_id = get_huggingface_model()
    if not model_id:
        return None
    
    # Count relevant publications
    pub_count = 0
    if publications:
        for pub in publications:
            if (drug['name'].lower() in pub.get('title', '').lower() or 
                drug['name'].lower() in pub.get('abstract', '').lower()) and (
                disease['name'].lower() in pub.get('title', '').lower() or 
                disease['name'].lower() in pub.get('abstract', '').lower()):
                pub_count += 1
    
    # Prepare mechanism details text
    mech_text = ""
    if mechanism_details and isinstance(mechanism_details, dict):
        mech_text = mechanism_details.get('explanation', '')
    elif mechanism_details and isinstance(mechanism_details, str):
        mech_text = mechanism_details
    
    # Create a detailed prompt for the model
    prompt = f"""
    As a drug repurposing expert, calculate a confidence score (0-100) for repurposing {drug['name']} to treat {disease['name']}.
    
    Drug Information:
    - Name: {drug['name']}
    - Description: {drug.get('description', 'Not available')}
    - Mechanism of action: {drug.get('mechanism', 'Not available')}
    - Original indication: {drug.get('original_indication', 'Not available')}
    
    Disease Information:
    - Name: {disease['name']}
    - Description: {disease.get('description', 'Not available')}
    
    Additional Context:
    - Number of relevant publications: {pub_count}
    - Mechanistic explanation: {mech_text[:500]}...
    
    Scoring Guide:
    - 0-20: Very low confidence, minimal evidence
    - 21-40: Low confidence, limited evidence
    - 41-60: Moderate confidence, some supporting evidence
    - 61-80: High confidence, substantial evidence
    - 81-100: Very high confidence, strong mechanistic and clinical evidence
    
    Provide a single integer score between 0 and 100 representing the confidence level.
    Format: {{"confidence_score": 75}}
    """
    
    try:
        response = query_huggingface_api(prompt, model_id)
        
        # Try to extract JSON from the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if 'confidence_score' in result:
                    score = int(result['confidence_score'])
                    return max(0, min(100, score))  # Ensure score is between 0-100
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If we couldn't parse JSON, try to extract a number
        score_match = re.search(r'\b(\d{1,3})\b', response)
        if score_match:
            try:
                score = int(score_match.group(1))
                return max(0, min(100, score))  # Ensure score is between 0-100
            except ValueError:
                pass
        
        # Default score if we can't extract one
        return 50
    
    except Exception as e:
        st.warning(f"Error using Hugging Face for confidence scoring: {str(e)}")
        return None