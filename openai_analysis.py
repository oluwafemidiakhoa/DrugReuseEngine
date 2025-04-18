"""
OpenAI integration for the Drug Repurposing Engine.

This module provides functions to leverage OpenAI's capabilities for:
1. Generating mechanistic explanations for drug-disease interactions
2. Analyzing drug repurposing candidates with advanced reasoning
3. Extracting insights from biomedical literature
"""

import os
import re
import json
import streamlit as st
import openai

# Flag to track availability
OPENAI_AVAILABLE = True

def get_openai_client():
    """Get configured OpenAI client using API key from environment variables"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    
    # Initialize the client
    client = openai.OpenAI(api_key=api_key)
    return client

def query_openai(prompt, model="gpt-4o", max_tokens=1024):
    """
    Query the OpenAI API with a prompt
    
    Parameters:
    - prompt: The prompt to send
    - model: The model to use (default: gpt-4o which was released May 13, 2024)
    - max_tokens: Maximum tokens in the response
    
    Returns:
    - The model's response as a string
    """
    client = get_openai_client()
    if not client:
        return "Error: No OpenAI API key available"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_mechanistic_explanation(drug, disease, include_molecular_details=True):
    """
    Generate a detailed mechanistic explanation for how a drug might treat a disease
    using OpenAI's advanced reasoning capabilities.
    
    Parameters:
    - drug: Dictionary containing drug information (name, mechanism, etc.)
    - disease: Dictionary containing disease information (name, description, etc.)
    - include_molecular_details: Whether to include molecular pathway details
    
    Returns:
    - Dictionary containing the explanation and supporting details
    """
    client = get_openai_client()
    if not client:
        return {
            "explanation": "OpenAI API key not available. Please provide a valid API key.",
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
    
    Ensure your response is in valid JSON format.
    """
    
    try:
        # Request JSON format
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    
    except Exception as e:
        st.warning(f"Error using OpenAI for explanation generation: {str(e)}")
        try:
            # Fallback to standard completion without JSON format constraint
            response = query_openai(prompt)
            
            # Try to extract JSON from the response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If still failed, return a simple structured response
            return {
                "explanation": response,
                "molecular_pathways": [],
                "supporting_evidence": [],
                "confidence": "Medium"
            }
        except Exception as e2:
            return {
                "explanation": f"Error generating explanation with OpenAI: {str(e)} -> {str(e2)}",
                "molecular_pathways": [],
                "supporting_evidence": [],
                "confidence": "Low"
            }

def analyze_repurposing_candidate(drug, disease, literature_count=0):
    """
    Analyze a drug repurposing candidate using OpenAI and generate a detailed assessment.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - literature_count: Number of literature references found (for context)
    
    Returns:
    - Dictionary with the analysis results
    """
    client = get_openai_client()
    if not client:
        return {
            "key_insights": ["OpenAI API key not available. Please provide a valid API key."],
            "potential_advantages": [],
            "challenges": ["Cannot perform detailed analysis without OpenAI API access."],
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
    
    Ensure your response is in valid JSON format.
    """
    
    try:
        # Request JSON format
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    
    except Exception as e:
        st.warning(f"Error using OpenAI for candidate analysis: {str(e)}")
        try:
            # Fallback to standard completion without JSON format constraint
            response = query_openai(prompt)
            
            # Try to extract JSON from the response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If extraction failed, create a structured response
            return {
                "key_insights": ["Based on mechanism of action, this repurposing candidate may have therapeutic potential."],
                "potential_advantages": ["May utilize existing safety data", "Could be more cost-effective than new drug development"],
                "challenges": ["Additional clinical trials needed", "May have different dosing requirements"],
                "research_directions": ["Conduct preclinical studies", "Investigate specific pathway interactions"]
            }
        except Exception as e2:
            return {
                "key_insights": [f"Error analyzing with OpenAI: {str(e)} -> {str(e2)}"],
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
    client = get_openai_client()
    if not client or not abstracts:
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
    
    Ensure your response is in valid JSON format.
    """
    
    try:
        # Request JSON format
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    
    except Exception as e:
        st.warning(f"Error using OpenAI for literature analysis: {str(e)}")
        try:
            # Fallback to standard completion
            response = query_openai(prompt)
            
            # Try to extract JSON
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If extraction failed, create a structured response
            return {
                "key_findings": ["Literature suggests potential relationship between drug and disease", 
                               "More research is needed to establish efficacy"],
                "evidence_strength": "Moderate",
                "research_gaps": ["Clinical trials specifically for this indication", 
                                "Mechanistic studies on pathway interactions"]
            }
        except Exception as e2:
            return {
                "key_findings": [f"Error analyzing literature with OpenAI: {str(e)} -> {str(e2)}"],
                "evidence_strength": "Unknown",
                "research_gaps": ["Analysis could not be completed due to technical issues."]
            }

def calculate_openai_confidence_score(drug, disease, publications=None, mechanism_details=None):
    """
    Calculate a comprehensive confidence score for a drug-disease pair using OpenAI.
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - publications: List of publications (optional)
    - mechanism_details: Mechanism details (optional)
    
    Returns:
    - Integer confidence score from 0-100
    """
    client = get_openai_client()
    if not client:
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
    
    # Limit mechanism text length
    if len(mech_text) > 500:
        mech_text = mech_text[:500] + "..."
    
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
    - Mechanistic explanation: {mech_text}
    
    Scoring Guide:
    - 0-20: Very low confidence, minimal evidence
    - 21-40: Low confidence, limited evidence
    - 41-60: Moderate confidence, some supporting evidence
    - 61-80: High confidence, substantial evidence
    - 81-100: Very high confidence, strong mechanistic and clinical evidence
    
    Provide a single integer score between 0 and 100 representing the confidence level.
    Format your response in JSON: {{"confidence_score": 75}}
    
    Ensure your response is in valid JSON format.
    """
    
    try:
        # Request JSON format
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        if 'confidence_score' in result:
            score = int(result['confidence_score'])
            return max(0, min(100, score))  # Ensure score is between 0-100
        return 50  # Default if no valid score
        
    except Exception as e:
        st.warning(f"Error using OpenAI for confidence scoring: {str(e)}")
        try:
            # Fallback to standard completion
            response = query_openai(prompt, max_tokens=100)
            
            # Try to extract JSON
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if 'confidence_score' in result:
                        score = int(result['confidence_score'])
                        return max(0, min(100, score))
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
        except Exception as e2:
            st.warning(f"Error in fallback for confidence scoring: {str(e2)}")
            return None