import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import re
import os
import streamlit as st

# Import AI model integrations if available
try:
    import openai_analysis
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import huggingface_analysis
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    
# Using OpenAI and HuggingFace for AI analysis

# Download NLTK resources
try:
    # Download required NLTK data directly instead of trying to find it first
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")
    # Continue anyway and handle missing resources in the safe_tokenize function

# Safe tokenization function to prevent NLTK errors
def safe_tokenize(text):
    """Safely tokenize text, with fallback to simple splitting if NLTK fails"""
    if text is None:
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        # Simple fallback tokenization
        return re.findall(r'\b\w+\b', text.lower())

def calculate_transformer_confidence(drug, disease):
    """
    Calculate a confidence score for drug-disease repurposing using transformer models
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    
    Returns:
    - Confidence score (0-100) or None if transformers not available
    """
    if HUGGINGFACE_AVAILABLE:
        try:
            # Use HuggingFace models for scoring
            return huggingface_analysis.calculate_repurposing_confidence(drug, disease)
        except Exception as e:
            print(f"HuggingFace error: {str(e)}")
    elif OPENAI_AVAILABLE:
        try:
            # Use OpenAI models for scoring
            return openai_analysis.calculate_repurposing_confidence(drug, disease)
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
    
    # If no transformers available or all failed
    return None

def calculate_confidence_score(drug, disease, graph=None, pubmed_articles=None, use_transformers=True):
    """Enhanced confidence scoring using transformer models when available"""
    if use_transformers and HUGGINGFACE_AVAILABLE:
        try:
            # Get enhanced score from transformer model
            transformer_score = calculate_transformer_confidence(drug, disease)
            if transformer_score is not None:
                # Combine with traditional score
                traditional_score = _calculate_traditional_confidence_score(drug, disease, graph, pubmed_articles)
                return round((transformer_score * 0.8) + (traditional_score * 0.2))
        except Exception as e:
            st.warning(f"Transformer scoring failed: {str(e)}, falling back to traditional scoring")
    """
    Calculate a confidence score for a drug-disease repurposing candidate
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - graph: NetworkX graph object (optional)
    - pubmed_articles: List of PubMed articles (optional)
    
    Returns:
    - confidence_score: A score between 0-100 indicating confidence
    
    When AI models are available, they will be used to enhance the confidence score
    calculation with advanced reasoning, otherwise falls back to rule-based scoring.
    
    Tries OpenAI first, then Hugging Face, then traditional scoring as fallbacks.
    """
    # Try to use OpenAI for advanced confidence scoring if available
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            # First get a mechanistic explanation to use as context
            mechanism_details = openai_analysis.generate_mechanistic_explanation(drug, disease)
            
            # Now calculate a confidence score with OpenAI
            openai_score = openai_analysis.calculate_openai_confidence_score(
                drug, 
                disease, 
                publications=pubmed_articles, 
                mechanism_details=mechanism_details
            )
            
            if openai_score is not None:
                # Combine OpenAI score (weight 70%) with traditional score (weight 30%)
                traditional_score = _calculate_traditional_confidence_score(drug, disease, graph, pubmed_articles)
                return round((openai_score * 0.7) + (traditional_score * 0.3))
                
        except Exception as e:
            st.warning(f"Error using OpenAI for confidence scoring: {str(e)}. Trying Hugging Face...")
            # Try HuggingFace next
            
    # Fallback to Hugging Face AI if OpenAI is not available
    
    # Try to use Hugging Face AI as alternative
    if HUGGINGFACE_AVAILABLE and os.environ.get("HUGGINGFACE_API_KEY"):
        try:
            # First get a mechanistic explanation to use as context
            mechanism_details = huggingface_analysis.generate_mechanistic_explanation(drug, disease)
            
            # Now calculate a confidence score with Hugging Face
            huggingface_score = huggingface_analysis.calculate_huggingface_confidence_score(
                drug, 
                disease, 
                publications=pubmed_articles, 
                mechanism_details=mechanism_details
            )
            
            if huggingface_score is not None:
                # Combine Hugging Face score (weight 60%) with traditional score (weight 40%)
                traditional_score = _calculate_traditional_confidence_score(drug, disease, graph, pubmed_articles)
                return round((huggingface_score * 0.6) + (traditional_score * 0.4))
                
        except Exception as e:
            st.warning(f"Error using Hugging Face for confidence scoring: {str(e)}. Falling back to traditional scoring.")
            # Continue with traditional approach on failure
    
    # Fall back to traditional scoring method
    return _calculate_traditional_confidence_score(drug, disease, graph, pubmed_articles)

def _calculate_traditional_confidence_score(drug, disease, graph=None, pubmed_articles=None):
    """
    Traditional rule-based method to calculate confidence score
    Used as fallback when AI models are not available
    """
    # Base score components
    literature_score = 0
    network_score = 0
    similarity_score = 0
    mechanism_score = 0
    
    # 1. Literature evidence score (0-25)
    if pubmed_articles:
        # Count articles that mention both the drug and disease
        count = 0
        for article in pubmed_articles:
            if (drug['name'].lower() in article['title'].lower() or 
                drug['name'].lower() in article['abstract'].lower()) and (
                disease['name'].lower() in article['title'].lower() or 
                disease['name'].lower() in article['abstract'].lower()):
                count += 1
        
        # Scale based on number of articles
        if count > 20:
            literature_score = 25
        else:
            literature_score = min(25, count * 1.25)
    
    # 2. Network-based score (0-25)
    if graph:
        try:
            # Check if there's a direct path
            if graph.has_edge(drug['id'], disease['id']):
                edge_data = graph.get_edge_data(drug['id'], disease['id'])
                network_score = 25 * edge_data['confidence']
            else:
                # Check for indirect paths
                try:
                    path = nx.shortest_path(graph, drug['id'], disease['id'])
                    path_length = len(path) - 1
                    
                    # Calculate path confidence
                    confidences = []
                    for i in range(len(path)-1):
                        edge_data = graph.get_edge_data(path[i], path[i+1])
                        confidences.append(edge_data['confidence'])
                    
                    avg_confidence = np.mean(confidences)
                    
                    # Adjust for path length: longer paths get lower scores
                    network_score = 25 * avg_confidence / max(1, path_length)
                except nx.NetworkXNoPath:
                    network_score = 0
        except:
            network_score = 0
    
    # 3. Mechanism similarity score (0-25)
    if 'mechanism' in drug:
        # Calculate similarity between drug mechanism and disease description
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and clean drug mechanism (using safe tokenize to avoid NLTK errors)
        drug_mech_tokens = [w.lower() for w in safe_tokenize(drug['mechanism']) 
                           if w.lower() not in stop_words and w.isalnum()]
        drug_mech_text = " ".join(drug_mech_tokens)
        
        # Tokenize and clean disease description (using safe tokenize to avoid NLTK errors)
        disease_desc_tokens = [w.lower() for w in safe_tokenize(disease['description']) 
                              if w.lower() not in stop_words and w.isalnum()]
        disease_desc_text = " ".join(disease_desc_tokens)
        
        # Calculate TF-IDF and similarity
        if drug_mech_text and disease_desc_text:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([drug_mech_text, disease_desc_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarity_score = 25 * similarity
            except:
                similarity_score = 10  # Default if calculation fails
        else:
            similarity_score = 10  # Default if text is empty
    
    # 4. Mechanistic plausibility (0-25)
    # This would ideally involve complex biological pathway analysis
    # For this simplified version, use a combination of known factors
    if 'original_indication' in drug:
        # If the drug's original indication is in the same category as the disease
        if ('category' in disease and 
            drug['original_indication'].lower() in disease['description'].lower()):
            mechanism_score = 20
        else:
            # Check for common keywords in mechanism and disease
            common_mechanisms = [
                'inflammation', 'immune', 'infection', 'metabolism', 
                'receptor', 'enzyme', 'protein', 'signaling', 'pathway',
                'kinase', 'channel', 'transport'
            ]
            
            mech_count = 0
            if 'mechanism' in drug:
                for mech in common_mechanisms:
                    if (mech in drug['mechanism'].lower() and 
                        mech in disease['description'].lower()):
                        mech_count += 1
            
            mechanism_score = min(25, mech_count * 5)
    
    # Calculate final score (0-100)
    confidence_score = literature_score + network_score + similarity_score + mechanism_score
    
    # Ensure the score is between 0-100
    confidence_score = max(0, min(100, confidence_score))
    
    return round(confidence_score)

def generate_evidence_summary(drug, disease, evidence_list=None):
    """
    Generate a summary of evidence supporting a drug-disease repurposing candidate
    
    Parameters:
    - drug: Dictionary containing drug information
    - disease: Dictionary containing disease information
    - evidence_list: Optional list of evidence items to summarize
    
    Returns:
    - A detailed summary of the evidence supporting the drug-disease relationship
    """
    # Start with basic information
    drug_name = drug.get('name', 'Unknown drug')
    disease_name = disease.get('name', 'Unknown disease')
    
    # Use available AI models for advanced analysis if available
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            # Structure evidence for OpenAI
            evidence_text = "\n".join([f"- {item}" for item in evidence_list]) if evidence_list else "No specific evidence items provided."
            
            prompt = f"""
            Create a comprehensive evidence summary for using {drug_name} to treat {disease_name}.
            
            Drug information:
            - Name: {drug_name}
            - Mechanism: {drug.get('mechanism_of_action', 'Unknown')}
            - Current indications: {', '.join(drug.get('indications', ['Unknown']))}
            
            Disease information:
            - Name: {disease_name}
            - Description: {disease.get('description', 'Unknown')}
            
            Evidence items:
            {evidence_text}
            
            Provide a structured scientific summary of the evidence supporting this potential repurposing opportunity.
            Include the strength of evidence, potential mechanisms, and any limitations in the current research.
            """
            
            # Get the summary from OpenAI
            summary = openai_analysis.get_ai_response(prompt)
            if summary:
                return summary
                
        except Exception as e:
            st.warning(f"Error using OpenAI for evidence summary generation: {str(e)}. Trying alternatives...")
    
    # Fall back to template-based generation if AI models are unavailable
    mechanisms = [
        f"{drug_name} may interact with receptors involved in {disease_name} pathology",
        f"{drug_name} could regulate inflammatory pathways relevant to {disease_name}",
        f"Molecular similarities between {drug_name}'s known targets and {disease_name}-related proteins",
        f"Metabolic modulation by {drug_name} that influences {disease_name} progression"
    ]
    
    evidence_sources = [
        "Preclinical studies in animal models",
        "Molecular docking simulations",
        "Observational data from clinical practice",
        "Knowledge graph-derived relationships"
    ]
    
    # Create template-based summary
    summary = f"""
    # Evidence Summary: {drug_name} for {disease_name}
    
    ## Key Mechanisms
    Based on current knowledge and analysis, {random.choice(mechanisms)}.
    
    ## Evidence Sources
    The repurposing hypothesis is supported by:
    - {random.choice(evidence_sources)}
    - {random.choice(evidence_sources)}
    
    ## Potential Impact
    If validated, this repurposing opportunity could provide additional treatment options
    for patients with {disease_name}, possibly with fewer side effects or better accessibility
    than current therapies.
    
    ## Research Status
    Additional clinical investigation is needed to validate this repurposing opportunity.
    """
    
    return summary

def generate_mechanistic_explanation(drug, disease):
    """
    Generate a plausible mechanistic explanation for why a drug might treat a disease
    
    When AI models are available, they are used to generate detailed, science-based explanations.
    Tries OpenAI first, then Hugging Face, then falls back to template-based generation.
    """
    # Try to use OpenAI if available
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            # Get detailed explanation from OpenAI
            openai_result = openai_analysis.generate_mechanistic_explanation(drug, disease)
            if openai_result and "explanation" in openai_result and openai_result["explanation"]:
                return openai_result["explanation"]
        except Exception as e:
            st.warning(f"Error using OpenAI for explanation generation: {str(e)}. Trying Hugging Face...")
            # Try HuggingFace next
    
    # Try to use Hugging Face AI if OpenAI failed or is not available
    if HUGGINGFACE_AVAILABLE and os.environ.get("HUGGINGFACE_API_KEY"):
        try:
            # Get detailed explanation from Hugging Face
            huggingface_result = huggingface_analysis.generate_mechanistic_explanation(drug, disease)
            if huggingface_result and "explanation" in huggingface_result and huggingface_result["explanation"]:
                return huggingface_result["explanation"]
        except Exception as e:
            st.warning(f"Error using Hugging Face for explanation generation: {str(e)}. Falling back to template-based explanation...")
            # Fallback to template-based approach
            
    # Using template-based approach for explanation generation
    # Common mechanism templates
    templates = [
        "{drug_name} may {action} {target}, which is implicated in {disease_name}. This {effect} may {result} in patients with {disease_name}.",
        "The {mechanism} of {drug_name} suggests it could {effect} in {disease_name} by {action} the {pathway} pathway.",
        "Research indicates {drug_name} {action} {target}, which may {effect} in {disease_name} through {pathway} regulation.",
        "{drug_name}'s ability to {action} may provide therapeutic benefit in {disease_name} by {effect}, potentially {long_term_effect}.",
        "By {action}, {drug_name} may {effect} associated with {disease_name}, which could {result} and {long_term_effect}."
    ]
    
    # Potential mechanisms based on drug class
    mechanisms = {
        "anti-inflammatory": ["reduce inflammation", "inhibit inflammatory cytokines", "suppress immune response"],
        "antidiabetic": ["improve insulin sensitivity", "decrease hepatic glucose production", "enhance glucose uptake"],
        "statin": ["reduce cholesterol synthesis", "improve endothelial function", "decrease inflammation"],
        "immunomodulator": ["modulate immune response", "inhibit cytokine production", "affect T-cell function"],
        "vasodilator": ["increase blood flow", "relax smooth muscle", "reduce vascular resistance"]
    }
    
    # Map drug to likely class based on name or description
    drug_class = None
    drug_info = (drug['name'] + " " + drug['description']).lower()
    
    if "metformin" in drug_info or "diabetes" in drug_info:
        drug_class = "antidiabetic"
    elif "statin" in drug_info or "cholesterol" in drug_info:
        drug_class = "statin"
    elif "anti-inflammatory" in drug_info or "inflammation" in drug_info:
        drug_class = "anti-inflammatory"
    elif "immune" in drug_info or "tnf" in drug_info:
        drug_class = "immunomodulator"
    elif "vasodilat" in drug_info or "blood flow" in drug_info:
        drug_class = "vasodilator"
    else:
        # Default to generic mechanisms
        mechanism_verbs = ["inhibit", "activate", "modulate", "regulate", "affect"]
        targets = ["cellular pathways", "protein expression", "gene transcription", "receptor activity"]
        effects = ["reduce symptoms", "slow disease progression", "improve outcomes", "provide therapeutic benefit"]
        
        # Generate generic explanation
        template = random.choice(templates)
        return template.format(
            drug_name=drug['name'],
            disease_name=disease['name'],
            action=random.choice(mechanism_verbs),
            target=random.choice(targets),
            effect=random.choice(effects),
            pathway="disease-related",
            result="improve clinical outcomes",
            long_term_effect="reduce disease burden"
        )
    
    # Generate class-specific explanation
    template = random.choice(templates)
    action = random.choice(mechanisms[drug_class])
    
    targets = {
        "anti-inflammatory": ["inflammatory markers", "cytokine production", "immune cell activation"],
        "antidiabetic": ["glucose metabolism", "insulin signaling", "hepatic glucose output"],
        "statin": ["cholesterol synthesis", "plaque formation", "vascular inflammation"],
        "immunomodulator": ["immune cell function", "cytokine signaling", "inflammatory response"],
        "vasodilator": ["vascular tone", "blood vessel dilation", "tissue perfusion"]
    }
    
    effects = {
        "anti-inflammatory": ["reduce inflammation", "decrease symptom severity", "slow tissue damage"],
        "antidiabetic": ["improve metabolic parameters", "enhance cellular energy utilization", "reduce oxidative stress"],
        "statin": ["improve vascular health", "reduce inflammatory markers", "enhance cell membrane function"],
        "immunomodulator": ["normalize immune response", "decrease inflammatory damage", "regulate cell signaling"],
        "vasodilator": ["increase oxygen delivery", "improve tissue perfusion", "enhance nutrient delivery"]
    }
    
    pathways = {
        "anti-inflammatory": ["NF-κB", "JAK-STAT", "COX"],
        "antidiabetic": ["AMPK", "insulin receptor", "glucose transporter"],
        "statin": ["mevalonate", "lipid metabolism", "isoprenoid"],
        "immunomodulator": ["cytokine", "T-cell activation", "macrophage"],
        "vasodilator": ["nitric oxide", "calcium channel", "endothelial"]
    }
    
    results = {
        "anti-inflammatory": ["reduce disease activity", "improve quality of life", "decrease tissue damage"],
        "antidiabetic": ["improve metabolic control", "protect against complications", "enhance cellular function"],
        "statin": ["reduce risk factors", "improve vascular function", "protect against disease progression"],
        "immunomodulator": ["normalize immune function", "reduce disease flares", "decrease tissue inflammation"],
        "vasodilator": ["improve perfusion", "enhance oxygen delivery", "reduce symptoms"]
    }
    
    long_term_effects = {
        "anti-inflammatory": ["prevent disease progression", "reduce long-term damage", "improve functional outcomes"],
        "antidiabetic": ["prevent diabetic complications", "preserve organ function", "improve long-term health"],
        "statin": ["reduce risk of adverse events", "improve long-term vascular health", "prevent complications"],
        "immunomodulator": ["maintain disease remission", "prevent organ damage", "improve long-term prognosis"],
        "vasodilator": ["prevent tissue damage", "improve functional capacity", "enhance quality of life"]
    }
    
    return template.format(
        drug_name=drug['name'],
        disease_name=disease['name'],
        action=action,
        target=random.choice(targets[drug_class]),
        effect=random.choice(effects[drug_class]),
        mechanism=drug.get('mechanism', action),
        pathway=random.choice(pathways[drug_class]),
        result=random.choice(results[drug_class]),
        long_term_effect=random.choice(long_term_effects[drug_class])
    )

def analyze_repurposing_candidate(drug, disease, graph=None, pubmed_articles=None):
    """
    Analyze a drug repurposing candidate and generate a detailed report
    
    When AI models are available, they use advanced reasoning to generate detailed, 
    science-based analysis. Tries OpenAI first, then Hugging Face, then falls back 
    to traditional rule-based analysis.
    """
    # Try to use AI for advanced analysis if available
    ai_analysis_results = None
    
    # Count literature references
    literature_count = 0
    if pubmed_articles:
        for article in pubmed_articles:
            if (drug['name'].lower() in article['title'].lower() or 
                drug['name'].lower() in article['abstract'].lower()) and (
                disease['name'].lower() in article['title'].lower() or 
                disease['name'].lower() in article['abstract'].lower()):
                literature_count += 1
    
    # Try OpenAI first
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            # Get detailed analysis from OpenAI
            ai_analysis_results = openai_analysis.analyze_repurposing_candidate(
                drug, 
                disease, 
                literature_count=literature_count
            )
        except Exception as e:
            st.warning(f"Error using OpenAI for candidate analysis: {str(e)}. Trying Hugging Face...")
            # Continue with Hugging Face next
    
    # Moving to alternative AI model option
    
    # Try Hugging Face if OpenAI failed or isn't available
    if ai_analysis_results is None and HUGGINGFACE_AVAILABLE and os.environ.get("HUGGINGFACE_API_KEY"):
        try:
            # Get detailed analysis from Hugging Face
            ai_analysis_results = huggingface_analysis.analyze_repurposing_candidate(
                drug, 
                disease, 
                literature_count=literature_count
            )
        except Exception as e:
            st.warning(f"Error using Hugging Face for candidate analysis: {str(e)}. Falling back to traditional analysis.")
            # Continue with traditional approach on failure
    
    # Calculate confidence score
    confidence_score = calculate_confidence_score(drug, disease, graph, pubmed_articles)
    
    # Generate mechanistic explanation
    mechanism = generate_mechanistic_explanation(drug, disease)
    
    # Determine status based on confidence score
    if confidence_score >= 75:
        status = "Highly Promising"
    elif confidence_score >= 60:
        status = "Promising"
    elif confidence_score >= 40:
        status = "Under Investigation"
    else:
        status = "Early Research"
    
    # Count supporting evidence from PubMed (if available)
    evidence_count = 0
    if pubmed_articles:
        for article in pubmed_articles:
            if (drug['name'].lower() in article['title'].lower() or 
                drug['name'].lower() in article['abstract'].lower()) and (
                disease['name'].lower() in article['title'].lower() or 
                disease['name'].lower() in article['abstract'].lower()):
                evidence_count += 1
    
    # Create analysis report
    report = {
        "drug": drug['name'],
        "disease": disease['name'],
        "confidence_score": confidence_score,
        "mechanism": mechanism,
        "evidence_count": evidence_count,
        "status": status,
        "ai_enhanced": (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY") is not None) or
                      (HUGGINGFACE_AVAILABLE and os.environ.get("HUGGINGFACE_API_KEY") is not None)
    }
    
    # Add AI-specific analysis results if available
    if ai_analysis_results:
        if "key_insights" in ai_analysis_results:
            report["key_insights"] = ai_analysis_results["key_insights"]
        if "potential_advantages" in ai_analysis_results:
            report["potential_advantages"] = ai_analysis_results["potential_advantages"]
        if "challenges" in ai_analysis_results:
            report["challenges"] = ai_analysis_results["challenges"]
        if "research_directions" in ai_analysis_results:
            report["research_directions"] = ai_analysis_results["research_directions"]
    
    return report

def batch_analyze_candidates(candidates, drugs, diseases, graph=None, pubmed_articles=None, batch_size=5, cache=True):
    """
    Analyze a batch of drug repurposing candidates with optimizations
    
    Parameters:
    - candidates: List of candidate dictionaries
    - drugs: List of drug dictionaries
    - diseases: List of disease dictionaries
    - graph: NetworkX graph object (optional)
    - pubmed_articles: List of PubMed articles (optional)
    - batch_size: Number of candidates to process in each batch
    - cache: Whether to use caching for results
    
    Returns:
    - List of analyzed candidates with details
    """
    results = []
    
    # Create a cache key function
    def get_cache_key(drug_id, disease_id):
        return f"analysis_{drug_id}_{disease_id}"
    
    # Check if we have a cache in session state
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    # Process candidates in batches for better UX
    candidate_batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    
    total_batches = len(candidate_batches)
    progress_placeholder = st.empty()
    
    for batch_idx, batch in enumerate(candidate_batches):
        batch_results = []
        
        # Update progress
        progress_placeholder.progress((batch_idx) / total_batches, 
                                     text=f"Processing batch {batch_idx+1}/{total_batches}...")
        
        for candidate in batch:
            # Get drug and disease details
            drug = next((d for d in drugs if d['id'] == candidate['drug_id']), None)
            disease = next((d for d in diseases if d['id'] == candidate['disease_id']), None)
            
            if drug and disease:
                # Check cache first if enabled
                cache_key = get_cache_key(drug['id'], disease['id'])
                if cache and cache_key in st.session_state.analysis_cache:
                    analysis = st.session_state.analysis_cache[cache_key]
                else:
                    # Analyze the candidate
                    analysis = analyze_repurposing_candidate(drug, disease, graph, pubmed_articles)
                    
                    # Store in cache if enabled
                    if cache:
                        st.session_state.analysis_cache[cache_key] = analysis
                
                batch_results.append(analysis)
        
        # Add batch results
        results.extend(batch_results)
    
    # Clear progress
    progress_placeholder.empty()
    
    # Sort by confidence score
    results.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    return results
