import pandas as pd
import requests
import xml.etree.ElementTree as ET
import json
import time
import random
import os
from nltk.tokenize import sent_tokenize
import streamlit as st

# API keys and base URLs
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")
UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest/"

def search_pubmed(query, max_results=20):
    """
    Search PubMed for articles matching the query
    """
    try:
        # Search for articles
        search_url = f"{PUBMED_BASE_URL}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        # Add API key if available
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY
        
        search_response = requests.get(search_url, params=params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            return []
            
        pmids = search_data["esearchresult"]["idlist"]
        
        if not pmids:
            return []
            
        # Get article details
        fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        # Add API key if available
        if PUBMED_API_KEY:
            fetch_params["api_key"] = PUBMED_API_KEY
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.content)
        articles = []
        
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article_id = article_elem.find(".//PMID").text
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else "No title available"
                
                abstract_elem = article_elem.find(".//AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else "No abstract available"
                
                year_elem = article_elem.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else "Unknown year"
                
                articles.append({
                    "pmid": article_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year
                })
            except Exception as e:
                st.error(f"Error parsing article: {str(e)}")
                continue
                
        return articles
        
    except requests.exceptions.RequestException as e:
        st.error(f"PubMed API request failed: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error in search_pubmed: {str(e)}")
        return []

def extract_drug_disease_relationships(articles):
    """
    Extract potential drug-disease relationships from article abstracts
    """
    relationships = []
    
    for article in articles:
        abstract = article["abstract"]
        sentences = sent_tokenize(abstract)
        
        for sentence in sentences:
            # Look for patterns indicating potential drug-disease relationships
            lower_sentence = sentence.lower()
            if any(term in lower_sentence for term in ["treat", "therapy", "effective", "efficacy", 
                                                    "administrat", "prescribe", "medication"]):
                relationships.append({
                    "source": article["pmid"],
                    "title": article["title"],
                    "text": sentence,
                    "year": article["year"]
                })
                
    return relationships

def fetch_drug_info(drug_name):
    """
    Fetch drug information from RxNorm API
    """
    try:
        # First get the RxCUI (RxNorm Concept Unique Identifier)
        base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        params = {
            "name": drug_name
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "idGroup" not in data or "rxnormId" not in data["idGroup"] or not data["idGroup"]["rxnormId"]:
            return None
            
        rxcui = data["idGroup"]["rxnormId"][0]
        
        # Now get detailed information using the RxCUI
        info_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
        info_response = requests.get(info_url)
        info_response.raise_for_status()
        info_data = info_response.json()
        
        if "allRelatedGroup" not in info_data or "conceptGroup" not in info_data["allRelatedGroup"]:
            return None
            
        drug_info = {
            "name": drug_name,
            "rxcui": rxcui,
            "related_concepts": []
        }
        
        for concept_group in info_data["allRelatedGroup"]["conceptGroup"]:
            if "conceptProperties" in concept_group:
                for concept in concept_group["conceptProperties"]:
                    drug_info["related_concepts"].append({
                        "name": concept.get("name", ""),
                        "category": concept_group.get("tty", "")
                    })
        
        return drug_info
        
    except requests.exceptions.RequestException as e:
        st.error(f"RxNorm API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error in fetch_drug_info: {str(e)}")
        return None

def get_umls_auth_token():
    """
    Get UMLS authentication token
    """
    if not UMLS_API_KEY:
        st.warning("UMLS API key not found. Using MeSH database as fallback.")
        return None
        
    try:
        # Request UMLS service ticket
        auth_url = f"{UMLS_BASE_URL}auth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "apikey": UMLS_API_KEY
        }
        
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"UMLS authentication failed: {str(e)}")
        return None

def fetch_disease_info(disease_name):
    """
    Fetch disease information from UMLS via their API
    Falls back to MeSH if UMLS authentication fails
    """
    # Try UMLS first
    umls_token = get_umls_auth_token()
    if umls_token:
        try:
            # Search for concept in UMLS
            search_url = f"{UMLS_BASE_URL}search/current"
            params = {
                "string": disease_name,
                "searchType": "exact",
                "ticket": umls_token
            }
            
            search_response = requests.get(search_url, params=params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if search_data.get('result', {}).get('results'):
                # Get the first concept
                concept = search_data['result']['results'][0]
                
                # Get concept details
                cui = concept['ui']
                concept_url = f"{UMLS_BASE_URL}content/current/CUI/{cui}"
                concept_params = {
                    "ticket": umls_token
                }
                
                concept_response = requests.get(concept_url, params=concept_params)
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                
                if concept_data.get('result'):
                    name = concept_data['result']['name']
                    # Get definition if available
                    definitions = []
                    for definition in concept_data['result'].get('definitions', []):
                        if definition.get('value'):
                            definitions.append(definition['value'])
                    
                    description = definitions[0] if definitions else "No description available"
                    
                    # Get semantic types
                    semantic_types = []
                    for semantic in concept_data['result'].get('semanticTypes', []):
                        if semantic.get('name'):
                            semantic_types.append(semantic['name'])
                    
                    return {
                        "name": name,
                        "umls_cui": cui,
                        "description": description,
                        "classification": semantic_types
                    }
        except Exception as e:
            st.warning(f"UMLS lookup failed, falling back to MeSH: {str(e)}")
    
    # Fallback to MeSH database
    try:
        # Search for disease in NCBI MeSH database
        search_url = f"{PUBMED_BASE_URL}esearch.fcgi"
        params = {
            "db": "mesh",
            "term": disease_name,
            "retmode": "json"
        }
        
        # Add API key if available
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY
        
        search_response = requests.get(search_url, params=params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"] or not search_data["esearchresult"]["idlist"]:
            return None
            
        mesh_id = search_data["esearchresult"]["idlist"][0]
        
        # Fetch detailed information
        fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi"
        fetch_params = {
            "db": "mesh",
            "id": mesh_id,
            "retmode": "xml"
        }
        
        # Add API key if available
        if PUBMED_API_KEY:
            fetch_params["api_key"] = PUBMED_API_KEY
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(fetch_response.content)
        descriptor = root.find(".//DescriptorRecord")
        
        if descriptor is None:
            return None
            
        # Extract disease information
        name_elem = descriptor.find(".//DescriptorName/String")
        name = name_elem.text if name_elem is not None else disease_name
        
        scope_note_elem = descriptor.find(".//ScopeNote")
        description = scope_note_elem.text if scope_note_elem is not None else "No description available"
        
        # Get tree numbers (classification)
        tree_numbers = []
        for tree_elem in descriptor.findall(".//TreeNumber"):
            if tree_elem.text:
                tree_numbers.append(tree_elem.text)
                
        return {
            "name": name,
            "mesh_id": mesh_id,
            "description": description,
            "classification": tree_numbers
        }
        
    except requests.exceptions.RequestException as e:
        st.error(f"MeSH API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error in fetch_disease_info: {str(e)}")
        return None

def load_initial_data():
    """
    Load initial sample data for the platform
    In a real implementation, this would fetch data from various sources
    """
    # Sample drugs - expanded list
    drugs = [
        {
            "id": "D001",
            "name": "Metformin",
            "description": "A biguanide hypoglycemic agent used in the treatment of non-insulin-dependent diabetes mellitus not responding to dietary modification.",
            "original_indication": "Type 2 Diabetes",
            "mechanism": "Decreases hepatic glucose production, decreases intestinal absorption of glucose, and improves insulin sensitivity"
        },
        {
            "id": "D002",
            "name": "Aspirin",
            "description": "A non-steroidal anti-inflammatory agent with analgesic, antipyretic, and antiplatelet properties.",
            "original_indication": "Pain, Fever, Inflammation",
            "mechanism": "Irreversibly inhibits cyclooxygenase-1 and 2 (COX-1 and COX-2) enzymes"
        },
        {
            "id": "D003",
            "name": "Atorvastatin",
            "description": "A selective, competitive inhibitor of HMG-CoA reductase used to lower cholesterol levels.",
            "original_indication": "Hypercholesterolemia",
            "mechanism": "Inhibits HMG-CoA reductase, reducing cholesterol synthesis"
        },
        {
            "id": "D004",
            "name": "Thalidomide",
            "description": "An immunomodulatory agent with anti-angiogenic and anti-inflammatory properties.",
            "original_indication": "Morning sickness (historically), now Multiple Myeloma",
            "mechanism": "Inhibits TNF-alpha production and angiogenesis"
        },
        {
            "id": "D005",
            "name": "Sildenafil",
            "description": "A selective inhibitor of cyclic guanosine monophosphate (cGMP)-specific phosphodiesterase type 5.",
            "original_indication": "Pulmonary arterial hypertension",
            "mechanism": "Inhibits PDE5, increasing cGMP levels leading to smooth muscle relaxation"
        },
        {
            "id": "D006",
            "name": "Donepezil",
            "description": "A cholinesterase inhibitor used to treat symptoms of mild to moderate Alzheimer's disease.",
            "original_indication": "Alzheimer's Disease",
            "mechanism": "Inhibits acetylcholinesterase, increasing acetylcholine levels in the brain"
        },
        {
            "id": "D007",
            "name": "Memantine",
            "description": "An NMDA receptor antagonist used to treat moderate to severe Alzheimer's disease.",
            "original_indication": "Alzheimer's Disease",
            "mechanism": "Blocks NMDA receptors, reducing glutamate excitotoxicity"
        },
        {
            "id": "D008",
            "name": "Fingolimod",
            "description": "A sphingosine-1-phosphate receptor modulator used to treat relapsing forms of multiple sclerosis.",
            "original_indication": "Multiple Sclerosis",
            "mechanism": "Prevents lymphocytes from leaving lymph nodes, reducing immune attacks on the central nervous system"
        },
        {
            "id": "D009",
            "name": "Lenalidomide",
            "description": "An immunomodulatory drug used to treat multiple myeloma and other blood cancers.",
            "original_indication": "Multiple Myeloma",
            "mechanism": "Modulates immune system and has anti-angiogenic properties"
        },
        {
            "id": "D010",
            "name": "Rituximab",
            "description": "A monoclonal antibody that targets CD20 antigen on B cells, used to treat certain cancers and autoimmune disorders.",
            "original_indication": "Non-Hodgkin's Lymphoma",
            "mechanism": "Binds to CD20 on B cells, triggering cell death"
        },
        {
            "id": "D011",
            "name": "Adalimumab",
            "description": "A monoclonal antibody that blocks TNF-alpha, used to treat various inflammatory diseases.",
            "original_indication": "Rheumatoid Arthritis",
            "mechanism": "Inhibits tumor necrosis factor (TNF) activity"
        },
        {
            "id": "D012",
            "name": "Etanercept",
            "description": "A fusion protein that acts as a TNF inhibitor, used to treat autoimmune diseases.",
            "original_indication": "Rheumatoid Arthritis",
            "mechanism": "Binds to and neutralizes TNF"
        },
        {
            "id": "D013",
            "name": "Tofacitinib",
            "description": "A Janus kinase (JAK) inhibitor used to treat rheumatoid arthritis and ulcerative colitis.",
            "original_indication": "Rheumatoid Arthritis",
            "mechanism": "Inhibits JAK enzymes, interfering with the JAK-STAT signaling pathway"
        },
        {
            "id": "D014",
            "name": "Dexamethasone",
            "description": "A corticosteroid used to treat various inflammatory and autoimmune conditions.",
            "original_indication": "Inflammation",
            "mechanism": "Binds to glucocorticoid receptors, reducing inflammation and suppressing the immune system"
        },
        {
            "id": "D015",
            "name": "Bortezomib",
            "description": "A proteasome inhibitor used to treat multiple myeloma and mantle cell lymphoma.",
            "original_indication": "Multiple Myeloma",
            "mechanism": "Inhibits 26S proteasome, leading to cell cycle arrest and apoptosis"
        }
    ]

    # Sample diseases - expanded list
    diseases = [
        {
            "id": "DIS001",
            "name": "Type 2 Diabetes",
            "description": "A metabolic disorder characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
            "category": "Metabolic"
        },
        {
            "id": "DIS002",
            "name": "Alzheimer's Disease",
            "description": "A progressive neurologic disorder that causes brain cells to die and the brain to shrink (atrophy).",
            "category": "Neurological"
        },
        {
            "id": "DIS003",
            "name": "Rheumatoid Arthritis",
            "description": "An autoimmune and inflammatory disease in which the immune system attacks healthy cells in the body by mistake.",
            "category": "Autoimmune"
        },
        {
            "id": "DIS004",
            "name": "Multiple Myeloma",
            "description": "A cancer that forms in a type of white blood cell called a plasma cell.",
            "category": "Cancer"
        },
        {
            "id": "DIS005",
            "name": "Cancer",
            "description": "A disease in which some of the body's cells grow uncontrollably and spread to other parts of the body.",
            "category": "Cancer"
        },
        {
            "id": "DIS006",
            "name": "Pulmonary Arterial Hypertension",
            "description": "High blood pressure in the arteries that supply the lungs (pulmonary arteries).",
            "category": "Cardiovascular"
        },
        {
            "id": "DIS007",
            "name": "Multiple Sclerosis",
            "description": "A disease in which the immune system eats away at the protective covering of nerves.",
            "category": "Autoimmune"
        },
        {
            "id": "DIS008",
            "name": "Parkinson's Disease",
            "description": "A brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination.",
            "category": "Neurological"
        },
        {
            "id": "DIS009",
            "name": "Psoriasis",
            "description": "A skin condition that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp.",
            "category": "Autoimmune"
        },
        {
            "id": "DIS010",
            "name": "Crohn's Disease",
            "description": "A type of inflammatory bowel disease that causes inflammation of the digestive tract, leading to abdominal pain, severe diarrhea, fatigue, weight loss and malnutrition.",
            "category": "Autoimmune"
        },
        {
            "id": "DIS011",
            "name": "Breast Cancer",
            "description": "A type of cancer that forms in the cells of the breasts and can occur in both men and women, but it's more common in women.",
            "category": "Cancer"
        },
        {
            "id": "DIS012",
            "name": "Heart Failure",
            "description": "A chronic condition in which the heart doesn't pump blood as well as it should.",
            "category": "Cardiovascular"
        },
        {
            "id": "DIS013",
            "name": "COVID-19",
            "description": "An infectious disease caused by the SARS-CoV-2 virus, primarily affecting the respiratory system.",
            "category": "Infectious"
        },
        {
            "id": "DIS014",
            "name": "Non-Hodgkin's Lymphoma",
            "description": "A type of cancer that begins in the lymphatic system, part of the body's germ-fighting immune system.",
            "category": "Cancer"
        },
        {
            "id": "DIS015",
            "name": "Asthma",
            "description": "A condition in which a person's airways become inflamed, narrow and swell and produce extra mucus, which makes it difficult to breathe.",
            "category": "Respiratory"
        }
    ]

    # Sample relationships (edges in the knowledge graph) - expanded
    relationships = [
        # Established treatments
        {"source": "D001", "target": "DIS001", "type": "treats", "confidence": 0.95},
        {"source": "D002", "target": "DIS003", "type": "treats", "confidence": 0.80},
        {"source": "D004", "target": "DIS004", "type": "treats", "confidence": 0.90},
        {"source": "D005", "target": "DIS006", "type": "treats", "confidence": 0.92},
        {"source": "D006", "target": "DIS002", "type": "treats", "confidence": 0.88},
        {"source": "D007", "target": "DIS002", "type": "treats", "confidence": 0.85},
        {"source": "D008", "target": "DIS007", "type": "treats", "confidence": 0.91},
        {"source": "D009", "target": "DIS004", "type": "treats", "confidence": 0.93},
        {"source": "D010", "target": "DIS014", "type": "treats", "confidence": 0.94},
        {"source": "D011", "target": "DIS003", "type": "treats", "confidence": 0.89},
        {"source": "D012", "target": "DIS003", "type": "treats", "confidence": 0.87},
        {"source": "D013", "target": "DIS003", "type": "treats", "confidence": 0.86},
        {"source": "D014", "target": "DIS003", "type": "treats", "confidence": 0.78},
        {"source": "D015", "target": "DIS004", "type": "treats", "confidence": 0.91},
        
        # Potential repurposing candidates
        {"source": "D003", "target": "DIS002", "type": "potential", "confidence": 0.65},
        {"source": "D002", "target": "DIS002", "type": "potential", "confidence": 0.55},
        {"source": "D001", "target": "DIS005", "type": "potential", "confidence": 0.40},
        {"source": "D003", "target": "DIS005", "type": "potential", "confidence": 0.45},
        {"source": "D014", "target": "DIS013", "type": "potential", "confidence": 0.72},
        {"source": "D006", "target": "DIS008", "type": "potential", "confidence": 0.48},
        {"source": "D010", "target": "DIS009", "type": "potential", "confidence": 0.58},
        {"source": "D011", "target": "DIS010", "type": "potential", "confidence": 0.75},
        {"source": "D001", "target": "DIS012", "type": "potential", "confidence": 0.42},
        {"source": "D007", "target": "DIS008", "type": "potential", "confidence": 0.53},
        {"source": "D008", "target": "DIS008", "type": "potential", "confidence": 0.39},
        {"source": "D013", "target": "DIS009", "type": "potential", "confidence": 0.61},
        {"source": "D005", "target": "DIS012", "type": "potential", "confidence": 0.44},
        {"source": "D012", "target": "DIS015", "type": "potential", "confidence": 0.51},
        {"source": "D002", "target": "DIS013", "type": "potential", "confidence": 0.62}
    ]

    # Sample repurposing candidates with mechanistic explanations
    candidates = [
        {
            "drug": "Metformin",
            "disease": "Cancer",
            "confidence_score": 72,
            "mechanism": "Metformin may inhibit cancer cell growth through activation of AMPK pathway, which leads to inhibition of mTOR signaling. It also appears to reduce insulin levels, which may slow the growth of cancer cells that are dependent on insulin for their proliferation.",
            "evidence_count": 28,
            "status": "Promising"
        },
        {
            "drug": "Aspirin",
            "disease": "Alzheimer's Disease",
            "confidence_score": 65,
            "mechanism": "Long-term use of aspirin may reduce inflammation in the brain, which is a contributing factor to Alzheimer's disease progression. Its anti-inflammatory properties may mitigate neuroinflammation and potentially reduce amyloid plaque formation.",
            "evidence_count": 15,
            "status": "Under Investigation"
        },
        {
            "drug": "Atorvastatin",
            "disease": "Alzheimer's Disease",
            "confidence_score": 58,
            "mechanism": "Statins like atorvastatin may reduce Alzheimer's risk by improving cerebral blood flow, reducing neuroinflammation, and potentially affecting amyloid-beta metabolism. Their cholesterol-lowering effects may also contribute to reduced risk as high cholesterol is associated with increased dementia risk.",
            "evidence_count": 12,
            "status": "Under Investigation"
        },
        {
            "drug": "Atorvastatin",
            "disease": "Cancer",
            "confidence_score": 49,
            "mechanism": "Atorvastatin may inhibit cancer cell proliferation by blocking the mevalonate pathway, which is essential for cellular functions including growth. This disruption may induce apoptosis in cancer cells and inhibit metastasis by affecting cell migration and invasion processes.",
            "evidence_count": 8,
            "status": "Early Research"
        }
    ]

    return drugs, diseases, relationships, candidates
