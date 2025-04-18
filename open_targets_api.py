"""
Open Targets Platform API Integration for the Drug Repurposing Engine.

This module provides functions to integrate with the Open Targets Platform API to retrieve:
1. Target-disease associations with evidence scores
2. Drug-target interactions
3. Disease information and phenotypes
4. Target information and properties

Open Targets is a public-private partnership that uses human genetics and genomics data
for systematic drug target identification and prioritization.

API Documentation: https://platform-api.opentargets.io/api/v4/graphql/schema
GraphQL Endpoint: https://api.platform.opentargets.org/api/v4/graphql

Implementation includes:
- Multiple retry mechanisms
- Response caching
- Fallback mechanisms for when API is unavailable
- Comprehensive error handling
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for API responses to reduce redundant calls
_api_cache = {}

# Function to execute GraphQL queries
def execute_gql_query(query, variables=None):
    """Execute a GraphQL query against the Open Targets API"""
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    try:
        response = requests.post(
            url,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"GraphQL query failed with status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error executing GraphQL query: {e}")
        return None

import requests
import json
import logging
import time
import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URLs for Open Targets APIs
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_REST_URL = "https://api.platform.opentargets.org/api/v4/platform"

# Constants for API
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
CACHE_EXPIRY = 7  # days

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Response caching functionality
def get_cache_key(endpoint: str, params: dict = None) -> str:
    """Generate a cache key from endpoint and parameters"""
    if params:
        param_str = json.dumps(params, sort_keys=True)
        return f"{endpoint}_{hash(param_str)}"
    return endpoint

def save_to_cache(key: str, data: Any) -> None:
    """Save response data to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    cache_data = {
        'timestamp': datetime.now(),
        'data': data
    }
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save to cache: {str(e)}")

def load_from_cache(key: str) -> Optional[Any]:
    """Load response data from cache if it exists and is not expired"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check if cache is expired
            if datetime.now() - cache_data['timestamp'] < timedelta(days=CACHE_EXPIRY):
                logger.info(f"Using cached data for {key}")
                return cache_data['data']
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
    
    return None

# Alternative data sources and fallback mechanisms
def get_default_disease_data() -> Dict[str, Dict[str, Any]]:
    """Return a minimal set of disease data for fallback"""
    return {
        'alzheimer': {
            'id': 'EFO_0000249',
            'name': 'Alzheimer disease',
            'description': 'A progressive disease of the brain affecting memory, thinking, behavior and emotion',
            'therapeuticAreas': ['central nervous system disease']
        },
        'diabetes': {
            'id': 'EFO_0001359',
            'name': 'Type 2 diabetes',
            'description': 'A metabolic disorder characterized by hyperglycemia',
            'therapeuticAreas': ['metabolic disease']
        },
        'cancer': {
            'id': 'EFO_0000311',
            'name': 'Cancer',
            'description': 'A disease characterized by abnormal cell growth',
            'therapeuticAreas': ['neoplasm']
        },
        'hypertension': {
            'id': 'EFO_0000537',
            'name': 'Hypertension',
            'description': 'A medical condition in which the blood pressure is persistently elevated',
            'therapeuticAreas': ['cardiovascular disease']
        }
    }

def search_drug(drug_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search for a drug in the Open Targets Platform.
    
    Args:
        drug_name (str): The drug name to search for
        
    Returns:
        Optional[List[Dict[str, Any]]]: Drug information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drug_search_{drug_name}"
        if cache_key in _api_cache:
            logger.info(f"Returning cached drug search results for '{drug_name}'")
            return _api_cache[cache_key]
            
        # Construct the API query
        query = """
        query DrugSearch($queryString: String!) {
            search(queryString: $queryString, entityNames: ["drug"], page: {index: 0, size: 10}) {
                hits {
                    id
                    entity
                    name
                    description
                }
            }
        }
        """
        
        variables = {"queryString": drug_name}
        
        # Execute the query
        response = execute_gql_query(query, variables)
        
        if not response or 'errors' in response:
            # Handle error
            logger.warning(f"Error searching for drug '{drug_name}': {response.get('errors', 'Unknown error')}")
            return None
            
        # Extract results
        hits = response.get('data', {}).get('search', {}).get('hits', [])
        
        # Format the results
        results = []
        for hit in hits:
            if hit.get('entity') == 'drug':
                results.append({
                    'id': hit.get('id'),
                    'name': hit.get('name'),
                    'description': hit.get('description', "No description available")
                })
                
        # Cache the results
        _api_cache[cache_key] = results
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching for drug '{drug_name}': {e}")
        return None

def search_disease(disease_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search for a disease in the Open Targets Platform.
    
    Args:
        disease_name (str): The disease name to search for
        
    Returns:
        Optional[List[Dict[str, Any]]]: Disease information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"disease_search_{disease_name}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # GraphQL query for disease search
        query = """
        query DiseaseSearch($query: String!) {
            search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: 10}) {
                total
                hits {
                    id
                    entity
                    name
                    description
                    highlights
                }
            }
        }
        """
        
        variables = {
            "query": disease_name
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'search' in response['data']:
            search_results = response['data']['search']
            
            if search_results['total'] > 0:
                diseases = []
                
                for hit in search_results['hits']:
                    if hit['entity'] == 'disease':
                        disease_info = {
                            'id': hit['id'],
                            'name': hit['name'],
                            'description': hit.get('description', 'No description available'),
                        }
                        
                        # Get additional disease details
                        details = get_disease_details(hit['id'])
                        if details:
                            disease_info.update(details)
                        
                        diseases.append(disease_info)
                
                # Cache the result
                save_to_cache(cache_key, diseases)
                return diseases
        
        # Try fallback approach
        if not response:
            default_data = get_default_disease_data()
            matched_diseases = []
            
            for key, value in default_data.items():
                if disease_name.lower() in key or disease_name.lower() in value.get('name', '').lower():
                    disease_info = {
                        'id': value['id'],
                        'name': value['name'],
                        'description': value.get('description', 'No description available'),
                        'therapeutic_areas': value.get('therapeuticAreas', []),
                    }
                    matched_diseases.append(disease_info)
            
            if matched_diseases:
                save_to_cache(cache_key, matched_diseases)
                return matched_diseases
                
        return None
    
    except Exception as e:
        logger.error(f"Error searching for disease {disease_name}: {str(e)}")
        return None

def get_disease_details(disease_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a disease from the Open Targets Platform.
    
    Args:
        disease_id (str): The Open Targets disease ID
        
    Returns:
        Optional[Dict[str, Any]]: Detailed disease information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"disease_details_{disease_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # GraphQL query for disease details
        query = """
        query DiseaseDetails($id: String!) {
            disease(efoId: $id) {
                id
                name
                description
                therapeuticAreas {
                    id
                    name
                }
                phenotypes {
                    id
                    name
                    url
                }
            }
        }
        """
        
        variables = {
            "id": disease_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'disease' in response['data']:
            disease_data = response['data']['disease']
            
            if disease_data:
                details = {
                    'therapeutic_areas': [
                        {'id': area['id'], 'name': area['name']}
                        for area in disease_data.get('therapeuticAreas', [])
                    ],
                    'phenotypes': [
                        {'id': phenotype['id'], 'name': phenotype['name'], 'url': phenotype.get('url')}
                        for phenotype in disease_data.get('phenotypes', [])
                    ]
                }
                
                # Cache the result
                save_to_cache(cache_key, details)
                return details
        
        # Try fallback approach
        if not response:
            default_data = get_default_disease_data()
            
            for _, value in default_data.items():
                if value['id'] == disease_id:
                    details = {
                        'therapeutic_areas': [
                            {'id': f"EFO_{i}", 'name': area}
                            for i, area in enumerate(value.get('therapeuticAreas', []))
                        ],
                        'phenotypes': []
                    }
                    save_to_cache(cache_key, details)
                    return details
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting disease details for {disease_id}: {str(e)}")
        return None

def search_target(target_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search for a gene/protein target in the Open Targets Platform.
    
    Args:
        target_name (str): The target name to search for (gene symbol or protein name)
        
    Returns:
        Optional[List[Dict[str, Any]]]: Target information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"target_search_{target_name}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for target search
        query = """
        query TargetSearch($query: String!) {
            search(queryString: $query, entityNames: ["target"], page: {index: 0, size: 10}) {
                total
                hits {
                    id
                    entity
                    name
                    description
                    highlights
                }
            }
        }
        """
        
        variables = {
            "query": target_name
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'search' in response['data']:
            search_results = response['data']['search']
            
            if search_results['total'] > 0:
                targets = []
                
                for hit in search_results['hits']:
                    if hit['entity'] == 'target':
                        target_info = {
                            'id': hit['id'],
                            'name': hit['name'],
                            'description': hit.get('description', 'No description available'),
                        }
                        
                        # Get additional target details
                        details = get_target_details(hit['id'])
                        if details:
                            target_info.update(details)
                        
                        targets.append(target_info)
                
                # Cache the result
                save_to_cache(cache_key, targets)
                return targets
        
        return None
    
    except Exception as e:
        logger.error(f"Error searching for target {target_name}: {str(e)}")
        return None

def get_target_details(target_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a target from the Open Targets Platform.
    
    Args:
        target_id (str): The Open Targets target ID (e.g., ENSG00000157764)
        
    Returns:
        Optional[Dict[str, Any]]: Detailed target information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"target_details_{target_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for target details
        query = """
        query TargetDetails($id: String!) {
            target(ensemblId: $id) {
                id
                approvedSymbol
                approvedName
                biotype
                genomicLocation {
                    chromosome
                    start
                    end
                }
                synonyms
                pathways {
                    name
                    id
                }
                functionDescriptions
                subcellularLocations
                targetClass {
                    id
                    name
                }
                chemicalProbes {
                    name
                    mechanismOfAction
                }
                druggability {
                    categories {
                        name
                        value
                    }
                }
            }
        }
        """
        
        variables = {
            "id": target_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'target' in response['data']:
            target_data = response['data']['target']
            
            if target_data:
                details = {
                    'approved_symbol': target_data.get('approvedSymbol'),
                    'approved_name': target_data.get('approvedName'),
                    'biotype': target_data.get('biotype'),
                    'genomic_location': target_data.get('genomicLocation'),
                    'synonyms': target_data.get('synonyms', []),
                    'pathways': target_data.get('pathways', []),
                    'function_descriptions': target_data.get('functionDescriptions', []),
                    'subcellular_locations': target_data.get('subcellularLocations', []),
                    'target_class': target_data.get('targetClass', []),
                    'chemical_probes': target_data.get('chemicalProbes', []),
                    'druggability': target_data.get('druggability', {}).get('categories', [])
                }
                
                # Cache the result
                save_to_cache(cache_key, details)
                return details
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting target details for {target_id}: {str(e)}")
        return None

def search_drugs(drug_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search for a drug in the Open Targets Platform.
    
    Args:
        drug_name (str): The drug name to search for
        
    Returns:
        Optional[List[Dict[str, Any]]]: Drug information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drug_search_{drug_name}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for drug search
        query = """
        query DrugSearch($query: String!) {
            search(queryString: $query, entityNames: ["drug"], page: {index: 0, size: 10}) {
                total
                hits {
                    id
                    entity
                    name
                    description
                    highlights
                }
            }
        }
        """
        
        variables = {
            "query": drug_name
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'search' in response['data']:
            search_results = response['data']['search']
            
            if search_results['total'] > 0:
                drugs = []
                
                for hit in search_results['hits']:
                    if hit['entity'] == 'drug':
                        drug_info = {
                            'id': hit['id'],
                            'name': hit['name'],
                            'description': hit.get('description', 'No description available'),
                        }
                        
                        # Get additional drug details
                        details = get_drug_details(hit['id'])
                        if details:
                            drug_info.update(details)
                        
                        drugs.append(drug_info)
                
                # Cache the results
                save_to_cache(cache_key, drugs)
                return drugs
        
        return None
    
    except Exception as e:
        logger.error(f"Error searching for drug {drug_name}: {str(e)}")
        return None

def get_drug_details(drug_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a drug from the Open Targets Platform.
    
    Args:
        drug_id (str): The Open Targets drug ID (e.g., CHEMBL1201580)
        
    Returns:
        Optional[Dict[str, Any]]: Detailed drug information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drug_details_{drug_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for drug details
        query = """
        query DrugDetails($id: String!) {
            drug(chemblId: $id) {
                id
                name
                synonyms
                tradeNames
                yearOfFirstApproval
                maximumClinicalTrialPhase
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targets {
                            id
                            name
                        }
                        references {
                            source
                            urls
                        }
                    }
                }
                indications {
                    rows {
                        disease {
                            id
                            name
                        }
                        phase
                        references {
                            source
                            urls
                        }
                    }
                }
                adverseEvents {
                    count
                    rows {
                        name
                        count
                        log2OddsRatio
                    }
                }
            }
        }
        """
        
        variables = {
            "id": drug_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'drug' in response['data']:
            drug_data = response['data']['drug']
            
            if drug_data:
                # Extract mechanisms of action
                mechanisms = []
                if drug_data.get('mechanismsOfAction') and drug_data['mechanismsOfAction'].get('rows'):
                    for row in drug_data['mechanismsOfAction']['rows']:
                        mechanism = {
                            'mechanism': row.get('mechanismOfAction'),
                            'targets': row.get('targets', [])
                        }
                        mechanisms.append(mechanism)
                
                # Extract indications
                indications = []
                if drug_data.get('indications') and drug_data['indications'].get('rows'):
                    for row in drug_data['indications']['rows']:
                        indication = {
                            'disease': row.get('disease'),
                            'phase': row.get('phase')
                        }
                        indications.append(indication)
                
                # Extract adverse events
                adverse_events = []
                if drug_data.get('adverseEvents') and drug_data['adverseEvents'].get('rows'):
                    adverse_events = drug_data['adverseEvents']['rows']
                
                details = {
                    'synonyms': drug_data.get('synonyms', []),
                    'trade_names': drug_data.get('tradeNames', []),
                    'year_of_first_approval': drug_data.get('yearOfFirstApproval'),
                    'maximum_clinical_trial_phase': drug_data.get('maximumClinicalTrialPhase'),
                    'mechanisms_of_action': mechanisms,
                    'indications': indications,
                    'adverse_events': adverse_events
                }
                
                # Cache the result
                save_to_cache(cache_key, details)
                return details
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting drug details for {drug_id}: {str(e)}")
        return None

def get_associations(target_id: str = None, disease_id: str = None, direct: bool = True, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
    """
    Get associations between targets and diseases from the Open Targets Platform.
    
    Args:
        target_id (str, optional): The Open Targets target ID
        disease_id (str, optional): The Open Targets disease ID
        direct (bool): Whether to use direct associations only
        limit (int): Maximum number of results to return
        
    Returns:
        Optional[List[Dict[str, Any]]]: Association information or None if not found
    """
    try:
        # Check cache first
        cache_key = f"associations_{target_id or ''}_{disease_id or ''}_{direct}_{limit}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # Construct the appropriate GraphQL query
        if target_id and not disease_id:
            # Get diseases associated with a target
            query = """
            query TargetAssociations($target_id: String!, $direct: Boolean, $limit: Int) {
                target(ensemblId: $target_id) {
                    associatedDiseases(direct: $direct) {
                        count
                        rows(limit: $limit) {
                            disease {
                                id
                                name
                            }
                            score
                            datatypeScores {
                                id
                                score
                            }
                        }
                    }
                }
            }
            """
            
            variables = {
                "target_id": target_id,
                "direct": direct,
                "limit": limit
            }
            
            response = make_graphql_request(query, variables)
            
            if response and 'data' in response and 'target' in response['data']:
                target_data = response['data']['target']
                
                if target_data and 'associatedDiseases' in target_data:
                    associated_diseases = target_data['associatedDiseases']
                    
                    if associated_diseases and 'rows' in associated_diseases:
                        associations = []
                        
                        for row in associated_diseases['rows']:
                            association = {
                                'target_id': target_id,
                                'disease_id': row['disease']['id'],
                                'disease_name': row['disease']['name'],
                                'overall_score': row['score'],
                                'datatype_scores': row['datatypeScores'] if 'datatypeScores' in row else []
                            }
                            
                            associations.append(association)
                        
                        # Cache the result
                        save_to_cache(cache_key, associations)
                        return associations
        
        elif disease_id and not target_id:
            # Get targets associated with a disease
            query = """
            query DiseaseAssociations($disease_id: String!, $direct: Boolean, $limit: Int) {
                disease(efoId: $disease_id) {
                    associatedTargets(direct: $direct) {
                        count
                        rows(limit: $limit) {
                            target {
                                id
                                approvedSymbol
                                approvedName
                            }
                            score
                            datatypeScores {
                                id
                                score
                            }
                        }
                    }
                }
            }
            """
            
            variables = {
                "disease_id": disease_id,
                "direct": direct,
                "limit": limit
            }
            
            response = make_graphql_request(query, variables)
            
            if response and 'data' in response and 'disease' in response['data']:
                disease_data = response['data']['disease']
                
                if disease_data and 'associatedTargets' in disease_data:
                    associated_targets = disease_data['associatedTargets']
                    
                    if associated_targets and 'rows' in associated_targets:
                        associations = []
                        
                        for row in associated_targets['rows']:
                            association = {
                                'disease_id': disease_id,
                                'target_id': row['target']['id'],
                                'target_symbol': row['target']['approvedSymbol'],
                                'target_name': row['target']['approvedName'],
                                'overall_score': row['score'],
                                'datatype_scores': row['datatypeScores'] if 'datatypeScores' in row else []
                            }
                            
                            associations.append(association)
                        
                        # Cache the result
                        save_to_cache(cache_key, associations)
                        return associations
        
        elif target_id and disease_id:
            # Get the association details between a specific target and disease
            query = """
            query TargetDiseaseAssociation($target_id: String!, $disease_id: String!, $direct: Boolean) {
                targetDiseasesConnection(
                    ensemblId: $target_id,
                    efoId: $disease_id,
                    direct: $direct
                ) {
                    rows {
                        target {
                            id
                            approvedSymbol
                        }
                        disease {
                            id
                            name
                        }
                        score
                        datatypeScores {
                            id
                            score
                        }
                    }
                }
            }
            """
            
            variables = {
                "target_id": target_id,
                "disease_id": disease_id,
                "direct": direct
            }
            
            response = make_graphql_request(query, variables)
            
            if response and 'data' in response and 'targetDiseasesConnection' in response['data']:
                connection_data = response['data']['targetDiseasesConnection']
                
                if connection_data and 'rows' in connection_data and len(connection_data['rows']) > 0:
                    row = connection_data['rows'][0]
                    
                    association = {
                        'target_id': row['target']['id'],
                        'target_symbol': row['target']['approvedSymbol'],
                        'disease_id': row['disease']['id'],
                        'disease_name': row['disease']['name'],
                        'overall_score': row['score'],
                        'datatype_scores': row['datatypeScores'] if 'datatypeScores' in row else []
                    }
                    
                    # Cache the result
                    save_to_cache(cache_key, [association])
                    return [association]
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting associations: {str(e)}")
        return None

def get_drug_indications(drug_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get the indications for a drug from the Open Targets Platform.
    
    Args:
        drug_id (str): The Open Targets drug ID (e.g., CHEMBL1201580)
        
    Returns:
        Optional[List[Dict[str, Any]]]: List of indications or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drug_indications_{drug_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for drug indications
        query = """
        query DrugIndications($id: String!) {
            drug(chemblId: $id) {
                id
                indications {
                    rows {
                        disease {
                            id
                            name
                        }
                        phase
                        references {
                            source
                            urls
                        }
                    }
                }
            }
        }
        """
        
        variables = {
            "id": drug_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'drug' in response['data']:
            drug_data = response['data']['drug']
            
            if drug_data and 'indications' in drug_data and 'rows' in drug_data['indications']:
                indications = []
                
                for row in drug_data['indications']['rows']:
                    indication = {
                        'disease_id': row['disease']['id'],
                        'disease_name': row['disease']['name'],
                        'phase': row['phase'],
                        'references': row.get('references', [])
                    }
                    
                    indications.append(indication)
                
                # Cache the result
                save_to_cache(cache_key, indications)
                return indications
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting drug indications for {drug_id}: {str(e)}")
        return None

def get_drug_targets(drug_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get the targets for a drug from the Open Targets Platform.
    
    Args:
        drug_id (str): The Open Targets drug ID (e.g., CHEMBL1201580)
        
    Returns:
        Optional[List[Dict[str, Any]]]: List of targets or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drug_targets_{drug_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for drug targets
        query = """
        query DrugTargets($id: String!) {
            drug(chemblId: $id) {
                id
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        actionType
                        targets {
                            id
                            approvedSymbol
                            approvedName
                        }
                    }
                }
            }
        }
        """
        
        variables = {
            "id": drug_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'drug' in response['data']:
            drug_data = response['data']['drug']
            
            if drug_data and 'mechanismsOfAction' in drug_data and 'rows' in drug_data['mechanismsOfAction']:
                targets = []
                
                for row in drug_data['mechanismsOfAction']['rows']:
                    for target in row.get('targets', []):
                        target_info = {
                            'target_id': target['id'],
                            'target_symbol': target['approvedSymbol'],
                            'target_name': target['approvedName'],
                            'mechanism_of_action': row.get('mechanismOfAction'),
                            'action_type': row.get('actionType')
                        }
                        
                        targets.append(target_info)
                
                # Cache the result
                save_to_cache(cache_key, targets)
                return targets
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting drug targets for {drug_id}: {str(e)}")
        return None

def find_repurposing_opportunities(disease_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Find potential drug repurposing opportunities for a disease.
    
    Args:
        disease_id (str): The Open Targets disease ID
        
    Returns:
        Optional[List[Dict[str, Any]]]: List of repurposing opportunities or None if not found
    """
    try:
        # Check cache first
        cache_key = f"repurposing_opportunities_{disease_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # Approach:
        # 1. Get associated targets for the disease
        # 2. For each target, get drugs that target it
        # 3. Filter out drugs that are already indicated for the disease
        # 4. Rank the opportunities by target association score
        
        # Get associated targets for the disease
        disease_targets = get_associations(disease_id=disease_id, direct=False, limit=20)
        
        if not disease_targets:
            return None
        
        # Get the disease info for later
        disease_info = None
        disease_details = get_disease_details(disease_id)
        if disease_details:
            disease_info = {
                'id': disease_id,
                'name': disease_details.get('name', 'Unknown disease')
            }
        
        opportunities = []
        
        for association in disease_targets:
            target_id = association['target_id']
            target_symbol = association['target_symbol']
            association_score = association['overall_score']
            
            # For each target, get drugs that target it
            target_drugs = get_drug_targets_by_target(target_id)
            
            if target_drugs:
                for drug in target_drugs:
                    # Get drug indications to check if the drug is already indicated for the disease
                    drug_indications = get_drug_indications(drug['id'])
                    
                    is_existing_indication = False
                    if drug_indications:
                        for indication in drug_indications:
                            if indication['disease_id'] == disease_id:
                                is_existing_indication = True
                                break
                    
                    if not is_existing_indication:
                        opportunity = {
                            'drug_id': drug['id'],
                            'drug_name': drug['name'],
                            'target_id': target_id,
                            'target_symbol': target_symbol,
                            'disease_id': disease_id,
                            'disease_name': disease_info['name'] if disease_info else 'Unknown disease',
                            'association_score': association_score,
                            'mechanism_of_action': drug.get('mechanism_of_action'),
                            'max_phase': drug.get('max_phase')
                        }
                        
                        opportunities.append(opportunity)
        
        # Sort by association score in descending order
        opportunities.sort(key=lambda x: x['association_score'], reverse=True)
        
        # Cache the result
        save_to_cache(cache_key, opportunities)
        return opportunities
    
    except Exception as e:
        logger.error(f"Error finding repurposing opportunities for disease {disease_id}: {str(e)}")
        return None

def get_drug_targets_by_target(target_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Helper function to get drugs that target a specific target.
    
    Args:
        target_id (str): The Open Targets target ID
        
    Returns:
        Optional[List[Dict[str, Any]]]: List of drugs or None if not found
    """
    try:
        # Check cache first
        cache_key = f"drugs_for_target_{target_id}"
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # GraphQL query for drugs targeting a specific target
        query = """
        query DrugsForTarget($id: String!) {
            target(ensemblId: $id) {
                id
                knownDrugs {
                    rows {
                        drug {
                            id
                            name
                            maximumClinicalTrialPhase
                        }
                        mechanismOfAction
                    }
                }
            }
        }
        """
        
        variables = {
            "id": target_id
        }
        
        # Make the GraphQL request
        response = make_graphql_request(query, variables)
        
        if response and 'data' in response and 'target' in response['data']:
            target_data = response['data']['target']
            
            if target_data and 'knownDrugs' in target_data and 'rows' in target_data['knownDrugs']:
                drugs = []
                
                for row in target_data['knownDrugs']['rows']:
                    drug_info = {
                        'id': row['drug']['id'],
                        'name': row['drug']['name'],
                        'max_phase': row['drug']['maximumClinicalTrialPhase'],
                        'mechanism_of_action': row.get('mechanismOfAction')
                    }
                    
                    drugs.append(drug_info)
                
                # Cache the result
                save_to_cache(cache_key, drugs)
                return drugs
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting drugs for target {target_id}: {str(e)}")
        return None

def make_graphql_request(query: str, variables: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """
    Make a GraphQL request to the Open Targets API.
    
    Args:
        query (str): The GraphQL query
        variables (Dict[str, Any], optional): Variables for the query
        
    Returns:
        Optional[Dict[str, Any]]: Response data or None if the request failed
    """
    # Generate cache key
    cache_key = f"graphql_{hash(query)}_{hash(str(variables) if variables else '')}"
    
    # Try to get from cache first
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached data for GraphQL query")
        return cached_data
    
    attempts = 0
    payload = {
        "query": query,
        "variables": variables if variables else {}
    }
    
    # Try alternative URLs if primary fails
    api_urls = [
        "https://platform-api.opentargets.org/api/v4/graphql",
        "https://api.platform.opentargets.org/api/v4/graphql",
        "https://api.platform.opentargets.org/api/v3/graphql"
    ]
    
    for api_url in api_urls:
        attempts = 0
        while attempts < MAX_RETRY_ATTEMPTS:
            try:
                logger.info(f"Attempting API request to {api_url}")
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if 'errors' in response_data:
                            logger.warning(f"GraphQL request returned errors: {response_data['errors']}")
                            # Try next attempt even with errors
                            attempts += 1
                            time.sleep(RETRY_DELAY * attempts)
                            continue
                        
                        # Cache the successful response
                        save_to_cache(cache_key, response_data)
                        return response_data
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {str(e)}")
                        attempts += 1
                        if attempts < MAX_RETRY_ATTEMPTS:
                            time.sleep(RETRY_DELAY * attempts)
                        continue
                elif response.status_code == 429 or response.status_code >= 500:
                    # Rate limiting or server error, retry after delay
                    logger.warning(f"Rate limit or server error (status {response.status_code}), retrying...")
                    attempts += 1
                    if attempts < MAX_RETRY_ATTEMPTS:
                        time.sleep(RETRY_DELAY * (attempts * 2))
                    continue
                else:
                    # Try next URL on client errors
                    logger.warning(f"GraphQL request failed with status code {response.status_code}")
                    try:
                        error_msg = response.json()
                        logger.warning(f"Error details: {error_msg}")
                    except:
                        logger.warning(f"Error response: {response.text[:200]}")
                    break  # Try next URL
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request exception: {str(e)}")
                attempts += 1
                if attempts < MAX_RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY * (attempts * 2))
                else:
                    # Try the next URL
                    break
    
    logger.error("All API endpoints and retry attempts failed for GraphQL request")
    
    # Try to get data from fallback methods
    if "disease" in query.lower():
        logger.info("Using fallback disease data")
        # Return fallback data for disease queries
        default_data = get_default_disease_data()
        if variables and 'query' in variables:
            query_term = variables['query'].lower()
            for key, value in default_data.items():
                if query_term in key or query_term in value.get('name', '').lower():
                    # Create a response structure that mimics the API response
                    mock_response = {
                        "data": {
                            "search": {
                                "hits": [
                                    {
                                        "id": value["id"],
                                        "name": value["name"],
                                        "description": value.get("description", ""),
                                        "entity": "disease"
                                    }
                                ],
                                "total": 1
                            }
                        }
                    }
                    # Cache this result too
                    save_to_cache(cache_key, mock_response)
                    logger.info(f"Using fallback data for {query_term}")
                    return mock_response
    
    return None

def make_rest_request(endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """
    Make a REST request to the Open Targets API.
    
    Args:
        endpoint (str): The API endpoint (e.g., '/target')
        params (Dict[str, Any], optional): Query parameters
        
    Returns:
        Optional[Dict[str, Any]]: Response data or None if the request failed
    """
    # Generate cache key
    cache_key = f"rest_{endpoint}_{hash(str(params) if params else '')}"
    
    # Try to get from cache first
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached data for REST request to {endpoint}")
        return cached_data
    
    attempts = 0
    url = f"{OPEN_TARGETS_REST_URL}{endpoint}"
    
    while attempts < MAX_RETRY_ATTEMPTS:
        try:
            response = requests.get(
                url,
                params=params,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Cache the successful response
                save_to_cache(cache_key, response_data)
                return response_data
            elif response.status_code == 429 or response.status_code >= 500:
                # Rate limiting or server error, retry after delay
                attempts += 1
                if attempts < MAX_RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY * (attempts * 2))
                continue
            else:
                logger.warning(f"REST request failed with status code {response.status_code}")
                try:
                    error_msg = response.json()
                    logger.warning(f"Error details: {error_msg}")
                except:
                    logger.warning(f"Error response: {response.text[:200]}")
                return None
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request exception: {str(e)}")
            attempts += 1
            if attempts < MAX_RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * (attempts * 2))
            continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {str(e)}")
            logger.warning(f"Response text: {response.text[:200]}")
            return None
    
    logger.error("All retry attempts failed for REST request")
    return None

def enrich_disease_data(disease_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich disease data with information from Open Targets.
    
    Args:
        disease_data (Dict[str, Any]): Existing disease data
        
    Returns:
        Dict[str, Any]: Enriched disease data
    """
    try:
        # Check if we already have Open Targets data
        if 'open_targets_id' in disease_data and disease_data['open_targets_id']:
            disease_id = disease_data['open_targets_id']
            disease_details = get_disease_details(disease_id)
            
            if disease_details:
                disease_data['therapeutic_areas'] = disease_details.get('therapeutic_areas', [])
                disease_data['phenotypes'] = disease_details.get('phenotypes', [])
        else:
            # Try to search by name
            disease_name = disease_data.get('name')
            if disease_name:
                search_results = search_disease(disease_name)
                
                if search_results and len(search_results) > 0:
                    top_result = search_results[0]
                    
                    disease_data['open_targets_id'] = top_result.get('id')
                    disease_data['description'] = top_result.get('description', disease_data.get('description', ''))
                    disease_data['therapeutic_areas'] = top_result.get('therapeutic_areas', [])
                    disease_data['phenotypes'] = top_result.get('phenotypes', [])
        
        return disease_data
    
    except Exception as e:
        logger.error(f"Error enriching disease data: {str(e)}")
        return disease_data

def enrich_drug_data(drug_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich drug data with information from Open Targets.
    
    Args:
        drug_data (Dict[str, Any]): Existing drug data
        
    Returns:
        Dict[str, Any]: Enriched drug data
    """
    try:
        # Check if we already have Open Targets data
        if 'chembl_id' in drug_data and drug_data['chembl_id']:
            drug_id = drug_data['chembl_id']
            drug_details = get_drug_details(drug_id)
            
            if drug_details:
                drug_data['mechanisms_of_action'] = drug_details.get('mechanisms_of_action', [])
                drug_data['indications'] = drug_details.get('indications', [])
                drug_data['maximum_clinical_trial_phase'] = drug_details.get('maximum_clinical_trial_phase')
                drug_data['year_of_first_approval'] = drug_details.get('year_of_first_approval')
        else:
            # Try to search by name
            drug_name = drug_data.get('name')
            if drug_name:
                search_results = search_drugs(drug_name)
                
                if search_results and len(search_results) > 0:
                    top_result = search_results[0]
                    
                    drug_data['chembl_id'] = top_result.get('id')
                    drug_data['description'] = top_result.get('description', drug_data.get('description', ''))
                    drug_data['mechanisms_of_action'] = top_result.get('mechanisms_of_action', [])
                    drug_data['indications'] = top_result.get('indications', [])
                    drug_data['maximum_clinical_trial_phase'] = top_result.get('maximum_clinical_trial_phase')
                    drug_data['year_of_first_approval'] = top_result.get('year_of_first_approval')
        
        return drug_data
    
    except Exception as e:
        logger.error(f"Error enriching drug data: {str(e)}")
        return drug_data
