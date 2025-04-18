"""
External Data Integration Module for the Drug Repurposing Engine.

This module provides an interface for retrieving data from external sources
such as ChEMBL and OpenFDA, with proper error handling, caching, and fallback mechanisms.
"""

import streamlit as st
import pandas as pd
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from data_sources import ChEMBLConnector, OpenFDAConnector, DrugBankConnector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data source connectors
chembl_connector = ChEMBLConnector()
openfda_connector = OpenFDAConnector()

# Caching configuration - adjust TTL based on how often the data changes
CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds

@st.cache_data(ttl=CACHE_TTL)
def search_drug_across_sources(drug_name: str, 
                              chembl_limit: int = 5, 
                              openfda_limit: int = 5,
                              include_chembl: bool = True,
                              include_openfda: bool = True) -> Dict[str, List[Dict]]:
    """
    Search for a drug across multiple data sources with caching
    
    Parameters:
    - drug_name: Name of the drug to search for
    - chembl_limit: Maximum number of results to fetch from ChEMBL
    - openfda_limit: Maximum number of results to fetch from OpenFDA
    - include_chembl: Whether to include ChEMBL results
    - include_openfda: Whether to include OpenFDA results
    
    Returns:
    - Dictionary containing results from each source
    """
    results = {}
    
    # Fetch data from ChEMBL
    if include_chembl:
        try:
            chembl_molecules = chembl_connector.search_molecule(drug_name, limit=chembl_limit)
            if chembl_molecules:
                results['chembl'] = chembl_connector.format_molecule_data(chembl_molecules)
            else:
                results['chembl'] = []
                logger.info(f"No ChEMBL results found for drug: {drug_name}")
        except Exception as e:
            results['chembl'] = []
            logger.error(f"Error fetching from ChEMBL: {str(e)}")
            st.warning(f"Unable to fetch data from ChEMBL: {str(e)}")
    
    # Fetch data from OpenFDA
    if include_openfda:
        try:
            openfda_drugs = openfda_connector.search_drug(drug_name, limit=openfda_limit)
            if openfda_drugs:
                results['openfda'] = openfda_connector.format_drug_data(openfda_drugs)
            else:
                results['openfda'] = []
                logger.info(f"No OpenFDA results found for drug: {drug_name}")
        except Exception as e:
            results['openfda'] = []
            logger.error(f"Error fetching from OpenFDA: {str(e)}")
            st.warning(f"Unable to fetch data from OpenFDA: {str(e)}")
    
    return results

@st.cache_data(ttl=CACHE_TTL)
def get_enriched_drug_data(drug_id: Optional[str], source: str) -> Optional[Dict[str, Any]]:
    """
    Get enriched data for a drug from its original source
    
    Parameters:
    - drug_id: ID of the drug in the source database, or None
    - source: Source database ('chembl' or 'openfda')
    
    Returns:
    - Dictionary with enriched drug data or None if not found
    """
    # Return None if drug_id is None or empty
    if drug_id is None or not drug_id.strip():
        logger.warning(f"No drug ID provided for {source}, cannot fetch enriched data")
        return None
        
    if source.lower() == 'chembl':
        try:
            # Get additional data from ChEMBL
            mechanisms = chembl_connector.get_drug_mechanisms(drug_id)
            indications = chembl_connector.get_drug_indications(drug_id)
            
            # Combine into enriched data
            return {
                'mechanisms': mechanisms,
                'indications': indications
            }
        except Exception as e:
            logger.error(f"Error getting enriched data from ChEMBL: {str(e)}")
            st.warning(f"Unable to fetch enriched data from ChEMBL: {str(e)}")
            return None
            
    elif source.lower() == 'openfda':
        try:
            # Get additional data from OpenFDA
            interactions = openfda_connector.get_drug_interactions(drug_id)
            adverse_events = openfda_connector.get_drug_adverse_events(drug_id, limit=5)
            
            # Combine into enriched data
            return {
                'interactions': interactions,
                'adverse_events': adverse_events
            }
        except Exception as e:
            logger.error(f"Error getting enriched data from OpenFDA: {str(e)}")
            st.warning(f"Unable to fetch enriched data from OpenFDA: {str(e)}")
            return None
    
    return None

def format_consolidated_drug_data(chembl_data: List[Dict], openfda_data: List[Dict]) -> List[Dict]:
    """
    Consolidate drug data from multiple sources, merging duplicates when possible
    
    Parameters:
    - chembl_data: List of drug dictionaries from ChEMBL
    - openfda_data: List of drug dictionaries from OpenFDA
    
    Returns:
    - Consolidated list of drug dictionaries with source information
    """
    # Track drugs by name to merge duplicates
    drug_dict = {}
    
    # Process ChEMBL data
    for drug in chembl_data or []:
        # Skip None or invalid entries
        if not drug or not isinstance(drug, dict):
            continue
            
        # Handle missing or None name
        if not drug.get('name'):
            drug_name = drug.get('id', 'Unknown Drug')
        else:
            drug_name = str(drug['name']).lower()
            
        if drug_name in drug_dict:
            # Merge with existing entry
            drug_dict[drug_name]['sources'].append('ChEMBL')
            if drug.get('id'):
                drug_dict[drug_name]['ids']['chembl'] = drug['id']
            # Add any missing fields
            if not drug_dict[drug_name].get('smiles') and drug.get('smiles'):
                drug_dict[drug_name]['smiles'] = drug['smiles']
        else:
            # Create new entry
            drug_dict[drug_name] = {
                'name': drug['name'],
                'description': drug['description'],
                'mechanism': drug.get('mechanism', ''),
                'sources': ['ChEMBL'],
                'ids': {'chembl': drug['id']},
                'smiles': drug.get('smiles', '')
            }
    
    # Process OpenFDA data
    for drug in openfda_data or []:
        # Skip None or invalid entries
        if not drug or not isinstance(drug, dict):
            continue
            
        # Handle missing or None name
        if not drug.get('name'):
            drug_name = drug.get('id', 'Unknown Drug')
        else:
            drug_name = str(drug['name']).lower()
            
        if drug_name in drug_dict:
            # Merge with existing entry
            drug_dict[drug_name]['sources'].append('OpenFDA')
            if drug.get('id'):
                drug_dict[drug_name]['ids']['openfda'] = drug['id']
            # Use OpenFDA mechanism if available and we don't have one
            if not drug_dict[drug_name].get('mechanism') and drug.get('mechanism'):
                drug_dict[drug_name]['mechanism'] = drug['mechanism']
            # Add brand name if available
            if drug.get('brand_name'):
                drug_dict[drug_name]['brand_name'] = drug['brand_name']
            # Add categories if available
            if drug.get('categories'):
                drug_dict[drug_name]['categories'] = drug['categories']
        else:
            # Create new entry with safe handling of potentially missing fields
            drug_dict[drug_name] = {
                'name': drug.get('name', drug_name),
                'description': drug.get('description', 'No description available'),
                'mechanism': drug.get('mechanism', ''),
                'sources': ['OpenFDA'],
                'ids': {'openfda': drug.get('id', f'unknown-{drug_name}')},
                'brand_name': drug.get('brand_name', ''),
                'categories': drug.get('categories', [])
            }
    
    # Convert dictionary back to list
    consolidated_drugs = list(drug_dict.values())
    
    # Add a unique ID for our system
    for i, drug in enumerate(consolidated_drugs):
        sources_str = "-".join(sorted(drug['sources']))
        drug['id'] = f"EXT-{sources_str}-{i+1}"
    
    return consolidated_drugs

def add_external_drug_to_database(drug_data: Dict[str, Any]) -> bool:
    """
    Add a drug from external sources to our database
    
    Parameters:
    - drug_data: Dictionary with drug data
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        from utils import add_drug
        
        # Format drug data for our database
        formatted_drug = {
            'id': drug_data['id'],
            'name': drug_data['name'],
            'description': drug_data.get('description', ''),
            'mechanism': drug_data.get('mechanism', ''),
            'source': ', '.join(drug_data.get('sources', [])),
            'structure': drug_data.get('smiles', '')
        }
        
        # Add drug to database
        result = add_drug(formatted_drug)
        
        return True if result else False
        
    except Exception as e:
        logger.error(f"Error adding drug to database: {str(e)}")
        st.error(f"Failed to add drug to database: {str(e)}")
        return False

def test_data_sources():
    """
    Test connectivity to data sources and return status
    
    Returns:
    - Dictionary with status information
    """
    status = {}
    
    # Test ChEMBL
    try:
        start_time = time.time()
        # Try a simple query
        result = chembl_connector.search_molecule("aspirin", limit=1)
        elapsed_time = time.time() - start_time
        
        if result:
            status['chembl'] = {
                'status': 'OK',
                'response_time': f"{elapsed_time:.2f}s",
                'message': f"Successfully retrieved {len(result)} results"
            }
        else:
            status['chembl'] = {
                'status': 'WARNING',
                'response_time': f"{elapsed_time:.2f}s",
                'message': "Connected but no data returned"
            }
    except Exception as e:
        status['chembl'] = {
            'status': 'ERROR',
            'message': str(e)
        }
    
    # Test OpenFDA with multiple fallbacks
    openfda_status = None
    
    # List of well-known drugs to try in order
    test_drugs = ["aspirin", "ibuprofen", "acetaminophen", "lisinopril", "metformin"]
    
    for drug in test_drugs:
        try:
            print(f"Testing OpenFDA connection with drug: {drug}")
            start_time = time.time()
            result = openfda_connector.search_drug(drug, limit=1)
            elapsed_time = time.time() - start_time
            
            if result:
                openfda_status = {
                    'status': 'OK',
                    'response_time': f"{elapsed_time:.2f}s",
                    'message': f"Successfully retrieved data for {drug}"
                }
                break  # Success, no need to try other drugs
        except Exception as e:
            print(f"OpenFDA test failed for {drug}: {str(e)}")
            # Continue to next drug on failure
    
    # If all test drugs failed, try direct API endpoint connection
    if not openfda_status:
        try:
            # Test direct connection to OpenFDA endpoint with a known good query
            print("Trying direct connection to OpenFDA API endpoint")
            start_time = time.time()
            
            # Create a temporary connector for direct API access
            from data_sources import DataSourceConnector
            connector = openfda_connector
            
            # Use a simple query that should always work
            url = "https://api.fda.gov/drug/label.json"
            params = {"search": "_exists_:openfda.brand_name", "limit": "1"}
            
            success, data = connector.handle_request(url, params=params)
            elapsed_time = time.time() - start_time
            
            if success and data and 'results' in data:
                openfda_status = {
                    'status': 'OK',
                    'response_time': f"{elapsed_time:.2f}s",
                    'message': "Connected successfully using direct API query"
                }
            else:
                openfda_status = {
                    'status': 'WARNING',
                    'response_time': f"{elapsed_time:.2f}s",
                    'message': "Connected but no data returned"
                }
        except Exception as e:
            openfda_status = {
                'status': 'ERROR',
                'message': str(e)
            }
    
    # Set the final status
    status['openfda'] = openfda_status or {
        'status': 'ERROR',
        'message': "Failed to connect to OpenFDA API after multiple attempts"
    }
    
    return status

# Example of how to use this module:
if __name__ == "__main__":
    # For testing this module directly
    test_results = test_data_sources()
    print("Data source connection test results:")
    print(json.dumps(test_results, indent=2))
    
    print("\nSearching for aspirin across sources...")
    results = search_drug_across_sources("aspirin")
    
    # Print summary of results
    for source, drugs in results.items():
        print(f"{source}: Found {len(drugs)} results")