from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from api.models.models import Disease, DiseaseSearchResponse, PubMedArticle, ExtractedRelationship
from api.security.auth import get_current_active_user, User

# Import core functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_disease_by_name, get_disease_by_id, get_pubmed_data, add_disease
from knowledge_graph import get_drug_disease_relationships

router = APIRouter(
    prefix="/diseases",
    tags=["diseases"],
    responses={404: {"description": "Disease not found"}},
)


@router.get("/", response_model=List[Disease])
async def read_diseases(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_active_user)):
    """
    Get all diseases in the database.
    
    - **skip**: Skip the first N diseases
    - **limit**: Limit the number of diseases returned
    """
    # Access session state from the main application
    import streamlit as st
    return st.session_state.diseases[skip : skip + limit]


@router.get("/{disease_id}", response_model=DiseaseSearchResponse)
async def read_disease(disease_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Get disease by ID with all related information.
    
    - **disease_id**: The ID of the disease to retrieve
    """
    disease = get_disease_by_id(disease_id)
    if not disease:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    # Get relationships from knowledge graph
    import streamlit as st
    relationships = get_drug_disease_relationships(st.session_state.graph, disease_id=disease_id)
    
    # Split into treats and potential relationships
    treats_relationships = [r for r in relationships if r['type'] == 'treats']
    potential_relationships = [r for r in relationships if r['type'] == 'potential']
    
    return DiseaseSearchResponse(
        disease=disease,
        relationships=treats_relationships,
        potential_repurposing=potential_relationships
    )


@router.get("/name/{disease_name}", response_model=DiseaseSearchResponse)
async def search_disease_by_name(
    disease_name: str, 
    fuzzy_match: bool = False,
    current_user: User = Depends(get_current_active_user)
):
    """
    Search for a disease by name.
    
    - **disease_name**: The name of the disease to search for
    - **fuzzy_match**: If True, performs a fuzzy match instead of exact match
    """
    disease = get_disease_by_name(disease_name)
    
    if not disease and fuzzy_match:
        # Implement fuzzy matching
        import streamlit as st
        all_diseases = st.session_state.diseases
        
        # Simple fuzzy match - find diseases that contain the search string
        matches = [d for d in all_diseases if disease_name.lower() in d['name'].lower()]
        if matches:
            disease = matches[0]  # Take the first match
    
    if not disease:
        return DiseaseSearchResponse(
            message=f"Disease '{disease_name}' not found"
        )
    
    # Get relationships from knowledge graph
    import streamlit as st
    relationships = get_drug_disease_relationships(st.session_state.graph, disease_id=disease['id'])
    
    # Split into treats and potential relationships
    treats_relationships = [r for r in relationships if r['type'] == 'treats']
    potential_relationships = [r for r in relationships if r['type'] == 'potential']
    
    return DiseaseSearchResponse(
        disease=disease,
        relationships=treats_relationships,
        potential_repurposing=potential_relationships
    )


@router.post("/", response_model=Disease, status_code=201)
async def create_disease(disease: Disease, current_user: User = Depends(get_current_active_user)):
    """
    Add a new disease to the database.
    
    - **disease**: Disease details to add
    """
    # Convert to dict for the add_disease function
    disease_dict = disease.dict()
    
    # Add the disease
    disease_id, message = add_disease(disease_dict)
    
    if "already exists" in message:
        raise HTTPException(status_code=400, detail=message)
    
    # Return the created disease
    return get_disease_by_id(disease_id)


@router.get("/{disease_id}/pubmed", response_model=dict)
async def get_disease_pubmed_data(
    disease_id: str, 
    max_results: int = Query(20, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get PubMed data for a specific disease.
    
    - **disease_id**: The ID of the disease
    - **max_results**: Maximum number of results to return (limited to 50)
    """
    disease = get_disease_by_id(disease_id)
    if not disease:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    # Search PubMed for the disease
    articles, relationships = get_pubmed_data(disease['name'])
    
    # Limit results
    articles = articles[:max_results]
    relationships = relationships[:max_results]
    
    return {
        "disease": disease,
        "articles": articles,
        "relationships": relationships
    }