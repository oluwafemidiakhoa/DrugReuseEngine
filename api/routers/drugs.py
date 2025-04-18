from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from api.models.models import Drug, DrugSearchResponse, PubMedArticle, ExtractedRelationship
from api.security.auth import get_current_active_user, User

# Import core functionality (these would be adapted to work with API)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_drug_by_name, get_drug_by_id, get_pubmed_data, add_drug
from knowledge_graph import get_drug_disease_relationships

router = APIRouter(
    prefix="/drugs",
    tags=["drugs"],
    responses={404: {"description": "Drug not found"}},
)


@router.get("/", response_model=List[Drug])
async def read_drugs(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_active_user)):
    """
    Get all drugs in the database.
    
    - **skip**: Skip the first N drugs
    - **limit**: Limit the number of drugs returned
    """
    # Access session state from the main application
    import streamlit as st
    return st.session_state.drugs[skip : skip + limit]


@router.get("/{drug_id}", response_model=DrugSearchResponse)
async def read_drug(drug_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Get drug by ID with all related information.
    
    - **drug_id**: The ID of the drug to retrieve
    """
    drug = get_drug_by_id(drug_id)
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    # Get relationships from knowledge graph
    import streamlit as st
    relationships = get_drug_disease_relationships(st.session_state.graph, drug_id=drug_id)
    
    # Split into treats and potential relationships
    treats_relationships = [r for r in relationships if r['type'] == 'treats']
    potential_relationships = [r for r in relationships if r['type'] == 'potential']
    
    return DrugSearchResponse(
        drug=drug,
        relationships=treats_relationships,
        potential_repurposing=potential_relationships
    )


@router.get("/name/{drug_name}", response_model=DrugSearchResponse)
async def search_drug_by_name(
    drug_name: str, 
    fuzzy_match: bool = False,
    current_user: User = Depends(get_current_active_user)
):
    """
    Search for a drug by name.
    
    - **drug_name**: The name of the drug to search for
    - **fuzzy_match**: If True, performs a fuzzy match instead of exact match
    """
    drug = get_drug_by_name(drug_name)
    
    if not drug and fuzzy_match:
        # Implement fuzzy matching
        import streamlit as st
        all_drugs = st.session_state.drugs
        
        # Simple fuzzy match - find drugs that contain the search string
        matches = [d for d in all_drugs if drug_name.lower() in d['name'].lower()]
        if matches:
            drug = matches[0]  # Take the first match
    
    if not drug:
        return DrugSearchResponse(
            message=f"Drug '{drug_name}' not found"
        )
    
    # Get relationships from knowledge graph
    import streamlit as st
    relationships = get_drug_disease_relationships(st.session_state.graph, drug_id=drug['id'])
    
    # Split into treats and potential relationships
    treats_relationships = [r for r in relationships if r['type'] == 'treats']
    potential_relationships = [r for r in relationships if r['type'] == 'potential']
    
    return DrugSearchResponse(
        drug=drug,
        relationships=treats_relationships,
        potential_repurposing=potential_relationships
    )


@router.post("/", response_model=Drug, status_code=201)
async def create_drug(drug: Drug, current_user: User = Depends(get_current_active_user)):
    """
    Add a new drug to the database.
    
    - **drug**: Drug details to add
    """
    # Convert to dict for the add_drug function
    drug_dict = drug.dict()
    
    # Add the drug
    drug_id, message = add_drug(drug_dict)
    
    if "already exists" in message:
        raise HTTPException(status_code=400, detail=message)
    
    # Return the created drug
    return get_drug_by_id(drug_id)


@router.get("/{drug_id}/pubmed", response_model=dict)
async def get_drug_pubmed_data(
    drug_id: str, 
    max_results: int = Query(20, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get PubMed data for a specific drug.
    
    - **drug_id**: The ID of the drug
    - **max_results**: Maximum number of results to return (limited to 50)
    """
    drug = get_drug_by_id(drug_id)
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    # Search PubMed for the drug
    articles, relationships = get_pubmed_data(drug['name'])
    
    # Limit results
    articles = articles[:max_results]
    relationships = relationships[:max_results]
    
    return {
        "drug": drug,
        "articles": articles,
        "relationships": relationships
    }