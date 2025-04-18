from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from api.models.models import KnowledgeGraphStats, PathAnalysisResponse, RepurposingCandidate
from api.security.auth import get_current_active_user, User

# Import core functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
from utils import get_drug_by_name, get_disease_by_name, regenerate_knowledge_graph, search_candidates
from knowledge_graph import find_paths_between, compute_centrality_measures
from ai_analysis import analyze_repurposing_candidate

router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge graph"],
    responses={404: {"description": "Not found"}},
)


@router.get("/stats", response_model=KnowledgeGraphStats)
async def get_knowledge_graph_stats(current_user: User = Depends(get_current_active_user)):
    """
    Get statistics about the current knowledge graph.
    """
    g = st.session_state.graph
    
    # Count nodes by type
    drug_nodes = sum(1 for n, data in g.nodes(data=True) if data.get('type') == 'drug')
    disease_nodes = sum(1 for n, data in g.nodes(data=True) if data.get('type') == 'disease')
    
    # Count edges by type
    edge_types = {}
    for _, _, data in g.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    return KnowledgeGraphStats(
        total_nodes=g.number_of_nodes(),
        total_edges=g.number_of_edges(),
        drug_nodes=drug_nodes,
        disease_nodes=disease_nodes,
        edge_types=edge_types
    )


@router.post("/regenerate", response_model=KnowledgeGraphStats)
async def regenerate_graph(current_user: User = Depends(get_current_active_user)):
    """
    Regenerate the knowledge graph from current data.
    """
    regenerate_knowledge_graph()
    return await get_knowledge_graph_stats(current_user)


@router.get("/path", response_model=PathAnalysisResponse)
async def analyze_path(
    source: str, 
    target: str, 
    max_length: int = Query(3, ge=1, le=5),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze paths between two nodes in the knowledge graph.
    
    - **source**: Source node ID (drug ID or disease ID)
    - **target**: Target node ID (drug ID or disease ID)
    - **max_length**: Maximum path length to search
    """
    g = st.session_state.graph
    
    # Verify nodes exist
    if source not in g.nodes:
        raise HTTPException(status_code=404, detail=f"Source node '{source}' not found in knowledge graph")
    if target not in g.nodes:
        raise HTTPException(status_code=404, detail=f"Target node '{target}' not found in knowledge graph")
    
    # Try to find paths
    source_type = g.nodes[source].get('type')
    target_type = g.nodes[target].get('type')
    
    paths_result = find_paths_between(g, source_type, target_type, max_length)
    
    # Filter to only paths between our specific source and target
    filtered_paths = []
    for path_entry in paths_result:
        path = path_entry.get('path', [])
        if path and path[0] == source and path[-1] == target:
            filtered_paths.append(path_entry)
    
    if not filtered_paths:
        return PathAnalysisResponse(
            found=False,
            message=f"No path found between '{source}' and '{target}' within {max_length} steps"
        )
    
    return PathAnalysisResponse(
        found=True,
        paths=filtered_paths
    )


@router.get("/centrality", response_model=Dict[str, Dict[str, float]])
async def get_centrality_measures(
    node_type: Optional[str] = None,
    top_n: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get centrality measures for nodes in the knowledge graph.
    
    - **node_type**: Filter by node type ('drug' or 'disease')
    - **top_n**: Return only the top N nodes for each centrality measure
    """
    g = st.session_state.graph
    
    centrality_dict = compute_centrality_measures(g)
    
    # Filter by node type if specified
    if node_type:
        for measure in centrality_dict:
            centrality_dict[measure] = {node: value for node, value in centrality_dict[measure].items() 
                                      if g.nodes[node].get('type') == node_type}
    
    # Limit to top N for each measure
    result = {}
    for measure, values in centrality_dict.items():
        # Sort by value descending
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        result[measure] = dict(sorted_items[:top_n])
    
    return result


@router.get("/repurposing", response_model=List[RepurposingCandidate])
async def list_repurposing_candidates(
    drug_name: Optional[str] = None,
    disease_name: Optional[str] = None,
    min_confidence: int = Query(0, ge=0, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """
    List drug repurposing candidates, optionally filtered by drug, disease, or minimum confidence score.
    
    - **drug_name**: Filter by drug name
    - **disease_name**: Filter by disease name
    - **min_confidence**: Minimum confidence score (0-100)
    """
    candidates = search_candidates(drug_name, disease_name, min_confidence)
    return candidates


@router.get("/repurposing/{drug_name}/{disease_name}", response_model=dict)
async def analyze_candidate(
    drug_name: str,
    disease_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze a specific drug-disease repurposing candidate.
    
    - **drug_name**: Name of the drug
    - **disease_name**: Name of the disease
    """
    # Get drug and disease
    drug = get_drug_by_name(drug_name)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not found")
    
    disease = get_disease_by_name(disease_name)
    if not disease:
        raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found")
    
    # Analyze the candidate
    analysis = analyze_repurposing_candidate(drug, disease, st.session_state.graph)
    
    # Convert any non-serializable objects
    for k, v in analysis.items():
        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            analysis[k] = str(v)
    
    return analysis