import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

def create_knowledge_graph(drugs, diseases, relationships):
    """
    Create a knowledge graph using NetworkX with advanced node and edge attributes
    for more sophisticated analysis
    """
    # Create a new directed graph to better represent relationship directionality
    G = nx.DiGraph()
    
    # Add drug nodes with enhanced metadata
    for drug in drugs:
        # Extract structured data from mechanism description for better analysis
        mechanism_keywords = extract_mechanism_keywords(drug.get('mechanism', ''))
        
        G.add_node(drug['id'], 
                  type='drug', 
                  name=drug['name'], 
                  description=drug['description'],
                  original_indication=drug['original_indication'],
                  mechanism=drug.get('mechanism', 'Unknown'),
                  mechanism_keywords=mechanism_keywords,
                  chemical_class=drug.get('chemical_class', 'Unknown'),
                  weight=1.0)  # Default node weight
    
    # Add disease nodes with enhanced metadata
    for disease in diseases:
        # Extract structured data from disease description
        pathways = extract_disease_pathways(disease.get('description', ''))
        
        G.add_node(disease['id'], 
                  type='disease', 
                  name=disease['name'], 
                  description=disease['description'],
                  category=disease['category'],
                  pathways=pathways,
                  systems=infer_body_systems(disease),
                  weight=1.0)  # Default node weight
    
    # Add edges (relationships) with enhanced metadata
    for rel in relationships:
        # Create bidirectional relationships but with type-specific attributes
        relationship_meta = extract_relationship_metadata(rel)
        
        # Add directed edge from source to target
        G.add_edge(rel['source'], rel['target'], 
                  type=rel['type'], 
                  confidence=rel['confidence'],
                  evidence_count=rel.get('evidence_count', 1),
                  weight=rel['confidence'],  # Weight edges by confidence for path algorithms
                  mechanism=relationship_meta.get('mechanism', ''),
                  references=relationship_meta.get('references', []))
        
        # For "treats" relationships, add a "treated_by" edge in reverse direction
        if rel['type'] == 'treats':
            G.add_edge(rel['target'], rel['source'], 
                      type='treated_by', 
                      confidence=rel['confidence'],
                      evidence_count=rel.get('evidence_count', 1),
                      weight=rel['confidence'],
                      mechanism=relationship_meta.get('mechanism', ''),
                      references=relationship_meta.get('references', []))
    
    # Add inferred relationships based on drug-disease similarities
    G = add_inferred_relationships(G, drugs, diseases)
    
    return G

def extract_mechanism_keywords(mechanism_text):
    """Extract key mechanism of action concepts from text description"""
    if not mechanism_text or mechanism_text == 'Unknown':
        return []
        
    # Simple keyword extraction - in a real system this would use NLP
    common_mechanisms = [
        "inhibitor", "agonist", "antagonist", "blocker", "modulator",
        "reuptake", "enzyme", "receptor", "channel", "transporter",
        "kinase", "phosphatase", "protease", "reductase", "oxidase"
    ]
    
    keywords = []
    for keyword in common_mechanisms:
        if keyword.lower() in mechanism_text.lower():
            keywords.append(keyword)
            
    return keywords

def extract_disease_pathways(description_text):
    """Extract potential biological pathways from disease description"""
    if not description_text:
        return []
        
    # Simple pathway keyword extraction - in a real system this would use NLP
    common_pathways = [
        "inflammatory", "immune", "metabolic", "signaling", "apoptosis",
        "cell cycle", "oxidative stress", "growth factor", "hormonal",
        "neuronal", "vascular", "mitochondrial", "genetic"
    ]
    
    pathways = []
    for pathway in common_pathways:
        if pathway.lower() in description_text.lower():
            pathways.append(pathway)
            
    return pathways

def infer_body_systems(disease):
    """Infer affected body systems from disease category and description"""
    category = disease.get('category', '').lower()
    description = disease.get('description', '').lower()
    
    systems = []
    
    # Map disease categories to body systems
    if 'cardiovascular' in category or 'heart' in description or 'vascular' in description:
        systems.append('cardiovascular')
    if 'neurological' in category or 'brain' in description or 'nerve' in description:
        systems.append('nervous')
    if 'respiratory' in category or 'lung' in description or 'airway' in description:
        systems.append('respiratory')
    if 'metabolic' in category or 'metabolism' in description:
        systems.append('metabolic')
    if 'autoimmune' in category or 'immune' in description:
        systems.append('immune')
    if 'cancer' in category or 'tumor' in description or 'oncology' in description:
        systems.append('multiple')
    
    return systems if systems else ['unknown']

def extract_relationship_metadata(relationship):
    """Extract and structure additional metadata from relationship"""
    metadata = {
        'mechanism': '',
        'references': []
    }
    
    # In a real implementation, this would parse relationship data
    # For now, we'll return placeholder metadata
    if 'mechanism' in relationship:
        metadata['mechanism'] = relationship['mechanism']
    
    if 'references' in relationship:
        metadata['references'] = relationship['references']
        
    return metadata

def add_inferred_relationships(G, drugs, diseases):
    """
    Add inferred relationships to the graph based on similarity analysis
    between drugs and diseases
    """
    # This would typically use more sophisticated algorithms
    # Here we'll implement a simple similarity-based inference
    
    # Find drugs that target similar mechanisms
    drug_by_mechanism = defaultdict(list)
    for drug_id in [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']:
        mechanism_keywords = G.nodes[drug_id].get('mechanism_keywords', [])
        for keyword in mechanism_keywords:
            drug_by_mechanism[keyword].append(drug_id)
    
    # Find diseases with similar pathways
    disease_by_pathway = defaultdict(list)
    for disease_id in [n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease']:
        pathways = G.nodes[disease_id].get('pathways', [])
        for pathway in pathways:
            disease_by_pathway[pathway].append(disease_id)
    
    # Add inferred relationships for drugs that share mechanisms
    for mechanism, similar_drugs in drug_by_mechanism.items():
        if len(similar_drugs) > 1:
            # For each pair of drugs with the same mechanism
            for i in range(len(similar_drugs)):
                for j in range(i+1, len(similar_drugs)):
                    drug1, drug2 = similar_drugs[i], similar_drugs[j]
                    
                    # Add similarity relationship if not already connected
                    if not G.has_edge(drug1, drug2):
                        G.add_edge(drug1, drug2, 
                                  type='similar_mechanism', 
                                  confidence=0.6,  # Default confidence for inferred relationship
                                  evidence_count=1,
                                  weight=0.6,
                                  mechanism=f"Both target {mechanism}")
    
    # Add inferred relationships for diseases that share pathways
    for pathway, similar_diseases in disease_by_pathway.items():
        if len(similar_diseases) > 1:
            # For each pair of diseases with the same pathway
            for i in range(len(similar_diseases)):
                for j in range(i+1, len(similar_diseases)):
                    disease1, disease2 = similar_diseases[i], similar_diseases[j]
                    
                    # Add similarity relationship if not already connected
                    if not G.has_edge(disease1, disease2):
                        G.add_edge(disease1, disease2, 
                                  type='similar_pathway', 
                                  confidence=0.6,  # Default confidence for inferred relationship
                                  evidence_count=1,
                                  weight=0.6,
                                  mechanism=f"Both involve {pathway} pathway")
    
    # Infer potential drug-disease relationships through transitivity
    for drug_id in [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']:
        # Find diseases this drug is known to treat
        treated_diseases = []
        for _, target in G.out_edges(drug_id):
            if G.nodes[target].get('type') == 'disease' and G.get_edge_data(drug_id, target).get('type') == 'treats':
                treated_diseases.append(target)
        
        # For each treated disease, find similar diseases
        for disease_id in treated_diseases:
            for _, similar_disease in G.out_edges(disease_id):
                if G.nodes[similar_disease].get('type') == 'disease' and G.get_edge_data(disease_id, similar_disease).get('type') == 'similar_pathway':
                    # If drug treats disease1, and disease1 is similar to disease2,
                    # then drug might potentially treat disease2
                    if not G.has_edge(drug_id, similar_disease):
                        G.add_edge(drug_id, similar_disease, 
                                  type='potential', 
                                  confidence=0.5,  # Lower confidence for inferred relationship
                                  evidence_count=1,
                                  weight=0.5,
                                  mechanism=f"Inferred via pathway similarity to {G.nodes[disease_id]['name']}")
    
    return G

def find_paths_between(G, source_type, target_type, max_length=3):
    """
    Find all paths between nodes of specific types with a maximum length
    """
    paths = []
    
    # Get all nodes of source_type and target_type
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == source_type]
    target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == target_type]
    
    # For each source-target pair, find all simple paths
    for source in source_nodes:
        for target in target_nodes:
            try:
                # Find all simple paths of length up to max_length
                for path in nx.all_simple_paths(G, source, target, cutoff=max_length):
                    if len(path) <= max_length + 1:  # +1 because the path includes both endpoints
                        # Get the node attributes for the path
                        node_names = [G.nodes[node]['name'] for node in path]
                        
                        # Get the edge attributes for the path
                        edge_types = []
                        edge_confidences = []
                        for i in range(len(path)-1):
                            edge_data = G.get_edge_data(path[i], path[i+1])
                            edge_types.append(edge_data['type'])
                            edge_confidences.append(edge_data['confidence'])
                        
                        # Append the path information
                        paths.append({
                            'path': path,
                            'node_names': node_names,
                            'edge_types': edge_types,
                            'edge_confidences': edge_confidences,
                            'avg_confidence': np.mean(edge_confidences) if edge_confidences else 0
                        })
            except nx.NetworkXNoPath:
                continue
    
    # Sort paths by average confidence
    paths.sort(key=lambda x: x['avg_confidence'], reverse=True)
    
    return paths

def get_drug_disease_relationships(G, drug_id=None, disease_id=None):
    """
    Get relationships between drugs and diseases
    If drug_id is provided, get all diseases related to that drug
    If disease_id is provided, get all drugs related to that disease
    If both are provided, get the specific relationship
    """
    relationships = []
    
    if drug_id and disease_id:
        # Get specific relationship
        if G.has_edge(drug_id, disease_id):
            edge_data = G.get_edge_data(drug_id, disease_id)
            drug_name = G.nodes[drug_id]['name']
            disease_name = G.nodes[disease_id]['name']
            relationships.append({
                'drug_id': drug_id,
                'drug_name': drug_name,
                'disease_id': disease_id,
                'disease_name': disease_name,
                'type': edge_data['type'],
                'confidence': edge_data['confidence']
            })
    elif drug_id:
        # Get all diseases related to this drug
        # Use edges() instead of out_edges() which only works with DiGraph
        for source, target, edge_data in G.edges(data=True):
            if source == drug_id:  # Only consider edges originating from this drug
                target_data = G.nodes[target]
                if target_data['type'] == 'disease':
                    relationships.append({
                        'drug_id': drug_id,
                        'drug_name': G.nodes[drug_id]['name'],
                        'disease_id': target,
                        'disease_name': target_data['name'],
                        'type': edge_data['type'],
                        'confidence': edge_data['confidence']
                    })
    elif disease_id:
        # Get all drugs related to this disease
        # NetworkX undirected graphs don't have in_edges, use edges instead
        for source, target, edge_data in G.edges(data=True):
            if target == disease_id and G.nodes[source]['type'] == 'drug':
                relationships.append({
                    'drug_id': source,
                    'drug_name': G.nodes[source]['name'],
                    'disease_id': disease_id,
                    'disease_name': G.nodes[disease_id]['name'],
                    'type': edge_data['type'],
                    'confidence': edge_data['confidence']
                })
            # Check the reverse direction as well for undirected graphs
            elif source == disease_id and G.nodes[target]['type'] == 'drug':
                relationships.append({
                    'drug_id': target,
                    'drug_name': G.nodes[target]['name'],
                    'disease_id': disease_id,
                    'disease_name': G.nodes[disease_id]['name'],
                    'type': edge_data['type'],
                    'confidence': edge_data['confidence']
                })
    else:
        # Get all drug-disease relationships
        for source, target, edge_data in G.edges(data=True):
            source_data = G.nodes[source]
            target_data = G.nodes[target]
            if source_data['type'] == 'drug' and target_data['type'] == 'disease':
                relationships.append({
                    'drug_id': source,
                    'drug_name': source_data['name'],
                    'disease_id': target,
                    'disease_name': target_data['name'],
                    'type': edge_data['type'],
                    'confidence': edge_data['confidence']
                })
    
    # Sort by confidence
    relationships.sort(key=lambda x: x['confidence'], reverse=True)
    
    return relationships

def compute_centrality_measures(G):
    """
    Compute various centrality measures for the knowledge graph
    """
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G)
    }
    
    # Try to compute eigenvector centrality safely
    try:
        # Get largest connected component for eigenvector centrality
        if nx.is_connected(G):
            centrality_measures['eigenvector'] = nx.eigenvector_centrality_numpy(G)
        else:
            # For disconnected graphs, compute on the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            eigen_centrality = nx.eigenvector_centrality_numpy(subgraph)
            
            # Map values back to original graph (nodes not in largest CC get 0)
            centrality_measures['eigenvector'] = {node: 0.0 for node in G.nodes()}
            for node, value in eigen_centrality.items():
                centrality_measures['eigenvector'][node] = value
    except Exception as e:
        # Fallback if eigenvector centrality fails
        print(f"Error computing eigenvector centrality: {str(e)}")
        centrality_measures['eigenvector'] = {node: 0.0 for node in G.nodes()}
    
    # Combine all measures into a single dataframe
    nodes_data = []
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        node_name = G.nodes[node]['name']
        
        node_data = {
            'id': node,
            'name': node_name,
            'type': node_type,
            'degree': centrality_measures['degree'][node],
            'betweenness': centrality_measures['betweenness'][node],
            'closeness': centrality_measures['closeness'][node],
            'eigenvector': centrality_measures['eigenvector'][node]
        }
        nodes_data.append(node_data)
    
    return pd.DataFrame(nodes_data)

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def generate_potential_repurposing_candidates(_G, min_confidence=0.4, page=1, per_page=50, max_candidates=200):
    """
    Generate potential drug repurposing candidates based on the knowledge graph
    with optimization for performance
    
    Parameters:
    - _G: NetworkX graph object
    - min_confidence: Minimum confidence threshold (0-1)
    - page: Page number for pagination
    - per_page: Number of candidates per page
    - max_candidates: Maximum total candidates to generate (limit for performance)
    
    Returns:
    - List of candidate dictionaries with pagination
    """
    # Create a key for storing in session state
    cache_key = "potential_candidates"
    
    # Check if we have already generated candidates in this session
    if cache_key in st.session_state and st.session_state[cache_key]:
        candidates = st.session_state[cache_key]
        
        # Return the requested page
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, len(candidates))
        return candidates[start_idx:end_idx]
    
    # Start with empty candidates list
    candidates = []
    
    # Get all drugs and diseases, but limit to most promising ones for performance
    drugs = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'drug']
    diseases = [n for n, attr in _G.nodes(data=True) if attr.get('type') == 'disease']
    
    # Precompute approved diseases for each drug to avoid repeated iterations
    approved_disease_map = {}
    for drug_id in drugs:
        approved_disease_map[drug_id] = []
    
    # Build the approved disease map in one pass
    for source, target, edge_data in _G.edges(data=True):
        if source in approved_disease_map and _G.nodes[target].get('type') == 'disease':
            if edge_data['type'] == 'treats' and edge_data['confidence'] >= 0.8:
                approved_disease_map[source].append(target)
    
    # Process direct relationships first (faster)
    for source, target, edge_data in _G.edges(data=True):
        if len(candidates) >= max_candidates:
            break  # Limit total candidates for performance
            
        # Filter to only drug->disease edges with 'potential' type
        if (_G.nodes[source].get('type') == 'drug' and 
            _G.nodes[target].get('type') == 'disease' and
            edge_data['type'] == 'potential' and 
            edge_data['confidence'] >= min_confidence and
            target not in approved_disease_map.get(source, [])):
            
            drug_name = _G.nodes[source]['name']
            disease_name = _G.nodes[target]['name']
            
            candidates.append({
                'drug_id': source,
                'drug_name': drug_name,
                'disease_id': target,
                'disease_name': disease_name,
                'confidence': edge_data['confidence'],
                'path_length': 1
            })
    
    # If we need more candidates, look for indirect relationships
    # (but only if we haven't reached our max candidates)
    if len(candidates) < max_candidates:
        # Limit the number of drug-disease pairs to consider for performance
        max_pairs_to_check = min(len(drugs) * len(diseases), 1000)
        pairs_checked = 0
        
        for drug_id in drugs:
            if len(candidates) >= max_candidates:
                break
                
            drug_data = _G.nodes[drug_id]
            drug_name = drug_data['name']
            approved_diseases = approved_disease_map.get(drug_id, [])
            
            for disease_id in diseases:
                if len(candidates) >= max_candidates or pairs_checked >= max_pairs_to_check:
                    break
                    
                pairs_checked += 1
                
                # Skip if disease is already approved for this drug
                if disease_id in approved_diseases:
                    continue
                
                # Skip if we already found a direct relationship
                if any(c['drug_id'] == drug_id and c['disease_id'] == disease_id for c in candidates):
                    continue
                
                disease_data = _G.nodes[disease_id]
                disease_name = disease_data['name']
                
                # Try to find an indirect relationship with limited search depth
                try:
                    # Use cutoff to limit search depth for performance
                    path = nx.shortest_path(_G, drug_id, disease_id, weight='weight')
                    if 2 <= len(path) <= 4:  # Paths of reasonable length
                        # Calculate average confidence along the path
                        confidences = []
                        for i in range(len(path)-1):
                            edge_data = _G.get_edge_data(path[i], path[i+1])
                            confidences.append(edge_data['confidence'])
                        
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence >= min_confidence:
                            candidates.append({
                                'drug_id': drug_id,
                                'drug_name': drug_name,
                                'disease_id': disease_id,
                                'disease_name': disease_name,
                                'confidence': avg_confidence,
                                'path_length': len(path) - 1
                            })
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
    
    # Sort by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Store in session state for future use
    st.session_state[cache_key] = candidates
    
    # Return the requested page
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(candidates))
    return candidates[start_idx:end_idx]
