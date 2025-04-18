"""
Cypher query interface for the Neo4j Graph Database integration.
This module provides tools for executing Cypher queries against a Neo4j database
and visualizing the results.
"""

import logging
import pandas as pd
import networkx as nx
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union

import plotly.graph_objects as go
import plotly.express as px

from neo4j_utils import driver
from caching import cached

logger = logging.getLogger(__name__)

def explain_cypher_query(query: str) -> str:
    """
    Explain a Cypher query in natural language
    
    Parameters:
    - query: Cypher query string
    
    Returns:
    - Explanation of the query in natural language
    """
    try:
        # Simple pattern matching for common Cypher patterns
        explanation = []
        
        # Check for MATCH statements
        match_patterns = re.findall(r'MATCH\s+\(([^)]+)\)(?:-\[([^]]*)\]->)?(?:\([^)]+\))?', query, re.IGNORECASE)
        for node, rel in match_patterns:
            if rel:
                explanation.append(f"Find {node.split(':')[0]} nodes with relationship {rel.split(':')[0] if ':' in rel else 'any'} to other nodes")
            else:
                node_type = node.split(':')[1].strip() if ':' in node else "any type"
                explanation.append(f"Find {node.split(':')[0]} nodes of type {node_type}")
        
        # Check for WHERE clauses
        where_clauses = re.findall(r'WHERE\s+([^{]+?)(RETURN|WITH|ORDER|LIMIT|$)', query, re.IGNORECASE)
        if where_clauses:
            for clause, _ in where_clauses:
                explanation.append(f"Filter results where {clause.strip()}")
        
        # Check for RETURN statements
        return_clauses = re.findall(r'RETURN\s+(.+?)($|\s+ORDER|\s+LIMIT)', query, re.IGNORECASE)
        if return_clauses:
            for clause, _ in return_clauses:
                explanation.append(f"Return the following data: {clause.strip()}")
        
        # Check for ORDER BY clauses
        order_clauses = re.findall(r'ORDER BY\s+(.+?)($|\s+LIMIT)', query, re.IGNORECASE)
        if order_clauses:
            for clause, _ in order_clauses:
                explanation.append(f"Order results by {clause.strip()}")
        
        # Check for LIMIT clauses
        limit_clauses = re.findall(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_clauses:
            explanation.append(f"Limit results to {limit_clauses[0]} records")
        
        if explanation:
            return "This query will:\n- " + "\n- ".join(explanation)
        else:
            return "This appears to be a Cypher query for Neo4j graph database, but I couldn't parse the specific operations."
            
    except Exception as e:
        logger.exception(f"Error explaining Cypher query: {str(e)}")
        return "Could not explain the query due to an error."

def suggest_common_queries(current_query: str = "") -> List[Dict[str, str]]:
    """
    Suggest common Cypher queries for biomedical knowledge graphs
    
    Parameters:
    - current_query: Current query string (to avoid suggesting duplicates)
    
    Returns:
    - List of suggested query strings
    """
    suggestions = [
        {
            "name": "Get all drugs",
            "query": "MATCH (d:Drug) RETURN d.id AS id, d.name AS name LIMIT 25",
            "description": "Retrieve basic information about drug nodes in the graph."
        },
        {
            "name": "Get all diseases",
            "query": "MATCH (d:Disease) RETURN d.id AS id, d.name AS name LIMIT 25",
            "description": "Retrieve basic information about disease nodes in the graph."
        },
        {
            "name": "Find treatments for a disease",
            "query": "MATCH (d:Drug)-[r:TREATS]->(dis:Disease) WHERE dis.name CONTAINS 'cancer' RETURN d.name AS drug, dis.name AS disease",
            "description": "Find drugs that are known to treat a specific disease."
        },
        {
            "name": "Find drugs targeting specific genes",
            "query": "MATCH (d:Drug)-[:TARGETS]->(g:Gene) WHERE g.name IN ['EGFR', 'BRAF', 'KRAS'] RETURN d.name AS drug, g.name AS gene",
            "description": "Find drugs that target specific genes of interest."
        },
        {
            "name": "Find common drug targets",
            "query": "MATCH (d1:Drug)-[:TARGETS]->(g:Gene)<-[:TARGETS]-(d2:Drug) WHERE d1.id <> d2.id RETURN d1.name AS drug1, d2.name AS drug2, g.name AS gene, COUNT(g) AS common_targets ORDER BY common_targets DESC LIMIT 20",
            "description": "Find pairs of drugs that share common gene targets."
        },
        {
            "name": "Find potential repurposing candidates",
            "query": "MATCH (d:Drug)-[:TARGETS]->(g:Gene)<-[:ASSOCIATED_WITH]-(dis:Disease) WHERE NOT (d)-[:TREATS]->(dis) RETURN d.name AS drug, dis.name AS disease, COLLECT(g.name) AS genes, COUNT(g) AS gene_count ORDER BY gene_count DESC LIMIT 20",
            "description": "Find potential drug repurposing candidates based on shared gene targets."
        },
        {
            "name": "Community analysis",
            "query": "MATCH (n) WHERE EXISTS(n.community_id) WITH n.community_id AS community, COLLECT(n.name) AS nodes, COUNT(*) AS size RETURN community, size, nodes[0..5] AS examples ORDER BY size DESC LIMIT 10",
            "description": "Analyze communities detected in the graph structure."
        },
        {
            "name": "Path between drug and disease",
            "query": "MATCH path = shortestPath((d:Drug {name: 'Metformin'})-[*..5]-(dis:Disease {name: 'Type 2 Diabetes'})) RETURN [node IN nodes(path) | CASE WHEN node:Drug THEN {id: node.id, name: node.name, type: 'Drug'} WHEN node:Disease THEN {id: node.id, name: node.name, type: 'Disease'} WHEN node:Gene THEN {id: node.id, name: node.name, type: 'Gene'} ELSE {id: node.id, name: node.name, type: 'Other'} END] AS path_nodes, [rel IN relationships(path) | type(rel)] AS path_rels",
            "description": "Find the shortest path between a drug and a disease in the graph."
        }
    ]
    
    # Filter out any suggestions that are too similar to current_query
    if current_query:
        return [s for s in suggestions if s["query"].lower().strip() != current_query.lower().strip()]
    
    return suggestions

def validate_cypher_query(query: str) -> Tuple[bool, str]:
    """
    Validate a Cypher query for syntax and security
    
    Parameters:
    - query: Cypher query string
    
    Returns:
    - Tuple (is_valid, message)
    """
    if not query.strip():
        return False, "Query is empty"
    
    # Check for dangerous operations (DELETE, REMOVE, etc.) that could modify the graph
    dangerous_ops = ['CREATE ', 'DELETE ', 'REMOVE ', 'SET ', 'MERGE ', 'DETACH ']
    for op in dangerous_ops:
        if op.lower() in query.lower():
            return False, f"Query contains potentially dangerous operation: {op.strip()}"
    
    # Basic syntax checks
    # Check for unbalanced parentheses
    if query.count('(') != query.count(')'):
        return False, "Unbalanced parentheses in query"
    
    # Check for unbalanced square brackets
    if query.count('[') != query.count(']'):
        return False, "Unbalanced square brackets in query"
    
    # Check for unbalanced curly braces
    if query.count('{') != query.count('}'):
        return False, "Unbalanced curly braces in query"
    
    # Check for missing semicolon at the end
    if not query.strip().endswith(';') and ';' not in query:
        return True, "Warning: Query is missing a semicolon at the end, but will be executed anyway."
    
    return True, "Query syntax appears valid"

@cached(max_age_seconds=3600, key_prefix="cypher_query")
def execute_cypher_query(query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[bool, Union[List[Dict[str, Any]], None], str]:
    """
    Execute a Cypher query against the Neo4j database
    
    Parameters:
    - query: Cypher query string
    - params: Dictionary of query parameters
    
    Returns:
    - Tuple (success, result, message)
    """
    if params is None:
        params = {}
    
    # Validate the query first
    is_valid, validation_message = validate_cypher_query(query)
    if not is_valid:
        return False, None, validation_message
    
    try:
        # Use global driver instance from neo4j_utils
        if not driver:
            return False, None, "No Neo4j connection available"
        
        # Add semicolon if missing
        if not query.strip().endswith(';'):
            query = query.strip() + ';'
        
        with driver.session() as session:
            result = session.run(query, **params)
            records = [record.data() for record in result]
            
            if not records:
                return True, [], "Query executed successfully, but returned no records."
            
            return True, records, f"Query executed successfully. {len(records)} records returned."
            
    except Exception as e:
        logger.exception(f"Error executing Cypher query: {str(e)}")
        return False, None, f"Error executing query: {str(e)}"

def visualize_query_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create visualizations for query results
    
    Parameters:
    - results: Dictionary with query results
    
    Returns:
    - Dictionary of Plotly figure objects
    """
    if not results:
        return {}
    
    visualizations = {}
    df = pd.DataFrame(results)
    
    # Create a network visualization if the results look like pairs of nodes
    if len(df.columns) >= 2 and len(df) > 0:
        try:
            visualizations['network'] = create_graph_visualization(df)
        except Exception as e:
            logger.exception(f"Error creating network visualization: {str(e)}")
    
    # Create a bar chart if there are numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        try:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            visualizations['bar'] = px.bar(
                df,
                x=cat_col,
                y=num_col,
                title=f"{num_col} by {cat_col}",
                template="plotly_white"
            )
        except Exception as e:
            logger.exception(f"Error creating bar chart: {str(e)}")
    
    # Create a pie chart if there is a column that looks like categories
    if len(df.columns) >= 2 and len(df) <= 10:
        try:
            label_col = df.columns[0]
            value_col = df.columns[1] if df.columns[1] in numeric_cols else None
            
            if value_col:
                visualizations['pie'] = px.pie(
                    df,
                    names=label_col,
                    values=value_col,
                    title=f"Distribution of {value_col} by {label_col}"
                )
        except Exception as e:
            logger.exception(f"Error creating pie chart: {str(e)}")
    
    return visualizations

def create_graph_visualization(df: pd.DataFrame) -> go.Figure:
    """
    Create a network visualization from query results
    
    Parameters:
    - df: DataFrame with query results
    
    Returns:
    - Plotly figure object
    """
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    # Try to identify source and target columns
    cols = list(df.columns)
    source_col, target_col = None, None
    
    # Look for columns that might represent nodes
    node_candidates = []
    for col in cols:
        if col.lower().endswith('name') or col.lower().endswith('id') or col.lower() in ['source', 'target', 'from', 'to']:
            node_candidates.append(col)
    
    # If we have at least two node candidates, use them as source and target
    if len(node_candidates) >= 2:
        source_col, target_col = node_candidates[0], node_candidates[1]
    else:
        # Otherwise, use the first two columns
        if len(cols) >= 2:
            source_col, target_col = cols[0], cols[1]
        else:
            raise ValueError("DataFrame must have at least two columns to create a graph visualization")
    
    # Add nodes and edges to the graph
    for _, row in df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        
        # Skip self-loops
        if source == target:
            continue
        
        # Add nodes if they don't exist
        if not G.has_node(source):
            G.add_node(source)
        if not G.has_node(target):
            G.add_node(target)
        
        # Add edge
        G.add_edge(source, target)
    
    # Use NetworkX's spring layout for node positions
    pos = nx.spring_layout(G)
    
    # Create lists for node positions
    node_x = []
    node_y = []
    node_text = []
    
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])
        node_text.append(str(node))
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=15,
            color='skyblue',
            line=dict(width=2, color='black')
        )
    )
    
    # Create lists for edge positions
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Graph Visualization of Query Results',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    
    return fig

def format_cypher_query(query: str) -> str:
    """
    Format a Cypher query for better readability
    
    Parameters:
    - query: Cypher query string
    
    Returns:
    - Formatted query string
    """
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Add line breaks after common clauses
    clauses = ['MATCH', 'WHERE', 'WITH', 'RETURN', 'ORDER BY', 'LIMIT', 'CREATE', 'MERGE', 'DELETE', 'REMOVE', 'SET']
    
    for clause in clauses:
        # Replace the clause if it appears as a whole word, preserving case
        pattern = r'(\s+)(' + clause + r')\b'
        query = re.sub(pattern, r'\n\2', query, flags=re.IGNORECASE)
    
    # Indent lines after the first
    lines = query.split('\n')
    for i in range(1, len(lines)):
        lines[i] = '  ' + lines[i].strip()
    
    return '\n'.join(lines)

def generate_cypher_from_natural_language(query: str) -> str:
    """
    Generate a Cypher query from a natural language description
    
    Parameters:
    - query: Natural language query string
    
    Returns:
    - Suggested Cypher query string
    """
    # Look for key entities and relationships in the query
    drug_pattern = r'drug(?:s)?\s+(?:called|named)?\s+["\']?([a-zA-Z0-9\s,]+)["\']?'
    disease_pattern = r'disease(?:s)?\s+(?:called|named)?\s+["\']?([a-zA-Z0-9\s,]+)["\']?'
    
    cypher_query = ""
    
    # Drug search
    if "all drugs" in query.lower():
        cypher_query = "MATCH (d:Drug) RETURN d.id AS id, d.name AS name LIMIT 25"
    
    # Disease search
    elif "all diseases" in query.lower():
        cypher_query = "MATCH (d:Disease) RETURN d.id AS id, d.name AS name LIMIT 25"
    
    # Drug-Disease relationship
    elif "treats" in query.lower() or "treatment" in query.lower():
        # Try to extract specific drug or disease names
        drug_match = re.search(drug_pattern, query, re.IGNORECASE)
        disease_match = re.search(disease_pattern, query, re.IGNORECASE)
        
        if drug_match and disease_match:
            drug_name = drug_match.group(1).strip()
            disease_name = disease_match.group(1).strip()
            cypher_query = f"MATCH (d:Drug)-[r:TREATS]->(dis:Disease) WHERE d.name CONTAINS '{drug_name}' AND dis.name CONTAINS '{disease_name}' RETURN d.name AS drug, dis.name AS disease"
        elif drug_match:
            drug_name = drug_match.group(1).strip()
            cypher_query = f"MATCH (d:Drug)-[r:TREATS]->(dis:Disease) WHERE d.name CONTAINS '{drug_name}' RETURN d.name AS drug, dis.name AS disease"
        elif disease_match:
            disease_name = disease_match.group(1).strip()
            cypher_query = f"MATCH (d:Drug)-[r:TREATS]->(dis:Disease) WHERE dis.name CONTAINS '{disease_name}' RETURN d.name AS drug, dis.name AS disease"
        else:
            cypher_query = "MATCH (d:Drug)-[r:TREATS]->(dis:Disease) RETURN d.name AS drug, dis.name AS disease LIMIT 25"
    
    # Drug targets
    elif "target" in query.lower() or "targets" in query.lower():
        drug_match = re.search(drug_pattern, query, re.IGNORECASE)
        
        if drug_match:
            drug_name = drug_match.group(1).strip()
            cypher_query = f"MATCH (d:Drug)-[:TARGETS]->(g:Gene) WHERE d.name CONTAINS '{drug_name}' RETURN d.name AS drug, g.name AS gene"
        else:
            cypher_query = "MATCH (d:Drug)-[:TARGETS]->(g:Gene) RETURN d.name AS drug, g.name AS gene LIMIT 25"
    
    # Common targets between drugs
    elif "common" in query.lower() and "target" in query.lower():
        cypher_query = "MATCH (d1:Drug)-[:TARGETS]->(g:Gene)<-[:TARGETS]-(d2:Drug) WHERE d1.id <> d2.id RETURN d1.name AS drug1, d2.name AS drug2, COLLECT(g.name) AS genes, COUNT(g) AS common_targets ORDER BY common_targets DESC LIMIT 20"
    
    # Repurposing candidates
    elif "repurpos" in query.lower() or "new use" in query.lower() or "candidate" in query.lower():
        cypher_query = "MATCH (d:Drug)-[:TARGETS]->(g:Gene)<-[:ASSOCIATED_WITH]-(dis:Disease) WHERE NOT (d)-[:TREATS]->(dis) RETURN d.name AS drug, dis.name AS disease, COLLECT(g.name) AS genes, COUNT(g) AS gene_count ORDER BY gene_count DESC LIMIT 20"
    
    # Community analysis
    elif "communit" in query.lower() or "cluster" in query.lower():
        cypher_query = "MATCH (n) WHERE EXISTS(n.community_id) WITH n.community_id AS community, COUNT(*) AS size RETURN community, size ORDER BY size DESC LIMIT 10"
    
    # Path between entities
    elif "path" in query.lower() or "connection" in query.lower() or "between" in query.lower():
        drug_match = re.search(drug_pattern, query, re.IGNORECASE)
        disease_match = re.search(disease_pattern, query, re.IGNORECASE)
        
        if drug_match and disease_match:
            drug_name = drug_match.group(1).strip()
            disease_name = disease_match.group(1).strip()
            cypher_query = f"MATCH path = shortestPath((d:Drug {{name: '{drug_name}'}})-[*..5]-(dis:Disease {{name: '{disease_name}'}})) RETURN [n IN nodes(path) | n.name] AS path_nodes, [r IN relationships(path) | type(r)] AS path_rels"
        else:
            cypher_query = "MATCH path = shortestPath((d:Drug)-[*..5]-(dis:Disease)) WHERE d.name CONTAINS 'Metformin' AND dis.name CONTAINS 'Type 2 Diabetes' RETURN [n IN nodes(path) | n.name] AS path_nodes, [r IN relationships(path) | type(r)] AS path_rels LIMIT 1"
    
    # Default for generic or unrecognized queries
    if not cypher_query:
        cypher_query = "MATCH (n) RETURN n.name AS name, labels(n)[0] AS type LIMIT 25"
    
    return cypher_query

def create_graph_from_results(results: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Create a NetworkX graph from query results
    
    Parameters:
    - results: List of dictionaries with query results
    
    Returns:
    - NetworkX graph object
    """
    G = nx.DiGraph()
    
    if not results:
        return G
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Try to identify source and target columns
    cols = list(df.columns)
    source_col, target_col = None, None
    
    # Look for columns that might represent nodes
    node_candidates = []
    for col in cols:
        if col.lower().endswith('name') or col.lower().endswith('id') or col.lower() in ['source', 'target', 'from', 'to']:
            node_candidates.append(col)
    
    # If we have at least two node candidates, use them as source and target
    if len(node_candidates) >= 2:
        source_col, target_col = node_candidates[0], node_candidates[1]
    else:
        # Otherwise, use the first two columns
        if len(cols) >= 2:
            source_col, target_col = cols[0], cols[1]
        else:
            # If there's only one column, create nodes but no edges
            for _, row in df.iterrows():
                G.add_node(str(row[cols[0]]))
            return G
    
    # Add nodes and edges to the graph
    for _, row in df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        
        # Skip self-loops
        if source == target:
            continue
        
        # Add nodes if they don't exist
        if not G.has_node(source):
            G.add_node(source)
        if not G.has_node(target):
            G.add_node(target)
        
        # Add edge
        G.add_edge(source, target)
    
    return G