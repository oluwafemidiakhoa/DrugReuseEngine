"""
Neo4j Graph Database Integration for the Drug Repurposing Engine.

This module provides utilities for connecting to a Neo4j graph database,
creating and managing graph data models, and performing advanced graph queries
for drug repurposing.

Requirements:
- neo4j Python driver
- py2neo for higher-level graph operations
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid

# Neo4j drivers
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship, Subgraph
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default connection parameters
DEFAULT_URI = "neo4j+s://9615a24a.databases.neo4j.io"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "1OiaKn9UryX4RRzF3tT4A09_A-NW9o3Stfl_H9rQTW4"

# Get Neo4j connection parameters from environment variables or use defaults
NEO4J_URI = os.getenv("NEO4J_URI", DEFAULT_URI)
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", DEFAULT_USERNAME)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD)

# Flag to check if Neo4j is available
NEO4J_AVAILABLE = False

# Global driver instance
driver = None

# Global py2neo Graph instance
graph = None

def initialize_neo4j():
    """
    Initialize the Neo4j connection and check availability
    
    Returns:
        bool: True if Neo4j is available, False otherwise
    """
    global NEO4J_AVAILABLE, driver, graph
    
    try:
        # Initialize the Neo4j driver with connection pooling for better performance
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            max_connection_lifetime=30 * 60,  # 30 minutes max connection lifetime
            max_connection_pool_size=50,      # Increase connection pool size
            connection_acquisition_timeout=2.0 # Faster timeout for better UX
        )
        
        # Verify connection with a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            test_value = result.single()["test"]
            
            if test_value == 1:
                logger.info("Neo4j connection successful")
                NEO4J_AVAILABLE = True
                
                # For Neo4j Aura database or Neo4j protocol, properly convert to bolt URL
                import re
                
                # Prepare a proper py2neo connection URI
                py2neo_uri = NEO4J_URI
                secure = False
                
                # Handle neo4j+s:// protocol
                if NEO4J_URI.startswith('neo4j+s://'):
                    match = re.search(r'neo4j\+s://([^:]+)', NEO4J_URI)
                    if match:
                        hostname = match.group(1)
                        py2neo_uri = f"bolt+s://{hostname}:7687"
                        secure = True
                # Handle neo4j:// protocol 
                elif NEO4J_URI.startswith('neo4j://'):
                    match = re.search(r'neo4j://([^:]+)', NEO4J_URI)
                    if match:
                        hostname = match.group(1)
                        py2neo_uri = f"bolt://{hostname}:7687"
                
                logger.info(f"Converting Neo4j URI to py2neo format: {py2neo_uri}")
                
                try:
                    # Try to connect with py2neo using the converted URI
                    # Add connection pool configuration for better performance
                    graph = Graph(
                        py2neo_uri, 
                        auth=(NEO4J_USERNAME, NEO4J_PASSWORD), 
                        secure=secure,
                        name="drug-repurposing-pool"  # Named connection pool
                    )
                    logger.info("py2neo Graph connection successful")
                except Exception as e:
                    logger.warning(f"py2neo Graph connection failed: {str(e)}")
                    
                    # Direct connection method for migration fallback
                    try:
                        # Initialize migration data structures even without py2neo
                        # Only execute this in development mode or when explicitly needed
                        # to avoid slowing down startup
                        logger.info("Skipping immediate data structure initialization for faster loading")
                        # We'll initialize data structures on demand when needed
                        NEO4J_AVAILABLE = True
                    except Exception as inner_e:
                        logger.warning(f"Failed to initialize fallback graph structures: {str(inner_e)}")
            else:
                logger.warning("Neo4j connection test returned unexpected result")
                NEO4J_AVAILABLE = False
    
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {str(e)}")
        NEO4J_AVAILABLE = False
    
    return NEO4J_AVAILABLE

def initialize_graph_data_structures():
    """
    Initialize data structures for graph operations without py2neo
    Used as a fallback when py2neo connection fails
    """
    global NEO4J_AVAILABLE
    
    # Create sample data for demonstration when Neo4j is not available
    # This allows the frontend to show graph visualizations
    try:
        # Use raw Cypher through the driver API to create nodes and relationships
        with driver.session() as session:
            # Create indexes and constraints
            try:
                # Neo4j 4.x syntax
                session.run("""
                    CREATE CONSTRAINT drug_id IF NOT EXISTS ON (d:Drug) ASSERT d.id IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT disease_id IF NOT EXISTS ON (d:Disease) ASSERT d.id IS UNIQUE
                """)
                session.run("""
                    CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)
                """)
                session.run("""
                    CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)
                """)
                session.run("""
                    CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)
                """)
            except Exception as e:
                # Try Neo4j 5.x syntax if 4.x fails
                logger.warning(f"Neo4j 4.x constraint syntax failed, trying Neo4j 5.x syntax: {str(e)}")
                try:
                    session.run("""
                        CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE
                    """)
                    session.run("""
                        CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE
                    """)
                    # Neo4j 5.x index syntax is the same
                    session.run("""
                        CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)
                    """)
                    session.run("""
                        CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)
                    """)
                    session.run("""
                        CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)
                    """)
                except Exception as e2:
                    logger.warning(f"Neo4j 5.x constraint syntax also failed: {str(e2)}")
                    # Continue without constraints as this is just demo data
            
            # Add some initial drug nodes
            session.run("""
                MERGE (d:Drug {id: 'D001', name: 'Metformin', type: 'small molecule'})
                SET d.mechanism = 'AMPK activation', d.approved = true, d.description = 'First-line medication for type 2 diabetes'
            """)
            session.run("""
                MERGE (d:Drug {id: 'D002', name: 'Aspirin', type: 'small molecule'})
                SET d.mechanism = 'Cyclooxygenase inhibition', d.approved = true, d.description = 'Anti-inflammatory, analgesic, and antipyretic drug'
            """)
            session.run("""
                MERGE (d:Drug {id: 'D003', name: 'Imatinib', type: 'small molecule'})
                SET d.mechanism = 'Tyrosine kinase inhibition', d.approved = true, d.description = 'Used to treat chronic myeloid leukemia and GI stromal tumors'
            """)
            
            # Add some disease nodes
            session.run("""
                MERGE (d:Disease {id: 'DIS001', name: 'Type 2 Diabetes', category: 'metabolic'})
                SET d.description = 'Chronic condition affecting the way the body metabolizes glucose'
            """)
            session.run("""
                MERGE (d:Disease {id: 'DIS002', name: 'Breast Cancer', category: 'oncology'})
                SET d.description = 'Cancer that forms in the cells of the breasts'
            """)
            session.run("""
                MERGE (d:Disease {id: 'DIS003', name: 'Chronic Myeloid Leukemia', category: 'oncology'})
                SET d.description = 'Cancer of the white blood cells characterized by increased growth of myeloid cells'
            """)
            
            # Add some gene nodes
            session.run("""
                MERGE (g:Gene {id: 'G001', symbol: 'ABL1'})
                SET g.name = 'ABL Proto-Oncogene 1, Non-Receptor Tyrosine Kinase'
            """)
            session.run("""
                MERGE (g:Gene {id: 'G002', symbol: 'PRKAA1'})
                SET g.name = 'Protein Kinase AMP-Activated Catalytic Subunit Alpha 1'
            """)
            session.run("""
                MERGE (g:Gene {id: 'G003', symbol: 'PTGS2'})
                SET g.name = 'Prostaglandin-Endoperoxide Synthase 2'
            """)
            
            # Create relationships
            session.run("""
                MATCH (d:Drug {id: 'D001'}), (dis:Disease {id: 'DIS001'})
                MERGE (d)-[r:TREATS]->(dis)
            """)
            session.run("""
                MATCH (d:Drug {id: 'D003'}), (dis:Disease {id: 'DIS003'})
                MERGE (d)-[r:TREATS]->(dis)
            """)
            session.run("""
                MATCH (d:Drug {id: 'D001'}), (g:Gene {id: 'G002'})
                MERGE (d)-[r:TARGETS]->(g)
            """)
            session.run("""
                MATCH (d:Drug {id: 'D003'}), (g:Gene {id: 'G001'})
                MERGE (d)-[r:TARGETS]->(g)
            """)
            session.run("""
                MATCH (dis:Disease {id: 'DIS001'}), (g:Gene {id: 'G002'})
                MERGE (dis)-[r:ASSOCIATED_WITH]->(g)
            """)
            session.run("""
                MATCH (dis:Disease {id: 'DIS003'}), (g:Gene {id: 'G001'})
                MERGE (dis)-[r:ASSOCIATED_WITH]->(g)
            """)
            session.run("""
                MATCH (d:Drug {id: 'D001'}), (dis:Disease {id: 'DIS002'})
                MERGE (d)-[r:POTENTIAL_TREATMENT]->(dis)
                SET r.confidence = 0.67, r.evidence = 'Preclinical data shows potential efficacy'
            """)
            
        logger.info("Created initial graph data structures successfully")
        
    except Exception as e:
        logger.error(f"Error initializing graph data structures: {str(e)}")
        return False
    
    return True

def close_connection():
    """Close the Neo4j connection if it exists"""
    global driver
    
    if driver:
        driver.close()
        logger.info("Neo4j connection closed")

def create_constraints_and_indexes():
    """
    Create constraints and indexes for the graph database
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create constraints and indexes")
        return False
    
    try:
        with driver.session() as session:
            try:
                # Neo4j 4.x syntax
                # Create constraints - each node type should have a unique ID
                session.run("CREATE CONSTRAINT drug_id IF NOT EXISTS ON (d:Drug) ASSERT d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT disease_id IF NOT EXISTS ON (d:Disease) ASSERT d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT gene_id IF NOT EXISTS ON (g:Gene) ASSERT g.id IS UNIQUE")
                session.run("CREATE CONSTRAINT pathway_id IF NOT EXISTS ON (p:Pathway) ASSERT p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT protein_id IF NOT EXISTS ON (p:Protein) ASSERT p.id IS UNIQUE")
                
                # Create indexes for properties we'll query frequently
                session.run("CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)")
                session.run("CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)")
                session.run("CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)")
                
                logger.info("Neo4j constraints and indexes created with Neo4j 4.x syntax")
                return True
            except Exception as e1:
                # Try Neo4j 5.x syntax if 4.x fails
                logger.warning(f"Neo4j 4.x constraint syntax failed, trying Neo4j 5.x syntax: {str(e1)}")
                try:
                    # Create constraints - using REQUIRE syntax for Neo4j 5.x
                    session.run("CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (p:Pathway) REQUIRE p.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE")
                    
                    # Create indexes for properties we'll query frequently
                    session.run("CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)")
                    session.run("CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)")
                    session.run("CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)")
                    
                    logger.info("Neo4j constraints and indexes created with Neo4j 5.x syntax")
                    return True
                except Exception as e2:
                    logger.warning(f"Neo4j 5.x constraint syntax also failed: {str(e2)}")
                    # Continue anyway, as this might be a Neo4j version compatibility issue
                    return False
    except Exception as e:
        logger.error(f"Error creating Neo4j constraints and indexes: {str(e)}")
        return False

def create_drug_node(drug_data: Dict[str, Any]) -> bool:
    """
    Create a Drug node in the Neo4j database
    
    Args:
        drug_data (Dict[str, Any]): Dictionary containing drug data
            Required keys: 'id', 'name'
            Optional keys: any other drug properties
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create Drug node")
        return False
    
    # Verify required fields
    if 'id' not in drug_data or 'name' not in drug_data:
        logger.error("Drug data must contain 'id' and 'name' fields")
        return False
    
    try:
        # Create drug node using py2neo for simplicity
        drug_node = Node("Drug", 
                        id=drug_data['id'], 
                        name=drug_data['name'])
        
        # Add all other properties
        for key, value in drug_data.items():
            if key not in ['id', 'name']:
                # Convert complex objects to JSON strings
                if isinstance(value, (dict, list)):
                    drug_node[key] = json.dumps(value)
                else:
                    drug_node[key] = value
        
        # Create or merge the node
        graph.merge(drug_node, "Drug", "id")
        logger.info(f"Drug node created: {drug_data['name']} (ID: {drug_data['id']})")
        return True
    
    except Exception as e:
        logger.error(f"Error creating Drug node: {str(e)}")
        return False

def create_disease_node(disease_data: Dict[str, Any]) -> bool:
    """
    Create a Disease node in the Neo4j database
    
    Args:
        disease_data (Dict[str, Any]): Dictionary containing disease data
            Required keys: 'id', 'name'
            Optional keys: any other disease properties
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create Disease node")
        return False
    
    # Verify required fields
    if 'id' not in disease_data or 'name' not in disease_data:
        logger.error("Disease data must contain 'id' and 'name' fields")
        return False
    
    try:
        # Create disease node using py2neo for simplicity
        disease_node = Node("Disease", 
                           id=disease_data['id'], 
                           name=disease_data['name'])
        
        # Add all other properties
        for key, value in disease_data.items():
            if key not in ['id', 'name']:
                # Convert complex objects to JSON strings
                if isinstance(value, (dict, list)):
                    disease_node[key] = json.dumps(value)
                else:
                    disease_node[key] = value
        
        # Create or merge the node
        graph.merge(disease_node, "Disease", "id")
        logger.info(f"Disease node created: {disease_data['name']} (ID: {disease_data['id']})")
        return True
    
    except Exception as e:
        logger.error(f"Error creating Disease node: {str(e)}")
        return False

def create_gene_node(gene_data: Dict[str, Any]) -> bool:
    """
    Create a Gene node in the Neo4j database
    
    Args:
        gene_data (Dict[str, Any]): Dictionary containing gene data
            Required keys: 'id', 'symbol'
            Optional keys: any other gene properties
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create Gene node")
        return False
    
    # Verify required fields
    if 'id' not in gene_data or 'symbol' not in gene_data:
        logger.error("Gene data must contain 'id' and 'symbol' fields")
        return False
    
    try:
        # Create gene node using py2neo for simplicity
        gene_node = Node("Gene", 
                        id=gene_data['id'], 
                        symbol=gene_data['symbol'])
        
        # Add all other properties
        for key, value in gene_data.items():
            if key not in ['id', 'symbol']:
                # Convert complex objects to JSON strings
                if isinstance(value, (dict, list)):
                    gene_node[key] = json.dumps(value)
                else:
                    gene_node[key] = value
        
        # Create or merge the node
        graph.merge(gene_node, "Gene", "id")
        logger.info(f"Gene node created: {gene_data['symbol']} (ID: {gene_data['id']})")
        return True
    
    except Exception as e:
        logger.error(f"Error creating Gene node: {str(e)}")
        return False

def create_relationship(from_id: str, to_id: str, relationship_type: str, 
                       properties: Dict[str, Any] = None) -> bool:
    """
    Create a relationship between two nodes in the Neo4j database
    
    Args:
        from_id (str): ID of the source node
        to_id (str): ID of the target node
        relationship_type (str): Type of the relationship (e.g., 'TREATS', 'TARGETS')
        properties (Dict[str, Any], optional): Properties for the relationship
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create relationship")
        return False
    
    if not properties:
        properties = {}
    
    try:
        # Create Cypher query to create or merge the relationship - optimized to avoid Cartesian product
        cypher = """
        MATCH (a:Drug|Disease|Gene|Protein|Pathway) 
        WHERE a.id = $from_id
        MATCH (b:Drug|Disease|Gene|Protein|Pathway) 
        WHERE b.id = $to_id
        MERGE (a)-[r:%s]->(b)
        SET r += $properties
        RETURN type(r) as rel_type
        """ % relationship_type
        
        with driver.session() as session:
            result = session.run(cypher, from_id=from_id, to_id=to_id, properties=properties)
            record = result.single()
            
            if record and record["rel_type"] == relationship_type:
                logger.info(f"Relationship created: ({from_id})-[{relationship_type}]->({to_id})")
                return True
            else:
                logger.warning(f"Failed to create relationship: ({from_id})-[{relationship_type}]->({to_id})")
                return False
    
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        return False

def import_drug_disease_relationships(relationships: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Import drug-disease relationships into the Neo4j database
    
    Args:
        relationships (List[Dict[str, Any]]): List of relationship dictionaries
            Each dictionary should have: 'source', 'target', 'type', and optional properties
    
    Returns:
        Tuple[int, int]: (success_count, error_count)
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot import relationships")
        return (0, len(relationships))
    
    success_count = 0
    error_count = 0
    
    for rel in relationships:
        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type', '').upper()
        
        if not source_id or not target_id or not rel_type:
            logger.warning(f"Skipping relationship with missing data: {rel}")
            error_count += 1
            continue
        
        # Prepare properties - exclude source, target, and type
        properties = {k: v for k, v in rel.items() if k not in ['source', 'target', 'type']}
        
        # Create the relationship
        if create_relationship(source_id, target_id, rel_type, properties):
            success_count += 1
        else:
            error_count += 1
    
    logger.info(f"Imported {success_count} relationships with {error_count} errors")
    return (success_count, error_count)

def migrate_from_postgres(drugs: List[Dict[str, Any]], diseases: List[Dict[str, Any]], 
                         relationships: List[Dict[str, Any]]) -> bool:
    """
    Migrate data from PostgreSQL to Neo4j
    
    Args:
        drugs (List[Dict[str, Any]]): List of drug dictionaries
        diseases (List[Dict[str, Any]]): List of disease dictionaries
        relationships (List[Dict[str, Any]]): List of relationship dictionaries
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot migrate data")
        return False
    
    try:
        # Create constraints and indexes
        create_constraints_and_indexes()
        
        # Import drugs
        drug_success = 0
        for drug in drugs:
            if create_drug_node(drug):
                drug_success += 1
        
        # Import diseases
        disease_success = 0
        for disease in diseases:
            if create_disease_node(disease):
                disease_success += 1
        
        # Import relationships
        rel_success, rel_errors = import_drug_disease_relationships(relationships)
        
        logger.info(f"Migration complete: {drug_success}/{len(drugs)} drugs, {disease_success}/{len(diseases)} diseases, {rel_success}/{len(relationships)} relationships")
        return True
    
    except Exception as e:
        logger.error(f"Error migrating data to Neo4j: {str(e)}")
        return False

def create_repurposing_candidate(candidate_data: Dict[str, Any]) -> bool:
    """
    Create a repurposing candidate in the Neo4j database
    
    Args:
        candidate_data (Dict[str, Any]): Candidate data dictionary
            Required keys: 'drug_id', 'disease_id', 'confidence_score'
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create repurposing candidate")
        return False
    
    # Verify required fields
    if 'drug_id' not in candidate_data or 'disease_id' not in candidate_data:
        logger.error("Candidate data must contain 'drug_id' and 'disease_id' fields")
        return False
    
    try:
        # Get properties for the relationship
        properties = {
            'confidence_score': candidate_data.get('confidence_score', 0),
            'mechanism': candidate_data.get('mechanism', 'Unknown'),
            'source': candidate_data.get('source', 'user'),
            'evidence': candidate_data.get('evidence', '{}'),
            'status': candidate_data.get('status', 'proposed'),
            'created_at': time.time()
        }
        
        # Create the POTENTIAL_TREATMENT relationship
        return create_relationship(
            candidate_data['drug_id'], 
            candidate_data['disease_id'], 
            'POTENTIAL_TREATMENT', 
            properties
        )
    
    except Exception as e:
        logger.error(f"Error creating repurposing candidate: {str(e)}")
        return False

def get_drug_disease_paths(drug_id: str, disease_id: str, max_length: int = 4) -> List[Dict[str, Any]]:
    """
    Find all paths between a drug and a disease in the Neo4j database
    
    Args:
        drug_id (str): ID of the drug node
        disease_id (str): ID of the disease node
        max_length (int): Maximum path length to consider
    
    Returns:
        List[Dict[str, Any]]: List of paths with metadata
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot find paths")
        return []
    
    try:
        # Cypher query to find paths
        cypher = """
        MATCH path = (d:Drug {id: $drug_id})-[*1..%d]-(dis:Disease {id: $disease_id})
        RETURN path
        LIMIT 10
        """ % max_length
        
        with driver.session() as session:
            result = session.run(cypher, drug_id=drug_id, disease_id=disease_id)
            
            paths = []
            for record in result:
                path = record["path"]
                
                # Extract nodes and relationships
                nodes = [dict(n) for n in path.nodes]
                relationships = [dict(r) for r in path.relationships]
                
                # Simplify by extracting just the key information
                simple_path = {
                    'nodes': [{'id': n['id'], 'name': n.get('name', n.get('symbol', 'Unknown')), 'labels': list(n.labels)} for n in path.nodes],
                    'relationships': [{'type': type(r).__name__, 'properties': dict(r)} for r in path.relationships],
                    'length': len(path.relationships)
                }
                
                paths.append(simple_path)
            
            return paths
    
    except Exception as e:
        logger.error(f"Error finding paths: {str(e)}")
        return []

def find_similar_drugs(drug_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Find drugs similar to the given drug based on shared targets, pathways, or mechanisms
    
    Args:
        drug_id (str): ID of the drug node
        limit (int): Maximum number of similar drugs to return
    
    Returns:
        List[Dict[str, Any]]: List of similar drugs with similarity scores
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot find similar drugs")
        return []
    
    try:
        # Cypher query to find similar drugs based on shared targets
        cypher = """
        MATCH (d1:Drug {id: $drug_id})-[r1]->(t:Gene|Protein|Pathway)<-[r2]-(d2:Drug)
        WHERE d1 <> d2
        WITH d2, count(t) AS shared_targets, collect(t.name) AS target_names
        RETURN d2.id AS drug_id, d2.name AS drug_name, 
               shared_targets AS similarity_score,
               target_names AS shared_entities
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        with driver.session() as session:
            result = session.run(cypher, drug_id=drug_id, limit=limit)
            
            similar_drugs = []
            for record in result:
                similar_drug = {
                    'drug_id': record["drug_id"],
                    'drug_name': record["drug_name"],
                    'similarity_score': record["similarity_score"],
                    'shared_entities': record["shared_entities"]
                }
                similar_drugs.append(similar_drug)
            
            return similar_drugs
    
    except Exception as e:
        logger.error(f"Error finding similar drugs: {str(e)}")
        return []

def find_repurposing_opportunities(disease_id: str, min_confidence: float = 0.5, 
                                 limit: int = 20) -> List[Dict[str, Any]]:
    """
    Find potential drug repurposing opportunities for a disease
    
    Args:
        disease_id (str): ID of the disease node
        min_confidence (float): Minimum confidence score (0.0-1.0)
        limit (int): Maximum number of opportunities to return
    
    Returns:
        List[Dict[str, Any]]: List of repurposing opportunities
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot find repurposing opportunities")
        return []
    
    try:
        # Cypher query to find repurposing opportunities
        # This looks for drugs that target genes associated with the disease
        # but don't already treat the disease
        cypher = """
        MATCH (dis:Disease {id: $disease_id})-[:ASSOCIATED_WITH]->(g:Gene)<-[:TARGETS]-(d:Drug)
        WHERE NOT (d)-[:TREATS]->(dis)
        WITH d, dis, count(g) AS common_targets, collect(g.symbol) AS target_genes
        WITH d, dis, common_targets, target_genes, 
             CASE WHEN common_targets > 5 THEN 0.9
                  WHEN common_targets > 3 THEN 0.7
                  WHEN common_targets > 1 THEN 0.5
                  ELSE 0.3
             END AS confidence_score
        WHERE confidence_score >= $min_confidence
        RETURN d.id AS drug_id, d.name AS drug_name, 
               dis.id AS disease_id, dis.name AS disease_name,
               confidence_score, common_targets, target_genes
        ORDER BY confidence_score DESC, common_targets DESC
        LIMIT $limit
        """
        
        with driver.session() as session:
            result = session.run(cypher, disease_id=disease_id, min_confidence=min_confidence, limit=limit)
            
            opportunities = []
            for record in result:
                opportunity = {
                    'drug_id': record["drug_id"],
                    'drug_name': record["drug_name"],
                    'disease_id': record["disease_id"],
                    'disease_name': record["disease_name"],
                    'confidence_score': record["confidence_score"],
                    'common_targets': record["common_targets"],
                    'target_genes': record["target_genes"]
                }
                opportunities.append(opportunity)
            
            return opportunities
    
    except Exception as e:
        logger.error(f"Error finding repurposing opportunities: {str(e)}")
        return []

def calculate_centrality_measures() -> Dict[str, List[Dict[str, Any]]]:
    """
    Calculate various centrality measures for nodes in the graph
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with node types and their centrality measures
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot calculate centrality measures")
        return {}
    
    try:
        # Calculate centrality measures for Drugs
        drug_cypher = """
        CALL gds.degree.stream('graph')
        YIELD nodeId, score
        MATCH (d:Drug) WHERE id(d) = nodeId
        RETURN d.id AS id, d.name AS name, score AS degree_centrality
        ORDER BY degree_centrality DESC
        LIMIT 10
        """
        
        # Calculate centrality measures for Diseases
        disease_cypher = """
        CALL gds.degree.stream('graph')
        YIELD nodeId, score
        MATCH (d:Disease) WHERE id(d) = nodeId
        RETURN d.id AS id, d.name AS name, score AS degree_centrality
        ORDER BY degree_centrality DESC
        LIMIT 10
        """
        
        # Calculate centrality measures for Genes
        gene_cypher = """
        CALL gds.degree.stream('graph')
        YIELD nodeId, score
        MATCH (g:Gene) WHERE id(g) = nodeId
        RETURN g.id AS id, g.symbol AS name, score AS degree_centrality
        ORDER BY degree_centrality DESC
        LIMIT 10
        """
        
        # First, we need to create a graph projection for analysis
        projection_cypher = """
        CALL gds.graph.project(
            'graph',
            ['Drug', 'Disease', 'Gene', 'Protein', 'Pathway'],
            {
                TREATS: {orientation: 'UNDIRECTED'},
                TARGETS: {orientation: 'UNDIRECTED'},
                ASSOCIATED_WITH: {orientation: 'UNDIRECTED'},
                POTENTIAL_TREATMENT: {orientation: 'UNDIRECTED'},
                PART_OF: {orientation: 'UNDIRECTED'}
            }
        )
        """
        
        with driver.session() as session:
            # Create graph projection
            try:
                session.run(projection_cypher)
            except Exception as e:
                # If the graph already exists, this will fail, which is fine
                pass
            
            # Calculate centrality for each node type
            results = {}
            
            # Drugs
            try:
                drug_result = session.run(drug_cypher)
                results['drugs'] = [dict(record) for record in drug_result]
            except Exception as e:
                logger.warning(f"Error calculating drug centrality: {str(e)}")
                results['drugs'] = []
            
            # Diseases
            try:
                disease_result = session.run(disease_cypher)
                results['diseases'] = [dict(record) for record in disease_result]
            except Exception as e:
                logger.warning(f"Error calculating disease centrality: {str(e)}")
                results['diseases'] = []
            
            # Genes
            try:
                gene_result = session.run(gene_cypher)
                results['genes'] = [dict(record) for record in gene_result]
            except Exception as e:
                logger.warning(f"Error calculating gene centrality: {str(e)}")
                results['genes'] = []
            
            return results
    
    except Exception as e:
        logger.error(f"Error calculating centrality measures: {str(e)}")
        return {}

def get_graph_statistics() -> Dict[str, Any]:
    """
    Get statistics about the graph database
    
    Returns:
        Dict[str, Any]: Dictionary with various statistics
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot get graph statistics")
        return {}
    
    try:
        # Cypher query to get basic statistics
        cypher = """
        MATCH (n)
        RETURN 
            count(n) AS total_nodes,
            sum(CASE WHEN 'Drug' IN labels(n) THEN 1 ELSE 0 END) AS drug_count,
            sum(CASE WHEN 'Disease' IN labels(n) THEN 1 ELSE 0 END) AS disease_count,
            sum(CASE WHEN 'Gene' IN labels(n) THEN 1 ELSE 0 END) AS gene_count,
            sum(CASE WHEN 'Protein' IN labels(n) THEN 1 ELSE 0 END) AS protein_count,
            sum(CASE WHEN 'Pathway' IN labels(n) THEN 1 ELSE 0 END) AS pathway_count
        """
        
        # Cypher query to get relationship statistics
        rel_cypher = """
        MATCH ()-[r]->()
        RETURN 
            count(r) AS total_relationships,
            sum(CASE WHEN type(r) = 'TREATS' THEN 1 ELSE 0 END) AS treats_count,
            sum(CASE WHEN type(r) = 'TARGETS' THEN 1 ELSE 0 END) AS targets_count,
            sum(CASE WHEN type(r) = 'ASSOCIATED_WITH' THEN 1 ELSE 0 END) AS associated_with_count,
            sum(CASE WHEN type(r) = 'POTENTIAL_TREATMENT' THEN 1 ELSE 0 END) AS potential_treatment_count
        """
        
        with driver.session() as session:
            # Get node statistics
            node_result = session.run(cypher)
            node_stats = dict(node_result.single())
            
            # Get relationship statistics
            rel_result = session.run(rel_cypher)
            rel_stats = dict(rel_result.single())
            
            # Combine the results
            stats = {**node_stats, **rel_stats}
            return stats
    
    except Exception as e:
        logger.error(f"Error getting graph statistics: {str(e)}")
        return {}

def is_connected() -> bool:
    """
    Check if the Neo4j connection is available
    
    Returns:
        bool: True if connected, False otherwise
    """
    return NEO4J_AVAILABLE

def execute_cypher(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query against the Neo4j database
    
    Args:
        query (str): Cypher query to execute
        params (Dict[str, Any], optional): Parameters for the query
    
    Returns:
        List[Dict[str, Any]]: Query results as a list of dictionaries
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot execute Cypher query")
        return []
    
    if params is None:
        params = {}
    
    try:
        with driver.session() as session:
            result = session.run(query, **params)
            records = [dict(record) for record in result]
            return records
    
    except Exception as e:
        logger.error(f"Error executing Cypher query: {str(e)}")
        return []

def initialize_demo_data() -> bool:
    """
    Initialize demo data for the Neo4j database
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot initialize demo data")
        return False
    
    try:
        # Create constraints and indexes
        create_constraints_and_indexes()
        
        # Create some demo drugs
        drugs = [
            {"id": "D001", "name": "Metformin", "mechanism": "Inhibits hepatic gluconeogenesis"},
            {"id": "D002", "name": "Aspirin", "mechanism": "Inhibits COX enzymes"},
            {"id": "D003", "name": "Ibuprofen", "mechanism": "Inhibits COX enzymes"},
            {"id": "D004", "name": "Atorvastatin", "mechanism": "HMG-CoA reductase inhibitor"},
            {"id": "D005", "name": "Losartan", "mechanism": "Angiotensin II receptor antagonist"}
        ]
        
        # Create some demo diseases
        diseases = [
            {"id": "DIS001", "name": "Type 2 Diabetes", "description": "Metabolic disorder characterized by high blood sugar"},
            {"id": "DIS002", "name": "Hypertension", "description": "High blood pressure"},
            {"id": "DIS003", "name": "Rheumatoid Arthritis", "description": "Autoimmune inflammatory disease"},
            {"id": "DIS004", "name": "Alzheimer's Disease", "description": "Neurodegenerative disease"},
            {"id": "DIS005", "name": "Hyperlipidemia", "description": "High levels of lipids in the blood"}
        ]
        
        # Create some demo genes
        genes = [
            {"id": "G001", "symbol": "PTGS2", "name": "Prostaglandin-Endoperoxide Synthase 2"},
            {"id": "G002", "symbol": "HMGCR", "name": "HMG-CoA Reductase"},
            {"id": "G003", "symbol": "AGTR1", "name": "Angiotensin II Receptor Type 1"},
            {"id": "G004", "symbol": "APP", "name": "Amyloid Beta Precursor Protein"},
            {"id": "G005", "symbol": "PPARG", "name": "Peroxisome Proliferator Activated Receptor Gamma"}
        ]
        
        # Import drugs
        for drug in drugs:
            create_drug_node(drug)
        
        # Import diseases
        for disease in diseases:
            create_disease_node(disease)
        
        # Import genes
        for gene in genes:
            create_gene_node(gene)
        
        # Create relationships
        relationships = [
            # Drug-Disease (TREATS)
            {"source": "D001", "target": "DIS001", "type": "TREATS", "confidence": 0.9},
            {"source": "D002", "target": "DIS003", "type": "TREATS", "confidence": 0.8},
            {"source": "D003", "target": "DIS003", "type": "TREATS", "confidence": 0.7},
            {"source": "D004", "target": "DIS005", "type": "TREATS", "confidence": 0.9},
            {"source": "D005", "target": "DIS002", "type": "TREATS", "confidence": 0.9},
            
            # Drug-Gene (TARGETS)
            {"source": "D001", "target": "G005", "type": "TARGETS", "confidence": 0.7},
            {"source": "D002", "target": "G001", "type": "TARGETS", "confidence": 0.9},
            {"source": "D003", "target": "G001", "type": "TARGETS", "confidence": 0.8},
            {"source": "D004", "target": "G002", "type": "TARGETS", "confidence": 0.9},
            {"source": "D005", "target": "G003", "type": "TARGETS", "confidence": 0.9},
            
            # Disease-Gene (ASSOCIATED_WITH)
            {"source": "DIS001", "target": "G005", "type": "ASSOCIATED_WITH", "confidence": 0.8},
            {"source": "DIS002", "target": "G003", "type": "ASSOCIATED_WITH", "confidence": 0.9},
            {"source": "DIS003", "target": "G001", "type": "ASSOCIATED_WITH", "confidence": 0.8},
            {"source": "DIS004", "target": "G004", "type": "ASSOCIATED_WITH", "confidence": 0.9},
            {"source": "DIS005", "target": "G002", "type": "ASSOCIATED_WITH", "confidence": 0.9},
            
            # Potential repurposing candidates
            {"source": "D001", "target": "DIS004", "type": "POTENTIAL_TREATMENT", "confidence": 0.6},
            {"source": "D002", "target": "DIS002", "type": "POTENTIAL_TREATMENT", "confidence": 0.5}
        ]
        
        # Import relationships
        import_drug_disease_relationships(relationships)
        
        logger.info("Neo4j demo data initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing Neo4j demo data: {str(e)}")
        return False

def search_entities(query: str, entity_types: List[str] = None, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for entities in the Neo4j database
    
    Args:
        query (str): Search query string
        entity_types (List[str], optional): List of entity types to search for
                                         (e.g., ["Drug", "Disease"])
        limit (int, optional): Maximum number of results per entity type
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with entity types and their search results
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot search entities")
        return {}
    
    if not entity_types:
        entity_types = ["Drug", "Disease", "Gene", "Protein", "Pathway"]
    
    try:
        results = {}
        
        for entity_type in entity_types:
            # Cypher query to search for entities by name
            cypher = f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.name) CONTAINS toLower($query) OR
                  toLower(n.id) CONTAINS toLower($query)
                  {" OR toLower(n.symbol) CONTAINS toLower($query)" if entity_type == "Gene" else ""}
            RETURN n.id AS id, n.name AS name,
                  {" n.symbol AS symbol," if entity_type == "Gene" else ""}
                  labels(n) AS labels
            LIMIT $limit
            """
            
            with driver.session() as session:
                result = session.run(cypher, query=query, limit=limit)
                results[entity_type.lower() + "s"] = [dict(record) for record in result]
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        return {}

def get_graph_for_visualization(limit: int = 200, node_types: List[str] = None, relationship_types: List[str] = None) -> nx.Graph:
    """
    Create a NetworkX graph from the Neo4j database for visualization purposes
    
    Args:
        limit (int): Maximum number of nodes to include (default: 200)
        node_types (List[str], optional): Filter by node types (e.g., ['Drug', 'Disease'])
        relationship_types (List[str], optional): Filter by relationship types (e.g., ['TREATS', 'TARGETS'])
    
    Returns:
        nx.Graph: NetworkX graph object
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot create graph")
        return nx.Graph()
    
    try:
        # Import caching utilities
        from caching import Cache
        
        # Generate a cache key for this specific combination of parameters
        cache_key = f"graph_viz_{limit}_{str(node_types)}_{str(relationship_types)}"
        
        # Try to get from cache first to improve performance
        cached_graph = Cache.get(cache_key, max_age=3600)  # Cache for 1 hour
        if cached_graph is not None:
            logger.info(f"Using cached graph with {cached_graph.number_of_nodes()} nodes and {cached_graph.number_of_edges()} edges")
            return cached_graph
        
        # If not in cache, generate the graph
        G = nx.Graph()
        
        # Build optimized Cypher query with more efficient filters and pagination
        # This improves performance by reducing data transfer between Neo4j and Python
        
        # Build node filters more efficiently
        node_filter = ""
        if node_types:
            node_filter = " WHERE " + " OR ".join([f"n:{t}" for t in node_types])
        
        rel_filter = ""
        if relationship_types:
            rel_filter = " WHERE " + " OR ".join([f"type(r) = '{t}'" for t in relationship_types])
        
        # Get nodes with a more optimized query that directly fetches essential properties
        # Rather than fetching ALL properties, specify only what we need
        cypher_nodes = f"""
        MATCH (n)
        {node_filter}
        RETURN n.id as id, n.name as name, labels(n)[0] as label, 
               n.description as description, n.category as category
        LIMIT {limit}
        """
        
        # Execute query for nodes with connection pooling
        with driver.session() as session:
            result = session.run(cypher_nodes)
            
            # Add nodes to the graph
            for record in result:
                node_id = record["id"]
                
                if not node_id:
                    continue
                
                # Create properties dictionary with only the properties we need
                properties = {
                    'id': node_id,
                    'name': record["name"],
                    'type': record["label"],
                    'description': record["description"],
                    'category': record["category"]
                }
                
                # Clean up None values for better compatibility
                properties = {k: v for k, v in properties.items() if v is not None}
                
                # Add node to graph
                G.add_node(node_id, **properties)
        
        # Get relationships with optimized query that reuses existing connections
        # and directly builds the relationship structure we need
        cypher_rels = f"""
        MATCH (a)-[r]->(b)
        {rel_filter}
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        RETURN a.id as source, b.id as target, type(r) as type, 
               r.confidence as confidence, r.evidence as evidence
        LIMIT {limit * 2}
        """
        
        # Execute query for relationships with batch processing
        with driver.session() as session:
            result = session.run(cypher_rels)
            
            # Process in batches for better performance
            batch_size = 100
            records = []
            batch_count = 0
            
            # Add edges to the graph - more efficiently using batches
            while True:
                # Get a batch of records
                batch = result.fetch(batch_size)
                if not batch:
                    break
                    
                for record in batch:
                    source = record["source"]
                    target = record["target"]
                    rel_type = record["type"]
                    
                    # Skip if source or target is missing
                    if not source or not target:
                        continue
                    
                    # Skip if source or target node not in graph
                    if source not in G.nodes or target not in G.nodes:
                        continue
                    
                    # Create properties dictionary with only what we need
                    properties = {
                        'type': rel_type,
                        'confidence': record["confidence"],
                        'evidence': record["evidence"]
                    }
                    
                    # Clean up None values
                    properties = {k: v for k, v in properties.items() if v is not None}
                    
                    # Add edge to graph
                    G.add_edge(source, target, **properties)
                
                batch_count += 1
                logger.debug(f"Processed relationship batch {batch_count}")
        
        logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Save to cache for future use
        Cache.set(cache_key, G)
        
        return G
    
    except Exception as e:
        logger.error(f"Error creating graph: {str(e)}")
        return nx.Graph()

def get_neighbors(node_id: str, relationship_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get neighbors of a node in the Neo4j database
    
    Args:
        node_id (str): ID of the node
        relationship_types (List[str], optional): List of relationship types to consider
        limit (int, optional): Maximum number of neighbors to return
    
    Returns:
        List[Dict[str, Any]]: List of neighbor nodes with relationship information
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j is not available, cannot get neighbors")
        return []
    
    relationship_clause = ""
    if relationship_types:
        relationship_clause = "WHERE type(r) IN $rel_types"
    
    try:
        # Cypher query to get neighbors
        cypher = f"""
        MATCH (n)-[r]-(m)
        WHERE n.id = $node_id
        {relationship_clause}
        RETURN m.id AS id, m.name AS name, labels(m) AS labels,
               type(r) AS relationship_type, properties(r) AS relationship_properties
        LIMIT $limit
        """
        
        with driver.session() as session:
            result = session.run(cypher, node_id=node_id, rel_types=relationship_types, limit=limit)
            
            neighbors = []
            for record in result:
                neighbor = {
                    'id': record["id"],
                    'name': record["name"],
                    'labels': record["labels"],
                    'relationship': {
                        'type': record["relationship_type"],
                        'properties': record["relationship_properties"]
                    }
                }
                neighbors.append(neighbor)
            
            return neighbors
    
    except Exception as e:
        logger.error(f"Error getting neighbors: {str(e)}")
        return []

# Initialize Neo4j connection when the module is imported
initialize_neo4j()

# Register cleanup function
import atexit
atexit.register(close_connection)