"""
Database utility functions for connecting to and working with the PostgreSQL database.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

def get_db_connection():
    """
    Create and return a database connection using environment variables.
    
    Returns:
        connection: A PostgreSQL database connection
    """
    try:
        # Get connection parameters from environment variables
        connection = psycopg2.connect(
            host=os.environ.get("PGHOST"),
            port=os.environ.get("PGPORT"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD"),
            database=os.environ.get("PGDATABASE")
        )
        # Set autocommit mode
        connection.autocommit = True
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def execute_query(query, params=None, fetch=False):
    """
    Execute a SQL query with optional parameters.
    
    Args:
        query (str): The SQL query to execute
        params (tuple|dict, optional): Parameters for the query
        fetch (bool): If True, fetch and return results
        
    Returns:
        If fetch is True, returns query results as a list of dicts.
        Otherwise, returns None.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
    except Exception as e:
        print(f"Error executing query: {e}")
        raise
    finally:
        if conn:
            conn.close()
            
def get_drugs(limit=100, offset=0):
    """
    Get a list of drugs from the database.
    
    Args:
        limit (int): Maximum number of drugs to return
        offset (int): Offset for pagination
        
    Returns:
        list: List of drug dictionaries
    """
    query = """
    SELECT * FROM drugs
    ORDER BY name
    LIMIT %s OFFSET %s
    """
    return execute_query(query, (limit, offset), fetch=True)

def get_drug_by_id(drug_id):
    """
    Get a drug by its ID.
    
    Args:
        drug_id (str): The ID of the drug
        
    Returns:
        dict: Drug data or None if not found
    """
    query = "SELECT * FROM drugs WHERE id = %s"
    results = execute_query(query, (drug_id,), fetch=True)
    return results[0] if results else None

def get_drug_by_name(drug_name):
    """
    Get a drug by its name.
    
    Args:
        drug_name (str): The name of the drug
        
    Returns:
        dict: Drug data or None if not found
    """
    query = "SELECT * FROM drugs WHERE name = %s"
    results = execute_query(query, (drug_name,), fetch=True)
    return results[0] if results else None

def add_drug(drug_data):
    """
    Add a new drug to the database.
    
    Args:
        drug_data (dict): Drug data including id, name, description, etc.
        
    Returns:
        dict: The added drug data
    """
    query = """
    INSERT INTO drugs (id, name, description, original_indication, mechanism)
    VALUES (%(id)s, %(name)s, %(description)s, %(original_indication)s, %(mechanism)s)
    RETURNING *
    """
    results = execute_query(query, drug_data, fetch=True)
    return results[0] if results else None

def get_diseases(limit=100, offset=0):
    """
    Get a list of diseases from the database.
    
    Args:
        limit (int): Maximum number of diseases to return
        offset (int): Offset for pagination
        
    Returns:
        list: List of disease dictionaries
    """
    query = """
    SELECT * FROM diseases
    ORDER BY name
    LIMIT %s OFFSET %s
    """
    return execute_query(query, (limit, offset), fetch=True)

def get_disease_by_id(disease_id):
    """
    Get a disease by its ID.
    
    Args:
        disease_id (str): The ID of the disease
        
    Returns:
        dict: Disease data or None if not found
    """
    query = "SELECT * FROM diseases WHERE id = %s"
    results = execute_query(query, (disease_id,), fetch=True)
    return results[0] if results else None

def get_disease_by_name(disease_name):
    """
    Get a disease by its name.
    
    Args:
        disease_name (str): The name of the disease
        
    Returns:
        dict: Disease data or None if not found
    """
    query = "SELECT * FROM diseases WHERE name = %s"
    results = execute_query(query, (disease_name,), fetch=True)
    return results[0] if results else None

def add_disease(disease_data):
    """
    Add a new disease to the database.
    
    Args:
        disease_data (dict): Disease data including id, name, description, etc.
        
    Returns:
        dict: The added disease data
    """
    query = """
    INSERT INTO diseases (id, name, description, category)
    VALUES (%(id)s, %(name)s, %(description)s, %(category)s)
    RETURNING *
    """
    results = execute_query(query, disease_data, fetch=True)
    return results[0] if results else None

def add_relationship(relationship_data):
    """
    Add a new relationship to the database.
    
    Args:
        relationship_data (dict): Relationship data
        
    Returns:
        dict: The added relationship data
    """
    query = """
    INSERT INTO relationships (
        source_id, source_type, target_id, target_type, 
        relationship_type, confidence, evidence_count
    )
    VALUES (
        %(source_id)s, %(source_type)s, %(target_id)s, %(target_type)s,
        %(relationship_type)s, %(confidence)s, %(evidence_count)s
    )
    ON CONFLICT (source_id, target_id, relationship_type) 
    DO UPDATE SET 
        confidence = %(confidence)s,
        evidence_count = %(evidence_count)s,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *
    """
    results = execute_query(query, relationship_data, fetch=True)
    return results[0] if results else None

def get_drug_disease_relationships(drug_id=None, disease_id=None):
    """
    Get relationships between drugs and diseases.
    
    Args:
        drug_id (str, optional): Filter by drug ID
        disease_id (str, optional): Filter by disease ID
        
    Returns:
        list: List of relationship dictionaries
    """
    query = """
    SELECT 
        r.id, r.source_id, r.target_id, r.relationship_type, r.confidence, r.evidence_count,
        d1.name as drug_name, d2.name as disease_name
    FROM relationships r
    JOIN drugs d1 ON r.source_id = d1.id
    JOIN diseases d2 ON r.target_id = d2.id
    WHERE r.source_type = 'drug' AND r.target_type = 'disease'
    """
    
    params = []
    if drug_id:
        query += " AND r.source_id = %s"
        params.append(drug_id)
    if disease_id:
        query += " AND r.target_id = %s"
        params.append(disease_id)
        
    query += " ORDER BY r.confidence DESC"
    
    return execute_query(query, tuple(params), fetch=True)

def get_repurposing_candidates(min_confidence=0, drug_name=None, disease_name=None):
    """
    Get drug repurposing candidates.
    
    Args:
        min_confidence (int): Minimum confidence score (0-100)
        drug_name (str, optional): Filter by drug name
        disease_name (str, optional): Filter by disease name
        
    Returns:
        list: List of repurposing candidate dictionaries
    """
    query = """
    SELECT 
        rc.id, rc.drug_id, rc.disease_id, rc.confidence_score, rc.mechanism,
        rc.evidence_count, rc.status, d1.name as drug, d2.name as disease
    FROM repurposing_candidates rc
    JOIN drugs d1 ON rc.drug_id = d1.id
    JOIN diseases d2 ON rc.disease_id = d2.id
    WHERE rc.confidence_score >= %s
    """
    
    params = [min_confidence]
    if drug_name:
        query += " AND d1.name = %s"
        params.append(drug_name)
    if disease_name:
        query += " AND d2.name = %s"
        params.append(disease_name)
        
    query += " ORDER BY rc.confidence_score DESC"
    
    return execute_query(query, tuple(params), fetch=True)

def add_repurposing_candidate(candidate_data):
    """
    Add a new repurposing candidate to the database.
    
    Args:
        candidate_data (dict): Candidate data
        
    Returns:
        dict: The added candidate data
    """
    query = """
    INSERT INTO repurposing_candidates (
        drug_id, disease_id, confidence_score, mechanism, 
        evidence_count, status
    )
    VALUES (
        %(drug_id)s, %(disease_id)s, %(confidence_score)s, %(mechanism)s,
        %(evidence_count)s, %(status)s
    )
    ON CONFLICT (drug_id, disease_id) 
    DO UPDATE SET 
        confidence_score = %(confidence_score)s,
        mechanism = %(mechanism)s,
        evidence_count = %(evidence_count)s,
        status = %(status)s,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *
    """
    results = execute_query(query, candidate_data, fetch=True)
    return results[0] if results else None

def get_knowledge_graph_stats():
    """
    Get statistics about the knowledge graph.
    
    Returns:
        dict: Statistics about the knowledge graph
    """
    query = """
    SELECT
        (SELECT COUNT(*) FROM drugs) as drug_nodes,
        (SELECT COUNT(*) FROM diseases) as disease_nodes,
        (SELECT COUNT(*) FROM genes) as gene_nodes,
        (SELECT COUNT(*) FROM pathways) as pathway_nodes,
        (SELECT COUNT(*) FROM relationships) as total_edges
    """
    results = execute_query(query, fetch=True)
    
    # Get edge type counts
    edge_type_query = """
    SELECT relationship_type, COUNT(*) as count
    FROM relationships
    GROUP BY relationship_type
    """
    edge_types = execute_query(edge_type_query, fetch=True)
    
    stats = results[0] if results else {}
    stats['edge_types'] = {et['relationship_type']: et['count'] for et in edge_types}
    stats['total_nodes'] = (
        stats.get('drug_nodes', 0) + 
        stats.get('disease_nodes', 0) + 
        stats.get('gene_nodes', 0) + 
        stats.get('pathway_nodes', 0)
    )
    
    return stats

def search_pubmed_articles(query, limit=20):
    """
    Search PubMed articles in the database.
    
    Args:
        query (str): Search query (searches title and abstract)
        limit (int): Maximum number of articles to return
        
    Returns:
        list: List of article dictionaries
    """
    search_query = """
    SELECT *
    FROM pubmed_articles
    WHERE to_tsvector('english', title || ' ' || COALESCE(abstract, '')) @@ plainto_tsquery('english', %s)
    ORDER BY year DESC
    LIMIT %s
    """
    return execute_query(search_query, (query, limit), fetch=True)

def get_user_by_username(username):
    """
    Get a user by username.
    
    Args:
        username (str): Username to look up
        
    Returns:
        dict: User data or None if not found
    """
    query = "SELECT * FROM users WHERE username = %s"
    results = execute_query(query, (username,), fetch=True)
    return results[0] if results else None

def create_user(user_data):
    """
    Create a new user.
    
    Args:
        user_data (dict): User data including username, email, hashed_password, etc.
        
    Returns:
        dict: The created user data
    """
    query = """
    INSERT INTO users (username, email, full_name, hashed_password)
    VALUES (%(username)s, %(email)s, %(full_name)s, %(hashed_password)s)
    RETURNING *
    """
    results = execute_query(query, user_data, fetch=True)
    return results[0] if results else None