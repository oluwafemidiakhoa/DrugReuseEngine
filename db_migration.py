"""
Database migration script to populate the PostgreSQL database with sample data.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

# Sample data
SAMPLE_DRUGS = [
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
        "original_indication": "Pain, Inflammation, Fever",
        "mechanism": "Inhibits cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis"
    },
    {
        "id": "D003",
        "name": "Atorvastatin",
        "description": "A selective, competitive inhibitor of HMG-CoA reductase, used to lower cholesterol levels.",
        "original_indication": "Hypercholesterolemia",
        "mechanism": "Inhibits HMG-CoA reductase, reducing cholesterol synthesis in the liver"
    },
    {
        "id": "D004",
        "name": "Losartan",
        "description": "An angiotensin II receptor antagonist used in the treatment of hypertension.",
        "original_indication": "Hypertension",
        "mechanism": "Blocks the binding of angiotensin II to AT1 receptors"
    },
    {
        "id": "D005",
        "name": "Sildenafil",
        "description": "A selective inhibitor of phosphodiesterase type 5 (PDE5), which enhances the effect of nitric oxide by inhibiting the degradation of cGMP.",
        "original_indication": "Erectile Dysfunction",
        "mechanism": "Inhibits PDE5, increasing cGMP levels and relaxing smooth muscle"
    }
]

SAMPLE_DISEASES = [
    {
        "id": "DIS001",
        "name": "Type 2 Diabetes",
        "description": "A metabolic disorder characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
        "category": "Metabolic"
    },
    {
        "id": "DIS002",
        "name": "Hypertension",
        "description": "A chronic condition in which the blood pressure in the arteries is persistently elevated.",
        "category": "Cardiovascular"
    },
    {
        "id": "DIS003",
        "name": "Alzheimer's Disease",
        "description": "A progressive neurologic disorder that causes brain cells to die and the brain to shrink.",
        "category": "Neurological"
    },
    {
        "id": "DIS004",
        "name": "Cancer",
        "description": "A disease in which some of the body's cells grow uncontrollably and spread to other parts of the body.",
        "category": "Oncological"
    },
    {
        "id": "DIS005",
        "name": "Rheumatoid Arthritis",
        "description": "An inflammatory disorder that affects joints, causing painful swelling that can eventually result in bone erosion and joint deformity.",
        "category": "Inflammatory"
    },
    {
        "id": "DIS006",
        "name": "Depression",
        "description": "A mental health disorder characterized by persistently depressed mood or loss of interest in activities.",
        "category": "Psychiatric"
    }
]

SAMPLE_RELATIONSHIPS = [
    {
        "source_id": "D001",
        "source_type": "drug",
        "target_id": "DIS001",
        "target_type": "disease",
        "relationship_type": "TREATS",
        "confidence": 0.95,
        "evidence_count": 48
    },
    {
        "source_id": "D002",
        "source_type": "drug",
        "target_id": "DIS002",
        "target_type": "disease",
        "relationship_type": "TREATS",
        "confidence": 0.72,
        "evidence_count": 35
    },
    {
        "source_id": "D003",
        "source_type": "drug",
        "target_id": "DIS002",
        "target_type": "disease",
        "relationship_type": "TREATS",
        "confidence": 0.85,
        "evidence_count": 42
    },
    {
        "source_id": "D004",
        "source_type": "drug",
        "target_id": "DIS002",
        "target_type": "disease",
        "relationship_type": "TREATS",
        "confidence": 0.92,
        "evidence_count": 53
    },
    {
        "source_id": "D001",
        "source_type": "drug",
        "target_id": "DIS004",
        "target_type": "disease",
        "relationship_type": "POTENTIAL",
        "confidence": 0.65,
        "evidence_count": 28
    },
    {
        "source_id": "D002",
        "source_type": "drug",
        "target_id": "DIS004",
        "target_type": "disease",
        "relationship_type": "POTENTIAL",
        "confidence": 0.58,
        "evidence_count": 23
    },
    {
        "source_id": "D003",
        "source_type": "drug",
        "target_id": "DIS003",
        "target_type": "disease",
        "relationship_type": "POTENTIAL",
        "confidence": 0.62,
        "evidence_count": 19
    }
]

SAMPLE_CANDIDATES = [
    {
        "drug_id": "D001",
        "disease_id": "DIS004",
        "confidence_score": 72,
        "mechanism": "Metformin may inhibit cancer cell growth through activation of AMPK pathway, which leads to inhibition of mTOR signaling.",
        "evidence_count": 28,
        "status": "Promising"
    },
    {
        "drug_id": "D002",
        "disease_id": "DIS004",
        "confidence_score": 65,
        "mechanism": "Aspirin may reduce cancer risk through inhibition of COX-2, which is overexpressed in many cancers, and through its anti-inflammatory effects.",
        "evidence_count": 23,
        "status": "Promising"
    },
    {
        "drug_id": "D003",
        "disease_id": "DIS003",
        "confidence_score": 68,
        "mechanism": "Statins may reduce Alzheimer's risk through anti-inflammatory effects and by affecting cholesterol metabolism in the brain.",
        "evidence_count": 19,
        "status": "Promising"
    },
    {
        "drug_id": "D001",
        "disease_id": "DIS005",
        "confidence_score": 52,
        "mechanism": "Metformin may reduce inflammation in rheumatoid arthritis through AMPK activation and subsequent NF-κB inhibition.",
        "evidence_count": 14,
        "status": "Needs Review"
    },
    {
        "drug_id": "D005",
        "disease_id": "DIS003",
        "confidence_score": 48,
        "mechanism": "Sildenafil may improve cognitive function in Alzheimer's through increased cerebral blood flow and potential reduction of amyloid-beta accumulation.",
        "evidence_count": 12,
        "status": "Needs Review"
    }
]

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
    ON CONFLICT (id) DO UPDATE SET
        name = %(name)s,
        description = %(description)s,
        original_indication = %(original_indication)s,
        mechanism = %(mechanism)s,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *
    """
    results = execute_query(query, drug_data, fetch=True)
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
    ON CONFLICT (id) DO UPDATE SET
        name = %(name)s,
        description = %(description)s,
        category = %(category)s,
        updated_at = CURRENT_TIMESTAMP
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

def migrate_drugs():
    """Migrate drugs data to database"""
    print("Migrating drugs data...")
    
    for drug in SAMPLE_DRUGS:
        try:
            # Add to database
            result = add_drug(drug)
            print(f"Added/updated drug: {drug['name']}")
        except Exception as e:
            print(f"Error adding drug {drug.get('name', 'unknown')}: {e}")

def migrate_diseases():
    """Migrate diseases data to database"""
    print("Migrating diseases data...")
    
    for disease in SAMPLE_DISEASES:
        try:
            # Add to database
            result = add_disease(disease)
            print(f"Added/updated disease: {disease['name']}")
        except Exception as e:
            print(f"Error adding disease {disease.get('name', 'unknown')}: {e}")

def migrate_relationships():
    """Migrate relationships data to database"""
    print("Migrating relationships data...")
    
    for rel in SAMPLE_RELATIONSHIPS:
        try:
            # Add to database
            result = add_relationship(rel)
            print(f"Added/updated relationship: {rel['source_id']} -> {rel['target_id']}")
        except Exception as e:
            print(f"Error adding relationship {rel.get('source_id', 'unknown')} -> {rel.get('target_id', 'unknown')}: {e}")

def migrate_repurposing_candidates():
    """Migrate repurposing candidates to database"""
    print("Migrating repurposing candidates data...")
    
    for candidate in SAMPLE_CANDIDATES:
        try:
            # Add to database
            result = add_repurposing_candidate(candidate)
            drug_id = candidate['drug_id']
            disease_id = candidate['disease_id']
            print(f"Added/updated repurposing candidate: {drug_id} -> {disease_id}")
        except Exception as e:
            print(f"Error adding repurposing candidate {candidate.get('drug_id', 'unknown')} -> {candidate.get('disease_id', 'unknown')}: {e}")

def migrate_all_data():
    """Migrate all data to database"""
    print("Starting data migration...")
    
    # Migrate in the correct order to maintain foreign key constraints
    migrate_drugs()
    migrate_diseases()
    migrate_relationships()
    migrate_repurposing_candidates()
    
    print("Data migration completed.")

if __name__ == "__main__":
    migrate_all_data()