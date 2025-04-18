"""
Script to migrate repurposing candidates data to the database.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

# Sample repurposing candidates data
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

def add_repurposing_candidate(candidate_data):
    """
    Add a new repurposing candidate to the database.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
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
            cur.execute(query, candidate_data)
            result = cur.fetchone()
            return result
    except Exception as e:
        print(f"Error adding repurposing candidate: {e}")
        raise
    finally:
        if conn:
            conn.close()

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

if __name__ == "__main__":
    migrate_repurposing_candidates()