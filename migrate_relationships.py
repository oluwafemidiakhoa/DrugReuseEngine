"""
Script to migrate relationships data to the database.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

# Sample relationships data
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

def add_relationship(relationship_data):
    """
    Add a new relationship to the database.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
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
            cur.execute(query, relationship_data)
            result = cur.fetchone()
            return result
    except Exception as e:
        print(f"Error adding relationship: {e}")
        raise
    finally:
        if conn:
            conn.close()

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

if __name__ == "__main__":
    migrate_relationships()