"""
Data migration script to populate the PostgreSQL database with initial sample data.
"""
import json
import os
from utils import initialize_session_state
from db_utils import (
    add_drug, add_disease, add_relationship, add_repurposing_candidate,
    get_drug_by_id, get_disease_by_id
)

def migrate_drugs():
    """Migrate drugs data from session state to database"""
    print("Migrating drugs data...")
    
    # Initialize session state to ensure we have data
    initialize_session_state()
    
    # Get drugs from session state
    drugs = getattr(initialize_session_state, "drugs", [])
    
    for drug in drugs:
        try:
            # Check if drug already exists
            existing = get_drug_by_id(drug['id'])
            if not existing:
                # Add to database
                add_drug(drug)
                print(f"Added drug: {drug['name']}")
            else:
                print(f"Drug already exists: {drug['name']}")
        except Exception as e:
            print(f"Error adding drug {drug.get('name', 'unknown')}: {e}")

def migrate_diseases():
    """Migrate diseases data from session state to database"""
    print("Migrating diseases data...")
    
    # Initialize session state to ensure we have data
    initialize_session_state()
    
    # Get diseases from session state
    diseases = getattr(initialize_session_state, "diseases", [])
    
    for disease in diseases:
        try:
            # Check if disease already exists
            existing = get_disease_by_id(disease['id'])
            if not existing:
                # Add to database
                add_disease(disease)
                print(f"Added disease: {disease['name']}")
            else:
                print(f"Disease already exists: {disease['name']}")
        except Exception as e:
            print(f"Error adding disease {disease.get('name', 'unknown')}: {e}")

def migrate_relationships():
    """Migrate relationships data from session state to database"""
    print("Migrating relationships data...")
    
    # Initialize session state to ensure we have data
    initialize_session_state()
    
    # Get relationships from session state
    relationships = getattr(initialize_session_state, "relationships", [])
    
    for rel in relationships:
        try:
            # Create a properly formatted relationship object
            relationship_data = {
                "source_id": rel["source"],
                "source_type": "drug",  # Assuming source is always a drug
                "target_id": rel["target"],
                "target_type": "disease",  # Assuming target is always a disease
                "relationship_type": rel["type"].upper(),
                "confidence": rel["confidence"],
                "evidence_count": 1  # Default value
            }
            
            # Add to database
            add_relationship(relationship_data)
            print(f"Added relationship: {rel['source']} -> {rel['target']}")
        except Exception as e:
            print(f"Error adding relationship {rel.get('source', 'unknown')} -> {rel.get('target', 'unknown')}: {e}")

def migrate_repurposing_candidates():
    """Migrate repurposing candidates from session state to database"""
    print("Migrating repurposing candidates data...")
    
    # Initialize session state to ensure we have data
    initialize_session_state()
    
    # Get repurposing candidates from session state
    candidates = getattr(initialize_session_state, "repurposing_candidates", [])
    
    for candidate in candidates:
        try:
            # Get drug and disease IDs
            drug_name = candidate.get("drug")
            disease_name = candidate.get("disease")
            
            # Find drug by name
            drug = None
            for d in initialize_session_state.drugs:
                if d.get("name") == drug_name:
                    drug = d
                    break
            
            # Find disease by name
            disease = None
            for d in initialize_session_state.diseases:
                if d.get("name") == disease_name:
                    disease = d
                    break
            
            if not drug or not disease:
                print(f"Could not find drug or disease for candidate: {drug_name} -> {disease_name}")
                continue
            
            # Create a properly formatted candidate object
            candidate_data = {
                "drug_id": drug["id"],
                "disease_id": disease["id"],
                "confidence_score": candidate["confidence_score"],
                "mechanism": candidate["mechanism"],
                "evidence_count": candidate["evidence_count"],
                "status": candidate["status"]
            }
            
            # Add to database
            add_repurposing_candidate(candidate_data)
            print(f"Added repurposing candidate: {drug_name} -> {disease_name}")
        except Exception as e:
            print(f"Error adding repurposing candidate {candidate.get('drug', 'unknown')} -> {candidate.get('disease', 'unknown')}: {e}")

def migrate_all_data():
    """Migrate all data from session state to database"""
    print("Starting data migration...")
    
    # Migrate in the correct order to maintain foreign key constraints
    migrate_drugs()
    migrate_diseases()
    migrate_relationships()
    migrate_repurposing_candidates()
    
    print("Data migration completed.")

if __name__ == "__main__":
    migrate_all_data()