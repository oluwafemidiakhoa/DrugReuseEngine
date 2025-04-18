from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class Drug(BaseModel):
    """Model for drug representation"""
    id: str
    name: str
    description: str
    original_indication: str
    mechanism: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "id": "D001",
                "name": "Metformin",
                "description": "A biguanide hypoglycemic agent used in the treatment of non-insulin-dependent diabetes mellitus not responding to dietary modification.",
                "original_indication": "Type 2 Diabetes",
                "mechanism": "Decreases hepatic glucose production, decreases intestinal absorption of glucose, and improves insulin sensitivity"
            }
        }


class Disease(BaseModel):
    """Model for disease representation"""
    id: str
    name: str
    description: str
    category: str

    class Config:
        schema_extra = {
            "example": {
                "id": "DIS001",
                "name": "Type 2 Diabetes",
                "description": "A metabolic disorder characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
                "category": "Metabolic"
            }
        }


class Relationship(BaseModel):
    """Model for drug-disease relationship"""
    source: str
    target: str
    type: str
    confidence: float

    class Config:
        schema_extra = {
            "example": {
                "source": "D001",
                "target": "DIS001",
                "type": "treats",
                "confidence": 0.95
            }
        }


class RepurposingCandidate(BaseModel):
    """Model for drug repurposing candidate"""
    drug: str
    disease: str
    confidence_score: int = Field(..., ge=0, le=100)
    mechanism: str
    evidence_count: int
    status: str

    class Config:
        schema_extra = {
            "example": {
                "drug": "Metformin",
                "disease": "Cancer",
                "confidence_score": 72,
                "mechanism": "Metformin may inhibit cancer cell growth through activation of AMPK pathway, which leads to inhibition of mTOR signaling.",
                "evidence_count": 28,
                "status": "Promising"
            }
        }


class PubMedArticle(BaseModel):
    """Model for PubMed article"""
    pmid: str
    title: str
    abstract: str
    year: str

    class Config:
        schema_extra = {
            "example": {
                "pmid": "12345678",
                "title": "Effects of Metformin on Cancer",
                "abstract": "This study explores the potential anticancer effects of Metformin...",
                "year": "2022"
            }
        }


class ExtractedRelationship(BaseModel):
    """Model for relationship extracted from literature"""
    source: str
    title: str
    text: str
    year: str

    class Config:
        schema_extra = {
            "example": {
                "source": "12345678",
                "title": "Effects of Metformin on Cancer",
                "text": "Metformin was found to be effective in treating certain types of cancer.",
                "year": "2022"
            }
        }


class DrugSearchResponse(BaseModel):
    """Response model for drug search"""
    drug: Optional[Drug] = None
    relationships: Optional[List[dict]] = None
    potential_repurposing: Optional[List[dict]] = None
    message: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "drug": {
                    "id": "D001",
                    "name": "Metformin",
                    "description": "A biguanide hypoglycemic agent...",
                    "original_indication": "Type 2 Diabetes",
                    "mechanism": "Decreases hepatic glucose production..."
                },
                "relationships": [
                    {"disease_name": "Type 2 Diabetes", "type": "treats", "confidence": 0.95}
                ],
                "potential_repurposing": [
                    {"disease_name": "Cancer", "confidence": 0.65}
                ]
            }
        }


class DiseaseSearchResponse(BaseModel):
    """Response model for disease search"""
    disease: Optional[Disease] = None
    relationships: Optional[List[dict]] = None
    potential_repurposing: Optional[List[dict]] = None
    message: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "disease": {
                    "id": "DIS001",
                    "name": "Type 2 Diabetes",
                    "description": "A metabolic disorder...",
                    "category": "Metabolic"
                },
                "relationships": [
                    {"drug_name": "Metformin", "type": "treats", "confidence": 0.95}
                ],
                "potential_repurposing": []
            }
        }


class KnowledgeGraphStats(BaseModel):
    """Model for knowledge graph statistics"""
    total_nodes: int
    total_edges: int
    drug_nodes: int
    disease_nodes: int
    edge_types: Dict[str, int]

    class Config:
        schema_extra = {
            "example": {
                "total_nodes": 11,
                "total_edges": 8,
                "drug_nodes": 5,
                "disease_nodes": 6,
                "edge_types": {"treats": 5, "potential": 3}
            }
        }


class PathAnalysisResponse(BaseModel):
    """Response model for path analysis"""
    found: bool
    paths: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "found": True,
                "paths": [
                    {
                        "path": ["D001", "DIS001"],
                        "node_names": ["Metformin", "Type 2 Diabetes"],
                        "edge_types": ["treats"],
                        "edge_confidences": [0.95],
                        "avg_confidence": 0.95
                    }
                ]
            }
        }


class Token(BaseModel):
    """Token model for authentication"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model containing user information"""
    username: Optional[str] = None


class User(BaseModel):
    """User model"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    """User model with hashed password for database storage"""
    hashed_password: str


class UserCreate(BaseModel):
    """Model for user creation"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "username": "researcher1",
                "email": "researcher@example.com",
                "password": "securepassword123",
                "full_name": "Dr. Jane Smith"
            }
        }