"""
Utility module to load environment variables from .env file
"""
import os
from dotenv import load_dotenv

def load_environment_variables():
    """
    Load environment variables from .env file
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Log whether important variables were loaded
    important_vars = [
        "NEO4J_URI", 
        "NEO4J_USERNAME", 
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY", 
        "HUGGINGFACE_API_KEY",
        "PUBMED_API_KEY",
        "UMLS_API_KEY"
    ]
    
    # Print the status of each variable (present or not)
    for var in important_vars:
        if os.getenv(var):
            print(f"Environment variable {var} loaded")
        else:
            print(f"Environment variable {var} not found")
    
    return True