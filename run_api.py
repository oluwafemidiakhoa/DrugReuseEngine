import uvicorn
import os
import sys
import streamlit as st
from utils import initialize_session_state

# Initialize Streamlit's session state for API usage
if 'session_state' not in globals():
    # Create a mock Streamlit session state for the API
    initialize_session_state()
    print("Initialized session state for API server")

from api.main import app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", 8000))

    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )