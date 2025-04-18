# Deployment Guide for Drug Repurposing Engine

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- All dependencies installed via pip (requirements in pyproject.toml)

## Configuration

The application is configured to run with:
- Streamlit frontend on port 5000
- FastAPI backend for API endpoints
- PostgreSQL database for data storage

## Local Development

1. Start the Streamlit server:
   ```
   streamlit run app.py --server.port 5000
   ```

2. Start the API server:
   ```
   python run_api.py
   ```

## Deployment Steps

1. Make sure `.streamlit/config.toml` has the following configuration:
   ```toml
   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = false
   address = "0.0.0.0"
   port = 5000
   
   [browser]
   gatherUsageStats = false
   ```

2. Ensure all environment variables are properly set:
   - `DATABASE_URL` for PostgreSQL connection
   - AI API keys if using OpenAI, HuggingFace, or Gemini services

3. For production deployment, it's recommended to:
   - Use a production WSGI server like gunicorn for the API
   - Set up proper logging and monitoring
   - Configure proper authentication for API endpoints
   - Use environment-specific configurations

## Troubleshooting

- If database connections fail, check PostgreSQL connection parameters
- If AI services aren't responding, verify API keys are valid
- For visualization issues, verify that Plotly and network packages are correctly installed