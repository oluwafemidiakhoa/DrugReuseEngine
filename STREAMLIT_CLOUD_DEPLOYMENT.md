# Streamlit Cloud Deployment Guide

This guide provides step-by-step instructions for deploying the Drug Repurposing Engine to Streamlit Cloud.

## Step 1: Prepare Your GitHub Repository

1. Make sure your GitHub repository contains the following files:
   - `streamlit_cloud_app.py` (renamed from the file we created specifically for Streamlit Cloud deployment)
   - `.streamlit/config.toml` (with the configuration we created)
   - `requirements.txt` (with all dependencies)
   - All other project files

2. The entry point should be `streamlit_cloud_app.py`, which includes both the Streamlit UI and API server.

## Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the deployment form:
   - Repository: `oluwafemidiakhoa/DrugReuseEngine`
   - Branch: `main` (or your default branch)
   - Main file path: `streamlit_cloud_app.py` (not app.py)
   - App URL: Choose a custom subdomain (e.g., drug-repurposing-engine)

## Step 3: Configure Secrets

After your app is deployed, add your secrets:

1. Go to your app settings in Streamlit Cloud
2. Click on "Secrets"
3. Add your secrets in TOML format:

```toml
[secrets]
OPENAI_API_KEY = "your_openai_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
HUGGINGFACE_API_KEY = "your_huggingface_api_key"
PUBMED_API_KEY = "your_pubmed_api_key"
UMLS_API_KEY = "your_umls_api_key"
DATABASE_URL = "your_database_url"
NEO4J_URI = "your_neo4j_uri"
NEO4J_USERNAME = "your_neo4j_username"
NEO4J_PASSWORD = "your_neo4j_password"
```

## Step 4: API Access in Streamlit Cloud

The `streamlit_cloud_app.py` file starts both the Streamlit app and the API server in a background thread.

- API endpoints will be available at: `https://your-app-name.streamlit.app/api/...`
- The API documentation will be accessible at: `https://your-app-name.streamlit.app/docs`

## Step 5: Database Setup

For the PostgreSQL database:

1. Use a cloud-hosted PostgreSQL provider:
   - [Heroku Postgres](https://www.heroku.com/postgres)
   - [AWS RDS](https://aws.amazon.com/rds/postgresql/)
   - [DigitalOcean](https://www.digitalocean.com/products/managed-databases/)
   - [Neon](https://neon.tech/)

2. Create a database and get the connection string
3. Add the connection string to your Streamlit Cloud secrets

For Neo4j:

1. Use [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) (free tier available)
2. Create a database and get the connection details
3. Add the connection details to your Streamlit Cloud secrets

## Step 6: Troubleshooting

Common issues and solutions:

1. **Dependencies issues**: Make sure all packages are in `requirements.txt`
2. **Memory errors**: Streamlit Cloud limits memory usage; optimize your code if needed
3. **API not starting**: Check logs for threading-related errors
4. **Database connection errors**: Verify connection strings and make sure the database is accessible

## Step 7: Custom Domain (Optional)

1. Purchase a domain (e.g., from Namecheap, GoDaddy)
2. In Streamlit Cloud settings, go to "Custom domain"
3. Follow the instructions to set up DNS records
4. Wait for DNS propagation (may take 24-48 hours)

## Final Notes

- The `streamlit_cloud_app.py` file embeds the API server, allowing both to run in Streamlit Cloud's single-process environment
- API keys and secrets are securely stored in Streamlit Cloud's secrets management system
- Database connections are established using the provided connection strings
- The app can be accessed at `https://your-app-name.streamlit.app`