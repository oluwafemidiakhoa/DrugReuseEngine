# GitHub Deployment Guide for Drug Repurposing Engine

This guide will help you migrate the Drug Repurposing Engine from Replit to your GitHub repository and deploy it with a custom URL.

## Step 1: Clone the Repository

If you're starting from scratch with your GitHub repository:

```bash
# Clone your repository
git clone https://github.com/oluwafemidiakhoa/DrugReuseEngine.git
cd DrugReuseEngine
```

## Step 2: Copy Files from Replit

You can download all files from Replit by:
1. Go to your Replit project
2. Click on the three dots (...) in the files panel
3. Select "Download as zip"
4. Extract the zip file
5. Copy all extracted files to your cloned GitHub repository folder

## Step 3: Set Up .gitignore

Create a `.gitignore` file to exclude unnecessary files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Environment variables
.env

# Replit specific
.replit
replit.nix
.cache/
.upm/

# Misc
.DS_Store
.idea/
.vscode/
```

## Step 4: Commit and Push to GitHub

```bash
# Add all files
git add .

# Commit changes
git commit -m "Migrate Drug Repurposing Engine from Replit"

# Push to GitHub
git push origin main
```

## Step 5: Deploy with Streamlit Cloud (Recommended)

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and the main file (app.py)
5. Set environment secrets for API keys
6. Click "Deploy"

This will give you a URL like `https://yourapp.streamlit.app`

## Step 6: Set Up Custom Domain (Optional)

If you want to use a custom domain:

1. Go to your app settings in Streamlit Cloud
2. Under "Custom domain", click "Add custom domain"
3. Follow the instructions to set up DNS records
4. Wait for DNS propagation (may take 24-48 hours)

## Step 7: Alternative Deployment Options

### Option 1: Heroku

1. Make sure your `Procfile` contains:
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

2. Deploy to Heroku:
   ```bash
   heroku create drug-repurposing-engine
   git push heroku main
   ```

3. Set up environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_key_here
   heroku config:set DATABASE_URL=your_db_url_here
   # Add all other required environment variables
   ```

### Option 2: AWS Elastic Beanstalk

1. Install the EB CLI:
   ```bash
   pip install awsebcli
   ```

2. Initialize EB:
   ```bash
   eb init
   ```

3. Create an environment:
   ```bash
   eb create drug-repurposing-production
   ```

4. Deploy:
   ```bash
   eb deploy
   ```

### Option 3: Google Cloud Run

1. Build a Docker container:
   ```
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8080
   CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0
   ```

2. Deploy to Cloud Run using the Google Cloud Console or CLI

## Notes on Database Migration

If you're using a PostgreSQL database:

1. Create a new database on your hosting provider
2. Export your data from Replit:
   ```bash
   pg_dump -U postgres -h your_replit_db_host -d your_db_name > database_backup.sql
   ```
3. Import to your new database:
   ```bash
   psql -U your_user -h your_new_db_host -d your_new_db_name < database_backup.sql
   ```
4. Update the DATABASE_URL environment variable in your deployment

## Running Locally

To run the application locally after cloning:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run the API server (in a separate terminal)
API_PORT=7000 python run_api.py
```

## Environment Variables

Make sure to set these environment variables in your deployment:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: PostgreSQL connection string
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key
- `PUBMED_API_KEY`: Your PubMed API key
- `UMLS_API_KEY`: Your UMLS API key
- Any other API keys used in the application

## Troubleshooting

- **Database connection issues**: Check your DATABASE_URL format and credentials
- **API errors**: Verify all API keys are set correctly
- **Streamlit app not loading**: Check the application logs for errors
- **Neo4j connection issues**: Update Neo4j credentials and connection strings

For additional help, refer to the documentation of your chosen deployment platform.