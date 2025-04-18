# Manual Export Instructions

Since we're encountering permission issues with direct GitHub pushes, here's how to manually export and upload your code to GitHub:

## Option 1: Download as ZIP from Replit

1. In Replit, click on the three dots (...) in the file explorer panel
2. Select "Download as zip"
3. Save the ZIP file to your local machine

## Option 2: Clone the repository locally and push from your machine

1. Clone the empty repository to your local machine:
   ```
   git clone https://github.com/oluwafemidiakhoa/DrugReuseEngine.git
   ```

2. Download the code from Replit (as in Option 1)
3. Extract the ZIP file
4. Copy all files into your local repository folder
5. Commit and push from your local machine:
   ```
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

## Option 3: Use GitHub's web interface to upload files

1. Go to https://github.com/oluwafemidiakhoa/DrugReuseEngine
2. Click on "Add file" > "Upload files"
3. Drag and drop files from your downloaded ZIP
4. Commit the changes

## Important files to include

Make sure these key files are included in your repository:
- app.py
- scientific_visualizations.py
- neo4j_utils.py
- db_utils.py
- openai_analysis.py
- pages/ directory (contains all page files)
- assets/ directory
- requirements.txt

## Additional deployment options

After uploading your code to GitHub, you can:
1. Deploy on Streamlit Cloud: https://share.streamlit.io/
2. Set up GitHub Actions for CI/CD
3. Connect to other cloud platforms like Heroku or Azure