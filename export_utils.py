"""
Export utilities for the Drug Repurposing Engine.
Provides functions to export data to various formats (PDF, CSV).
"""
import base64
import io
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import streamlit as st


def candidates_to_dataframe(candidates):
    """
    Convert a list of candidate dictionaries to a DataFrame.
    
    Args:
        candidates: List of candidate dictionaries
        
    Returns:
        pd.DataFrame: DataFrame with candidate data
    """
    data = []
    for c in candidates:
        # Create a row for each candidate
        row = {
            'Drug': c.get('drug', ''),
            'Disease': c.get('disease', ''),
            'Confidence Score': c.get('confidence_score', 0),
            'Mechanism': c.get('mechanism', '')[:100] + '...' if len(c.get('mechanism', '')) > 100 else c.get('mechanism', ''),
            'Source': c.get('source', ''),
            'Status': c.get('status', '')
        }
        data.append(row)
    
    return pd.DataFrame(data)


def relationships_to_dataframe(relationships):
    """
    Convert a list of relationship dictionaries to a DataFrame.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        pd.DataFrame: DataFrame with relationship data
    """
    data = []
    for r in relationships:
        # Create a row for each relationship
        row = {
            'Source': r.get('source_name', r.get('source', '')),
            'Target': r.get('target_name', r.get('target', '')),
            'Type': r.get('type', ''),
            'Evidence': r.get('evidence', '')[:100] + '...' if len(r.get('evidence', '')) > 100 else r.get('evidence', ''),
            'Source Type': r.get('source_type', ''),
            'Target Type': r.get('target_type', '')
        }
        data.append(row)
    
    return pd.DataFrame(data)


def generate_pdf_download_link(pdf_bytes, filename, link_text="Download PDF"):
    """
    Generate a download link for a PDF file.
    
    Args:
        pdf_bytes: Bytes of the PDF file
        filename: Name of the file to download
        link_text: Text to display for the download link
        
    Returns:
        str: HTML for the download link
    """
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def generate_csv_download_link(df, filename, link_text="Download CSV"):
    """
    Generate a download link for a CSV file.
    
    Args:
        df: Pandas DataFrame to export
        filename: Name of the file to download
        link_text: Text to display for the download link
        
    Returns:
        str: HTML for the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def create_pdf(title, content_sections):
    """
    Create a PDF with a title and content sections.
    
    Args:
        title: Title of the PDF
        content_sections: List of dictionaries with 'heading' and 'content' keys
        
    Returns:
        bytes: PDF file as bytes
    """
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)
    
    # Add date
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="R")
    pdf.ln(5)
    
    # Add content sections
    for section in content_sections:
        # Add section heading
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, section.get('heading', ''), ln=True)
        
        # Add section content
        pdf.set_font("Arial", "", 10)
        
        # Handle different content types
        content = section.get('content', '')
        if isinstance(content, str):
            # Text content - split into lines to prevent overflow
            pdf.multi_cell(0, 5, content)
        elif isinstance(content, pd.DataFrame):
            # DataFrame content - convert to table
            df = content
            
            # Calculate column widths based on content
            col_widths = []
            for col in df.columns:
                col_width = max(
                    pdf.get_string_width(str(col)),
                    max(pdf.get_string_width(str(val)) for val in df[col].astype(str))
                )
                col_widths.append(col_width + 6)  # Add padding
            
            # Scale column widths if they exceed page width
            page_width = pdf.w - 2 * pdf.l_margin
            if sum(col_widths) > page_width:
                scale_factor = page_width / sum(col_widths)
                col_widths = [w * scale_factor for w in col_widths]
            
            # Add table header
            pdf.set_font("Arial", "B", 10)
            for i, col in enumerate(df.columns):
                pdf.cell(col_widths[i], 7, str(col), border=1)
            pdf.ln()
            
            # Add table rows
            pdf.set_font("Arial", "", 10)
            for _, row in df.iterrows():
                for i, col in enumerate(df.columns):
                    pdf.cell(col_widths[i], 6, str(row[col])[:50], border=1)
                pdf.ln()
        
        pdf.ln(5)
    
    # Return PDF as bytes
    return pdf.output(dest="S").encode("latin-1")


def create_repurposing_candidates_pdf(candidates):
    """
    Create a PDF report for repurposing candidates.
    
    Args:
        candidates: List of candidate dictionaries
        
    Returns:
        bytes: PDF file as bytes
    """
    # Convert candidates to DataFrame
    df = candidates_to_dataframe(candidates)
    
    # Create content sections
    content_sections = [
        {
            'heading': 'Summary',
            'content': f"This report contains {len(candidates)} drug repurposing candidates identified by the Drug Repurposing Engine. Each candidate represents a potential new therapeutic application for an existing drug."
        },
        {
            'heading': 'Repurposing Candidates',
            'content': df
        },
        {
            'heading': 'Methodology',
            'content': """
            Drug repurposing candidates were identified using a combination of methods:
            1. Knowledge graph analysis
            2. Literature-based discovery
            3. Molecular similarity analysis
            4. Network-based drug-disease connections
            5. AI-assisted mechanistic reasoning
            
            Confidence scores are calculated based on evidence from multiple sources, including
            published literature, molecular pathways, gene expression patterns, and structural similarities.
            """
        }
    ]
    
    # Create PDF
    return create_pdf("Drug Repurposing Candidates Report", content_sections)


def create_drug_disease_pdf(drug, disease, relationships, candidates):
    """
    Create a PDF report for a drug-disease pair.
    
    Args:
        drug: Drug dictionary
        disease: Disease dictionary
        relationships: List of relationship dictionaries
        candidates: List of candidate dictionaries
        
    Returns:
        bytes: PDF file as bytes
    """
    drug_name = drug.get('name', 'Unknown Drug')
    disease_name = disease.get('name', 'Unknown Disease')
    
    # Create content sections
    content_sections = [
        {
            'heading': 'Drug Information',
            'content': f"""
            Name: {drug_name}
            ID: {drug.get('id', 'Unknown')}
            Original Indication: {drug.get('original_indication', 'Unknown')}
            Mechanism of Action: {drug.get('mechanism', 'Unknown')}
            
            Description: {drug.get('description', 'No description available.')}
            """
        },
        {
            'heading': 'Disease Information',
            'content': f"""
            Name: {disease_name}
            ID: {disease.get('id', 'Unknown')}
            Category: {disease.get('category', 'Unknown')}
            
            Description: {disease.get('description', 'No description available.')}
            """
        }
    ]
    
    # Add relationships if any
    if relationships:
        rel_df = relationships_to_dataframe(relationships)
        content_sections.append({
            'heading': 'Known Relationships',
            'content': rel_df
        })
    
    # Add repurposing candidates if any
    if candidates:
        # Add specific candidate information
        for candidate in candidates:
            content_sections.append({
                'heading': f"Repurposing Analysis (Confidence: {candidate.get('confidence_score', 0)}%)",
                'content': f"""
                Status: {candidate.get('status', 'Unknown')}
                Source: {candidate.get('source', 'Unknown')}
                
                Proposed Mechanism:
                {candidate.get('mechanism', 'No mechanism proposed.')}
                
                Evidence Base:
                {candidate.get('evidence_text', 'No detailed evidence available.')}
                """
            })
    
    # Add analysis summary and references
    content_sections.append({
        'heading': 'Analysis Methodology',
        'content': """
        This report was generated using the Drug Repurposing Engine, which analyzes potential 
        new uses for existing drugs based on multiple lines of evidence:
        
        1. Knowledge graph relationships and path analysis
        2. Biomedical literature co-occurrences
        3. Molecular and genetic pathway overlap
        4. Network-based drug-disease similarity
        5. AI-assisted mechanistic reasoning
        
        Confidence scores reflect the strength and diversity of supporting evidence.
        """
    })
    
    # Create PDF
    title = f"Drug-Disease Analysis: {drug_name} for {disease_name}"
    return create_pdf(title, content_sections)