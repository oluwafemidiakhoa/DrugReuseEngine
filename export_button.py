"""
Floating export button component for Streamlit applications.
"""
import streamlit as st
import base64
from one_click_export import get_current_context, export_to_format

def floating_export_button():
    """
    Add a floating export button to the Streamlit application.
    
    Returns:
        dict: Context information for the current view
    """
    # Add custom CSS for the floating button
    st.markdown("""
    <style>
    .floating-export-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }
    
    .floating-export-button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #0068C9;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .floating-export-button:hover {
        background-color: #004A8F;
        transform: scale(1.05);
    }
    
    .floating-export-icon {
        font-size: 24px;
    }
    
    .floating-export-menu {
        position: absolute;
        bottom: 65px;
        right: 0;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        padding: 10px;
        display: none;
    }
    
    .floating-export-menu.show {
        display: block;
    }
    
    .floating-export-menu-item {
        display: block;
        padding: 8px 16px;
        text-decoration: none;
        color: #333;
        transition: background-color 0.2s;
        border-radius: 4px;
        margin-bottom: 4px;
        cursor: pointer;
    }
    
    .floating-export-menu-item:hover {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add HTML for the floating button
    st.markdown("""
    <div class="floating-export-container">
        <div class="floating-export-button" id="export-button">
            <div class="floating-export-icon">📥</div>
        </div>
        <div id="exportMenu" class="floating-export-menu">
            <div class="floating-export-menu-item" id="export-pdf">Export to PDF</div>
            <div class="floating-export-menu-item" id="export-csv">Export to CSV</div>
        </div>
    </div>
    
    <script>
    // Add event listeners when the DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        const exportButton = document.getElementById('export-button');
        const exportMenu = document.getElementById('exportMenu');
        const exportPdf = document.getElementById('export-pdf');
        const exportCsv = document.getElementById('export-csv');
        
        // Toggle menu on button click
        exportButton.addEventListener('click', function() {
            exportMenu.classList.toggle('show');
        });
        
        // Handle export to PDF
        exportPdf.addEventListener('click', function() {
            // Use custom event to communicate with Streamlit
            const event = new CustomEvent('streamlit:export', {
                detail: { format: 'pdf' }
            });
            window.dispatchEvent(event);
            exportMenu.classList.remove('show');
        });
        
        // Handle export to CSV
        exportCsv.addEventListener('click', function() {
            // Use custom event to communicate with Streamlit
            const event = new CustomEvent('streamlit:export', {
                detail: { format: 'csv' }
            });
            window.dispatchEvent(event);
            exportMenu.classList.remove('show');
        });
        
        // Close the menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!exportButton.contains(event.target) && !exportMenu.contains(event.target)) {
                exportMenu.classList.remove('show');
            }
        });
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Get the current context for export operations
    context = get_current_context()
    
    # Create a container for the download links
    download_container = st.empty()
    
    # Add a button clicked detection and export logic
    if st.button("Export PDF", key="pdf_export_hidden", help="Export to PDF"):
        download_container.markdown(export_to_format('pdf', context))
        
    if st.button("Export CSV", key="csv_export_hidden", help="Export to CSV"):
        download_container.markdown(export_to_format('csv', context))
    
    return context