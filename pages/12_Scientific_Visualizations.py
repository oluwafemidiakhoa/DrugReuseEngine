import streamlit as st
import sys
sys.path.append('.')
from utils import initialize_session_state
from scientific_visualizations import show_scientific_visualizations

st.set_page_config(page_title="Scientific Visualizations", page_icon="📊", layout="wide")
initialize_session_state()

# Show the scientific visualizations
show_scientific_visualizations()

# Add detailed explanation at the bottom
st.markdown("""
---
## About Our Scientific Visualizations

These advanced visualizations leverage cutting-edge data science and visualization techniques to transform complex biomedical data into actionable insights. They represent the intersection of pharmaceutical science, bioinformatics, and data visualization.

### Scientific Value

These visualizations provide unique value to researchers by:

1. **Revealing Hidden Patterns**: Exposing relationships between drugs, targets, and diseases that might not be apparent through traditional analysis
2. **Contextualizing Historical Trends**: Placing drug repurposing success in historical context with key events and technological advancements
3. **Enabling Multi-dimensional Analysis**: Comparing candidates across multiple parameters simultaneously
4. **Depicting Complex Hierarchies**: Illustrating the cascading effects from drug to target to pathway to disease
5. **Facilitating Comparison**: Presenting confidence scores in an intuitive, visual format

### Methodology

Each visualization is generated using a combination of:

- Sophisticated data processing algorithms that transform raw biomedical data
- Advanced statistical methods for calculating similarity metrics and confidence scores
- State-of-the-art visualization techniques using Plotly's scientific plotting capabilities
- Customized layouts and color schemes optimized for scientific interpretation

_These visualizations represent the cutting edge of biomedical data science, providing researchers with powerful tools for drug repurposing discovery._
""")