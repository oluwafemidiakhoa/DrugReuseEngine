import streamlit as st
import sys
sys.path.append('.')
from utils import initialize_session_state
from interactive_3d_vis import show_3d_mechanism_explorer

st.set_page_config(page_title="3D Mechanism Viewer", page_icon="🔍", layout="wide")
initialize_session_state()

# Show the 3D mechanism explorer
show_3d_mechanism_explorer()

# Add detailed explanation at the bottom
st.markdown("""
---
### About 3D Mechanism Visualization

This cutting-edge visualization technology allows researchers to explore drug-target interactions and 
molecular pathways in an immersive 3D environment. Understanding the spatial relationships between drugs 
and their targets is crucial for drug repurposing efforts.

#### Key Features

- **Interactive 3D protein-drug binding simulation**: Watch as the drug molecule approaches and binds to its target protein
- **Conformational changes visualization**: See how proteins change shape during binding events
- **Pathway network exploration**: Explore the complex network of signaling pathways affected by the drug
- **Interactive controls**: Rotate, zoom, and play animations to get different perspectives

#### Scientific Value

These visualizations help researchers:
1. Understand possible binding modes for repurposed drugs
2. Identify potential off-target interactions
3. Develop hypotheses about molecular mechanisms
4. Communicate complex biological concepts to diverse audiences

_This feature uses advanced computational models to generate simulations based on known pharmacological properties._
""")