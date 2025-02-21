import streamlit as st
import os

# Configure the main page
st.set_page_config(
    page_title="Operational Research Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open(os.path.join("assets", "styles.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main page content
st.title("📊 Operational Research Analysis Tool")
st.markdown("""
Welcome to the Operational Research Analysis Tool - a platform designed to make 
advanced analytical methods accessible to everyone.

### 🚀 Features
- **No coding required** - Just upload your data and select variables
- **Natural language explanations** - Get plain-English interpretations of results
- **Interactive visualizations** - Explore your data visually
- **AI-powered insights** - Ask questions about your analysis in natural language

### 📋 Available Tools
- **Regression Analysis** - Understand relationships between variables
- More tools coming soon!

### 🔒 Privacy & Security
- Your data remains private and is processed locally
- API keys are stored only in your browser session
- We never store or share your uploaded data or credentials

### 📚 Getting Started
Select a tool from the sidebar to begin your analysis.
""")


# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ for operational researchers and business analysts. "
    "[GitHub Repository]()"
)