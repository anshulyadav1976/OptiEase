# OptiEase 📊

> Making Operations Research accessible to everyone through AI-powered optimization

OptiEase is an intelligent web application that democratizes advanced operations research techniques by combining powerful optimization algorithms with AI-driven guidance. Built for businesses and analysts who need sophisticated optimization solutions without the complexity.

## 🌟 Features

### Current Features

#### 1. Regression Analysis Tool
- Multiple regression types support
- Automated variable selection
- Interactive visualizations
- AI-powered insights and interpretations
- Natural language explanations of results

#### 2. Network Optimization Tool 🕸️
- **Intelligent Problem Detection**
  - Upload your network data
  - Provide context in plain English
  - AI automatically identifies the optimal problem type
  - Get guided recommendations for configuration

- **Supported Network Problems**
  - Minimum Cost Flow
  - Maximum Flow
  - Transportation Problem
  - Shortest Path
  - Minimum Spanning Tree

- **Interactive Visualization**
  - Dynamic network graphs
  - Flow visualization
  - Path highlighting
  - Cost/capacity labeling

- **AI-Powered Guidance**
  - Context-aware configuration suggestions
  - Parameter recommendations
  - Business-focused explanations
  - Natural language interpretations

### 🚀 Weekly Feature Updates
We're continuously improving OptiEase with new features and optimizations. Coming soon:
- Linear Programming Module
- Integer Programming Solutions
- Queueing Theory Analysis
- Inventory Optimization
- And more!

## 🛠️ Technical Stack

### Backend
- Python 3.8+
- NetworkX for graph operations
- PuLP for optimization
- Streamlit for web interface
- Pandas & NumPy for data handling
- Matplotlib & Seaborn for visualization

### AI Integration
- OpenAI/Anthropic for natural language processing
- Custom prompt engineering for domain-specific insights
- Context-aware recommendation system

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/anshulyadav1976/OptiEase.git
cd OptiEase
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
- Create a `.env` file in the root directory
- Add your API keys:
  ```
  OPENAI_API_KEY=your_key_here
  ANTHROPIC_API_KEY=your_key_here
  ```

5. Run the application:
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### 1. Regression Analysis
1. Upload your dataset (CSV/Excel)
2. Select dependent and independent variables
3. Choose regression type
4. Get instant analysis with visualizations and AI explanations

### 2. Network Optimization
1. Upload network data
2. Describe your problem context
3. Review AI's problem identification
4. Configure network parameters with AI guidance
5. Get optimized solution with visual insights

## 🏗️ Project Structure
```
OptiEase/
├── app.py                 # Main application entry
├── requirements.txt       # Project dependencies
├── assets/               # Static assets and styles
├── components/           # Reusable UI components
├── pages/                # Application pages
│   ├── regression_analysis.py
│   └── network_optimization.py
└── utils/                # Utility functions
    ├── api.py           # API integration
    └── network_solvers.py # Network optimization algorithms
```

## 🤝 Contributing
We welcome contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 UI/UX enhancements

.



## 🙏 Acknowledgments
- NetworkX team for the excellent graph library
- PuLP developers for the optimization framework
- Streamlit team for the amazing web framework
- OpenAI/Anthropic for powerful language models

## 📬 Contact
- LinkedIn: https://www.linkedin.com/in/anshulyadav1976/
---

Built with ❤️ for making operations research accessible to everyone.

*Note: This is an actively developed project. New features are shipped weekly!* 