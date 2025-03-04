import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pulp import *
import sys
import os
import json

# Add the root directory to the path to import from utils and components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.api_settings import render_api_settings
from utils.api import get_explanation

# Import the network solvers
from utils.network_solvers import (
    solve_min_cost_flow,
    solve_max_flow,
    solve_transportation,
    solve_shortest_path,
    solve_min_spanning_tree,
    format_solution_report
)

# Configure page
st.set_page_config(page_title="Network Optimization", page_icon="🕸️", layout="wide")
st.title("🕸️ Network Optimization")
st.markdown("""
This tool helps you analyze and optimize network-based problems using advanced optimization techniques.
Upload your network data, and our AI will help identify and solve the appropriate network optimization problem.
""")

# Get API settings from the component
api_settings = render_api_settings()

# File upload section
st.sidebar.header("📤 Upload Network Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    help="Upload your network data. The file should contain network structure information."
)

# After file upload and before data loading, add:
# Context input
st.header("📝 Problem Context")
user_context = st.text_area(
    "Describe your problem and data context",
    help="Provide any relevant information about your network problem, data structure, business context, or specific requirements.",
    height=150
)

def detect_network_problem(df, api_settings, user_context=""):
    """Use LLM to detect the type of network problem based on the data structure and user context."""
    # Prepare data summary for LLM
    columns_desc = df.dtypes.to_string()
    data_sample = df.head().to_string()
    unique_values = {col: df[col].nunique() for col in df.columns}
    
    prompt = f"""
    Analyze this network data and identify the most suitable network optimization problem type.
    
    User Context:
    {user_context}
    
    Data Structure:
    {columns_desc}
    
    Sample Data:
    {data_sample}
    
    Column Unique Values:
    {json.dumps(unique_values, indent=2)}
    
    Based on both the data structure and the user's context:
    
    1. Identify which type of network problem this data represents:
       - Minimum Cost Flow
       - Maximum Flow
       - Transportation Problem
       - Shortest Path
       - Minimum Spanning Tree
    
    2. Explain why this type was chosen, considering:
       - The user's described context
       - The data structure and content
       - The business implications
    
    3. Explain what each column represents in the network context
    
    4. Provide specific recommendations for:
       - Which nodes should be sources/sinks
       - What the weights/costs represent
       - Any special considerations for this problem type
    """
    
    response = get_explanation(prompt, "network problem identification", api_settings)
    return response

def create_network_visualization(G, pos, problem_type):
    """Create a network visualization based on the problem type."""
    plt.figure(figsize=(12, 8))
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, arrowsize=20)
    nx.draw_networkx_labels(G, pos)
    
    # Add edge labels based on problem type
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if problem_type in ['Minimum Cost Flow', 'Maximum Flow']:
        edge_labels = {k: f"c:{v}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title(f"Network Visualization - {problem_type}")
    return plt.gcf()

# Main analysis workflow
if uploaded_file is not None:
    try:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Data loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        # Privacy notice
        st.info("🔒 Your data is processed locally and is not stored or shared with third parties. Only the analysis context is sent to the AI provider when generating explanations.")

        # Data preview
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head())
            st.write("Data Types:")
            st.write(df.dtypes)

        # Problem Detection with context
        st.header("Problem Identification")
        problem_analysis = detect_network_problem(df, api_settings, user_context)
        st.markdown("### AI Analysis of Your Network Data")
        st.write(problem_analysis)

        # Network Structure Configuration
        st.header("Network Configuration")
        
        # Get LLM guidance for configuration
        config_guidance_prompt = f"""
        Based on the following information:
        
        User Context:
        {user_context}
        
        Data Structure:
        {df.dtypes.to_string()}
        
        Sample Data:
        {df.head().to_string()}
        
        Please provide specific guidance for configuring the network:
        1. Which column should be used as the source column and why?
        2. Which column should be used as the target column and why?
        3. Which column represents weights/costs and why?
        4. Are there any capacity constraints in the data?
        
        Explain your recommendations in simple terms, focusing on how they relate to the user's context.
        """
        
        config_guidance = get_explanation(config_guidance_prompt, "configuration guidance", api_settings)
        
        with st.expander("🤖 AI Configuration Guidance", expanded=True):
            st.write(config_guidance)
        
        col1, col2 = st.columns(2)
        with col1:
            source_col = st.selectbox("Select Source Column", df.columns)
            target_col = st.selectbox("Select Target Column", [col for col in df.columns if col != source_col])
            
        with col2:
            weight_col = st.selectbox("Select Weight/Cost Column", 
                                    [col for col in df.columns if col not in [source_col, target_col]])
            
            if "capacity" in df.columns:
                capacity_col = st.selectbox("Select Capacity Column (optional)", 
                                          ["None"] + [col for col in df.columns if col not in [source_col, target_col, weight_col]])

        # Create NetworkX graph
        if st.button("Generate Network Model", type="primary"):
            st.header("Network Model")
            
            # Create graph
            G = nx.from_pandas_edgelist(df, source_col, target_col, weight_col)
            
            # Visualization
            with st.expander("Network Visualization", expanded=True):
                pos = nx.spring_layout(G)
                fig = create_network_visualization(G, pos, "Network Model")
                st.pyplot(fig)
                plt.close()

            # Network Statistics
            with st.expander("Network Statistics", expanded=True):
                st.markdown("### Basic Network Metrics")
                metrics = {
                    "Number of Nodes": G.number_of_nodes(),
                    "Number of Edges": G.number_of_edges(),
                    "Average Degree": np.mean([d for n, d in G.degree()]),
                    "Is Connected": nx.is_connected(G),
                    "Density": nx.density(G)
                }
                
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")
                
                # Get LLM explanation of network metrics
                metrics_prompt = f"""
                Explain what these network metrics mean for the given network:
                {json.dumps(metrics, indent=2)}
                
                Focus on what these values suggest about the network structure and its implications for optimization.
                """
                metrics_explanation = get_explanation(metrics_prompt, "network metrics", api_settings)
                st.markdown("### Metrics Interpretation")
                st.write(metrics_explanation)

            # Optimization Model
            with st.expander("Optimization Model", expanded=True):
                st.markdown("### Mathematical Model")
                # Generate and display the mathematical model based on the problem type
                # This will be expanded based on the specific problem type detected
                
                # Example for a basic minimum cost flow problem
                model = LpProblem("Network_Optimization", LpMinimize)
                
                # Display model structure
                st.latex(r"""
                \text{Minimize } \sum_{(i,j) \in A} c_{ij}x_{ij} \\
                \text{Subject to:} \\
                \sum_{j:(i,j) \in A} x_{ij} - \sum_{j:(j,i) \in A} x_{ji} = b_i \quad \forall i \in N \\
                0 \leq x_{ij} \leq u_{ij} \quad \forall (i,j) \in A
                """)
                
                # Get LLM explanation of the model
                model_explanation = get_explanation(
                    "Explain this mathematical model in simple terms, focusing on what each component represents.",
                    "optimization model",
                    api_settings
                )
                st.markdown("### Model Interpretation")
                st.write(model_explanation)

            # Problem-Specific Parameters
            st.header("Problem Configuration")
            problem_type = st.selectbox(
                "Select Problem Type",
                ["Minimum Cost Flow", "Maximum Flow", "Transportation", 
                 "Shortest Path", "Minimum Spanning Tree"]
            )
            
            # Before problem-specific inputs, add:
            if problem_type:
                param_guidance_prompt = f"""
                Based on:
                
                User Context:
                {user_context}
                
                Selected Problem Type: {problem_type}
                Network Statistics:
                {json.dumps(metrics, indent=2)}
                
                Provide specific recommendations for setting the problem parameters:
                
                1. For sources/sinks selection:
                   - Which nodes should be selected as sources/sinks?
                   - What capacity/demand values would make sense?
                
                2. For flow/capacity values:
                   - What range of values would be appropriate?
                   - Are there any constraints to consider?
                
                3. Special considerations:
                   - Any specific nodes that need attention?
                   - Any business constraints to keep in mind?
                
                Explain your recommendations in business terms.
                """
                
                param_guidance = get_explanation(param_guidance_prompt, "parameter guidance", api_settings)
                
                with st.expander("🤖 AI Parameter Guidance", expanded=True):
                    st.write(param_guidance)

            # Problem-specific inputs
            if problem_type == "Minimum Cost Flow":
                sources = st.multiselect("Select Source Nodes", list(G.nodes()))
                sinks = st.multiselect("Select Sink Nodes", 
                                     [n for n in G.nodes() if n not in sources])
                
                # Node demands
                st.subheader("Node Demands")
                demands = {}
                for node in G.nodes():
                    if node in sources or node in sinks:
                        demands[node] = st.number_input(f"Demand for {node}", 
                                                      value=100 if node in sources else -100,
                                                      step=10)
            
            elif problem_type == "Maximum Flow":
                source = st.selectbox("Select Source Node", list(G.nodes()))
                sink = st.selectbox("Select Sink Node", 
                                  [n for n in G.nodes() if n != source])
            
            elif problem_type == "Transportation":
                sources = st.multiselect("Select Source Nodes", list(G.nodes()))
                sinks = st.multiselect("Select Sink Nodes", 
                                     [n for n in G.nodes() if n not in sources])
                
                # Supply and demand
                st.subheader("Supply and Demand")
                supply = {}
                demand = {}
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Supply at Sources")
                    for node in sources:
                        supply[node] = st.number_input(f"Supply at {node}", 
                                                     value=100, step=10)
                
                with col2:
                    st.write("Demand at Sinks")
                    for node in sinks:
                        demand[node] = st.number_input(f"Demand at {node}", 
                                                     value=100, step=10)
            
            elif problem_type == "Shortest Path":
                source = st.selectbox("Select Start Node", list(G.nodes()))
                target = st.selectbox("Select End Node", 
                                    [n for n in G.nodes() if n != source])
            
            # Solve button
            if st.button("Solve Problem", type="primary"):
                st.header("Solution")
                
                try:
                    # Get edge weights and capacities
                    weights = nx.get_edge_attributes(G, 'weight')
                    capacities = nx.get_edge_attributes(G, 'capacity') if 'capacity' in G.edges[list(G.edges())[0]] else None
                    
                    # Solve based on problem type
                    if problem_type == "Minimum Cost Flow":
                        solution = solve_min_cost_flow(G, sources, sinks, demands, weights, capacities)
                        
                    elif problem_type == "Maximum Flow":
                        solution = solve_max_flow(G, source, sink, capacities)
                        
                    elif problem_type == "Transportation":
                        solution = solve_transportation(sources, sinks, weights, supply, demand)
                        
                    elif problem_type == "Shortest Path":
                        solution = solve_shortest_path(G, source, target, weights)
                        
                    elif problem_type == "Minimum Spanning Tree":
                        solution = solve_min_spanning_tree(G, weights)
                    
                    # Format and display results
                    results_df = format_solution_report(problem_type, solution, G)
                    st.dataframe(results_df)
                    
                    # Update visualization with solution
                    st.subheader("Solution Visualization")
                    if problem_type in ["Minimum Cost Flow", "Maximum Flow", "Transportation"]:
                        # Color edges based on flow values
                        pos = nx.spring_layout(G)
                        plt.figure(figsize=(12, 8))
                        
                        # Draw nodes
                        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                             node_size=500)
                        
                        # Draw edges with varying widths based on flow
                        flow_dict = solution[0]
                        edge_widths = [1 + 2 * flow_dict.get((u, v), 0) 
                                     for u, v in G.edges()]
                        nx.draw_networkx_edges(G, pos, width=edge_widths)
                        
                        # Labels
                        nx.draw_networkx_labels(G, pos)
                        edge_labels = {(u, v): f"{flow_dict.get((u, v), 0):.1f}"
                                     for u, v in G.edges()}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels)
                        
                    elif problem_type == "Shortest Path":
                        # Highlight shortest path
                        path = solution[0]
                        pos = nx.spring_layout(G)
                        plt.figure(figsize=(12, 8))
                        
                        # Draw all nodes and edges
                        nx.draw_networkx_nodes(G, pos, node_color='lightgray')
                        nx.draw_networkx_edges(G, pos, edge_color='lightgray')
                        
                        # Highlight path
                        path_edges = list(zip(path[:-1], path[1:]))
                        nx.draw_networkx_nodes(G, pos, nodelist=path, 
                                             node_color='lightblue')
                        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                             edge_color='blue', width=2)
                        
                        # Labels
                        nx.draw_networkx_labels(G, pos)
                        edge_labels = nx.get_edge_attributes(G, 'weight')
                        nx.draw_networkx_edge_labels(G, pos, edge_labels)
                        
                    elif problem_type == "Minimum Spanning Tree":
                        # Highlight MST edges
                        mst_edges = solution[0]
                        pos = nx.spring_layout(G)
                        plt.figure(figsize=(12, 8))
                        
                        # Draw all nodes and edges
                        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
                        nx.draw_networkx_edges(G, pos, edge_color='lightgray')
                        
                        # Highlight MST
                        nx.draw_networkx_edges(G, pos, 
                                             edgelist=[(u, v) for u, v, _ in mst_edges],
                                             edge_color='blue', width=2)
                        
                        # Labels
                        nx.draw_networkx_labels(G, pos)
                        edge_labels = nx.get_edge_attributes(G, 'weight')
                        nx.draw_networkx_edge_labels(G, pos, edge_labels)
                    
                    st.pyplot(plt.gcf())
                    plt.close()
                    
                    # Get LLM explanation of the solution
                    solution_prompt = f"""
                    Explain the solution for this {problem_type} problem:
                    
                    Results:
                    {results_df.to_string()}
                    
                    Focus on:
                    1. What the solution means in practical terms
                    2. Key insights from the results
                    3. Any recommendations based on the solution
                    """
                    solution_explanation = get_explanation(solution_prompt, 
                                                        "optimization solution", 
                                                        api_settings)
                    st.markdown("### Solution Interpretation")
                    st.write(solution_explanation)
                    
                except Exception as e:
                    st.error(f"Error solving the problem: {str(e)}")
                    st.write("Please check your input parameters and try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data format and try again.")

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ for operational researchers and business analysts. "
    "[GitHub Repository](https://github.com/anshulyadav1976/OptiEase)"
) 