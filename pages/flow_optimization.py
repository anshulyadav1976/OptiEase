import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pulp import *
import sys
import os
import json
from typing import Dict, List, Tuple, Optional
import io
from datetime import datetime
from utils.flow_animation import FlowAnimator
from utils.complex_flows import MultiCommodityFlow, TimeDependentFlow, FlowWithUncertainty

# Add the root directory to the path to import from utils and components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.api_settings import render_api_settings
from utils.api import get_explanation
from utils.flow_analysis import FlowAnalyzer
from utils.flow_visualization import FlowVisualizer

# Configure page
st.set_page_config(page_title="Flow Optimization", page_icon="🌊", layout="wide")
st.title("🌊 Flow Optimization")
st.markdown("""
This tool helps you solve maximum flow and minimum-cost flow problems using advanced algorithms.
Simply upload your network data, describe your problem, and get AI-guided optimization results.
""")

# Get API settings from the component
api_settings = render_api_settings()

def explain_algorithms(problem_type: str, data_info: Dict, api_settings: Dict) -> str:
    """Get AI explanation of suitable algorithms based on problem type and data."""
    prompt = f"""
    For a {problem_type} problem with the following characteristics:
    - Number of nodes: {data_info.get('num_nodes', 'unknown')}
    - Number of edges: {data_info.get('num_edges', 'unknown')}
    - Data structure: {data_info.get('data_structure', 'unknown')}
    
    Please explain:
    1. Which algorithms are available for solving this problem?
    2. What are the pros and cons of each algorithm?
    3. Which algorithm would you recommend for this specific case and why?
    4. Are there any special considerations for the recommended algorithm?
    
    Explain in simple terms that a non-technical person can understand.
    Focus on practical implications rather than mathematical details.
    """
    return get_explanation(prompt, "algorithm explanation", api_settings)

def validate_flow_data(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """Validate the uploaded data for flow problems."""
    required_cols = ['source', 'target', 'capacity']
    optional_cols = ['cost']
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}", {}
    
    # Convert string columns to string type explicitly
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    
    # Convert numeric columns to float
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    
    # Check for negative values in capacity
    if (df['capacity'] < 0).any():
        return False, "Capacity values cannot be negative", {}
    
    # Check for NaN values
    if df.isna().any().any():
        return False, "Dataset contains invalid or missing values", {}
    
    # Gather data info
    info = {
        'num_nodes': len(set(df['source']).union(set(df['target']))),
        'num_edges': len(df),
        'has_costs': 'cost' in df.columns,
        'data_structure': df.dtypes.to_dict(),
        'nodes': sorted(list(set(df['source']).union(set(df['target'])))),
        'edges': [(row['source'], row['target']) for _, row in df.iterrows()]
    }
    
    return True, "Data validation successful", info

def create_network_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Create a NetworkX graph from the dataframe."""
    G = nx.from_pandas_edgelist(
        df,
        source='source',
        target='target',
        edge_attr=['capacity', 'cost'] if 'cost' in df.columns else ['capacity'],
        create_using=nx.DiGraph
    )
    return G

def solve_max_flow(G: nx.DiGraph, source: str, sink: str, algorithm: str) -> Dict:
    """Solve maximum flow problem using specified algorithm."""
    if algorithm == "Ford-Fulkerson":
        flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity')
    elif algorithm == "Edmonds-Karp":
        flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity', flow_func=nx.edmonds_karp)
    elif algorithm == "Preflow-Push":
        flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity', flow_func=nx.preflow_push)
    
    return {
        'flow_value': flow_value,
        'flow_dict': flow_dict
    }

def solve_min_cost_flow(G: nx.DiGraph, source: str, sink: str, demand: float) -> Dict:
    """Solve minimum cost flow problem."""
    # Create demand dictionary
    demand_dict = {node: 0 for node in G.nodes()}
    demand_dict[source] = demand
    demand_dict[sink] = -demand
    
    # Solve min cost flow
    flow_dict = nx.min_cost_flow(G, demand_dict)
    cost = nx.cost_of_flow(G, flow_dict)
    
    return {
        'flow_dict': flow_dict,
        'total_cost': cost
    }

def visualize_flow_solution(G: nx.DiGraph, flow_dict: Dict, source: str, sink: str) -> plt.Figure:
    """Create visualization of the flow solution."""
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                          node_color='lightgreen', 
                          node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=[sink], 
                          node_color='lightcoral', 
                          node_size=500)
    
    # Draw edges with varying widths based on flow
    edge_widths = [1 + 2 * flow_dict[u][v] / G[u][v]['capacity'] 
                  for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True,
                          edge_color='gray', arrowstyle='->', 
                          connectionstyle='arc3, rad=0.1')
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    edge_labels = {(u, v): f"{flow_dict[u][v]}/{G[u][v]['capacity']}"
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Flow Network Solution")
    return plt.gcf()

# File upload section
st.sidebar.header("📤 Upload Network Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    help="Upload your network data. Required columns: source, target, capacity. Optional: cost"
)

# Main workflow
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Validate data
        is_valid, message, data_info = validate_flow_data(df)
        
        if not is_valid:
            st.error(message)
            # Get AI help for fixing data issues
            help_prompt = f"""
            The uploaded data has the following issue: {message}
            
            Please explain:
            1. What this error means in simple terms
            2. How to fix this issue
            3. What the correct data format should look like
            
            Include an example of properly formatted data.
            """
            help_explanation = get_explanation(help_prompt, "data help", api_settings)
            with st.expander("🤔 Need Help?"):
                st.write(help_explanation)
        else:
            st.success(f"Data loaded successfully! Nodes: {data_info['num_nodes']}, Edges: {data_info['num_edges']}")
            
            # Create graph
            G = create_network_graph(df)
            
            # AI Guidance for Problem Type
            with st.expander("🤖 Need help choosing the right problem type?", expanded=True):
                problem_guidance_prompt = f"""
                Based on the network data:
                - Number of nodes: {data_info['num_nodes']}
                - Number of edges: {data_info['num_edges']}
                - Has cost information: {data_info['has_costs']}
                - Network structure: {data_info['edges']}
                
                Please help the user choose the most appropriate problem type by:
                1. Explaining each available problem type
                2. Recommending the best type for their network
                3. Explaining the trade-offs between different choices
                4. Providing examples of when to use each type
                
                Keep the explanation business-focused and practical.
                """
                problem_guidance = get_explanation(problem_guidance_prompt, "problem guidance", api_settings)
                st.write(problem_guidance)
                
                user_problem_question = st.text_input(
                    "Ask any question about problem types",
                    placeholder="e.g., What's the difference between Maximum Flow and Minimum Cost Flow?"
                )
                if user_problem_question:
                    answer = get_explanation(user_problem_question, "problem type question", api_settings)
                    st.write(answer)
            
            # Problem Type Selection
            st.header("Problem Configuration")
            problem_type = st.selectbox(
                "Select Problem Type",
                ["Maximum Flow", "Minimum Cost Flow", "Multi-Commodity Flow", 
                 "Time-Dependent Flow", "Flow with Uncertainty"],
                help="Choose the type of flow problem you want to solve"
            )
            
            # AI Guidance for Node Selection
            with st.expander("🤖 Need help choosing source and sink nodes?", expanded=True):
                node_guidance_prompt = f"""
                Based on the network structure:
                Nodes: {data_info['nodes']}
                Edges: {data_info['edges']}
                Problem Type: {problem_type}
                
                Please help the user choose appropriate source and sink nodes by:
                1. Analyzing the network structure
                2. Identifying potential source nodes (nodes with many outgoing edges)
                3. Identifying potential sink nodes (nodes with many incoming edges)
                4. Recommending the best source-sink pairs
                5. Explaining why these nodes would be good choices
                
                Keep the explanation practical and business-focused.
                """
                node_guidance = get_explanation(node_guidance_prompt, "node guidance", api_settings)
                st.write(node_guidance)
                
                user_node_question = st.text_input(
                    "Ask any question about node selection",
                    placeholder="e.g., Which nodes would be best for my use case?"
                )
                if user_node_question:
                    answer = get_explanation(user_node_question, "node selection question", api_settings)
                    st.write(answer)
            
            # Get problem-specific configuration
            if problem_type == "Multi-Commodity Flow":
                st.subheader("Commodity Configuration")
                num_commodities = st.number_input(
                    "Number of Commodities",
                    min_value=1,
                    max_value=10,
                    value=2
                )
                
                commodities = []
                for i in range(num_commodities):
                    st.markdown(f"#### Commodity {i+1}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        source = st.selectbox(f"Source {i+1}", sorted(G.nodes()), key=f"src_{i}")
                    with col2:
                        sink = st.selectbox(f"Sink {i+1}", 
                                          [n for n in sorted(G.nodes()) if n != source],
                                          key=f"sink_{i}")
                    with col3:
                        demand = st.number_input(f"Demand {i+1}", 
                                               min_value=0.0,
                                               value=100.0,
                                               key=f"demand_{i}")
                    
                    priority = st.slider(f"Priority {i+1}", 
                                       min_value=0.1,
                                       max_value=1.0,
                                       value=1.0,
                                       key=f"priority_{i}")
                    
                    commodities.append({
                        'id': f"commodity_{i+1}",
                        'source': source,
                        'sink': sink,
                        'demand': demand,
                        'priority': priority
                    })
            
            elif problem_type == "Time-Dependent Flow":
                st.subheader("Time Configuration")
                num_periods = st.number_input(
                    "Number of Time Periods",
                    min_value=2,
                    max_value=24,
                    value=8
                )
                
                # Time-varying capacity
                st.markdown("#### Capacity Profiles")
                edges_to_vary = st.multiselect(
                    "Select Edges with Time-Varying Capacity",
                    [(u, v) for u, v in G.edges()]
                )
                
                capacity_profiles = {}
                for u, v in edges_to_vary:
                    st.markdown(f"Edge {u} → {v}")
                    profile = []
                    cols = st.columns(num_periods)
                    for t in range(num_periods):
                        with cols[t]:
                            cap = st.number_input(
                                f"t{t}",
                                min_value=0.0,
                                value=G[u][v]['capacity'],
                                key=f"cap_{u}_{v}_{t}"
                            )
                            profile.append(cap)
                    capacity_profiles[(u, v)] = profile
                
                # Time-varying demand
                st.markdown("#### Demand Profile")
                source = st.selectbox("Source", sorted(G.nodes()))
                sink = st.selectbox("Sink", [n for n in sorted(G.nodes()) if n != source])
                
                demand_profile = []
                cols = st.columns(num_periods)
                for t in range(num_periods):
                    with cols[t]:
                        demand = st.number_input(
                            f"t{t}",
                            min_value=0.0,
                            value=100.0,
                            key=f"demand_{t}"
                        )
                        demand_profile.append(demand)
            
            elif problem_type == "Flow with Uncertainty":
                st.subheader("Uncertainty Configuration")
                
                # Node Selection
                col1, col2 = st.columns(2)
                with col1:
                    source = st.selectbox("Select Source Node", sorted(G.nodes()), key="source_uncertainty")
                with col2:
                    sink = st.selectbox("Select Sink Node", 
                                      [n for n in sorted(G.nodes()) if n != source],
                                      key="sink_uncertainty")
                
                num_scenarios = st.number_input(
                    "Number of Scenarios",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                scenarios = []
                for i in range(num_scenarios):
                    st.markdown(f"#### Scenario {i+1}")
                    probability = st.slider(
                        f"Probability",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/num_scenarios,
                        key=f"prob_{i}"
                    )
                    
                    # Capacity factors
                    st.markdown("Capacity Factors")
                    capacity_factors = {}
                    for u, v in G.edges():
                        factor = st.slider(
                            f"Edge {u} → {v}",
                            min_value=0.5,
                            max_value=1.5,
                            value=1.0,
                            key=f"cap_{i}_{u}_{v}"
                        )
                        capacity_factors[(u, v)] = factor
                    
                    # Demand factors
                    st.markdown("Demand Factors")
                    demand_factors = {}
                    factor = st.slider(
                        "Demand Factor",
                        min_value=0.5,
                        max_value=1.5,
                        value=1.0,
                        key=f"dem_{i}"
                    )
                    demand_factors[(source, sink)] = factor
                    
                    scenarios.append({
                        'probability': probability,
                        'capacity_factors': capacity_factors,
                        'demand_factors': demand_factors
                    })
            
            # Get algorithm explanation
            algo_explanation = explain_algorithms(problem_type, data_info, api_settings)
            with st.expander("🤖 Algorithm Recommendations", expanded=True):
                st.write(algo_explanation)
            
            # Solve button
            if st.button("Solve Problem", type="primary"):
                st.header("Solution")
                
                try:
                    # Solve based on problem type
                    if problem_type == "Maximum Flow":
                        algorithm = st.selectbox(
                            "Select Algorithm",
                            ["Ford-Fulkerson", "Edmonds-Karp", "Preflow-Push"],
                            help="Choose the algorithm to solve the maximum flow problem"
                        )
                        solution = solve_max_flow(G, source, sink, algorithm)
                        st.subheader(f"Maximum Flow: {solution['flow_value']:.2f}")
                    
                    elif problem_type == "Minimum Cost Flow":
                        demand = st.number_input("Enter Required Flow Amount", 
                                               min_value=0.0,
                                               value=100.0)
                        solution = solve_min_cost_flow(G, source, sink, demand)
                        st.subheader(f"Total Cost: {solution['total_cost']:.2f}")
                    
                    elif problem_type == "Multi-Commodity Flow":
                        # Create and solve multi-commodity flow problem
                        mcf = MultiCommodityFlow(G)
                        for commodity in commodities:
                            mcf.add_commodity(**commodity)
                        
                        solution = mcf.solve()
                        st.subheader(f"Total Cost: {solution['total_cost']:.2f}")
                        
                        # Display commodity-specific results
                        st.markdown("### Results by Commodity")
                        for commodity in commodities:
                            with st.expander(f"Commodity {commodity['id']}"):
                                flow_df = pd.DataFrame([
                                    {
                                        "From": u,
                                        "To": v,
                                        "Flow": solution['flows'][commodity['id']][(u, v)],
                                        "Capacity": G[u][v]['capacity']
                                    }
                                    for u, v in G.edges()
                                ])
                                st.dataframe(flow_df)
                    
                    elif problem_type == "Time-Dependent Flow":
                        # Create and solve time-dependent flow problem
                        tdf = TimeDependentFlow(G, num_periods)
                        
                        # Set capacity profiles
                        for edge, profile in capacity_profiles.items():
                            tdf.set_capacity_profile(edge, profile)
                        
                        # Set demand profile
                        tdf.set_demand_profile(source, sink, demand_profile)
                        
                        solution = tdf.solve()
                        st.subheader(f"Total Cost: {solution['total_cost']:.2f}")
                        
                        # Display time-dependent results
                        st.markdown("### Results by Time Period")
                        for t in range(num_periods):
                            with st.expander(f"Time Period {t}"):
                                flow_df = pd.DataFrame([
                                    {
                                        "From": u,
                                        "To": v,
                                        "Flow": solution['flows'][t][(u, v)],
                                        "Capacity": (capacity_profiles.get((u, v), 
                                                   [G[u][v]['capacity']]*num_periods)[t])
                                    }
                                    for u, v in G.edges()
                                ])
                                st.dataframe(flow_df)
                    
                    else:  # Flow with Uncertainty
                        # Create and solve uncertain flow problem
                        uflow = FlowWithUncertainty(G, num_scenarios)
                        for scenario in scenarios:
                            uflow.add_scenario(**scenario)
                        
                        solution = uflow.solve()
                        st.subheader(f"Expected Cost: {solution['expected_cost']:.2f}")
                        
                        # Display scenario-specific results
                        st.markdown("### Results by Scenario")
                        for i, scenario in enumerate(scenarios):
                            with st.expander(f"Scenario {i+1} (Probability: {scenario['probability']:.2f})"):
                                flow_df = pd.DataFrame([
                                    {
                                        "From": u,
                                        "To": v,
                                        "Flow": solution['scenario_flows'][i][(u, v)],
                                        "Capacity": G[u][v]['capacity'] * 
                                                  scenario['capacity_factors'].get((u, v), 1.0)
                                    }
                                    for u, v in G.edges()
                                ])
                                st.dataframe(flow_df)
                    
                    # Create animator and visualizer
                    animator = FlowAnimator(G, solution['flow_dict'] 
                                         if problem_type in ["Maximum Flow", "Minimum Cost Flow"]
                                         else solution['flows'][0])
                    visualizer = FlowVisualizer(G, solution['flow_dict'], source, sink)
                    
                    # Visualization options
                    st.header("🎯 Visualization")
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Network Graph", "Flow Animation", "Flow Explorer", 
                         "Flow Paths", "Utilization Heatmap", "Sankey Diagram"]
                    )
                    
                    if viz_type == "Network Graph":
                        fig = visualizer.create_network_graph()
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Flow Animation":
                        fig = animator.create_flow_propagation_animation()
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Flow Explorer":
                        fig = animator.create_interactive_flow_explorer()
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        # Use existing visualization code
                        fig = visualize_flow_solution(G, solution['flow_dict'], source, sink)
                        st.pyplot(fig)
                    
                    # Get solution explanation
                    solution_prompt = f"""
                    Explain this {problem_type} solution in simple terms:
                    
                    Key Results:
                    - {'Maximum Flow' if problem_type == 'Maximum Flow' else 'Total Cost'}: {solution.get('flow_value', solution.get('total_cost')):.2f}
                    - Number of edges with flow: {sum(1 for u, v in G.edges() if solution['flow_dict'][u][v] > 0)}
                    - Average edge utilization: {flow_df['Utilization'].str.rstrip('%').astype(float).mean():.1f}%
                    
                    Please explain:
                    1. What these results mean in practical terms
                    2. Key insights from the solution
                    3. Any potential bottlenecks or areas for improvement
                    4. Recommendations based on these results
                    
                    Use business-friendly language and avoid technical jargon.
                    """
                    solution_explanation = get_explanation(solution_prompt, "solution explanation", api_settings)
                    
                    st.markdown("### 📊 Solution Interpretation")
                    st.write(solution_explanation)
                    
                    # Q&A Section
                    st.header("❓ Ask Questions")
                    user_question = st.text_input(
                        "Ask any question about the solution",
                        placeholder="e.g., What does the maximum flow value mean for my business?"
                    )
                    
                    if user_question:
                        qa_prompt = f"""
                        Question: {user_question}
                        
                        Context:
                        - Problem Type: {problem_type}
                        - Solution Results: {json.dumps(solution, default=str)}
                        - Network Size: {data_info['num_nodes']} nodes, {data_info['num_edges']} edges
                        
                        Please provide a clear, business-focused answer that a non-technical person can understand.
                        """
                        answer = get_explanation(qa_prompt, "question answering", api_settings)
                        st.write(answer)
                    
                    # After finding the solution, create analyzer and visualizer
                    analyzer = FlowAnalyzer(G, solution['flow_dict'], source, sink)
                    
                    # Enhanced Analysis Section
                    st.header("📊 Network Analysis")
                    
                    # Performance Metrics
                    metrics = analyzer.get_performance_metrics()
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Flow", 
                                f"{metrics['flow_metrics']['total_flow']:.2f}",
                                help="Total flow through the network")
                        st.metric("Active Paths",
                                metrics['flow_metrics']['num_active_paths'],
                                help="Number of paths carrying flow")
                    
                    with col2:
                        st.metric("Average Utilization",
                                f"{metrics['utilization_metrics']['mean']*100:.1f}%",
                                help="Average capacity utilization across all edges")
                        st.metric("Network Connectivity",
                                metrics['resilience_metrics']['edge_connectivity'],
                                help="Minimum number of edges to remove to disconnect the network")
                    
                    with col3:
                        st.metric("Critical Points",
                                metrics['resilience_metrics']['num_critical_edges'],
                                help="Number of critical edges (bottlenecks)")
                        st.metric("Average Path Length",
                                f"{metrics['flow_metrics']['avg_path_length']:.1f}",
                                help="Average number of edges in active paths")
                    
                    # Network Visualization Section
                    st.header("🎯 Network Visualization")
                    
                    # Visualization type selector
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Network Graph", "Flow Paths", "Utilization Heatmap", "Sankey Diagram"]
                    )
                    
                    if viz_type == "Network Graph":
                        fig = visualizer.create_network_graph()
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "Flow Paths":
                        path_flows = analyzer.get_path_flows()
                        fig = visualizer.create_path_visualization(path_flows)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Path details
                        with st.expander("📍 Path Details"):
                            for i, path in enumerate(path_flows):
                                st.write(f"Path {i+1}:")
                                st.write(f"- Nodes: {' → '.join(path['path'])}")
                                st.write(f"- Flow: {path['flow']:.2f}")
                                st.write(f"- Length: {path['length']} edges")
                                st.write(f"- Bottleneck Capacity: {path['bottleneck']:.2f}")
                                st.write("---")
                        
                    elif viz_type == "Utilization Heatmap":
                        fig = visualizer.create_utilization_heatmap()
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:  # Sankey Diagram
                        fig = visualizer.create_sankey_diagram()
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Network Analysis Section
                    st.header("🔍 Detailed Analysis")
                    
                    # Bottleneck Analysis
                    bottleneck_analysis = analyzer.bottleneck_analysis()
                    with st.expander("🚧 Bottleneck Analysis", expanded=True):
                        if bottleneck_analysis['bottlenecks']:
                            st.warning(f"Found {len(bottleneck_analysis['bottlenecks'])} critical bottlenecks")
                            for b in bottleneck_analysis['bottlenecks']:
                                st.write(f"Edge {b['edge'][0]} → {b['edge'][1]}:")
                                st.write(f"- Utilization: {b['utilization']*100:.1f}%")
                                st.write(f"- Flow: {b['flow']:.2f}")
                                st.write(f"- Capacity: {b['capacity']:.2f}")
                        else:
                            st.success("No critical bottlenecks found!")
                            
                        if bottleneck_analysis['near_bottlenecks']:
                            st.info(f"Found {len(bottleneck_analysis['near_bottlenecks'])} potential future bottlenecks")
                            for b in bottleneck_analysis['near_bottlenecks']:
                                st.write(f"Edge {b['edge'][0]} → {b['edge'][1]}:")
                                st.write(f"- Utilization: {b['utilization']*100:.1f}%")
                    
                    # Resilience Analysis
                    resilience = analyzer.analyze_resilience()
                    with st.expander("🛡️ Network Resilience"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Edge Connectivity", resilience['edge_connectivity'])
                            st.write("Minimum edges to disconnect network")
                        with col2:
                            st.metric("Node Connectivity", resilience['node_connectivity'])
                            st.write("Minimum nodes to disconnect network")
                        
                        if resilience['critical_edges']:
                            st.warning("Critical Edges (Single Points of Failure):")
                            for edge in resilience['critical_edges']:
                                st.write(f"- {edge[0]} → {edge[1]}")
                    
                    # Node Importance Analysis
                    importance = analyzer.node_importance_analysis()
                    with st.expander("🎯 Node Importance"):
                        # Create node importance dataframe
                        importance_df = pd.DataFrame({
                            'Node': list(importance['flow_centrality'].keys()),
                            'Flow Centrality': list(importance['flow_centrality'].values()),
                            'Betweenness': list(importance['betweenness_centrality'].values())
                        }).sort_values('Flow Centrality', ascending=False)
                        
                        st.dataframe(importance_df)
                    
                    # Improvement Suggestions
                    suggestions = analyzer.generate_improvement_suggestions()
                    if suggestions:
                        st.header("💡 Improvement Suggestions")
                        
                        for suggestion in suggestions:
                            if suggestion['priority'] == 'high':
                                st.error(suggestion['reason'])
                            elif suggestion['priority'] == 'medium':
                                st.warning(suggestion['reason'])
                            else:
                                st.info(suggestion['reason'])
                            
                            if suggestion['type'] == 'capacity_increase':
                                st.write(f"Suggested Action: Increase capacity of edge "
                                       f"{suggestion['edge'][0]} → {suggestion['edge'][1]} "
                                       f"by {suggestion['suggested_increase']:.2f}")
                            elif suggestion['type'] == 'redundancy':
                                st.write(f"Suggested Action: Add alternative path around "
                                       f"{suggestion['edge'][0]} → {suggestion['edge'][1]}")
                            elif suggestion['type'] == 'node_capacity':
                                st.write(f"Suggested Action: Consider load balancing around node {suggestion['node']}")
                    
                    # Get AI insights
                    insights_prompt = f"""
                    Based on the network analysis:
                    
                    Performance Metrics:
                    {json.dumps(metrics, indent=2)}
                    
                    Bottleneck Analysis:
                    {json.dumps(bottleneck_analysis, indent=2)}
                    
                    Resilience Analysis:
                    {json.dumps(resilience, indent=2)}
                    
                    Please provide:
                    1. Key insights about the network's performance
                    2. Potential risks and vulnerabilities
                    3. Specific recommendations for improvement
                    4. Business impact analysis
                    
                    Focus on practical implications and use business-friendly language.
                    """
                    insights = get_explanation(insights_prompt, "network insights", api_settings)
                    
                    st.header("🎯 AI Insights")
                    st.write(insights)
                    
                    # Export Section
                    st.header("📥 Export Results")
                    
                    # Create summary report
                    summary_df = analyzer.generate_summary_report()
                    
                    # Export options
                    export_format = st.selectbox(
                        "Select Export Format",
                        ["Excel", "CSV", "JSON"]
                    )
                    
                    if export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer) as writer:
                            summary_df.to_excel(writer, sheet_name="Summary", index=False)
                            flow_df.to_excel(writer, sheet_name="Flow Details", index=False)
                            importance_df.to_excel(writer, sheet_name="Node Importance", index=False)
                        
                        st.download_button(
                            label="Download Excel Report",
                            data=buffer.getvalue(),
                            file_name=f"flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    
                    elif export_format == "CSV":
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV Report",
                            data=csv,
                            file_name=f"flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    else:  # JSON
                        json_data = {
                            'summary': summary_df.to_dict('records'),
                            'flow_details': flow_df.to_dict('records'),
                            'metrics': metrics,
                            'bottlenecks': bottleneck_analysis,
                            'resilience': resilience,
                            'suggestions': suggestions
                        }
                        
                        st.download_button(
                            label="Download JSON Report",
                            data=json.dumps(json_data, indent=2),
                            file_name=f"flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"Error solving the problem: {str(e)}")
                    st.write("Please check your input parameters and try again.")
                    
                    # Get error explanation
                    error_prompt = f"""
                    An error occurred while solving the problem: {str(e)}
                    
                    Please explain:
                    1. What might have caused this error in simple terms
                    2. How to fix this issue
                    3. How to prevent this issue in the future
                    
                    Explain in non-technical terms.
                    """
                    error_explanation = get_explanation(error_prompt, "error explanation", api_settings)
                    with st.expander("🔍 Error Analysis"):
                        st.write(error_explanation)
    
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.write("Please check your file format and try again.")

# Add help section
with st.sidebar.expander("📖 How to Use This Tool"):
    help_prompt = """
    Explain how to use this flow optimization tool for non-technical users:
    1. What kind of problems can it solve?
    2. What data format is needed?
    3. How to interpret the results?
    4. Common use cases and examples
    
    Keep the explanation simple and business-focused.
    """
    help_text = get_explanation(help_prompt, "tool help", api_settings)
    st.write(help_text)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ for making network optimization accessible to everyone. "
    "Need help? Use the Q&A section to ask any questions!"
) 