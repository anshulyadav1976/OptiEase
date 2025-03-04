import networkx as nx
from pulp import *
import pandas as pd

def solve_min_cost_flow(G, sources, sinks, demands, costs, capacities=None):
    """
    Solve a minimum cost flow problem.
    
    Args:
        G (nx.DiGraph): NetworkX directed graph
        sources (list): Source nodes
        sinks (list): Sink nodes
        demands (dict): Node demands (positive for sources, negative for sinks)
        costs (dict): Edge costs
        capacities (dict, optional): Edge capacities
    
    Returns:
        dict: Flow values for each edge
        float: Total cost
        str: Solution status
    """
    # Create optimization problem
    prob = LpProblem("MinCostFlow", LpMinimize)
    
    # Create flow variables
    flows = LpVariable.dicts("flow", 
                            ((i, j) for i, j in G.edges()), 
                            lowBound=0,
                            upBound=(capacities.get((i, j)) if capacities else None))
    
    # Objective function
    prob += lpSum(costs[i, j] * flows[i, j] for i, j in G.edges())
    
    # Flow conservation constraints
    for node in G.nodes():
        outflow = lpSum(flows[i, j] for i, j in G.out_edges(node))
        inflow = lpSum(flows[i, j] for j, i in G.in_edges(node))
        prob += outflow - inflow == demands.get(node, 0)
    
    # Solve
    prob.solve()
    
    # Get results
    flow_dict = {(i, j): flows[i, j].value() for i, j in G.edges()}
    total_cost = value(prob.objective)
    status = LpStatus[prob.status]
    
    return flow_dict, total_cost, status

def solve_max_flow(G, source, sink, capacities):
    """
    Solve a maximum flow problem.
    
    Args:
        G (nx.DiGraph): NetworkX directed graph
        source (str): Source node
        sink (str): Sink node
        capacities (dict): Edge capacities
    
    Returns:
        dict: Flow values for each edge
        float: Maximum flow value
    """
    return nx.maximum_flow(G, source, sink, capacity=capacities)

def solve_transportation(sources, sinks, costs, supply, demand):
    """
    Solve a transportation problem.
    
    Args:
        sources (list): Source nodes
        sinks (list): Sink nodes
        costs (dict): Transportation costs
        supply (dict): Supply at each source
        demand (dict): Demand at each sink
    
    Returns:
        dict: Shipping quantities
        float: Total cost
        str: Solution status
    """
    # Create optimization problem
    prob = LpProblem("Transportation", LpMinimize)
    
    # Create variables
    routes = [(i, j) for i in sources for j in sinks]
    shipments = LpVariable.dicts("ship", routes, lowBound=0)
    
    # Objective function
    prob += lpSum(costs[i, j] * shipments[i, j] for i, j in routes)
    
    # Supply constraints
    for i in sources:
        prob += lpSum(shipments[i, j] for j in sinks) <= supply[i]
    
    # Demand constraints
    for j in sinks:
        prob += lpSum(shipments[i, j] for i in sources) >= demand[j]
    
    # Solve
    prob.solve()
    
    # Get results
    ship_dict = {(i, j): shipments[i, j].value() for i, j in routes}
    total_cost = value(prob.objective)
    status = LpStatus[prob.status]
    
    return ship_dict, total_cost, status

def solve_shortest_path(G, source, target, weights):
    """
    Solve a shortest path problem.
    
    Args:
        G (nx.Graph): NetworkX graph
        source (str): Source node
        target (str): Target node
        weights (dict): Edge weights
    
    Returns:
        list: Path nodes
        float: Path length
    """
    path = nx.shortest_path(G, source, target, weight=weights)
    length = nx.shortest_path_length(G, source, target, weight=weights)
    return path, length

def solve_min_spanning_tree(G, weights):
    """
    Solve a minimum spanning tree problem.
    
    Args:
        G (nx.Graph): NetworkX graph
        weights (dict): Edge weights
    
    Returns:
        list: MST edges
        float: Total weight
    """
    mst_edges = nx.minimum_spanning_edges(G, weight=weights, data=True)
    total_weight = sum(w for _, _, w in mst_edges)
    return list(mst_edges), total_weight

def format_solution_report(problem_type, solution_data, G):
    """
    Format the solution results into a pandas DataFrame for display.
    
    Args:
        problem_type (str): Type of network problem
        solution_data (tuple): Solution data from respective solver
        G (nx.Graph): NetworkX graph
    
    Returns:
        pd.DataFrame: Formatted solution report
    """
    if problem_type == "Minimum Cost Flow":
        flow_dict, total_cost, status = solution_data
        df = pd.DataFrame([
            {"From": i, "To": j, "Flow": flow, "Cost": G[i][j]["weight"]}
            for (i, j), flow in flow_dict.items()
        ])
        df["Total Cost"] = df["Flow"] * df["Cost"]
        
    elif problem_type == "Maximum Flow":
        flow_dict, max_flow = solution_data
        df = pd.DataFrame([
            {"From": i, "To": j, "Flow": flow, "Capacity": G[i][j]["capacity"]}
            for (i, j), flow in flow_dict.items()
        ])
        
    elif problem_type == "Transportation":
        ship_dict, total_cost, status = solution_data
        df = pd.DataFrame([
            {"From": i, "To": j, "Quantity": qty, "Cost": G[i][j]["weight"]}
            for (i, j), qty in ship_dict.items()
        ])
        df["Total Cost"] = df["Quantity"] * df["Cost"]
        
    elif problem_type == "Shortest Path":
        path, length = solution_data
        path_edges = list(zip(path[:-1], path[1:]))
        df = pd.DataFrame([
            {"From": i, "To": j, "Distance": G[i][j]["weight"]}
            for i, j in path_edges
        ])
        
    elif problem_type == "Minimum Spanning Tree":
        mst_edges, total_weight = solution_data
        df = pd.DataFrame([
            {"From": i, "To": j, "Weight": w}
            for i, j, w in mst_edges
        ])
        
    return df 