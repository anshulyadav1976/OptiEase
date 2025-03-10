import networkx as nx
import pulp
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MultiCommodityFlow:
    """Handle multi-commodity flow problems."""
    
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.commodities = []
        self.demands = {}
        
    def add_commodity(self, 
                     commodity_id: str,
                     source: str,
                     sink: str,
                     demand: float,
                     priority: float = 1.0):
        """Add a commodity with its source, sink, and demand."""
        self.commodities.append({
            'id': commodity_id,
            'source': source,
            'sink': sink,
            'demand': demand,
            'priority': priority
        })
        
    def solve(self) -> Dict:
        """Solve the multi-commodity flow problem."""
        # Create optimization problem
        prob = pulp.LpProblem("MultiCommodityFlow", pulp.LpMinimize)
        
        # Create flow variables for each commodity on each edge
        flows = {}
        for commodity in self.commodities:
            flows[commodity['id']] = {}
            for u, v in self.G.edges():
                flows[commodity['id']][(u, v)] = pulp.LpVariable(
                    f"flow_{commodity['id']}_{u}_{v}",
                    0,
                    self.G[u][v]['capacity']
                )
        
        # Objective: Minimize total weighted cost
        total_cost = pulp.lpSum(
            commodity['priority'] * 
            self.G[u][v].get('cost', 1) * 
            flows[commodity['id']][(u, v)]
            for commodity in self.commodities
            for u, v in self.G.edges()
        )
        prob += total_cost
        
        # Capacity constraints
        for u, v in self.G.edges():
            prob += (pulp.lpSum(flows[c['id']][(u, v)] 
                              for c in self.commodities) 
                    <= self.G[u][v]['capacity'])
        
        # Flow conservation constraints
        for commodity in self.commodities:
            for node in self.G.nodes():
                outflow = pulp.lpSum(flows[commodity['id']][(node, v)]
                                   for v in self.G.successors(node))
                inflow = pulp.lpSum(flows[commodity['id']][(u, node)]
                                  for u in self.G.predecessors(node))
                
                if node == commodity['source']:
                    prob += outflow - inflow == commodity['demand']
                elif node == commodity['sink']:
                    prob += outflow - inflow == -commodity['demand']
                else:
                    prob += outflow - inflow == 0
        
        # Solve
        prob.solve()
        
        # Extract solution
        solution = {
            'status': pulp.LpStatus[prob.status],
            'total_cost': pulp.value(prob.objective),
            'flows': {
                c['id']: {
                    (u, v): pulp.value(flows[c['id']][(u, v)])
                    for u, v in self.G.edges()
                }
                for c in self.commodities
            }
        }
        
        return solution

class TimeDependentFlow:
    """Handle time-dependent flow problems."""
    
    def __init__(self, G: nx.DiGraph, time_periods: int):
        self.G = G
        self.time_periods = time_periods
        self.time_varying_capacity = {}
        self.time_varying_demand = {}
        
    def set_capacity_profile(self, 
                           edge: Tuple[str, str], 
                           profile: List[float]):
        """Set time-varying capacity for an edge."""
        if len(profile) != self.time_periods:
            raise ValueError("Profile length must match time periods")
        self.time_varying_capacity[edge] = profile
        
    def set_demand_profile(self, 
                         source: str,
                         sink: str,
                         profile: List[float]):
        """Set time-varying demand between source and sink."""
        if len(profile) != self.time_periods:
            raise ValueError("Profile length must match time periods")
        self.time_varying_demand[(source, sink)] = profile
        
    def solve(self) -> Dict:
        """Solve the time-dependent flow problem."""
        # Create optimization problem
        prob = pulp.LpProblem("TimeDependentFlow", pulp.LpMinimize)
        
        # Create flow variables for each edge at each time period
        flows = {}
        for t in range(self.time_periods):
            flows[t] = {}
            for u, v in self.G.edges():
                flows[t][(u, v)] = pulp.LpVariable(
                    f"flow_{t}_{u}_{v}",
                    0,
                    self.time_varying_capacity.get((u, v), [self.G[u][v]['capacity']])[t]
                )
        
        # Objective: Minimize total cost over time
        total_cost = pulp.lpSum(
            self.G[u][v].get('cost', 1) * flows[t][(u, v)]
            for t in range(self.time_periods)
            for u, v in self.G.edges()
        )
        prob += total_cost
        
        # Time-dependent capacity constraints
        for t in range(self.time_periods):
            for u, v in self.G.edges():
                capacity = self.time_varying_capacity.get(
                    (u, v), 
                    [self.G[u][v]['capacity']] * self.time_periods
                )[t]
                prob += flows[t][(u, v)] <= capacity
        
        # Flow conservation constraints for each time period
        for t in range(self.time_periods):
            for node in self.G.nodes():
                outflow = pulp.lpSum(flows[t][(node, v)]
                                   for v in self.G.successors(node))
                inflow = pulp.lpSum(flows[t][(u, node)]
                                  for u in self.G.predecessors(node))
                
                # Check if node is source or sink in any demand profile
                net_flow = 0
                for (src, snk), profile in self.time_varying_demand.items():
                    if node == src:
                        net_flow += profile[t]
                    elif node == snk:
                        net_flow -= profile[t]
                
                prob += outflow - inflow == net_flow
        
        # Solve
        prob.solve()
        
        # Extract solution
        solution = {
            'status': pulp.LpStatus[prob.status],
            'total_cost': pulp.value(prob.objective),
            'flows': {
                t: {
                    (u, v): pulp.value(flows[t][(u, v)])
                    for u, v in self.G.edges()
                }
                for t in range(self.time_periods)
            }
        }
        
        return solution

class FlowWithUncertainty:
    """Handle flow problems with uncertain parameters."""
    
    def __init__(self, G: nx.DiGraph, num_scenarios: int):
        self.G = G
        self.num_scenarios = num_scenarios
        self.scenarios = []
        
    def add_scenario(self, 
                    capacity_factors: Dict[Tuple[str, str], float],
                    demand_factors: Dict[Tuple[str, str], float],
                    probability: float):
        """Add a scenario with capacity and demand uncertainty."""
        self.scenarios.append({
            'capacity_factors': capacity_factors,
            'demand_factors': demand_factors,
            'probability': probability
        })
        
    def solve(self) -> Dict:
        """Solve the flow problem under uncertainty."""
        # Create optimization problem
        prob = pulp.LpProblem("FlowWithUncertainty", pulp.LpMinimize)
        
        # Create flow variables for each scenario
        flows = {}
        for i, scenario in enumerate(self.scenarios):
            flows[i] = {}
            for u, v in self.G.edges():
                flows[i][(u, v)] = pulp.LpVariable(
                    f"flow_s{i}_{u}_{v}",
                    0,
                    self.G[u][v]['capacity'] * 
                    scenario['capacity_factors'].get((u, v), 1.0)
                )
        
        # Objective: Minimize expected cost
        expected_cost = pulp.lpSum(
            scenario['probability'] * 
            pulp.lpSum(self.G[u][v].get('cost', 1) * flows[i][(u, v)]
                      for u, v in self.G.edges())
            for i, scenario in enumerate(self.scenarios)
        )
        prob += expected_cost
        
        # Capacity constraints for each scenario
        for i, scenario in enumerate(self.scenarios):
            for u, v in self.G.edges():
                capacity = (self.G[u][v]['capacity'] * 
                          scenario['capacity_factors'].get((u, v), 1.0))
                prob += flows[i][(u, v)] <= capacity
        
        # Flow conservation constraints for each scenario
        for i, scenario in enumerate(self.scenarios):
            for node in self.G.nodes():
                outflow = pulp.lpSum(flows[i][(node, v)]
                                   for v in self.G.successors(node))
                inflow = pulp.lpSum(flows[i][(u, node)]
                                  for u in self.G.predecessors(node))
                
                # Apply demand factors if node is source or sink
                net_flow = 0
                for (src, snk), base_demand in self.G.graph.get('demands', {}).items():
                    if node == src:
                        net_flow += (base_demand * 
                                   scenario['demand_factors'].get((src, snk), 1.0))
                    elif node == snk:
                        net_flow -= (base_demand * 
                                   scenario['demand_factors'].get((src, snk), 1.0))
                
                prob += outflow - inflow == net_flow
        
        # Solve
        prob.solve()
        
        # Extract solution
        solution = {
            'status': pulp.LpStatus[prob.status],
            'expected_cost': pulp.value(prob.objective),
            'scenario_flows': {
                i: {
                    (u, v): pulp.value(flows[i][(u, v)])
                    for u, v in self.G.edges()
                }
                for i in range(len(self.scenarios))
            }
        }
        
        return solution 