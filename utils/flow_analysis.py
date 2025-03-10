import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.stats import percentileofscore

class FlowAnalyzer:
    def __init__(self, G: nx.DiGraph, flow_dict: Dict, source: str, sink: str):
        self.G = G
        self.flow_dict = flow_dict
        self.source = source
        self.sink = sink
        
    def get_path_flows(self) -> List[Dict]:
        """Analyze all paths and their flows."""
        all_paths = list(nx.all_simple_paths(self.G, self.source, self.sink))
        path_flows = []
        
        for path in all_paths:
            # Calculate flow along this path
            path_edges = list(zip(path[:-1], path[1:]))
            path_flow = min(self.flow_dict[u][v] for u, v in path_edges)
            
            if path_flow > 0:
                path_flows.append({
                    'path': path,
                    'flow': path_flow,
                    'length': len(path) - 1,
                    'bottleneck': min((self.G[u][v]['capacity'] for u, v in path_edges))
                })
        
        return sorted(path_flows, key=lambda x: x['flow'], reverse=True)

    def analyze_resilience(self) -> Dict:
        """Analyze network resilience and reliability."""
        # Edge connectivity
        edge_conn = nx.edge_connectivity(self.G)
        
        # Node connectivity
        node_conn = nx.node_connectivity(self.G)
        
        # Critical edges (bridges)
        bridges = list(nx.bridges(self.G.to_undirected()))
        
        # Critical nodes (articulation points)
        cut_nodes = list(nx.articulation_points(self.G.to_undirected()))
        
        # Alternative paths analysis
        alt_paths = {}
        for node in self.G.nodes():
            if node not in [self.source, self.sink]:
                # Remove node and check if path exists
                H = self.G.copy()
                H.remove_node(node)
                alt_paths[node] = nx.has_path(H, self.source, self.sink)
        
        return {
            'edge_connectivity': edge_conn,
            'node_connectivity': node_conn,
            'critical_edges': bridges,
            'critical_nodes': cut_nodes,
            'alternative_paths': alt_paths
        }

    def capacity_utilization_analysis(self) -> Dict:
        """Analyze capacity utilization patterns."""
        utilization = {}
        for u, v in self.G.edges():
            flow = self.flow_dict[u][v]
            capacity = self.G[u][v]['capacity']
            utilization[(u, v)] = flow / capacity
        
        util_values = list(utilization.values())
        
        return {
            'utilization': utilization,
            'mean_utilization': np.mean(util_values),
            'median_utilization': np.median(util_values),
            'std_utilization': np.std(util_values),
            'percentiles': {
                '25th': np.percentile(util_values, 25),
                '75th': np.percentile(util_values, 75),
                '90th': np.percentile(util_values, 90)
            }
        }

    def bottleneck_analysis(self) -> Dict:
        """Identify and analyze bottlenecks."""
        bottlenecks = []
        near_bottlenecks = []
        
        for u, v in self.G.edges():
            flow = self.flow_dict[u][v]
            capacity = self.G[u][v]['capacity']
            utilization = flow / capacity
            
            if utilization > 0.9:
                bottlenecks.append({
                    'edge': (u, v),
                    'utilization': utilization,
                    'flow': flow,
                    'capacity': capacity
                })
            elif utilization > 0.7:
                near_bottlenecks.append({
                    'edge': (u, v),
                    'utilization': utilization,
                    'flow': flow,
                    'capacity': capacity
                })
        
        return {
            'bottlenecks': bottlenecks,
            'near_bottlenecks': near_bottlenecks,
            'num_bottlenecks': len(bottlenecks),
            'num_near_bottlenecks': len(near_bottlenecks)
        }

    def node_importance_analysis(self) -> Dict:
        """Analyze the importance of each node in the network."""
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.G)
        
        # Flow centrality (custom metric based on flow passing through node)
        flow_centrality = {}
        for node in self.G.nodes():
            if node in [self.source, self.sink]:
                continue
            incoming_flow = sum(self.flow_dict.get(u, {}).get(node, 0) 
                              for u in self.G.predecessors(node))
            outgoing_flow = sum(self.flow_dict.get(node, {}).get(v, 0) 
                              for v in self.G.successors(node))
            flow_centrality[node] = (incoming_flow + outgoing_flow) / 2
        
        return {
            'betweenness_centrality': betweenness,
            'flow_centrality': flow_centrality
        }

    def generate_improvement_suggestions(self) -> List[Dict]:
        """Generate suggestions for network improvement."""
        suggestions = []
        
        # Analyze bottlenecks
        bottleneck_analysis = self.bottleneck_analysis()
        if bottleneck_analysis['bottlenecks']:
            for bottleneck in bottleneck_analysis['bottlenecks']:
                suggestions.append({
                    'type': 'capacity_increase',
                    'priority': 'high',
                    'edge': bottleneck['edge'],
                    'current_capacity': bottleneck['capacity'],
                    'suggested_increase': bottleneck['capacity'] * 0.5,
                    'reason': f"Critical bottleneck with {bottleneck['utilization']*100:.1f}% utilization"
                })
        
        # Analyze path redundancy
        resilience = self.analyze_resilience()
        if resilience['critical_edges']:
            for edge in resilience['critical_edges']:
                suggestions.append({
                    'type': 'redundancy',
                    'priority': 'medium',
                    'edge': edge,
                    'reason': "Single point of failure - consider adding alternative path"
                })
        
        # Analyze node importance
        importance = self.node_importance_analysis()
        for node, centrality in importance['flow_centrality'].items():
            if centrality > np.mean(list(importance['flow_centrality'].values())) + \
               np.std(list(importance['flow_centrality'].values())):
                suggestions.append({
                    'type': 'node_capacity',
                    'priority': 'medium',
                    'node': node,
                    'reason': "High flow concentration - consider load balancing"
                })
        
        return suggestions

    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic flow metrics
        total_flow = sum(self.flow_dict[self.source][v] 
                        for v in self.G.successors(self.source))
        
        # Capacity utilization
        util_analysis = self.capacity_utilization_analysis()
        
        # Path analysis
        path_flows = self.get_path_flows()
        
        # Resilience metrics
        resilience = self.analyze_resilience()
        
        return {
            'flow_metrics': {
                'total_flow': total_flow,
                'num_active_paths': len(path_flows),
                'avg_path_length': np.mean([p['length'] for p in path_flows]) if path_flows else 0
            },
            'utilization_metrics': {
                'mean': util_analysis['mean_utilization'],
                'median': util_analysis['median_utilization'],
                'std_dev': util_analysis['std_utilization']
            },
            'resilience_metrics': {
                'edge_connectivity': resilience['edge_connectivity'],
                'node_connectivity': resilience['node_connectivity'],
                'num_critical_edges': len(resilience['critical_edges'])
            }
        }

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a comprehensive summary report."""
        metrics = self.get_performance_metrics()
        bottlenecks = self.bottleneck_analysis()
        suggestions = self.generate_improvement_suggestions()
        
        report_data = []
        
        # Flow Performance
        report_data.append({
            'Category': 'Flow Performance',
            'Metric': 'Total Flow',
            'Value': f"{metrics['flow_metrics']['total_flow']:.2f}"
        })
        report_data.append({
            'Category': 'Flow Performance',
            'Metric': 'Active Paths',
            'Value': str(metrics['flow_metrics']['num_active_paths'])
        })
        
        # Utilization
        report_data.append({
            'Category': 'Utilization',
            'Metric': 'Mean Utilization',
            'Value': f"{metrics['utilization_metrics']['mean']*100:.1f}%"
        })
        report_data.append({
            'Category': 'Utilization',
            'Metric': 'Bottlenecks',
            'Value': str(bottlenecks['num_bottlenecks'])
        })
        
        # Resilience
        report_data.append({
            'Category': 'Resilience',
            'Metric': 'Edge Connectivity',
            'Value': str(metrics['resilience_metrics']['edge_connectivity'])
        })
        report_data.append({
            'Category': 'Resilience',
            'Metric': 'Critical Points',
            'Value': str(len(suggestions))
        })
        
        return pd.DataFrame(report_data) 