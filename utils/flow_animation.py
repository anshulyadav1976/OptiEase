import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd

class FlowAnimator:
    def __init__(self, G: nx.DiGraph, flow_dict: Dict, source: str, sink: str):
        self.G = G
        self.flow_dict = flow_dict
        self.source = source
        self.sink = sink
        self.pos = nx.spring_layout(G)
        
    def create_flow_propagation_animation(self) -> go.Figure:
        """Create an animation showing flow propagation through the network."""
        # Get all paths from source to sink
        paths = list(nx.all_simple_paths(self.G, self.source, self.sink))
        
        # Calculate flow along each path
        path_flows = []
        for path in paths:
            path_edges = list(zip(path[:-1], path[1:]))
            flow = min(self.flow_dict[u][v] for u, v in path_edges)
            if flow > 0:
                path_flows.append({
                    'path': path,
                    'flow': flow,
                    'edges': path_edges
                })
        
        # Sort paths by flow value
        path_flows.sort(key=lambda x: x['flow'], reverse=True)
        
        # Create frames for animation
        frames = []
        
        # Initial frame (empty network)
        initial_frame = self._create_network_frame([], {})
        frames.append(go.Frame(data=initial_frame, name='init'))
        
        # Create frames for each path
        current_flows = {}
        for i, path_data in enumerate(path_flows):
            # Update flows
            for u, v in path_data['edges']:
                current_flows[(u, v)] = current_flows.get((u, v), 0) + path_data['flow']
            
            # Create frame
            frame_data = self._create_network_frame(
                active_paths=path_flows[:i+1],
                current_flows=current_flows
            )
            frames.append(go.Frame(
                data=frame_data,
                name=f'step{i+1}'
            ))
        
        # Create figure with animation
        fig = go.Figure(
            data=initial_frame,
            frames=frames,
            layout=go.Layout(
                title="Flow Propagation Animation",
                showlegend=True,
                hovermode='closest',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 500}
                        }]
                    }, {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }]
                }],
                sliders=[{
                    'currentvalue': {
                        'prefix': 'Step: ',
                        'visible': True
                    },
                    'steps': [
                        {
                            'label': f'Step {i}',
                            'method': 'animate',
                            'args': [[f'step{i}'], {
                                'mode': 'immediate',
                                'frame': {'duration': 0, 'redraw': True},
                                'transition': {'duration': 500}
                            }]
                        }
                        for i in range(len(path_flows) + 1)
                    ]
                }]
            )
        )
        
        return fig
    
    def _create_network_frame(self, active_paths: List[Dict], current_flows: Dict) -> List[go.Scatter]:
        """Create network traces for a single animation frame."""
        traces = []
        
        # Create node trace
        node_x, node_y = [], []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            in_flow = sum(current_flows.get((u, node), 0) for u in self.G.predecessors(node))
            out_flow = sum(current_flows.get((node, v), 0) for v in self.G.successors(node))
            
            node_info = f"Node: {node}<br>"
            node_info += f"Current In Flow: {in_flow:.2f}<br>"
            node_info += f"Current Out Flow: {out_flow:.2f}"
            node_text.append(node_info)
            
            # Node styling
            if node == self.source:
                node_color.append('#2ecc71')
                node_size.append(30)
            elif node == self.sink:
                node_color.append('#e74c3c')
                node_size.append(30)
            else:
                node_color.append('#3498db')
                node_size.append(20)
        
        # Add node trace
        traces.append(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[str(node) for node in self.G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            ),
            name='Nodes'
        ))
        
        # Add edge traces for each active path
        for i, path_data in enumerate(active_paths):
            path_edges = path_data['edges']
            flow = path_data['flow']
            
            for u, v in path_edges:
                x0, y0 = self.pos[u]
                x1, y1 = self.pos[v]
                
                # Calculate edge styling
                current_flow = current_flows.get((u, v), 0)
                capacity = self.G[u][v]['capacity']
                utilization = current_flow / capacity
                
                traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=2 + 4 * utilization,
                        color=f'rgba({int(255*utilization)}, {int(255*(1-utilization))}, 0, 0.6)'
                    ),
                    hovertext=f"Path {i+1}<br>"
                             f"Flow: {current_flow:.2f}<br>"
                             f"Capacity: {capacity:.2f}<br>"
                             f"Utilization: {utilization*100:.1f}%",
                    name=f'Path {i+1}'
                ))
        
        return traces
    
    def create_interactive_flow_explorer(self) -> go.Figure:
        """Create an interactive flow explorer with multiple linked views."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Network Graph",
                "Flow Distribution",
                "Path Analysis",
                "Utilization Heatmap"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Network Graph (top left)
        self._add_network_graph(fig, row=1, col=1)
        
        # 2. Flow Distribution (top right)
        self._add_flow_distribution(fig, row=1, col=2)
        
        # 3. Path Analysis (bottom left)
        self._add_path_analysis(fig, row=2, col=1)
        
        # 4. Utilization Heatmap (bottom right)
        self._add_utilization_heatmap(fig, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title="Interactive Flow Explorer",
            hovermode='closest'
        )
        
        return fig
    
    def _add_network_graph(self, fig: go.Figure, row: int, col: int):
        """Add network graph to the subplot."""
        # Add nodes
        node_x, node_y = [], []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            in_flow = sum(self.flow_dict.get(u, {}).get(node, 0) 
                         for u in self.G.predecessors(node))
            out_flow = sum(self.flow_dict.get(node, {}).get(v, 0) 
                          for v in self.G.successors(node))
            
            node_text.append(f"Node: {node}<br>"
                           f"In Flow: {in_flow:.2f}<br>"
                           f"Out Flow: {out_flow:.2f}")
            
            if node == self.source:
                node_color.append('#2ecc71')
                node_size.append(30)
            elif node == self.sink:
                node_color.append('#e74c3c')
                node_size.append(30)
            else:
                node_color.append('#3498db')
                node_size.append(20)
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[str(node) for node in self.G.nodes()],
                hovertext=node_text,
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=2, color='white')
                ),
                name='Nodes'
            ),
            row=row, col=col
        )
        
        # Add edges
        for u, v in self.G.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            
            flow = self.flow_dict[u][v]
            capacity = self.G[u][v]['capacity']
            utilization = flow / capacity
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=2 + 4 * utilization,
                        color=f'rgba({int(255*utilization)}, {int(255*(1-utilization))}, 0, 0.6)'
                    ),
                    hovertext=f"Edge {u} → {v}<br>"
                             f"Flow: {flow:.2f}<br>"
                             f"Capacity: {capacity:.2f}<br>"
                             f"Utilization: {utilization*100:.1f}%",
                    showlegend=False
                ),
                row=row, col=col
            )
    
    def _add_flow_distribution(self, fig: go.Figure, row: int, col: int):
        """Add flow distribution histogram to the subplot."""
        flows = [self.flow_dict[u][v] for u, v in self.G.edges()]
        
        fig.add_trace(
            go.Histogram(
                x=flows,
                name='Flow Distribution',
                nbinsx=20,
                marker_color='#3498db'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Flow Value", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_path_analysis(self, fig: go.Figure, row: int, col: int):
        """Add path analysis plot to the subplot."""
        paths = list(nx.all_simple_paths(self.G, self.source, self.sink))
        path_data = []
        
        for path in paths:
            path_edges = list(zip(path[:-1], path[1:]))
            flow = min(self.flow_dict[u][v] for u, v in path_edges)
            if flow > 0:
                path_data.append({
                    'path_id': len(path_data) + 1,
                    'flow': flow,
                    'length': len(path) - 1
                })
        
        df = pd.DataFrame(path_data)
        
        fig.add_trace(
            go.Scatter(
                x=df['length'],
                y=df['flow'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['flow'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Path {i}" for i in df['path_id']],
                name='Paths'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Path Length", row=row, col=col)
        fig.update_yaxes(title_text="Flow Value", row=row, col=col)
    
    def _add_utilization_heatmap(self, fig: go.Figure, row: int, col: int):
        """Add utilization heatmap to the subplot."""
        nodes = list(self.G.nodes())
        matrix = np.zeros((len(nodes), len(nodes)))
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if self.G.has_edge(u, v):
                    matrix[i, j] = self.flow_dict[u][v] / self.G[u][v]['capacity']
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=nodes,
                y=nodes,
                colorscale='RdYlGn_r',
                name='Utilization'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="To Node", row=row, col=col)
        fig.update_yaxes(title_text="From Node", row=row, col=col) 