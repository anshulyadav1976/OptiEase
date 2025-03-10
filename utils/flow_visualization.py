import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

class FlowVisualizer:
    def __init__(self, G: nx.DiGraph, flow_dict: Dict, source: str, sink: str):
        self.G = G
        self.flow_dict = flow_dict
        self.source = source
        self.sink = sink
        self.pos = nx.spring_layout(G)
        
    def create_network_graph(self, show_labels: bool = True) -> go.Figure:
        """Create interactive network visualization."""
        # Prepare node data
        node_x, node_y = [], []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            in_flow = sum(self.flow_dict.get(u, {}).get(node, 0) 
                         for u in self.G.predecessors(node))
            out_flow = sum(self.flow_dict.get(node, {}).get(v, 0) 
                          for v in self.G.successors(node))
            
            node_info = f"Node: {node}<br>"
            node_info += f"In Flow: {in_flow:.2f}<br>"
            node_info += f"Out Flow: {out_flow:.2f}"
            node_text.append(node_info)
            
            # Node styling
            if node == self.source:
                node_color.append('#2ecc71')  # Green
                node_size.append(30)
            elif node == self.sink:
                node_color.append('#e74c3c')  # Red
                node_size.append(30)
            else:
                node_color.append('#3498db')  # Blue
                node_size.append(20)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=[str(node) for node in self.G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )
        
        # Prepare edge data
        edge_traces = []
        for u, v in self.G.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            
            # Calculate edge styling based on flow
            flow = self.flow_dict[u][v]
            capacity = self.G[u][v]['capacity']
            utilization = flow / capacity
            
            # Edge width based on flow
            width = 2 + 4 * utilization
            
            # Edge color based on utilization
            if utilization > 0.9:
                color = '#e74c3c'  # Red for high utilization
            elif utilization > 0.7:
                color = '#f39c12'  # Orange for medium utilization
            else:
                color = '#2ecc71'  # Green for low utilization
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f"From: {u} To: {v}<br>"
                     f"Flow: {flow:.2f}<br>"
                     f"Capacity: {capacity:.2f}<br>"
                     f"Utilization: {utilization*100:.1f}%"
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[*edge_traces, node_trace])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Network Flow Visualization",
                x=0.5,
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="Source",
                    x=self.pos[self.source][0],
                    y=self.pos[self.source][1] - 0.1,
                    showarrow=False
                ),
                dict(
                    text="Sink",
                    x=self.pos[self.sink][0],
                    y=self.pos[self.sink][1] - 0.1,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def create_path_visualization(self, path_flows: List[Dict]) -> go.Figure:
        """Create visualization of flow paths."""
        # Create base network
        fig = self.create_network_graph(show_labels=False)
        
        # Add path traces
        colors = px.colors.qualitative.Set3
        for i, path_data in enumerate(path_flows):
            path = path_data['path']
            flow = path_data['flow']
            
            # Create path trace
            path_x = []
            path_y = []
            for node in path:
                x, y = self.pos[node]
                path_x.append(x)
                path_y.append(y)
            
            fig.add_trace(go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(
                    width=2 + 2 * (flow / max(p['flow'] for p in path_flows)),
                    color=colors[i % len(colors)]
                ),
                name=f"Path {i+1} (Flow: {flow:.2f})"
            ))
        
        fig.update_layout(
            title="Flow Paths Visualization",
            showlegend=True
        )
        
        return fig
    
    def create_utilization_heatmap(self) -> go.Figure:
        """Create heatmap of edge utilizations."""
        # Calculate utilizations
        utilization_matrix = {}
        for u in self.G.nodes():
            utilization_matrix[u] = {}
            for v in self.G.nodes():
                if self.G.has_edge(u, v):
                    utilization_matrix[u][v] = self.flow_dict[u][v] / self.G[u][v]['capacity']
                else:
                    utilization_matrix[u][v] = 0
        
        # Convert to matrix form
        nodes = list(self.G.nodes())
        matrix = [[utilization_matrix[u][v] for v in nodes] for u in nodes]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=nodes,
            y=nodes,
            colorscale='RdYlGn_r',
            text=[[f"{val*100:.1f}%" if val > 0 else "" for val in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Edge Utilization Heatmap",
            xaxis_title="To Node",
            yaxis_title="From Node"
        )
        
        return fig
    
    def create_sankey_diagram(self) -> go.Figure:
        """Create Sankey diagram of the flow network."""
        # Prepare node labels
        nodes = list(self.G.nodes())
        node_labels = [f"{node}" for node in nodes]
        
        # Prepare links
        links = []
        for u, v in self.G.edges():
            links.append({
                'source': nodes.index(u),
                'target': nodes.index(v),
                'value': self.flow_dict[u][v],
                'label': f"Flow: {self.flow_dict[u][v]:.2f}"
            })
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=["#2ecc71" if n == self.source else
                      "#e74c3c" if n == self.sink else
                      "#3498db" for n in nodes]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                label=[link['label'] for link in links]
            )
        )])
        
        fig.update_layout(
            title="Flow Network Sankey Diagram",
            font_size=10
        )
        
        return fig
    
    def create_time_series_plot(self, time_data: List[Dict]) -> go.Figure:
        """Create time series plot of flow variations."""
        df = pd.DataFrame(time_data)
        
        fig = go.Figure()
        
        # Add total flow line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_flow'],
            mode='lines+markers',
            name='Total Flow',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Add utilization line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['avg_utilization'],
            mode='lines+markers',
            name='Average Utilization',
            yaxis='y2',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title='Flow and Utilization Over Time',
            xaxis_title='Time',
            yaxis_title='Total Flow',
            yaxis2=dict(
                title='Average Utilization',
                overlaying='y',
                side='right',
                tickformat='%'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_dashboard(self, metrics: Dict) -> List[go.Figure]:
        """Create a set of performance dashboard visualizations."""
        figures = []
        
        # Flow distribution gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['utilization_metrics']['mean'] * 100,
            title={'text': "Average Network Utilization"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        figures.append(fig_gauge)
        
        # Performance metrics bar chart
        metrics_data = {
            'Metric': ['Total Flow', 'Active Paths', 'Edge Connectivity'],
            'Value': [
                metrics['flow_metrics']['total_flow'],
                metrics['flow_metrics']['num_active_paths'],
                metrics['resilience_metrics']['edge_connectivity']
            ]
        }
        fig_metrics = px.bar(
            metrics_data,
            x='Metric',
            y='Value',
            title="Key Performance Metrics"
        )
        figures.append(fig_metrics)
        
        return figures 