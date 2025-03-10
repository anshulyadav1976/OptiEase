# OptiEase API Reference

## Overview

The OptiEase API provides programmatic access to advanced network flow optimization capabilities. This reference documents the classes, methods, and data structures available for integration.

## Core Classes

### MultiCommodityFlow

```python
from utils.complex_flows import MultiCommodityFlow

mcf = MultiCommodityFlow(G: nx.DiGraph)
```

#### Methods

##### `add_commodity`
```python
def add_commodity(
    commodity_id: str,
    source: str,
    sink: str,
    demand: float,
    priority: float = 1.0
) -> None
```

**Parameters:**
- `commodity_id`: Unique identifier for the commodity
- `source`: Source node name
- `sink`: Sink node name
- `demand`: Required flow amount
- `priority`: Priority weight (default: 1.0)

##### `solve`
```python
def solve() -> Dict[str, Any]
```

**Returns:**
```python
{
    'status': str,  # Solution status
    'total_cost': float,  # Total weighted cost
    'flows': {
        'commodity_id': {
            (u, v): flow_amount  # Edge flows per commodity
        }
    }
}
```

### TimeDependentFlow

```python
from utils.complex_flows import TimeDependentFlow

tdf = TimeDependentFlow(G: nx.DiGraph, time_periods: int)
```

#### Methods

##### `set_capacity_profile`
```python
def set_capacity_profile(
    edge: Tuple[str, str],
    profile: List[float]
) -> None
```

**Parameters:**
- `edge`: Tuple of (source, target) nodes
- `profile`: List of capacities per time period

##### `set_demand_profile`
```python
def set_demand_profile(
    source: str,
    sink: str,
    profile: List[float]
) -> None
```

**Parameters:**
- `source`: Source node name
- `sink`: Sink node name
- `profile`: List of demands per time period

##### `solve`
```python
def solve() -> Dict[str, Any]
```

**Returns:**
```python
{
    'status': str,  # Solution status
    'total_cost': float,  # Total cost
    'flows': {
        time_period: {
            (u, v): flow_amount  # Edge flows per period
        }
    }
}
```

### FlowWithUncertainty

```python
from utils.complex_flows import FlowWithUncertainty

uflow = FlowWithUncertainty(G: nx.DiGraph, num_scenarios: int)
```

#### Methods

##### `add_scenario`
```python
def add_scenario(
    capacity_factors: Dict[Tuple[str, str], float],
    demand_factors: Dict[Tuple[str, str], float],
    probability: float
) -> None
```

**Parameters:**
- `capacity_factors`: Edge capacity multipliers
- `demand_factors`: Demand multipliers
- `probability`: Scenario probability

##### `solve`
```python
def solve() -> Dict[str, Any]
```

**Returns:**
```python
{
    'status': str,  # Solution status
    'expected_cost': float,  # Expected total cost
    'scenario_flows': {
        scenario_id: {
            (u, v): flow_amount  # Edge flows per scenario
        }
    }
}
```

## Analysis Tools

### FlowAnalyzer

```python
from utils.flow_analysis import FlowAnalyzer

analyzer = FlowAnalyzer(
    G: nx.DiGraph,
    flow_dict: Dict,
    source: str,
    sink: str
)
```

#### Methods

##### `get_performance_metrics`
```python
def get_performance_metrics() -> Dict[str, Dict[str, float]]
```

**Returns:**
```python
{
    'flow_metrics': {
        'total_flow': float,
        'num_active_paths': int,
        'avg_path_length': float
    },
    'utilization_metrics': {
        'mean': float,
        'max': float,
        'min': float
    },
    'resilience_metrics': {
        'edge_connectivity': int,
        'node_connectivity': int,
        'num_critical_edges': int
    }
}
```

##### `bottleneck_analysis`
```python
def bottleneck_analysis() -> Dict[str, List[Dict]]
```

**Returns:**
```python
{
    'bottlenecks': [
        {
            'edge': Tuple[str, str],
            'utilization': float,
            'flow': float,
            'capacity': float
        }
    ],
    'near_bottlenecks': [
        {
            'edge': Tuple[str, str],
            'utilization': float
        }
    ]
}
```

##### `analyze_resilience`
```python
def analyze_resilience() -> Dict[str, Any]
```

**Returns:**
```python
{
    'edge_connectivity': int,
    'node_connectivity': int,
    'critical_edges': List[Tuple[str, str]],
    'vulnerability_score': float
}
```

## Visualization Tools

### FlowVisualizer

```python
from utils.flow_visualization import FlowVisualizer

visualizer = FlowVisualizer(
    G: nx.DiGraph,
    flow_dict: Dict,
    source: str,
    sink: str
)
```

#### Methods

##### `create_network_graph`
```python
def create_network_graph() -> go.Figure
```

**Returns:**
- Plotly figure object with interactive network visualization

##### `create_utilization_heatmap`
```python
def create_utilization_heatmap() -> go.Figure
```

**Returns:**
- Plotly figure object with edge utilization heatmap

##### `create_sankey_diagram`
```python
def create_sankey_diagram() -> go.Figure
```

**Returns:**
- Plotly figure object with Sankey flow diagram

### FlowAnimator

```python
from utils.flow_animation import FlowAnimator

animator = FlowAnimator(
    G: nx.DiGraph,
    flow_dict: Dict
)
```

#### Methods

##### `create_flow_propagation_animation`
```python
def create_flow_propagation_animation() -> go.Figure
```

**Returns:**
- Plotly figure object with animated flow visualization

##### `create_interactive_flow_explorer`
```python
def create_interactive_flow_explorer() -> go.Figure
```

**Returns:**
- Plotly figure object with interactive flow exploration tools

## Data Structures

### Network Graph
```python
import networkx as nx

G = nx.DiGraph()
G.add_edge(u, v, capacity=cap, cost=cost)
```

Required edge attributes:
- `capacity`: float
- `cost`: float (optional)

### Flow Dictionary
```python
flow_dict = {
    (u, v): flow_amount  # For basic flows
    commodity_id: {(u, v): flow_amount}  # For multi-commodity
    time_period: {(u, v): flow_amount}  # For time-dependent
    scenario_id: {(u, v): flow_amount}  # For uncertainty
}
```

### Solution Format
```python
solution = {
    'status': str,  # Solution status
    'objective_value': float,  # Cost or flow value
    'flows': Dict,  # Flow dictionary
    'metrics': Dict  # Performance metrics
}
```

## Integration Examples

### Basic Flow Problem
```python
import networkx as nx
from utils.complex_flows import MultiCommodityFlow

# Create network
G = nx.DiGraph()
G.add_edge('A', 'B', capacity=100, cost=5)
G.add_edge('B', 'C', capacity=80, cost=3)

# Create solver
mcf = MultiCommodityFlow(G)

# Add commodities
mcf.add_commodity('prod1', 'A', 'C', 50, 1.0)
mcf.add_commodity('prod2', 'A', 'C', 30, 0.8)

# Solve
solution = mcf.solve()

# Analyze
analyzer = FlowAnalyzer(G, solution['flows']['prod1'], 'A', 'C')
metrics = analyzer.get_performance_metrics()

# Visualize
visualizer = FlowVisualizer(G, solution['flows']['prod1'], 'A', 'C')
fig = visualizer.create_network_graph()
```

### Time-Dependent Flow
```python
# Create solver
tdf = TimeDependentFlow(G, time_periods=24)

# Set profiles
tdf.set_capacity_profile(('A', 'B'), [100]*8 + [80]*8 + [120]*8)
tdf.set_demand_profile('A', 'C', [50]*8 + [30]*8 + [70]*8)

# Solve
solution = tdf.solve()

# Analyze time periods
for t in range(24):
    analyzer = FlowAnalyzer(G, solution['flows'][t], 'A', 'C')
    metrics = analyzer.get_performance_metrics()
```

### Uncertainty Analysis
```python
# Create solver
uflow = FlowWithUncertainty(G, num_scenarios=3)

# Add scenarios
uflow.add_scenario(
    capacity_factors={('A', 'B'): 0.8, ('B', 'C'): 0.9},
    demand_factors={('A', 'C'): 0.7},
    probability=0.3
)

# Solve
solution = uflow.solve()

# Analyze scenarios
for scenario in solution['scenario_flows']:
    analyzer = FlowAnalyzer(G, solution['scenario_flows'][scenario], 'A', 'C')
    metrics = analyzer.get_performance_metrics()
```

## Error Handling

### Common Errors

#### InvalidGraphError
```python
class InvalidGraphError(Exception):
    """Raised when graph structure is invalid."""
    pass
```

#### InfeasibleFlowError
```python
class InfeasibleFlowError(Exception):
    """Raised when no feasible flow exists."""
    pass
```

#### ParameterError
```python
class ParameterError(Exception):
    """Raised when parameters are invalid."""
    pass
```

### Error Handling Example
```python
try:
    solution = mcf.solve()
except InvalidGraphError as e:
    print(f"Graph error: {e}")
except InfeasibleFlowError as e:
    print(f"No feasible solution: {e}")
except ParameterError as e:
    print(f"Invalid parameters: {e}")
```

## Best Practices

### Performance
1. Limit graph size for interactive use
2. Use appropriate time periods
3. Balance scenario count
4. Cache results when possible

### Memory Management
1. Clear large data structures
2. Use generators for large datasets
3. Implement proper cleanup

### Thread Safety
1. Create new solver instances
2. Avoid global state
3. Use proper locking

### Error Handling
1. Validate input data
2. Check feasibility
3. Handle edge cases
4. Provide clear messages
``` 