# OptiEase Technical Documentation

## Architecture Overview

### Core Components

1. **Flow Optimization Engine** (`utils/complex_flows.py`)
   - `MultiCommodityFlow`: Handles multi-commodity flow problems
   - `TimeDependentFlow`: Manages time-varying flow problems
   - `FlowWithUncertainty`: Handles flow problems with uncertain parameters

2. **Visualization System** (`utils/flow_visualization.py`)
   - Network graph visualization
   - Flow animation
   - Interactive flow explorer
   - Multiple view types (heatmap, Sankey, etc.)

3. **Analysis Engine** (`utils/flow_analysis.py`)
   - Performance metrics calculation
   - Bottleneck detection
   - Network resilience analysis
   - Node importance assessment

4. **User Interface** (`pages/flow_optimization.py`)
   - Problem type selection
   - Parameter configuration
   - Results visualization
   - Export functionality

## Flow Optimization Engine

### MultiCommodityFlow Class

```python
class MultiCommodityFlow:
    def __init__(self, G: nx.DiGraph):
        """Initialize with NetworkX DiGraph."""
        
    def add_commodity(self, commodity_id: str, source: str, sink: str, 
                     demand: float, priority: float = 1.0):
        """Add a commodity with its properties."""
        
    def solve(self) -> Dict:
        """Solve the multi-commodity flow problem."""
```

#### Key Features
- Multiple commodity support with priorities
- Individual source-sink pairs
- Capacity sharing between commodities
- Priority-based optimization

### TimeDependentFlow Class

```python
class TimeDependentFlow:
    def __init__(self, G: nx.DiGraph, time_periods: int):
        """Initialize with graph and number of time periods."""
        
    def set_capacity_profile(self, edge: Tuple[str, str], profile: List[float]):
        """Set time-varying capacity for an edge."""
        
    def set_demand_profile(self, source: str, sink: str, profile: List[float]):
        """Set time-varying demand between source and sink."""
```

#### Key Features
- Time-varying capacities
- Dynamic demand profiles
- Period-specific optimization
- Temporal consistency constraints

### FlowWithUncertainty Class

```python
class FlowWithUncertainty:
    def __init__(self, G: nx.DiGraph, num_scenarios: int):
        """Initialize with graph and number of scenarios."""
        
    def add_scenario(self, capacity_factors: Dict, demand_factors: Dict, 
                    probability: float):
        """Add a scenario with its parameters."""
```

#### Key Features
- Multiple scenario support
- Probability-weighted optimization
- Capacity uncertainty modeling
- Demand uncertainty handling

## Visualization System

### Network Graph Visualization
- Interactive network layout
- Flow magnitude representation
- Capacity utilization coloring
- Node/edge highlighting

### Flow Animation
- Dynamic flow propagation
- Time-series visualization
- Path-based animation
- Interactive controls

### Analysis Views
- Utilization heatmap
- Sankey diagram
- Path flow visualization
- Performance dashboards

## Analysis Engine

### Performance Metrics
- Total flow calculation
- Path flow analysis
- Utilization statistics
- Cost metrics

### Network Analysis
- Bottleneck detection
- Critical edge identification
- Resilience assessment
- Node importance calculation

### Improvement Analysis
- Capacity enhancement suggestions
- Alternative path recommendations
- Load balancing proposals
- Risk mitigation strategies

## User Interface

### Problem Configuration
- Problem type selection
- Parameter input forms
- Validation rules
- Dynamic updates

### Results Display
- Multiple visualization options
- Interactive exploration
- Metric dashboards
- Export functionality

## Data Structures

### Network Graph
```python
G = nx.DiGraph()
G.add_edge(u, v, capacity=cap, cost=cost)
```

### Flow Solution
```python
solution = {
    'status': str,
    'total_cost': float,
    'flows': Dict[str, Dict[Tuple[str, str], float]]
}
```

### Analysis Results
```python
metrics = {
    'flow_metrics': Dict[str, float],
    'utilization_metrics': Dict[str, float],
    'resilience_metrics': Dict[str, Any]
}
```

## Extension Guide

### Adding New Problem Types
1. Create new class in `complex_flows.py`
2. Implement required methods:
   - `__init__`
   - Problem-specific configuration
   - `solve`
3. Update UI in `flow_optimization.py`

### Adding Visualizations
1. Add visualization method in `flow_visualization.py`
2. Implement data transformation
3. Create Plotly/Matplotlib figure
4. Add to UI options

### Adding Analysis Features
1. Add analysis method in `flow_analysis.py`
2. Implement metric calculation
3. Add visualization support
4. Update UI components

## Best Practices

### Code Style
- Type hints for all functions
- Comprehensive docstrings
- Clear variable names
- Modular design

### Performance
- Efficient graph algorithms
- Optimized data structures
- Caching when appropriate
- Batch processing for large datasets

### Error Handling
- Input validation
- Graceful error recovery
- Clear error messages
- User feedback

### Testing
- Unit tests for core functions
- Integration tests for workflows
- Performance benchmarks
- Edge case handling

## Dependencies

### Core Libraries
- NetworkX: Graph operations
- PuLP: Optimization solver
- Pandas: Data handling
- NumPy: Numerical operations

### Visualization
- Plotly: Interactive plots
- Matplotlib: Static plots
- Streamlit: Web interface

## Future Development

### Planned Features
- Additional problem types
- Enhanced visualizations
- Advanced analysis tools
- Performance optimizations

### Integration Points
- Custom solvers
- External data sources
- Additional export formats
- API endpoints 