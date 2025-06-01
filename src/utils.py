import plotly.graph_objects as go
import numpy as np
from scipy import stats


def plot_monte_carlo_paths(simulation_data, n_to_plot=100, title='Monte Carlo Simulation', 
                          xlabel='Days', ylabel='Simulated Price', color='blue', 
                          alpha=0.1, width=800, height=400):
    """
    Create a Monte Carlo simulation path plot using Plotly.
    
    Parameters:
    -----------
    simulation_data : array-like
        2D array where each row is a simulation path
    n_to_plot : int, optional
        Number of simulation paths to plot (default: 100)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    color : str, optional
        Color for the simulation paths (default: 'blue')
    alpha : float, optional
        Transparency level (0-1, default: 0.1)
    width : int, optional
        Plot width in pixels
    height : int, optional
        Plot height in pixels
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    
    fig = go.Figure()
    
    # Convert color name to RGBA with alpha
    color_map = {
        'blue': f'rgba(0, 0, 255, {alpha})',
        'red': f'rgba(255, 0, 0, {alpha})',
        'green': f'rgba(0, 128, 0, {alpha})',
        'orange': f'rgba(255, 165, 0, {alpha})',
        'purple': f'rgba(128, 0, 128, {alpha})',
        'black': f'rgba(0, 0, 0, {alpha})',
        'gray': f'rgba(128, 128, 128, {alpha})',
    }
    
    # Use color mapping or assume it's already in rgba format
    line_color = color_map.get(color.lower(), color)
    
    # Plot the specified number of simulations
    n_sims = min(n_to_plot, len(simulation_data))
    
    for i in range(n_sims):
        fig.add_trace(go.Scatter(
            x=list(range(len(simulation_data[i]))),
            y=simulation_data[i],
            mode='lines',
            line=dict(color=line_color, width=1),
            showlegend=False,
            hovertemplate='Day: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='plotly_white',
        showlegend=False,
        hovermode='closest'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig



def plot_kde_comparison(data_dict, title='KDE Comparison', xlabel='Value', ylabel='Density'):
    """
    Create a KDE plot comparing multiple datasets using Plotly.

    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are labels and values are data arrays
        Example: {'QQQ': qqq_ending, 'VOO': voo_ending}
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label

    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """

    fig = go.Figure()

    # Default colors for multiple traces
    colors = [
        'rgba(31, 119, 180, 0.3)',  # Blue
        'rgba(255, 127, 14, 0.3)',  # Orange
        'rgba(44, 160, 44, 0.3)',  # Green
        'rgba(214, 39, 40, 0.3)',  # Red
        'rgba(148, 103, 189, 0.3)',  # Purple
        'rgba(140, 86, 75, 0.3)',  # Brown
        'rgba(227, 119, 194, 0.3)',  # Pink
        'rgba(127, 127, 127, 0.3)',  # Gray
    ]

    for i, (label, data) in enumerate(data_dict.items()):
        # Create KDE
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        y = kde(x_range)

        # Add trace
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y,
            fill='tozeroy',
            name=label,
            line=dict(width=2),
            fillcolor=colors[i % len(colors)]
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig