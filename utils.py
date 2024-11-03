import plotly.graph_objects as go
from typing import Dict

# Define color constants for visual consistency
GAUGE_GREEN = "rgba(0, 255, 0, 0.8)"
GAUGE_YELLOW = "rgba(255, 255, 0, 0.8)"
GAUGE_RED = "rgba(255, 0, 0, 0.8)"
LIGHT_BLUE = "rgba(0, 123, 255, 0.8)"
DARK_BG = "rgba(0, 0, 0, 0)"
LIGHT_BG = "rgba(255, 255, 255, 0)"

def create_gauge_chart(probability: float, theme: str = "dark") -> go.Figure:
    """
    Create a gauge chart that visualizes churn probability.

    Parameters:
    - probability (float): The churn probability value (0 to 1).
    - theme (str): Theme color for background and font, dark or light.

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly gauge chart figure.

    Raises:
    - ValueError: If probability is not between 0 and 1.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    # Set gauge color based on churn risk level
    color = GAUGE_GREEN if probability < 0.3 else GAUGE_YELLOW if probability < 0.6 else GAUGE_RED
    
    # Define font and background color based on theme
    font_color = 'white' if theme == "dark" else 'black'
    bg_color = DARK_BG if theme == "dark" else LIGHT_BG

    # Create the gauge chart figure
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,  # Convert to percentage for display
            title={'text': "Churn Probability", 'font': {'color': font_color}},
            number={'font': {'color': font_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': font_color, 'tickformat': "%"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                    {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': font_color, 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        )
    )

    # Update layout with background, font colors, and size
    fig.update_layout(
        paper_bgcolor=bg_color,
        font={'color': font_color},
        width=400, height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_model_probability_chart(probabilities: Dict[str, float], theme: str = "dark") -> go.Figure:
    """
    Create a horizontal bar chart for model churn probabilities.

    Parameters:
    - probabilities (dict): Dictionary of model names and their churn
    probabilities.
    - theme (str): Theme color for background and font, either dark or light.

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly bar chart figure.
    """
    # Extract model names and probabilities
    models = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Define font and background color based on theme
    font_color = 'white' if theme == "dark" else 'black'
    bg_color = DARK_BG if theme == "dark" else LIGHT_BG

    # Create the bar chart figure
    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation='h',
                text=[f'{p:.2%}' for p in probs],  # Display as percentages
                textposition='auto',
                marker=dict(color=LIGHT_BLUE)
            )
        ]
    )

    # Update layout with title, axis labels, background, and font colors
    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=font_color),
        paper_bgcolor=bg_color
    )
    return fig

def create_percentile_bar_chart(percentiles: Dict[str, float], theme: str = "dark") -> go.Figure:
    """
    Create a vertical bar chart for customer metrics percentiles.

    Parameters:
    - percentiles (dict): Dictionary of metric names and their percentiles.
    - theme (str): Theme color for background and font, either dark or light.

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly bar chart figure.
    """
    # Extract metric names and percentile values
    metrics = list(percentiles.keys())
    values = list(percentiles.values())
    
    # Define font and background color based on theme
    font_color = 'white' if theme == "dark" else 'black'
    bg_color = DARK_BG if theme == "dark" else LIGHT_BG

    # Create the bar chart figure
    fig = go.Figure(
        go.Bar(
            x=metrics,
            y=values,
            text=[f"{v}%" for v in values],  # Display values as percentages
            textposition='auto',
            marker=dict(color=LIGHT_BLUE)
        )
    )

    # Update layout with title, axis labels, background, and font colors
    fig.update_layout(
        title="Customer Percentiles Across Metrics",
        xaxis_title="Metrics",
        yaxis_title="Percentile (%)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=font_color),
        paper_bgcolor=bg_color
    )
    return fig
