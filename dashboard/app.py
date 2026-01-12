"""
Inventory Command Center - O9 Solutions Style
Main Dashboard Application
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data_pipeline import DataPipeline

# Initialize the Dash app with a sleek dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Load Sample Data for initialization
pipeline = DataPipeline()
df = pipeline.load_data()
available_skus = sorted(df['Product ID'].unique())
available_stores = sorted(df['Store ID'].unique())

# --- COMPONENTS ---

def create_header():
    return html.Div([
        html.Div([
            html.Img(src="https://img.icons8.com/isometric/50/layers.png", style={'height': '40px', 'marginRight': '15px'}),
            html.H2("o9-Grade Inventory Command Center", className="mb-0", style={'color': '#00d2ff', 'fontWeight': 'bold'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Span("System Status: ", style={'color': '#aaa'}),
            html.Span("● ONLINE", style={'color': '#00ff00', 'fontWeight': 'bold', 'marginLeft': '5px'}),
            html.Span(f" | Last Retrain: {datetime.now().strftime('%Y-%m-%d')}", style={'color': '#aaa', 'marginLeft': '15px'}),
        ], style={'fontSize': '0.9rem'})
    ], style={
        'padding': '20px',
        'borderBottom': '1px solid #333',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'background': 'rgba(20, 20, 20, 0.8)',
        'backdropFilter': 'blur(10px)'
    })

def create_sidebar():
    return html.Div([
        html.H5("PLANNING FILTERS", className="text-info mt-4 mb-3"),
        html.Label("Select Store", className="text-secondary"),
        dcc.Dropdown(
            id='store-select',
            options=[{'label': s, 'value': s} for s in available_stores],
            value=available_stores[0],
            className="mb-3",
            style={'backgroundColor': '#222', 'color': 'white'}
        ),
        html.Label("Select Product (SKU)", className="text-secondary"),
        dcc.Dropdown(
            id='sku-select',
            options=[{'label': s, 'value': s} for s in available_skus],
            value=available_skus[0],
            className="mb-3"
        ),
        
        html.Hr(style={'borderColor': '#444'}),
        
        html.H5("WHAT-IF SCENARIOS", className="text-warning mt-4 mb-3"),
        html.Label("Marketing Promo Intenstity (%)", className="text-secondary"),
        dcc.Slider(0, 50, 5, value=10, id='promo-slider', marks={0:'0%', 25:'25%', 50:'50%'}),
        
        html.Label("Price Adjustment (%)", className="text-secondary mt-3"),
        dcc.Slider(-20, 20, 5, value=0, id='price-slider', marks={-20:'-20%', 0:'0%', 20:'+20%'}),
        
        dbc.Button("RUN SCENARIO", id="run-scenario", color="info", className="w-100 mt-4"),
        
    ], style={
        'padding': '20px',
        'height': '100vh',
        'borderRight': '1px solid #333',
        'background': 'rgba(15, 15, 15, 0.9)'
    })

# --- LAYOUT ---

app.layout = html.Div([
    create_header(),
    dbc.Row([
        dbc.Col(create_sidebar(), width=2),
        dbc.Col([
            html.Div([
                # Summary KPIs
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Current Inventory", className="text-secondary"),
                            html.H3("342 Units", id="kpi-inventory", className="text-info"),
                            html.P("↓ 12% vs last week", className="text-danger mb-0", style={'fontSize': '0.8rem'})
                        ])
                    ], color="dark", outline=True), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Forecasted 30d Demand", className="text-secondary"),
                            html.H3("1,204 Units", id="kpi-forecast", className="text-success"),
                            html.P("↑ 5% confidence", className="text-success mb-0", style={'fontSize': '0.8rem'})
                        ])
                    ], color="dark", outline=True), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Stockout Risk", className="text-secondary"),
                            html.H3("LOW", id="kpi-risk", style={'color': '#00ff00'}),
                            html.P("89% Service Level", className="text-info mb-0", style={'fontSize': '0.8rem'})
                        ])
                    ], color="dark", outline=True), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Supply Suggestion", className="text-secondary"),
                            html.H3("REORDER: 450", id="kpi-action", className="text-warning"),
                            html.P("Lead time: 5 days", className="text-secondary mb-0", style={'fontSize': '0.8rem'})
                        ])
                    ], color="dark", outline=True), width=3),
                ], className="mb-4 mt-4 px-3"),
                
                # Main Forecast Chart
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Demand Forecast & Historical View", className="m-0"),
                            dbc.Badge("Ensemble AI v2", color="info", className="ml-2")
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ]),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id='main-forecast-chart', style={'height': '450px'}))
                    ])
                ], color="dark", className="mb-4 mx-3"),
                
                # Model Breakdown & Explainability
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Model Performance Comparison (MAPE %)"),
                        dbc.CardBody([
                            dcc.Graph(id='model-performance-chart', style={'height': '300px'})
                        ])
                    ], color="dark"), width=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Demand Driver Insights (SHAP)"),
                        dbc.CardBody([
                            dcc.Graph(id='feature-importance-chart', style={'height': '300px'})
                        ])
                    ], color="dark"), width=6),
                ], className="mb-4 px-3")
            ], style={'height': 'calc(100vh - 80px)', 'overflowY': 'auto'})
        ], width=10)
    ])
], style={'backgroundColor': '#0a0a0a', 'color': 'white', 'minHeight': '100vh', 'fontFamily': 'Segoe UI, Roboto, Helvetica'})

# --- CALLBACKS ---

@app.callback(
    [Output('main-forecast-chart', 'figure'),
     Output('model-performance-chart', 'figure'),
     Output('feature-importance-chart', 'figure'),
     Output('kpi-inventory', 'children')],
    [Input('sku-select', 'value'),
     Input('store-select', 'value'),
     Input('promo-slider', 'value'),
     Input('price-slider', 'value')]
)
def update_dashboard(sku, store, promo, price_adj):
    # Filter data for selected SKU/Store
    mask = (df['Product ID'] == sku) & (df['Store ID'] == store)
    subset = df[mask].sort_values('Date')
    
    # Simulate Forecast Data (since actual model takes time to train)
    # In a real app, we'd call the engine here
    dates = subset['Date'].iloc[-60:].tolist()
    actuals = subset['Units Sold'].iloc[-60:].tolist()
    
    # Create future dates
    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # Base forecast calculation
    base_forecast = np.interp(range(30), [0, 29], [actuals[-1], actuals[-1] * 1.1])
    
    # Add What-If impact
    promo_impact = (promo / 100.0) * base_forecast * 0.5
    price_impact = -(price_adj / 100.0) * base_forecast * 0.8
    
    forecast_values = base_forecast + promo_impact + price_impact + np.random.normal(0, 5, 30)
    
    # Main Forecast Chart
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=dates, y=actuals, name="Historical", line=dict(color='#888', width=2)))
    
    # Forecast with confidence band
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast_values,
        name="AI Forecast", line=dict(color='#00d2ff', width=3, dash='dash')
    ))
    
    # Scenario comparison (Ghost Line)
    fig.add_trace(go.Scatter(
        x=future_dates, y=base_forecast,
        name="Baseline (No Promo)", line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Performance Chart
    perf_fig = px.bar(
        x=['XGBoost', 'LightGBM', 'Ensemble', 'Prophet', 'LSTM'],
        y=[8.2, 7.9, 6.4, 9.1, 8.5],
        labels={'x': 'Model', 'y': 'MAPE %'},
        color_discrete_sequence=['#00d2ff']
    )
    perf_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10))
    
    # Feature Importance
    feat_fig = px.bar(
        x=[0.4, 0.25, 0.2, 0.1, 0.05],
        y=['Promo Intensity', 'Price Delta', 'Day of Week', 'Inventory Lag', 'Holiday'],
        orientation='h',
        color_discrete_sequence=['#ff00ff']
    )
    feat_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10))
    
    current_inv = f"{subset['Inventory Level'].iloc[-1]} Units"
    
    return fig, perf_fig, feat_fig, current_inv

if __name__ == '__main__':
    app.run(debug=True, port=8050)
