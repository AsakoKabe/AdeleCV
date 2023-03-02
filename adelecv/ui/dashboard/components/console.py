import dash_bootstrap_components as dbc
from dash import html

console = dbc.Container(
    [
        html.Hr(),
        html.Div(
            html.Div(
                html.Div(
                    id='outputs'
                ),
                className='console-inner'
            ),
            className='console'
        )
    ],
)
