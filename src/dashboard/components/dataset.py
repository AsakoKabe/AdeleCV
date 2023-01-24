import dash_bootstrap_components as dbc
from dash import html

controls = [
    dbc.Form(
        [
            # html.Br(),
            dbc.Label("Path Dataset"),
            dbc.Input(id='dataset-path')
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Type Dataset"),
            dbc.RadioItems(
                id="select-type-dataset",
                options=[{"label": x, "value": x} for x in ['COCOSemantic', 'ImageMaskSemantic']],
                value="COCOSemantic",
                inline=True
            ),
        ]
    ),
    dbc.Form(
        [
            # html.Br(),
            # dbc.Label("Load Dataset"),
            dbc.Button(
                id='submit-button-state',
                children='Load Dataset'
            ),
        ]
    )
]

dataset = dbc.Container(
    [
        html.Hr(),
        html.Div(
            id='hidden-div',
            style={'display': 'none'}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(controls, body=True),
                    ],
                    md=2,
                ),
                dbc.Col(
                    html.Div(
                        html.Iframe(
                            src="http://localhost:5151/datasets/",
                            style={"height": "100%", "width": "100%"}
                        )
                    )
                )
            ]
        ),
    ],
)
