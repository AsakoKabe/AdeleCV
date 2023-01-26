import dash_bootstrap_components as dbc
from dash import html, dcc

controls = [
    dbc.Form(
        [
            dbc.Label("Architectures"),
            dcc.Dropdown(
                ["DeepLabV3MobileNet", "LRASPPMobileNetV3"],
                id='architectures',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Learning rate"),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Input(id='lr-from', placeholder='from', type='number', min='0', value=0.001)]
                    ),
                    dbc.Col(
                        [dbc.Input(id='lr-to', placeholder='to', type='number', min='0', value=0.3)]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Optimizers"),
            dcc.Dropdown(
                ["Adam", "RMSprop", "SGD"],
                id='optimizers',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Loss functions"),
            dcc.Dropdown(
                ["CrossEntropyLoss"],
                id='loss-fns',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Num epoch"),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Input(id='epoch-from', placeholder='from', type='number', min='1', value=2)]
                    ),
                    dbc.Col(
                        [dbc.Input(id='epoch-to', placeholder='to', type='number', min='1', value=5)]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Hyperparameter optimizer"),
            dcc.Dropdown(
                [
                    "RandomSampler", "GridSampler", "TPESampler", "CmaEsSampler",
                    "NSGAIISampler", "QMCSampler", "MOTPESampler"
                ],
                id='strategy',
                value='TPESampler'
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Num trials"),
            dbc.Input(id='num-trials', type='number', min='1', value=10)
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Device"),
            dcc.Dropdown(
                ["GPU", "CPU"],
                id='device',
                value='GPU'
            )
        ]
    ),
    dbc.Form(
        [
            html.Br(),
            # dbc.Label("Load Dataset"),
            dbc.Button(
                id='submit-button-train',
                children='Start'
            ),
        ]
    )
]

train_board = dbc.Container(
    [
        html.Hr(),
        html.Div(
            id='hidden-div-train',
            style={'display': 'none'}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(controls, body=True),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        # html.Div(
                        #     html.Iframe(
                        #         src="http://localhost:5151/datasets/",
                        #         style={"height": "100vh", "width": "100%"},
                        #     )
                        # ),
                    ],
                    # md=6
                )
            ],
        ),
    ],
)
