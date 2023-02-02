import dash_bootstrap_components as dbc
from dash import html, dcc

from data.dataset import dataset_types

controls = [
    dbc.Form(
        [
            # html.Br(),
            dbc.Label("Path Dataset"),
            dbc.Input(id='dataset-path', value=r'F:\dataset\ph2')
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Type Dataset"),
            dcc.Dropdown(
                [dataset_type.__name__ for dataset_type in dataset_types],
                id='dataset-type'
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label('Img size'),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Input(id='img-height', placeholder='Height', type='number', min='0', value=256)]
                    ),
                    dbc.Col(
                        [dbc.Input(id='img-width', placeholder='Width', type='number', min='0', value=256)]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label('Split dataset'),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Input(id='train-size', placeholder='Train', type='number', min='0', max='1', value=0.7)]
                    ),
                    dbc.Col(
                        [dbc.Input(id='val-size', placeholder='Val', type='number', min='0', max='1', value=0.2)]
                    ),
                    dbc.Col(
                        [dbc.Input(id='test-size', placeholder='Test', type='number', min='0', max='1', value=0.1)]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label('Batch size'),
            dbc.Input(id='batch-size', type='number', min='1', value=16)
        ]
    ),
    dbc.Form(
        [
            html.Br(),
            # dbc.Label("Load Dataset"),
            dbc.Button(
                id='submit-button-dataset',
                children='Load Dataset'
            ),
        ]
    ),
    # dbc.Form(
    #     [
    #         html.Br(),
    #         dbc.Button(
    #             "Hide settings",
    #             id="collapse-dataset-settings-btn",
    #         ),
    #     ]
    # )
]

dataset = dbc.Container(
    [
        html.Hr(),
        html.Div(
            id='hidden-div',
            style={'display': 'none'}
        ),
        dbc.Button(
            "Hide settings",
            id="collapse-dataset-settings-btn",
            style={"margin-bottom": "1%"}
        ),
        dbc.Row(
            [
                dbc.Collapse(
                    dbc.Card(controls, body=True),
                    id="collapse-dataset-settings",
                    is_open=True,
                    class_name='col-md-3',
                    dimension='width',
                ),

                dbc.Col(
                    [
                        html.Div(
                            html.Iframe(
                                src="http://localhost:5151/datasets/",
                                style={"height": "100vh", "width": "100%"},
                            )
                        ),
                    ],
                    # md=6
                )
            ],
        ),
    ],
)
