import dash_bootstrap_components as dbc
from dash import dcc, html

from adelecv.api.config import Settings
from adelecv.api.models.segmentations import (get_encoders, get_losses,
                                              get_models, get_optimize_scores,
                                              get_pretrained_weights,
                                              get_torch_optimizers)
from adelecv.api.optimize.segmentations import get_hp_optimizers

controls = [
    # dbc.Label("Model hyper params", class_name='font-weight-bold'),
    dbc.Form(
        [
            dbc.Label("Architectures"),
            dcc.Dropdown(
                get_models(),
                id='architectures',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Encoder"),
            dcc.Dropdown(
                get_encoders(),
                id='encoders',
                multi=True,
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Pretrained weight"),
            dcc.Dropdown(
                get_pretrained_weights(),
                id='pretrained-weight',
                multi=True,
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Learning rate"),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Input(
                            id='lr-from', placeholder='from', type='number',
                            min='0',
                        )]
                    ),
                    dbc.Col(
                        [dbc.Input(
                            id='lr-to', placeholder='to', type='number',
                            min='0',
                        )]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Optimizers"),
            dcc.Dropdown(
                get_torch_optimizers(),
                id='optimizers',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Loss functions"),
            dcc.Dropdown(
                get_losses(),
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
                        [dbc.Input(
                            id='epoch-from', placeholder='from', type='number',
                            min='1'
                        )]
                    ),
                    dbc.Col(
                        [dbc.Input(
                            id='epoch-to', placeholder='to', type='number',
                            min='1'
                        )]
                    )
                ]
            ),
        ]
    ),
    html.Hr(),
    # dbc.Label("Algorith optimize params", class_name='font-weight-bold'),
    dbc.Form(
        [
            dbc.Label("Hyperparameter optimizer"),
            dcc.Dropdown(
                get_hp_optimizers(),
                id='strategy',
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Num trials"),
            dbc.Input(id='num-trials', type='number', min='1')
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Optimize score"),
            dcc.Dropdown(
                get_optimize_scores(),
                id='optimize-score',
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Device"),
            dcc.Dropdown(
                ["GPU", "CPU"],
                id='device',
            )
        ]
    ),
    # todo: block but
    dbc.Form(
        [
            html.Br(),
            # dbc.Label("Load Dataset"),
            dbc.Button(
                id='submit-button-train',
                children='Start'
            ),
        ]
    ),
]


def train_board() -> dbc.Container:
    return dbc.Container(
        [
            html.Hr(),
            dbc.Button(
                "Hide settings",
                id="collapse-train-settings-btn",
                style={"margin-bottom": "1%"}
            ),
            dbc.Row(
                [
                    dbc.Collapse(
                        dbc.Card(controls, body=True),
                        id="collapse-train-settings",
                        is_open=True,
                        class_name='col-md-3',
                        dimension='width',
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                html.Iframe(
                                    src=f"http://localhost:"
                                        f"{Settings.TENSORBOARD_PORT}/",
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
