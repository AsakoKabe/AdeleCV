import dash_bootstrap_components as dbc
from dash import html, dcc
import segmentation_models_pytorch as smp


controls = [
    dbc.Form(
        [
            dbc.Label("Architectures"),
            dcc.Dropdown(
                [
                    "Unet",
                    "UnetPlusPlus",
                    "MAnet",
                    "Linknet",
                    "FPN",
                    "PSPNet",
                    "DeepLabV3",
                    "DeepLabV3Plus",
                    "PAN",
                ],
                id='architectures',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Encoder"),
            dcc.Dropdown(
                smp.encoders.get_encoder_names(),
                id='encoders',
                multi=True,
                value=['mobilenet_v2'],
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
                        [dbc.Input(id='lr-to', placeholder='to', type='number', min='0', value=0.001)]
                    )
                ]
            ),
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Optimizers"),
            dcc.Dropdown(
                ["AdamW", "RMSprop", "SGD"],
                id='optimizers',
                multi=True
            )
        ]
    ),
    dbc.Form(
        [
            dbc.Label("Loss functions"),
            dcc.Dropdown(
                [
                    "JaccardLoss",
                    "DiceLoss",
                    "FocalLoss",
                    "LovaszLoss",
                    "SoftBCEWithLogitsLoss",
                    "SoftCrossEntropyLoss",
                    "TverskyLoss",
                    "MCCLoss",
                ],
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
    # dbc.Form(
    #     [
    #         html.Br(),
    #         dbc.Button(
    #             "Hide settings",
    #             id="collapse-train-settings-btn",
    #         ),
    #     ]
    # )
]

train_board = dbc.Container(
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
                                src="http://localhost:6006/",
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