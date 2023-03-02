import dash_bootstrap_components as dbc
from dash import html

description = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2('Dataset'),
                        html.P(
                            'Loading dataset and displaying a window with'
                            ' fiftyone.\n\n\n'
                            'To load dataset you need to specify the path'
                            ' to the dataset, format of the dataset,'
                            'the size of the image, split train val test,'
                            ' batch size.'
                            )
                    ],
                    className='col-md-4',
                    style={"padding": "1%", 'white-space': 'pre-wrap'}
                ),
                dbc.Col(
                    [
                        html.H2('Train'),
                        html.P(
                            'Hyperparams selection, optimization parameters'
                            ' setting, tensorboard window.\n\n'
                            'Dataset must be loaded to run. '
                            'After setting up the parameters,'
                            ' notifications about '
                            'the start of training will appear.'
                            ' The tensorboard window displays statistics'
                            ' on model '
                            'training.'
                            )
                    ],
                    className='col-md-4',
                    style={"padding": "1%", 'white-space': 'pre-wrap'}
                ),

            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2('Table Models'),
                        html.P(
                            'Table with training results and management of '
                            'export and convert of weights.\n\n'
                            'You can hide unnecessary columns in the table. '
                            'It can be exported in .csv format. To'
                            'convert or export, you need to select the '
                            'desired models and select the necessary'
                            'action'
                            'in the menu.'
                            )
                    ],
                    className='col-md-4',
                    style={"padding": "1%", 'white-space': 'pre-wrap'}
                ),
                dbc.Col(
                    [
                        html.H2('Console'),
                        html.P('Displaying all logs.\n\n\n')
                    ],
                    className='col-md-4',
                    style={"padding": "1%", 'white-space': 'pre-wrap'}
                ),
            ]
        )
    ]
)
