import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html


def table_models(df: pd.DataFrame) -> dbc.Container:
    cols = [
        {"name": i, "id": i, "hideable": True, "selectable": True,
         'editable': i == 'name'}
        for i in df.columns
    ]

    return dbc.Container(
        [
            html.Hr(),
            dbc.DropdownMenu(
                label="Menu",
                id='table-menu',
                children=[
                    dbc.DropdownMenuItem(
                        "Export weights", id='export-weights'
                        ),
                    dbc.DropdownMenu(
                        label="Convert weights",
                        id='convert-weights',
                        children=[
                            dbc.DropdownMenuItem(
                                "ONNX", id='convert-weights-format-onnx'
                            ),
                        ],
                        color="secondary",
                        direction="end",
                        style={"margin-bottom": "1%"}
                    ),
                ],
                style={"margin-bottom": "1%"}
            ),
            dcc.Download(id="download-weights"),
            dcc.Download(id="download-converted-onnx"),
            html.Div(
                [
                    dash_table.DataTable(
                        id='stats-models-table',
                        columns=cols,
                        data=df.to_dict('records'),
                        # editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="multi",
                        row_deletable=True,
                        page_action="native",
                        page_current=0,
                        page_size=10,
                        css=[
                            {'selector': 'table',
                             'rule': 'table-layout: fixed'},
                        ],
                        style_cell={
                            'width': f'{len(df.columns)}',
                            'textOverflow': 'ellipsis',
                            'overflow': 'hidden'
                        },
                        export_format='csv',
                    ),
                ],
                # className='table'
            )
        ]
    )
