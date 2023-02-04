from dash import html, dash_table
import dash_bootstrap_components as dbc


def table_models(df):
    cols = [{"name": i, "id": i, "hideable": True, "selectable": True} for i in df.columns]

    return dbc.Container(
        [
            html.Div(
                [
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        columns=cols,
                        data=df.to_dict('records'),
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="multi",
                        row_deletable=True,
                        page_action="native",
                        page_current=0,
                        page_size=10,
                        css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                        style_cell={
                            'width': '{}%'.format(len(df.columns)),
                            'textOverflow': 'ellipsis',
                            'overflow': 'hidden'
                        },
                        export_format='xlsx',
                    ),
                ]
            )
        ]
    )
