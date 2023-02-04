from dash import html, dash_table
from ..app import _task


def table_models():
    return html.Div(
        [
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True} for i in _task.stats_models.columns
                ],
                data=_task.stats_models.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
            ),
            html.Div(id='datatable-interactivity-container')
        ]
    )
