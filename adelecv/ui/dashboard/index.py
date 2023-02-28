from dash import html, dcc, Output, Input

import callbacks  # pylint: disable=unused-import,import-error
from adelecv.ui.dashboard.components import nav, dataset, train_board, table_models, console
from adelecv.ui.dashboard.app import app, _task


content = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
        dcc.Interval(
            id="interval-notifications"
        ),
        dcc.Interval(
            id="interval-console-log",
            # interval=5000,
        ),
    ]
)

app.layout = html.Div([nav, content])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")],
    prevent_initial_call=True,
)
def render_page_content(pathname):
    if pathname == '/':
        return "Page empty"
    if pathname == '/dataset':
        return dataset
    if pathname == '/train':
        return train_board
    if pathname == '/table-models':
        return table_models(_task.stats_models)
    if pathname == '/log-console':
        return console

    return "ERROR 404: Page not found!"


if __name__ == "__main__":
    app.run(
        port=8080,
        # debug=True
    )
