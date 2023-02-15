import os
import shutil

from dash import html, dcc, Output, Input

import callbacks  # pylint: disable=unused-import,import-error
from config import get_settings
from ui.dashboard.components import nav, dataset, train_board, table_models
from .app import app, _task


content = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
        dcc.Interval(
            id="interval-notifications"
        ),
        dcc.Store(id='count-notifications'),
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

    return "ERROR 404: Page not found!"


if __name__ == "__main__":
    if os.path.exists(f'{get_settings().TMP_PATH.as_posix()}'):
        shutil.rmtree(f'{get_settings().TMP_PATH.as_posix()}')
    # set debug to false when deploying app
    app.run_server(port=8080, debug=True)
    # cache.close()
    if os.path.exists(f'{get_settings().TMP_PATH.as_posix()}'):
        shutil.rmtree(f'{get_settings().TMP_PATH.as_posix()}')
