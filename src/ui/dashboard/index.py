import os
import shutil

from dash import html, dcc, Output, Input

from api.config import get_settings
from ui.dashboard.components import nav, dataset, train_board, table_models
from ui.dashboard.app import app, _task
from callbacks import *


content = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content")
    ]
)

app.layout = html.Div([nav, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == '/':
        return "Page empty"
    elif pathname == '/dataset':
        return dataset
    elif pathname == '/train':
        return train_board
    elif pathname == '/table-models':
        return table_models(_task.stats_models)
    else:
        return "ERROR 404: Page not found!"


if __name__ == "__main__":
    if os.path.exists(f'{get_settings().TMP_PATH.as_posix()}'):
        shutil.rmtree(f'{get_settings().TMP_PATH.as_posix()}')
    # set debug to false when deploying app
    app.run_server(debug=True, port=8080)
    # cache.close()
    if os.path.exists(f'{get_settings().TMP_PATH.as_posix()}'):
        shutil.rmtree(f'{get_settings().TMP_PATH.as_posix()}')
