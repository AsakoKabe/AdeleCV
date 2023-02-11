import os
import shutil

from dash import html, dcc, Output, Input

from api.logs import enable_logs
from config import get_settings
from ui.dashboard.components import nav, dataset, train_board, table_models
from ui.dashboard.app import app, _task
import callbacks

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
    app.run_server(port=8080, debug=True)
    # cache.close()
    if os.path.exists(f'{get_settings().TMP_PATH.as_posix()}'):
        shutil.rmtree(f'{get_settings().TMP_PATH.as_posix()}')
