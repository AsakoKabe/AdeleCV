import logging

from dash import Input, Output
from dash.exceptions import PreventUpdate

from adelecv.api.config import Settings
from adelecv.ui.dashboard.app import app


@app.callback(
    Input("interval-console-log", "n_intervals"),
    Output("outputs", "children"),
)
def add_logs_to_console(
        n_intervals,
):
    if n_intervals is None:
        raise PreventUpdate()

    logger = logging.getLogger(Settings.LOGGER_NAME)
    console_data = '\n'.join(logger.handlers[1].logs[::-1])

    return console_data
