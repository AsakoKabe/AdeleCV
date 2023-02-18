from dash import Input
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashLogger

from api.logs import get_logger
from ui.dashboard.app import app


def _add_log(log, dash_logger: DashLogger):
    level, message = log.split(' - ')
    if level == 'INFO':
        dash_logger.info(message)
    elif level == 'WARNING':
        dash_logger.warning(message)
    elif level == 'ERROR':
        dash_logger.error(message)
    # elif level == 'DEBUG':
    #     dash_logger.debug(message)
    else:
        dash_logger.log(level, message)


@app.callback(
    Input("interval-notifications", "n_intervals"),
    log=True
)
def notify(
        n_intervals,
        dash_logger: DashLogger
):
    if n_intervals is None:
        raise PreventUpdate()

    logger = get_logger()
    for log in logger.handlers[0].pop_logs():
        _add_log(log, dash_logger)
