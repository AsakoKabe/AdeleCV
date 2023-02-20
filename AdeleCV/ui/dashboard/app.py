import logging

import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, LogTransform, NoOutputTransform

from api.logs import enable_logs, LogMonitoringHandler
from config import get_settings
from ui.dashboard.task import SegmentationTask
from ui.dashboard.utils import setup_notifications_log_config, LogConsoleHandler

# celery + docker + redis
app = DashProxy(
    __name__,
    update_title=None,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    transforms=[LogTransform(log_config=setup_notifications_log_config()), NoOutputTransform()],
    prevent_initial_callbacks=True,
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AutoDL-CV"

_task = SegmentationTask()

enable_logs(LogMonitoringHandler())
enable_logs(
    LogConsoleHandler(),
    get_settings().LOGGER_NAME,
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
