import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, LogTransform, NoOutputTransform

from api.logs import enable_logs, LogMonitoringHandler
from api.task import SegmentationTask

# celery + docker + redis
app = DashProxy(
    __name__,
    update_title=None,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    transforms=[LogTransform(), NoOutputTransform()],
    prevent_initial_callbacks=True,
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AutoDL-CV"

_task = SegmentationTask()

enable_logs(LogMonitoringHandler)
