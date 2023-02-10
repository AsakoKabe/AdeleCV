import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, LogTransform

from api.task import SegmentationTask

# celery + docker + redis
app = DashProxy(
    __name__,
    update_title=None,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    transforms=[LogTransform(
        # log_config=log_confid
    )],
    prevent_initial_callbacks=True,
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AutoDL-CV"

_task = SegmentationTask()

