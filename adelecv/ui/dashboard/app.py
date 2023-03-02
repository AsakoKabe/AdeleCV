import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, LogTransform, NoOutputTransform

from adelecv.ui.dashboard.task import SegmentationTask
from adelecv.ui.dashboard.utils import setup_notifications_log_config

app = DashProxy(
    __name__,
    update_title=None,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
    transforms=[LogTransform(log_config=setup_notifications_log_config()),
                NoOutputTransform()],
    prevent_initial_callbacks=True,
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AdeleCV"

_task = SegmentationTask()
