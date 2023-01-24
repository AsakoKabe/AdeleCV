import dash
import dash_bootstrap_components as dbc

from train.trainer import Trainer


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AutoDL-CV"

_trainer = Trainer()

