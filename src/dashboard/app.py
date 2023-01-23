import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input

from dashboard.components import nav, controls


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

srv = app.server

app.config.suppress_callback_exceptions = True

app.title = "AutoDL-CV"

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
        return html.Div(
            [
                controls
            ],
            className="m-5",
        )
    else:
        return "ERROR 404: Page not found!"


if __name__ == "__main__":
    # set debug to false when deploying app
    app.run_server(debug=True, port=8080)
