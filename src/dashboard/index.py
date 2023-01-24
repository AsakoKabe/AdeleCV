from dash import html, dcc, Output, Input
from dashboard.components import nav, controls, dataset

from app import app
from dashboard.components import nav
import callbacks


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
    elif pathname == '/dataset':
        return dataset
    else:
        return "ERROR 404: Page not found!"


if __name__ == "__main__":
    # set debug to false when deploying app
    app.run_server(debug=True, port=8080)
