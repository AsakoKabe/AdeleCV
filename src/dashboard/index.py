import os
import shutil

from dash import html, dcc, Output, Input


from dashboard.components import nav, dataset, train_board, table_models
from app import app
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
        return "Page empty"
    elif pathname == '/dataset':
        return dataset
    elif pathname == '/train':
        return train_board
    elif pathname == '/table-models':
        return table_models()
    else:
        return "ERROR 404: Page not found!"


if __name__ == "__main__":
    # set debug to false when deploying app
    app.run_server(debug=True, port=8080)
    if os.path.exists(f'{os.getenv("TMP_PATH")}'):
        shutil.rmtree(f'{os.getenv("TMP_PATH")}')
