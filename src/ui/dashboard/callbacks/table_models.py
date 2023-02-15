from dash import Output, Input, State, dcc
from dash.exceptions import PreventUpdate

from ui.dashboard.app import app, _task


@app.callback(
    Output("download-weights", "data"),
    Input("export-weights", "n_clicks"),
    State('stats-models-table', "derived_virtual_data"),
    State('stats-models-table', "derived_virtual_selected_rows"),
    prevent_initial_call=True
)
def export_weights(n_clicks, rows, derived_virtual_selected_rows):
    if not n_clicks:
        raise PreventUpdate()

    id_selected = {rows[i]['_id'] for i in derived_virtual_selected_rows}
    zip_path = _task.export_weights(id_selected)

    return dcc.send_file(zip_path.as_posix())


@app.callback(
    Input("convert-weights", "n_clicks"),
    State('stats-models-table', "derived_virtual_data"),
    State('stats-models-table', "derived_virtual_selected_rows"),
    prevent_initial_call=True
)
def convert_weights(n_clicks, rows, derived_virtual_selected_rows):
    if not n_clicks:
        raise PreventUpdate()

    print('convert weights', derived_virtual_selected_rows)
    print(rows)
    # return ''
