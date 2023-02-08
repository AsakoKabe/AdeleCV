from dash import Output, Input, State, dcc

from ui.dashboard.app import app, _task


@app.callback(
    Output("download-weights", "data"),
    Input("export-weights", "n_clicks"),
    State('stats-models-table', "derived_virtual_data"),
    State('stats-models-table', "derived_virtual_selected_rows"),
    prevent_initial_call=True
)
def export_weights(n_clicks, rows, derived_virtual_selected_rows):
    id_selected = set([rows[i]['_id'] for i in derived_virtual_selected_rows])
    zip_path = _task.export_weights(id_selected)

    return dcc.send_file(zip_path.as_posix())


@app.callback(
    Output('hidden-div-table2', component_property='children'),
    Input("convert-weights", "n_clicks"),
    State('stats-models-table', "derived_virtual_data"),
    State('stats-models-table', "derived_virtual_selected_rows"),
    prevent_initial_call=True
)
def convert_weights(n_clicks, rows, derived_virtual_selected_rows):
    print('convert weights', derived_virtual_selected_rows)
    print(rows)
    return ''
