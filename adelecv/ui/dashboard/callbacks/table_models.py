from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate

from adelecv.api.config import Settings
from adelecv.api.modification_models import ExportWeights, ConvertWeights
from adelecv.ui.dashboard.app import app, _task


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
    zip_path = ExportWeights(Settings.WEIGHTS_PATH).create_zip(id_selected)

    return dcc.send_file(zip_path.as_posix())


@app.callback(
    Output("download-converted-onnx", "data"),
    Input("convert-weights-format-onnx", "n_clicks"),
    State('stats-models-table', "derived_virtual_data"),
    State('stats-models-table', "derived_virtual_selected_rows"),
    prevent_initial_call=True
)
def convert_weights(n_clicks, rows, derived_virtual_selected_rows):
    if not n_clicks:
        raise PreventUpdate()

    id_selected = {rows[i]['_id'] for i in derived_virtual_selected_rows}
    img_shape = _task.img_shape[0], _task.img_shape[0], 3
    zip_path = ConvertWeights(
        img_shape, Settings.WEIGHTS_PATH
    ).run(id_selected, 'onnx')

    return dcc.send_file(zip_path.as_posix())
