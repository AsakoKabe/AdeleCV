from dash import Output, Input, State, dcc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashLogger

from ui.dashboard.app import app, _task
from api.data.segmentations import types


@app.callback(
    # Output('hidden-div', component_property='children'),
    Input('submit-button-segmentations', 'n_clicks'),
    State('segmentations-type', 'value'),
    State('segmentations-path', 'value'),
    State('img-height', 'value'),
    State('img-width', 'value'),
    State('train-size', 'value'),
    State('val-size', 'value'),
    State('test-size', 'value'),
    State('batch-size', 'value'),
    prevent_initial_call=True,
    # running=[
    #     (Output("submit-button-segmentations", "disabled"), True, False),
    # ],
    log=True
)
def update_dataset_params(
    n_clicks,
    *args,
    dash_logger: DashLogger,
):
    if not n_clicks:
        raise PreventUpdate()

    param_names = [
        "dataset_type",
        "dataset_path",
        "img_height",
        "img_width",
        "train_size",
        "val_size",
        "test_size",
        "batch_size"
    ]
    dataset_params = dict(zip(param_names, args))
    if all(dataset_params.values()):
        _task.load_dataset(
            dataset_path=dataset_params["dataset_path"],
            dataset_type=getattr(types, dataset_params["dataset_type"]),
            img_size=(dataset_params["img_height"], dataset_params["img_width"]),
            split=(dataset_params["train_size"], dataset_params["val_size"], dataset_params["test_size"]),
            batch_size=dataset_params["batch_size"],
        )
    dash_logger.info("Here goes some info")
    dash_logger.warning("This is a warning")
    dash_logger.error("Some error occurred")
    dash_logger.info("Here goes some info")


@app.callback(
    Output("collapse-segmentations-settings", "is_open"),
    [Input("collapse-segmentations-settings-btn", "n_clicks")],
    [State("collapse-segmentations-settings", "is_open")],
    prevent_initial_call=True
)
def collapse_dataset(n, is_open):
    if n:
        return not is_open
    return is_open