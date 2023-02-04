from dash import Output, Input, html, State

from app import app
from data.dataset import types
from app import _task


@app.callback(
    Output('hidden-div', component_property='children'),
    Input('submit-button-dataset', 'n_clicks'),
    State('dataset-type', 'value'),
    State('dataset-path', 'value'),
    State('img-height', 'value'),
    State('img-width', 'value'),
    State('train-size', 'value'),
    State('val-size', 'value'),
    State('test-size', 'value'),
    State('batch-size', 'value'),
)
def update_dataset_params(
        *args
):
    param_names = [
        "n_clicks",
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
        _task.create_dataset_session()

    return ''


@app.callback(
    Output('hidden-div-train', component_property='children'),
    Input('submit-button-train', 'n_clicks'),
    State('architectures', 'value'),
    State('lr-from', 'value'),
    State('lr-to', 'value'),
    State('optimizers', 'value'),
    State('loss-fns', 'value'),
    State('epoch-from', 'value'),
    State('epoch-to', 'value'),
    State('strategy', 'value'),
    State('num-trials', 'value'),
    State('device', 'value'),
)
def update_train_params(
        *args
):
    param_names = [
        "n_clicks",
        "architectures",
        "lr_from",
        "lr_to",
        "optimizers",
        "loss_fns",
        "epoch_from",
        "epoch_to",
        "strategy",
        "num_trials",
        "device"
    ]
    train_params = dict(zip(param_names, args))
    if all(train_params.values()):
        train_params['lr_range'] = (train_params['lr_from'], train_params['lr_to'])
        train_params['epoch_range'] = (train_params['epoch_from'], train_params['epoch_to'])
        print(train_params)
        _task.create_optimizer(train_params)
        _task.run_optimize()

    return ''


@app.callback(
    Output("collapse-dataset-settings", "is_open"),
    [Input("collapse-dataset-settings-btn", "n_clicks")],
    [State("collapse-dataset-settings", "is_open")],
)
def collapse_dataset(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-train-settings", "is_open"),
    [Input("collapse-train-settings-btn", "n_clicks")],
    [State("collapse-train-settings", "is_open")],
)
def collapse_train(n, is_open):
    if n:
        return not is_open
    return is_open
