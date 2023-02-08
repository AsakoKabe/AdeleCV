from dash import Output, Input, State, dcc

from ui.dashboard.app import app, _task


@app.callback(
    Output('hidden-div-train', component_property='children'),
    Input('submit-button-train', 'n_clicks'),
    State('architectures', 'value'),
    State('encoders', 'value'),
    State('lr-from', 'value'),
    State('lr-to', 'value'),
    State('optimizers', 'value'),
    State('loss-fns', 'value'),
    State('epoch-from', 'value'),
    State('epoch-to', 'value'),
    State('strategy', 'value'),
    State('num-trials', 'value'),
    State('device', 'value'),
    prevent_initial_call=True,
    # running=[
    #     (Output("submit-button-train", "disabled"), True, False),
    # ],
)
def update_train_params(
        *args
):
    param_names = [
        "n_clicks",
        "architectures",
        "encoders",
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
        _task.train(train_params)

    return ''


@app.callback(
    Output("collapse-train-settings", "is_open"),
    [Input("collapse-train-settings-btn", "n_clicks")],
    [State("collapse-train-settings", "is_open")],
    prevent_initial_call=True
)
def collapse_train(n, is_open):
    if n:
        return not is_open
    return is_open

