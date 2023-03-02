from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from adelecv.api.logs import get_logger
from adelecv.ui.dashboard.app import _task, app


@app.callback(
    Input('submit-button-train', 'n_clicks'),
    State('architectures', 'value'),
    State('encoders', 'value'),
    State('pretrained-weight', 'value'),
    State('lr-from', 'value'),
    State('lr-to', 'value'),
    State('optimizers', 'value'),
    State('loss-fns', 'value'),
    State('epoch-from', 'value'),
    State('epoch-to', 'value'),
    State('strategy', 'value'),
    State('num-trials', 'value'),
    State('optimize-score', 'value'),
    State('device', 'value'),
    prevent_initial_call=True,
    # running=[
    #     (Output("submit-button-train", "disabled"), True, False),
    # ],
)
def update_train_params(
        n_clicks,
        *args,
):
    if not n_clicks:
        raise PreventUpdate()

    param_names = [
        "architectures",
        "encoders",
        'pretrained_weight',
        "lr_from",
        "lr_to",
        "optimizers",
        "loss_fns",
        "epoch_from",
        "epoch_to",
        "strategy",
        "num_trials",
        'optimize_score',
        "device",
    ]
    train_params = dict(zip(param_names, args))
    if all(train_params.values()):
        train_params['lr_range'] = (
            train_params['lr_from'], train_params['lr_to'])
        train_params['epoch_range'] = (
            train_params['epoch_from'], train_params['epoch_to'])
        try:
            _task.train(train_params)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger = get_logger()
            logger.error(e)


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
