from dash import Output, Input, html, State

from dashboard.components import controls
from app import app
from data.dataset import types
from app import _trainer


@app.callback(
    Output('hidden-div', component_property='children'),
    Input('submit-button-state', 'n_clicks'),
    State('dataset-type', 'value'),
    State('dataset-path', 'value'),
    State('img-height', 'value'),
    State('img-width', 'value'),
    State('train-size', 'value'),
    State('val-size', 'value'),
    State('test-size', 'value'),
    State('batch-size', 'value'),
)
def update_scene_selection(
        n_clicks,
        dataset_type,
        dataset_path,
        img_height,
        img_width,
        train_size,
        val_size,
        test_size,
        batch_size
):
    if dataset_type is not None:
        _trainer.load_dataset(
            dataset_path=dataset_path,
            dataset_type=getattr(types, dataset_type),
            img_size=(img_height, img_width),
            split=(train_size, val_size, test_size),
            batch_size=batch_size
        )
        _trainer.create_dataset_session()

    return ''
