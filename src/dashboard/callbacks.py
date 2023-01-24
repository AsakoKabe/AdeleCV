from dash import Output, Input, html, State

from dashboard.components import controls
from app import app
from data.dataset import types
from app import _trainer


@app.callback(
    Output('hidden-div', component_property='children'),
    Input('submit-button-state', 'n_clicks'),
    State('select-type-dataset', 'value'),
    State('dataset-path', 'value'),
)
def update_scene_selection(n_clicks, dataset_type, dataset_path):
    print(dataset_type, dataset_path)
    if dataset_path is not None:
        _trainer.load_dataset(dataset_path, getattr(types, dataset_type), (640, 640))
        _trainer.create_dataset_session()

    return ''
