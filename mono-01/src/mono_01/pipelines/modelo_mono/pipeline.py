"""
This is a boilerplate pipeline 'modelo_mono'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, mod_regrecion_lineal, evaluacion_regrecion_lineal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["pos_proceso", "params:model_options"],
            outputs=["X_train_modelo1", "X_test_modelo1", "y_train_modelo1", "y_test_modelo1"],
        ),
        node(
            func=mod_regrecion_lineal,
            inputs=["X_train_modelo1", "y_train_modelo1"],
            outputs="regressor",
        ),
        node(
            func=evaluacion_regrecion_lineal,
            inputs=["regressor", "X_test_modelo1", "y_test_modelo1"],
            outputs=None,
        ),
    ])
