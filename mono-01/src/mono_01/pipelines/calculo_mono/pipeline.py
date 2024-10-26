"""
This is a boilerplate pipeline 'calculo_mono'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import  split_data2, mod_SGDClassifier, evaluacion_SGDClassifier, mod_tree_model_clasicicacion, evaluacion_tree_model_clasicicacion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data2,
            inputs=["pos_proceso", "params:model_options2"],
            outputs=["X_train_clasi", "X_test_clasi", "y_train_clasi", "y_test_clasi"],
            name="split_data_node_02"
        ),
        node(
            func=mod_SGDClassifier,
            inputs=["X_train_clasi", "y_train_clasi"],
            outputs="model_SGDClassifier",
        ),
        node(
            func=evaluacion_SGDClassifier,
            inputs=["model_SGDClassifier", "X_test_clasi", "y_test_clasi"],
            outputs=None,
        ),
                node(
            func=mod_tree_model_clasicicacion,
            inputs=["X_train_clasi", "y_train_clasi"],
            outputs="model_tree_clasicicacion",
        ),
        node(
            func=evaluacion_tree_model_clasicicacion,
            inputs=["model_tree_clasicicacion", "X_test_clasi", "y_test_clasi"],
            outputs=None,
        ),

    ])