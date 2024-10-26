"""
This is a boilerplate pipeline 'calculo_mono'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data2, mod_regresion_logistica,evaluacion_regresion_logistica, mod_arbol_decision, evaluacion_arbol_decision, mod_svc, evaluacion_svc


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=split_data2,
            inputs=["pos_proceso", "params:model_options2"],
            outputs=["X_train_clasi", "X_test_clasi", "y_train_clasi", "y_test_clasi"],
            name="split_data_node_02"
        ),
        node(
            func=mod_regresion_logistica,
            inputs=["X_train_clasi", "y_train_clasi"],
            outputs="logistic_model",
        ),
        node(
            func=evaluacion_regresion_logistica,
            inputs=["logistic_model", "X_test_clasi", "y_test_clasi"],
            outputs=None,
        ),
        node(
            func=mod_arbol_decision,
            inputs=["X_train_clasi", "y_train_clasi"],
            outputs="decision_tree_model",
        ),
        node(
            func=evaluacion_arbol_decision,
            inputs=["decision_tree_model", "X_test_clasi", "y_test_clasi"],
            outputs=None,
        ),
        node(
            func=mod_svc,
            inputs=["X_train_clasi", "y_train_clasi"],
            outputs="svc_model",
        ),
        node(
            func=evaluacion_svc,
            inputs=["svc_model", "X_test_clasi", "y_test_clasi"],
            outputs=None,
        ),
    ])