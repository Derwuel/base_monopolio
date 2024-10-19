"""
This is a boilerplate pipeline 'data_mono'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import pre_proceso

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=pre_proceso,
            inputs="Base_clientes_Monopoly-0",
            outputs="pos_proceso",
            name="pre_proceso_node",
        ),
    ])
