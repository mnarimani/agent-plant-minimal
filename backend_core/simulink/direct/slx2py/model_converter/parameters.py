from __future__ import annotations

from .blocks import MatlabFunction
from .model import Model
from .stateflow import _Assignment, _Conditional


def extract_model_parameters(model: Model) -> list[dict[str, str]]:
    parameters = []
    for block in model.blocks.values():
        if not isinstance(block, MatlabFunction):
            continue
        interface_names = set(block.chart.input_names) | set(block.chart.output_names)
        assigned_names = dict.fromkeys(_assigned_names(block.chart.statements))
        for name in assigned_names:
            if name not in interface_names:
                parameters.append({"name": name, "block_name": block.name, "block_id": block.block_id})
    return parameters


def _assigned_names(statements):
    for statement in statements:
        if isinstance(statement, _Assignment):
            yield statement.name
        elif isinstance(statement, _Conditional):
            for branch in statement.branches:
                yield from _assigned_names(branch.statements)
