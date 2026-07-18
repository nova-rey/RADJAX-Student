"""Legacy scalar objective fixture shared by focused learning-loop tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearObjective:
    def evaluate(self, parameters, batch):
        x = float(batch.inputs["token_ids"]["x"])
        target = float(batch.targets["target"]["y"])
        prediction = parameters["trunk.weight"] * x + parameters["head.weight"]
        error = prediction - target
        return error * error, {
            "head.weight": 2.0 * error,
            "trunk.bias": 0.0,
            "trunk.weight": 2.0 * error * x,
        }
