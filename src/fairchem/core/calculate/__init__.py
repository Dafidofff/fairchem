"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# These are loaded lazily to avoid pulling in ray/serve at import time.
# Import directly if needed:
#   from fairchem.core.calculate._batch import InferenceBatcher
#   from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
InferenceBatcher = None
FAIRChemCalculator = None
FormationEnergyCalculator = None
InferenceSettings = None

__all__ = [
    "FAIRChemCalculator",
    "FormationEnergyCalculator",
    "InferenceBatcher",
    "InferenceSettings",
]
