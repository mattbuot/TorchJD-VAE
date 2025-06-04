from typing import Literal

import torch
from torch import Tensor
from torchjd.aggregation._pref_vector_utils import (
    pref_vector_to_str_suffix,
    pref_vector_to_weighting,
)
from torchjd.aggregation.bases import _WeightedAggregator, _Weighting
from torchjd.aggregation.mean import _MeanWeighting
from torchjd.aggregation.upgrad import _UPGradWrapper


class PairwiseUpgradAggregator(_WeightedAggregator):
    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: Literal["quadprog"] = "quadprog",
        final_aggregation: Literal["mean", "upgrad"] = "mean",
    ):
        weighting = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
        self._pref_vector = pref_vector
        self._final_aggregation = final_aggregation

        super().__init__(
            weighting=PairwiseUpgradAggregatorWrapper(
                weighting=weighting, norm_eps=norm_eps, reg_eps=reg_eps, solver=solver, final_aggregation=final_aggregation,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, norm_eps="
            f"{self.weighting.norm_eps}, reg_eps={self.weighting.reg_eps}, "
            f"solver={repr(self.weighting.solver)})"
        )

    def __str__(self) -> str:
        return f"PairwiseUpgrad_{self._final_aggregation}{pref_vector_to_str_suffix(self._pref_vector)}"


class PairwiseUpgradAggregatorWrapper(_Weighting):

    def __init__(
        self,
        weighting: _Weighting,
        norm_eps: float,
        reg_eps: float,
        solver: Literal["quadprog"],
        final_aggregation: Literal["mean", "upgrad"],
    ):
        super().__init__()
        self._upgrad_wrapper = _UPGradWrapper(
            weighting=weighting, norm_eps=norm_eps, reg_eps=reg_eps, solver=solver
        )
        self.final_aggregation = final_aggregation

    def forward(self, matrix: Tensor) -> Tensor:
        
        pairwise_aggregations = torch.stack([
            self._upgrad_wrapper(matrix[i:i + 2]) for i in range(0, matrix.size(0), 2)
        ]).flatten()
        return pairwise_aggregations / matrix.size(0)
        # .view(1, -1)
        # paired_matrix = torch.mul(pairwise_aggregations, matrix)

        # if self.final_aggregation == "upgrad":
        #     return self._upgrad_wrapper(paired_matrix)
        # elif self.final_aggregation == "mean":
        #     return paired_matrix.mean(dim=0)
        # else:
        #     raise ValueError(f"Unknown final aggregation method: {self.final_aggregation}")
        
