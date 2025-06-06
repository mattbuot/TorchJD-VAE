from typing import Iterable, Sequence

from torch import Tensor
from torchjd._autojac._transform import (
    Accumulate,
    Aggregate,
    Diagonalize,
    Init,
    Jac,
    OrderedSet,
    Select,
    Transform,
)
from torchjd._autojac._utils import (
    as_checked_ordered_set,
    check_optional_positive_chunk_size,
    get_leaf_tensors,
)
from torchjd.aggregation import Aggregator


def vae_backward(
    losses: Sequence[Tensor],
    encoder_params: Iterable[Tensor],
    decoder_params: Iterable[Tensor],
    aggregator: Aggregator,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    check_optional_positive_chunk_size(parallel_chunk_size)

    losses = as_checked_ordered_set(losses, "losses")
    encoder_params = OrderedSet(encoder_params)
    decoder_params = OrderedSet(decoder_params)

    if len(losses) == 0:
        raise ValueError("`losses` cannot be empty")

    if inputs is None:
        inputs = get_leaf_tensors(tensors=losses, excluded=set())
    else:
        inputs = OrderedSet(inputs)

    transform = _create_transform(
        losses=losses,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        aggregator=aggregator,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    transform.check_keys(set())

    transform({})


def _create_transform(
    losses: OrderedSet[Tensor],
    encoder_params: OrderedSet[Tensor],
    decoder_params: OrderedSet[Tensor],
    aggregator: Aggregator,
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> Transform:
    """Creates the Encoder-Decoder transform."""

    to_differentiate = encoder_params + decoder_params

    # Transform that creates gradient outputs containing only ones.
    init = Init(losses)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(losses)

    # Transform that computes the required Jacobians.
    jac = Jac(losses, to_differentiate, parallel_chunk_size, retain_graph)

    encoder_accumulate = Aggregate(aggregator, encoder_params) << Select(encoder_params)
    decoder_accumulate = Aggregate(aggregator, decoder_params) << Select(decoder_params)

    return Accumulate() << (encoder_accumulate | decoder_accumulate) << jac << diag << init
