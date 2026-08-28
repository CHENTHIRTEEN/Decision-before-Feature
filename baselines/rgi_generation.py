"""SeedSequence-driven random expression-tree instances for the AS-LGBM baseline.

The public AS-LGBM repository calls these functions RGI (randomly generated
instances).  Each instance is a full expression tree.  Internal nodes are
drawn from the repository's weighted unary/binary operator list and leaves
encode either a variable or a constant.  The representation is kept as RPN
so that it can be written to a table without using executable source code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from optimizers.seeding import make_indexed_rng


RGI_GENERATION_STREAM = 5101
RGI_OPERATOR_CODES = tuple(range(10, 25))
RGI_UNARY_CODES = tuple(range(10, 21))
RGI_BINARY_CODES = tuple(range(21, 25))
RGI_OPERATOR_WEIGHTS = np.asarray(
    [2, 2, 2, 4, 4, 2, 2, 3, 3, 3, 3, 40, 40, 20, 20],
    dtype=float,
)

# The numerical codes and weights follow the public repository.  The names
# are used only for readable diagnostics; the RPN payload stores numerical
# codes and terminal encodings.
RGI_OPERATOR_NAMES = {
    10: "sin",
    11: "cos",
    12: "tan",
    13: "exp",
    14: "log_abs",
    15: "sqrt_abs",
    16: "abs",
    17: "square",
    18: "cube",
    19: "negate",
    20: "reciprocal",
    21: "add",
    22: "subtract",
    23: "multiply",
    24: "divide",
}
RGI_OPERATOR_ARITY = {
    code: 1 if code in RGI_UNARY_CODES else 2 for code in RGI_OPERATOR_CODES
}

_NUMERICAL_LIMIT = 1.0e100
_DENOMINATOR_FLOOR = 1.0e-12


@dataclass
class RGINode:
    """A node in an RGI expression tree."""

    code: float | int
    children: tuple["RGINode", ...] = ()

    @property
    def is_operator(self) -> bool:
        return isinstance(self.code, (int, np.integer))

    def to_rpn(self) -> list[float | int]:
        values: list[float | int] = []
        for child in self.children:
            values.extend(child.to_rpn())
        values.append(self.code)
        return values


@dataclass
class RGIInstance:
    """Serializable metadata and expression for one generated problem."""

    instance_id: int
    instance_seed: int
    tree_depth: int
    dimension: int
    lower_bound: float
    upper_bound: float
    rpn: tuple[float | int, ...]

    @property
    def expression(self) -> str:
        return serialize_rpn(self.rpn)

    @property
    def terminal_count(self) -> int:
        return sum(
            1
            for token in self.rpn
            if not _is_operator_token(token)
        )

    @property
    def operator_count(self) -> int:
        return len(self.rpn) - self.terminal_count

    @property
    def unary_operator_count(self) -> int:
        return sum(
            1
            for token in self.rpn
            if _is_operator_token(token) and int(token) in RGI_UNARY_CODES
        )

    @property
    def binary_operator_count(self) -> int:
        return sum(
            1
            for token in self.rpn
            if _is_operator_token(token) and int(token) in RGI_BINARY_CODES
        )

    def to_record(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "instance_seed": self.instance_seed,
            "tree_depth": self.tree_depth,
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "rpn": self.expression,
            "terminal_count": self.terminal_count,
            "operator_count": self.operator_count,
            "unary_operator_count": self.unary_operator_count,
            "binary_operator_count": self.binary_operator_count,
        }


@dataclass
class RGIObjective:
    """Decoded vectorized objective for one RGI instance."""

    rpn: tuple[float | int, ...]
    dimension: int
    lower_bound: float = -10.0
    upper_bound: float = 10.0

    def __call__(self, x: np.ndarray | Sequence[float]) -> float | np.ndarray:
        return evaluate_rpn(self.rpn, x, dimension=self.dimension)


def _is_operator_token(token: object) -> bool:
    if isinstance(token, (int, np.integer)):
        return int(token) in RGI_OPERATOR_ARITY
    if isinstance(token, (float, np.floating)):
        value = float(token)
        return value.is_integer() and int(value) in RGI_OPERATOR_ARITY
    return False


def _random_terminal(rng: np.random.Generator) -> float:
    """Encode a variable in [0, 1) or a constant in [1, 2)."""
    return float(rng.uniform(0.0, 2.0))


def generate_expression_tree(
    rng: np.random.Generator,
    *,
    depth: int,
) -> RGINode:
    """Generate one full expression tree at exactly ``depth``."""
    if depth < 0:
        raise ValueError("tree depth must be non-negative")
    if depth == 0:
        return RGINode(code=_random_terminal(rng))

    probabilities = RGI_OPERATOR_WEIGHTS / np.sum(RGI_OPERATOR_WEIGHTS)
    code = int(rng.choice(RGI_OPERATOR_CODES, p=probabilities))
    arity = RGI_OPERATOR_ARITY[code]
    children = tuple(
        generate_expression_tree(rng, depth=depth - 1)
        for _ in range(arity)
    )
    return RGINode(code=code, children=children)


def serialize_rpn(rpn: Iterable[float | int]) -> str:
    """Serialize RPN as JSON rather than executable Python text."""
    return json.dumps(list(rpn), ensure_ascii=False, separators=(",", ":"))


def parse_rpn(value: str | Sequence[float | int]) -> tuple[float | int, ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("RGI RPN must be a JSON list") from exc
    else:
        parsed = list(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("RGI RPN must be a non-empty list")

    normalized: list[float | int] = []
    for token in parsed:
        if isinstance(token, bool) or not isinstance(token, (int, float)):
            raise ValueError(f"RGI RPN contains a non-numeric token: {token!r}")
        if _is_operator_token(token):
            normalized.append(int(token))
        else:
            numeric = float(token)
            if not 0.0 <= numeric < 2.0:
                raise ValueError(
                    "RGI terminal codes must be in [0, 2), "
                    f"got {numeric}"
                )
            normalized.append(numeric)

    stack_depth = 0
    for token in normalized:
        if not _is_operator_token(token):
            stack_depth += 1
            continue
        arity = RGI_OPERATOR_ARITY[int(token)]
        if stack_depth < arity:
            raise ValueError("RGI RPN has insufficient operands")
        stack_depth -= arity - 1
    if stack_depth != 1:
        raise ValueError("RGI RPN does not decode to one expression")
    return tuple(normalized)


def generate_rgi_instance(
    *,
    instance_id: int,
    root_seed: int,
    dimension: int = 10,
    min_depth: int = 5,
    max_depth: int = 8,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
) -> RGIInstance:
    """Generate one deterministic RGI instance from explicit integer indices."""
    if instance_id < 1:
        raise ValueError("instance_id must be at least 1")
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("depth range is invalid")
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("bounds must be finite")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")

    # Event 1 is the instance identifier stream; event 2 is the tree stream.
    # The fields are deliberately explicit so changing one experimental index
    # cannot silently reuse a different random sequence.
    seed_rng = make_indexed_rng(
        seed=root_seed,
        unit_number=instance_id,
        stream_code=RGI_GENERATION_STREAM,
        generation=0,
        target=dimension,
        event=1,
    )
    tree_rng = make_indexed_rng(
        seed=root_seed,
        unit_number=instance_id,
        stream_code=RGI_GENERATION_STREAM,
        generation=0,
        target=dimension,
        event=2,
    )
    tree_depth = int(tree_rng.integers(min_depth, max_depth + 1))
    tree = generate_expression_tree(tree_rng, depth=tree_depth)
    return RGIInstance(
        instance_id=int(instance_id),
        instance_seed=int(seed_rng.integers(0, 2**32, dtype=np.uint32)),
        tree_depth=tree_depth,
        dimension=int(dimension),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        rpn=tuple(tree.to_rpn()),
    )


def generate_rgi_instances(
    *,
    count: int,
    root_seed: int,
    start_instance_id: int = 1,
    dimension: int = 10,
    min_depth: int = 5,
    max_depth: int = 8,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
) -> Iterable[RGIInstance]:
    """Yield a batch of independently keyed RGI instances."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if start_instance_id < 1:
        raise ValueError("start_instance_id must be at least 1")
    for instance_id in range(start_instance_id, start_instance_id + count):
        yield generate_rgi_instance(
            instance_id=instance_id,
            root_seed=root_seed,
            dimension=dimension,
            min_depth=min_depth,
            max_depth=max_depth,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


def decode_rgi(
    rpn: str | Sequence[float | int],
    *,
    dimension: int,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
) -> RGIObjective:
    return RGIObjective(
        rpn=parse_rpn(rpn),
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _safe_denominator(value: np.ndarray) -> np.ndarray:
    sign = np.where(value < 0.0, -1.0, 1.0)
    return np.where(np.abs(value) < _DENOMINATOR_FLOOR, sign * _DENOMINATOR_FLOOR, value)


def _sanitize(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        value,
        nan=0.0,
        posinf=_NUMERICAL_LIMIT,
        neginf=-_NUMERICAL_LIMIT,
    )


def _apply_unary(code: int, value: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if code == 10:
            result = np.sin(value)
        elif code == 11:
            result = np.cos(value)
        elif code == 12:
            result = np.tan(value)
        elif code == 13:
            result = np.exp(np.clip(value, -50.0, 50.0))
        elif code == 14:
            result = np.log(np.abs(value) + _DENOMINATOR_FLOOR)
        elif code == 15:
            result = np.sqrt(np.abs(value))
        elif code == 16:
            result = np.abs(value)
        elif code == 17:
            result = np.square(value)
        elif code == 18:
            result = np.power(value, 3.0)
        elif code == 19:
            result = -value
        elif code == 20:
            result = 1.0 / _safe_denominator(value)
        else:
            raise ValueError(f"unsupported unary RGI operator code: {code}")
    return _sanitize(result)


def _apply_binary(code: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if code == 21:
            result = left + right
        elif code == 22:
            result = left - right
        elif code == 23:
            result = left * right
        elif code == 24:
            result = left / _safe_denominator(right)
        else:
            raise ValueError(f"unsupported binary RGI operator code: {code}")
    return _sanitize(result)


def evaluate_rpn(
    rpn: str | Sequence[float | int],
    x: np.ndarray | Sequence[float],
    *,
    dimension: int,
) -> float | np.ndarray:
    """Evaluate an RPN expression on one point or an ``(n, dimension)`` array."""
    # ``decode_rgi`` stores an already validated tuple.  Avoid reparsing it
    # on every objective evaluation, which matters for a large FE budget.
    tokens = parse_rpn(rpn) if isinstance(rpn, str) else tuple(rpn)
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    values = np.asarray(x, dtype=float)
    scalar_input = values.ndim == 1
    if scalar_input:
        if values.shape[0] != dimension:
            raise ValueError(f"expected a point of dimension {dimension}, got {values.shape}")
        values = values.reshape(1, dimension)
    elif values.ndim == 2:
        if values.shape[1] != dimension:
            raise ValueError(
                f"expected an array with {dimension} columns, got {values.shape}"
            )
    else:
        raise ValueError("x must have shape (dimension,) or (n, dimension)")

    stack: list[np.ndarray] = []
    for token in tokens:
        if not _is_operator_token(token):
            numeric = float(token)
            if numeric < 1.0:
                index = min(int(numeric * dimension), dimension - 1)
                stack.append(values[:, index])
            else:
                stack.append(np.full(len(values), 10.0 * (numeric - 1.0)))
            continue

        code = int(token)
        arity = RGI_OPERATOR_ARITY[code]
        if len(stack) < arity:
            raise ValueError("RGI RPN has insufficient operands")
        if arity == 1:
            stack.append(_apply_unary(code, stack.pop()))
        else:
            right = stack.pop()
            left = stack.pop()
            stack.append(_apply_binary(code, left, right))

    if len(stack) != 1:
        raise ValueError("RGI RPN does not decode to one output")
    output = _sanitize(stack[0])
    return float(output[0]) if scalar_input else output


__all__ = [
    "RGI_BINARY_CODES",
    "RGIInstance",
    "RGIObjective",
    "RGINode",
    "RGI_OPERATOR_ARITY",
    "RGI_OPERATOR_NAMES",
    "RGI_OPERATOR_WEIGHTS",
    "RGI_UNARY_CODES",
    "decode_rgi",
    "evaluate_rpn",
    "generate_expression_tree",
    "generate_rgi_instance",
    "generate_rgi_instances",
    "parse_rpn",
    "serialize_rpn",
]
