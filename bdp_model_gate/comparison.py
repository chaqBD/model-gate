"""Champion against challenger: the statistics behind the comparison checks.

## Both models, the same rows, one interval on the difference

The champion's score from last year's model card is not a baseline: it was
measured on different data, and comparing across datasets measures the data.
So the champion is re-scored on *this* validation set, and every bootstrap
resample scores **both** models on the **same** rows. The statistic is the
difference itself, `Δ = metric(challenger) − metric(champion)`.

The tempting alternative — compute an interval for each model and check
whether they overlap — is wrong, and badly. Two models scored on the same rows
make correlated errors, so their difference is far more precise than either
score alone; overlapping intervals call a real improvement `UNCERTAIN` almost
every time. `tests/test_comparison.py` measures that rather than asserting it.

Row-order invariance comes free: resampling runs over `canonical_order`, so
sorting the validation set cannot move a verdict here any more than anywhere
else.

## Δ is signed so that positive always means "challenger better"

`MetricSpec.greater_is_better` already records each metric's direction, so an
RMSE that fell and an AUC that rose both read as a positive Δ. One rule then
serves every metric, and a reader never has to remember which way round an
error metric goes.

## Non-inferiority, read three ways like every other interval

See `ComparisonConfig` for the table. The default asks only that the
challenger is not *materially worse* — retraining on fresh data, a cheaper
model or a simpler one are legitimate replacements that are not measurably
better. `require="superior"` is for change processes that say a replacement
must earn its risk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .actuarial import exposure_array
from .calibration import brier_score, expected_calibration_error
from .classes import favourable_mask, resolve_favourable
from .exceptions import GateConfigurationError
from .metrics import (
    AUTO,
    BUILTIN_METRICS,
    MetricSetting,
    resolve_metric,
    to_class_labels,
    to_hard_labels,
    validate_metric,
)
from .task import BINARY, MULTICLASS, REGRESSION, resolve_task
from .uncertainty import BLOCK, POINT, UNCERTAIN_FLAG, Interval, bootstrap

logger = get_logger("comparison")

NON_INFERIOR = "non_inferior"
SUPERIOR = "superior"
REQUIREMENTS = (NON_INFERIOR, SUPERIOR)

#: The challenger is materially worse than the champion. Blocking.
WORSE_FLAG = "CHALLENGER_WORSE"
#: `require="superior"` and the challenger is not measurably better. Blocking.
NOT_BETTER_FLAG = "CHALLENGER_NOT_BETTER"

#: Outcomes, recorded in metadata. Only two of them are flags: an improvement
#: is not a finding, and reporting one as a non-OK flag would route every
#: better model to a human.
IMPROVED = "improved"
NON_INFERIOR_OUTCOME = "non_inferior"
NOT_IMPROVED = "not_improved"
WORSE = "worse"
UNDECIDED = "uncertain"

#: Calibration measures usable as guardrails beside the built-in metrics.
#: Both are lower-is-better and need positive-class probabilities, so they
#: apply to binary classification only.
CALIBRATION_GUARDRAILS = ("brier", "ece")

#: Integer and float predictions with at most this many distinct values in a
#: regression segment column are treated as categories; more are cut into
#: quintiles.
MAX_SEGMENT_LEVELS = 10


def validate_comparison_config(config: Any) -> None:
    """Rejects an unusable `ComparisonConfig` while the suite is being built."""
    if config.require not in REQUIREMENTS:
        raise GateConfigurationError(
            f"comparison.require={config.require!r} — must be one of {', '.join(REQUIREMENTS)}"
        )
    if config.metric is not None:
        validate_metric(config.metric)
    if config.margin is not None and float(config.margin) < 0:
        raise GateConfigurationError("comparison.margin must be zero or positive")
    if float(config.margin_relative) < 0:
        raise GateConfigurationError("comparison.margin_relative must be zero or positive")
    if float(config.min_improvement) < 0:
        raise GateConfigurationError("comparison.min_improvement must be zero or positive")
    if isinstance(config.guardrails, dict):
        for name, margin in config.guardrails.items():
            if margin is None or float(margin) < 0:
                raise GateConfigurationError(
                    f"comparison.guardrails[{name!r}] must be a margin of zero or more"
                )
    for name in config.guardrails:
        if name not in CALIBRATION_GUARDRAILS and name not in BUILTIN_METRICS:
            valid = ", ".join([*CALIBRATION_GUARDRAILS, *sorted(BUILTIN_METRICS)])
            raise GateConfigurationError(
                f"comparison.guardrails names {name!r}, which is not a metric — valid: {valid}"
            )
    for ratio in ("max_latency_ratio", "max_cost_ratio"):
        if float(getattr(config, ratio)) <= 0:
            raise GateConfigurationError(f"comparison.{ratio} must be positive")
    if int(config.min_segment_size) < 2:
        raise GateConfigurationError("comparison.min_segment_size must be at least 2")


# --------------------------------------------------------------------------
# Scoring the champion on the challenger's rows
# --------------------------------------------------------------------------


def prediction_kind(values: Any, task: str) -> str:
    """`"values"`, `"matrix"`, `"probability"` or `"labels"`.

    Matters because a ranking metric scored on one model's probabilities and
    the other's hard labels compares two different things, and the difference
    looks like a model effect.
    """
    arr = np.asarray(values)
    if task == REGRESSION:
        return "values"
    if arr.ndim == 2:
        return "matrix"
    if arr.dtype.kind in "fc" and arr.size:
        finite = arr[np.isfinite(arr)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            if not np.all(np.isin(finite, (0.0, 1.0))):
                return "probability"
    return "labels"


@dataclass
class Compared:
    """Both models' predictions for the same rows, as the checks read them."""

    task: str
    y_true: np.ndarray
    challenger: np.ndarray
    champion: np.ndarray
    challenger_kind: str
    champion_kind: str
    challenger_threshold: float
    champion_threshold: float
    exposure: np.ndarray | None = None
    class_order: Any = None

    @property
    def n(self) -> int:
        return len(self.y_true)


def score_champion(champion: Any, X: pd.DataFrame, task: str, challenger_pred: Any) -> np.ndarray:
    """The champion's predictions for `X`, in the form the challenger's take.

    Probabilities when the challenger supplied probabilities and the champion
    can produce them, a probability matrix for a matrix, otherwise whatever
    `.predict()` returns. Supplied predictions are used as given.
    """
    if champion.predictions_only:
        return np.asarray(champion.predictions)

    adapter = champion.adapter
    frame = champion.frame_for(X)
    kind = prediction_kind(challenger_pred, task)
    if task == BINARY and kind == "probability" and adapter.can_predict_proba:
        return adapter.predict_positive_proba(frame)
    if task == MULTICLASS and kind == "matrix" and adapter.can_predict_proba:
        return adapter.predict_proba_matrix(frame)
    return adapter.predict(frame)


def compared(context: Any, decision_threshold: float) -> Compared:
    """Both models' predictions, computed once per run and cached on the champion.

    Raises `GateConfigurationError` when the champion cannot be scored — the
    integrity check turns that into a blocking finding, and every other
    comparison check reports NOT_APPLICABLE pointing at it.
    """
    champion = context.champion
    task = resolve_task(context)
    key = (id(context.X), id(context.y_pred), task, decision_threshold)
    cached = getattr(champion, "_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    try:
        champion_pred = np.asarray(score_champion(champion, context.X, task, context.y_pred))
    except GateConfigurationError:
        raise
    except Exception as exc:
        raise GateConfigurationError(
            f"the champion could not score context.X: {type(exc).__name__}: {exc}"
        ) from exc

    challenger_pred = np.asarray(context.y_pred)
    if len(champion_pred) != len(challenger_pred):
        raise GateConfigurationError(
            f"the champion produced {len(champion_pred)} prediction(s) for {len(challenger_pred)} "
            "row(s) — Champion(predictions=...) must be row-aligned to context.X"
        )
    if champion_pred.ndim != challenger_pred.ndim:
        raise GateConfigurationError(
            f"the champion's predictions have {champion_pred.ndim} dimension(s) and the "
            f"challenger's {challenger_pred.ndim} — one is a probability matrix and the other "
            "is not, so the two cannot be scored by the same metric"
        )

    result = Compared(
        task=task,
        y_true=np.asarray(context.y_true),
        challenger=challenger_pred,
        champion=champion_pred,
        challenger_kind=prediction_kind(challenger_pred, task),
        champion_kind=prediction_kind(champion_pred, task),
        challenger_threshold=decision_threshold,
        champion_threshold=(
            champion.decision_threshold
            if champion.decision_threshold is not None
            else decision_threshold
        ),
        exposure=exposure_array(context),
        class_order=getattr(context, "class_order", None),
    )
    champion._cache = (key, result)
    return result


# --------------------------------------------------------------------------
# Metrics, as the comparison calls them
# --------------------------------------------------------------------------


@dataclass
class ComparisonMetric:
    """One metric, ready to score either model on any subset of rows.

    Exposure is passed per call rather than bound in, because the bootstrap
    indexes rows: a weight vector bound to the full frame would be paired
    with the wrong rows on every resample.
    """

    name: str
    greater_is_better: bool
    needs_hard_labels: bool
    needs_probability: bool
    fn: Callable[..., float]
    weighted: bool = False
    is_fallback: bool = False

    @property
    def sign(self) -> float:
        return 1.0 if self.greater_is_better else -1.0

    def prepare(self, compared_: Compared, which: str) -> np.ndarray:
        """One model's predictions in the shape this metric wants."""
        raw = compared_.challenger if which == "challenger" else compared_.champion
        threshold = (
            compared_.challenger_threshold
            if which == "challenger"
            else compared_.champion_threshold
        )
        if self.needs_probability:
            return positive_probability(raw, compared_)
        if not self.needs_hard_labels:
            return raw
        if compared_.task == MULTICLASS:
            return np.asarray(to_class_labels(raw, compared_.class_order))
        return np.asarray(to_hard_labels(raw, threshold))

    def score(self, y_true: Any, predicted: Any, weights: Any = None) -> float:
        if self.weighted and weights is not None:
            return float(self.fn(y_true, predicted, sample_weight=weights))
        return float(self.fn(y_true, predicted))


def resolve_comparison_metric(
    setting: MetricSetting | str,
    task: str,
    *,
    average: str = "macro",
    class_order: Any = None,
    exposure: Any = None,
) -> ComparisonMetric:
    """A metric name or callable, resolved for the comparison."""
    if setting == "brier":
        return ComparisonMetric("brier", False, False, True, _binary_metric(brier_score))
    if setting == "ece":
        return ComparisonMetric(
            "ece", False, False, True, _binary_metric(expected_calibration_error)
        )

    resolved = resolve_metric(setting, task, average=average, class_order=class_order)
    weighted = False
    if exposure is not None and isinstance(setting, str):
        name = resolved.name if setting == AUTO else setting
        weighted = BUILTIN_METRICS[name].supports_weight
    return ComparisonMetric(
        name=resolved.name,
        greater_is_better=resolved.greater_is_better,
        needs_hard_labels=resolved.needs_hard_labels,
        needs_probability=False,
        fn=resolved.fn,
        weighted=weighted,
        is_fallback=resolved.is_fallback,
    )


def _binary_metric(fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[..., float]:
    def scored(y_true: Any, y_prob: Any) -> float:
        return float(fn(np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)))

    return scored


def positive_probability(raw: np.ndarray, compared_: Compared) -> np.ndarray:
    """Positive-class probabilities, or a `GateConfigurationError` saying why not."""
    if compared_.task != BINARY or prediction_kind(raw, BINARY) != "probability":
        raise GateConfigurationError(
            "a calibration guardrail needs positive-class probabilities from both models on a "
            "binary task"
        )
    return raw.astype(float)


def favourable_outcome(values: np.ndarray, compared_: Compared, threshold: float) -> np.ndarray:
    """0/1 "got the favourable outcome" per row, for parity and churn."""
    if compared_.task == MULTICLASS:
        favourable = resolve_favourable(None, compared_.class_order, MULTICLASS)
        if favourable is None:
            raise GateConfigurationError(
                "multiclass comparison of outcomes needs context.class_order or "
                "context.favourable_classes"
            )
        labels = to_class_labels(values, compared_.class_order)
        return favourable_mask(labels, favourable).astype(float)
    return np.asarray(to_hard_labels(values, threshold), dtype=float)


# --------------------------------------------------------------------------
# The paired interval, and reading it
# --------------------------------------------------------------------------


def resampling_frame(compared_: Compared, *extra: np.ndarray) -> pd.DataFrame:
    """The columns the canonical order is derived from.

    Every array the statistic reads, so two runs on the same rows in a
    different order resample identically. A probability matrix contributes
    one column per class.
    """
    columns: dict[str, Any] = {"y_true": compared_.y_true}
    for name, values in (("challenger", compared_.challenger), ("champion", compared_.champion)):
        if values.ndim == 2:
            for j in range(values.shape[1]):
                columns[f"{name}_{j}"] = values[:, j]
        else:
            columns[name] = values
    if compared_.exposure is not None:
        columns["exposure"] = compared_.exposure
    for i, values in enumerate(extra):
        columns[f"extra_{i}"] = values
    return pd.DataFrame(columns)


def paired_interval(
    statistic: Callable[[np.ndarray], float],
    frame: pd.DataFrame,
    uncertainty: Any,
    *,
    level: float | None = None,
) -> Interval | None:
    """A bootstrap interval for a paired statistic, honouring `UncertaintyConfig`.

    `level` overrides the configured confidence level — the guardrails and the
    segments widen theirs for multiplicity.
    """
    if not uncertainty.compute_intervals:
        return None
    return bootstrap(
        statistic,
        frame,
        samples=uncertainty.bootstrap_samples,
        level=uncertainty.confidence_level if level is None else level,
        random_state=uncertainty.random_state,
        min_rows=uncertainty.min_rows_for_interval,
    )


def bonferroni(level: float, comparisons: int) -> float:
    """The per-comparison level that keeps the family-wise level at `level`."""
    return 1.0 - (1.0 - level) / max(1, comparisons)


def delta_statistic(
    metric: ComparisonMetric, compared_: Compared
) -> tuple[Callable[[np.ndarray], float], float, float]:
    """`(statistic over positions, challenger score, champion score)`.

    The statistic is the signed difference on a set of row positions. One
    function serves the point estimate and every resample, so the interval
    cannot describe a different number from the one the verdict came from.
    """
    y = compared_.y_true
    challenger = metric.prepare(compared_, "challenger")
    champion = metric.prepare(compared_, "champion")
    weights = compared_.exposure if metric.weighted else None

    def score(predicted: np.ndarray, positions: np.ndarray) -> float:
        w = None if weights is None else weights[positions]
        return metric.score(y[positions], predicted[positions], w)

    def statistic(positions: np.ndarray) -> float:
        return metric.sign * (score(challenger, positions) - score(champion, positions))

    everything = np.arange(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return statistic, score(challenger, everything), score(champion, everything)


@dataclass(frozen=True)
class Decision:
    flag: str
    blocking: bool
    outcome: str
    note: str


def decide(
    interval: Interval | None,
    point: float,
    *,
    margin: float,
    min_improvement: float,
    require: str,
    on_uncertain: str,
    blocking: bool = True,
) -> Decision:
    """Reads Δ's interval against `−margin` and `min_improvement`.

    The passing side is inclusive, as everywhere else in this library: an
    interval whose lower end sits exactly on `−margin` is non-inferior. An
    *improvement* is a positive claim, so it needs the interval strictly above
    `min_improvement`.

    Without an interval — too few rows, or intervals switched off — the point
    estimate decides, and the note says so.
    """
    low, high = (interval.low, interval.high) if interval is not None else (point, point)
    risk = NOT_BETTER_FLAG if require == SUPERIOR else WORSE_FLAG
    basis = (
        f" — {interval.level * 100:g}% interval [{interval.low:+.4f}, {interval.high:+.4f}]"
        if interval is not None
        else " — no interval (too few rows, or intervals are off), so the point estimate decides"
    )

    if high < -margin:
        return Decision(WORSE_FLAG, blocking, WORSE, f"{basis} sits entirely below −{margin:.4f}")
    if low > min_improvement:
        return Decision(
            "OK", blocking, IMPROVED, f"{basis} sits entirely above {min_improvement:g}"
        )

    if require == SUPERIOR:
        if high <= min_improvement:
            return Decision(
                NOT_BETTER_FLAG,
                blocking,
                NOT_IMPROVED,
                f"{basis} never clears {min_improvement:g}, and comparison.require='superior'",
            )
        straddled = f"{basis} straddles {min_improvement:g}: improvement is not established"
    else:
        if low >= -margin:
            return Decision(
                "OK", blocking, NON_INFERIOR_OUTCOME, f"{basis} stays above −{margin:.4f}"
            )
        straddled = (
            f"{basis} reaches below −{margin:.4f}: this sample cannot rule out a "
            "material regression"
        )

    if on_uncertain == BLOCK:
        return Decision(risk, blocking, UNDECIDED, straddled + " (on_uncertain='block')")
    if on_uncertain == POINT:
        decided = decide(
            None,
            point,
            margin=margin,
            min_improvement=min_improvement,
            require=require,
            on_uncertain=on_uncertain,
            blocking=blocking,
        )
        return Decision(
            decided.flag,
            decided.blocking,
            decided.outcome,
            straddled
            + f" (accepted on the point estimate: on_uncertain='point' → {decided.outcome})",
        )
    return Decision(UNCERTAIN_FLAG, False, UNDECIDED, straddled)


def margin_for(config: Any, champion_score: float) -> tuple[float, bool]:
    """`(margin, relative?)` — the absolute `margin` if set, else relative."""
    if config.margin is not None:
        return float(config.margin), False
    return float(config.margin_relative) * abs(float(champion_score)), True


def segment_labels(values: pd.Series) -> pd.Series:
    """Groups for a segment column: its own levels, or quintiles if it has many."""
    if values.dtype.kind in "iuf" and values.nunique(dropna=True) > MAX_SEGMENT_LEVELS:
        ranked = values.rank(method="first")
        return pd.qcut(ranked, 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    if values.dtype.kind == "f" and np.all(np.mod(values.dropna(), 1) == 0):
        # A 0/1 indicator stored as float reads as "broker=1", not "broker=1.0".
        return values.map(lambda v: str(int(v)) if pd.notna(v) else v)
    return values.astype(str)


__all__ = [
    "CALIBRATION_GUARDRAILS",
    "Compared",
    "ComparisonMetric",
    "Decision",
    "IMPROVED",
    "NON_INFERIOR",
    "NOT_BETTER_FLAG",
    "REQUIREMENTS",
    "SUPERIOR",
    "WORSE_FLAG",
    "bonferroni",
    "compared",
    "decide",
    "delta_statistic",
    "favourable_outcome",
    "margin_for",
    "paired_interval",
    "prediction_kind",
    "resampling_frame",
    "resolve_comparison_metric",
    "score_champion",
    "segment_labels",
    "validate_comparison_config",
]
