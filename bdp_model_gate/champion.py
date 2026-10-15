"""The model already in production, as an input to the gate.

A gate run answers *is this model good enough?* A replacement raises a second
question that governance usually cares about more: **is it better than the
model it replaces, and what changes for the people it scores?** A challenger
can clear every absolute threshold and still be worse than the champion.

`Champion` is how the incumbent reaches the gate. It is scored on the same
rows as the challenger, by the comparison checks in
`bdp_model_gate.structured.comparison_checks`.

Four ways to supply one, most to least convenient:

    Champion.from_file("v3.joblib")              a file; format chosen by extension
    Champion.from_loader("pkg.module:factory")   anything a file cannot describe
    Champion(model=estimator)                    an object already in memory
    Champion(predict_fn=fn)                      the callables ModelAdapter takes
    Champion(predictions=y_old)                  scored elsewhere, same rows as X

## What `from_file` will and will not load

joblib and pickle files, `.skops`, `.onnx` and `.keras` / `.h5` are loaded
directly, each framework imported only when its file is asked for. A `.pt`
file and a TensorFlow SavedModel are **refused** with a pointer to
`from_loader`: a `.pt` is either a `state_dict`, which cannot be loaded
without the class that defines the network, or TorchScript, whose tensor
dtype, device and batching are the caller's business. `ModelAdapter` already
draws that line — this library never converts a DataFrame into a tensor,
because a wrong guess produces plausible numbers rather than an error.

## An old pickle may not load cleanly, and a warning is not good enough

The champion was serialised under last year's scikit-learn; the challenger's
environment has this year's. Unpickling across versions either fails, or
**warns and loads a subtly different model**. A comparison against the wrong
champion is worse than none, so a version warning raised while loading is an
error here. The message names the way out: score the champion in its own
environment and pass `Champion(predictions=...)`, which is exactly as valid
for every comparison check except the operational one.

## Loading a pickle runs code

The champion comes from a registry the team controls, so that is acceptable,
but it should be a visible and checkable choice. `from_file` hashes the file
**before** loading it, and `sha256=` pins it: a mismatch refuses to unpickle
anything, rather than running unknown bytes and reporting the problem after.
The hash, size and loader are recorded in the report, so "the model we
compared against is the one in production" is something the record proves.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .exceptions import GateConfigurationError
from .model import ModelAdapter

logger = get_logger("champion")

#: Extensions `from_file` loads with joblib. `.pkl` and `.pickle` fall back to
#: the standard library when joblib is not installed; `.joblib` needs joblib.
JOBLIB_SUFFIXES = (".joblib", ".pkl", ".pickle")
SKOPS_SUFFIXES = (".skops",)
ONNX_SUFFIXES = (".onnx",)
KERAS_SUFFIXES = (".keras", ".h5")
#: Recognised, and refused with an explanation rather than a guess.
TORCH_SUFFIXES = (".pt", ".pth")

_SUPPORTED = JOBLIB_SUFFIXES + SKOPS_SUFFIXES + ONNX_SUFFIXES + KERAS_SUFFIXES

#: A warning raised while deserialising whose class or text matches this is a
#: version mismatch between the environment that saved the model and this one.
#: scikit-learn raises `InconsistentVersionWarning`; XGBoost and LightGBM say
#: "version" in prose. Matched by name so scikit-learn need not be imported.
_VERSION_WARNING = re.compile(r"version", re.IGNORECASE)

_PREDICTIONS_FALLBACK = (
    "Score the champion in the environment it was trained in and pass the result "
    "as Champion(predictions=...) — or --champion-col on the command line. Scores "
    "computed on the same rows are exactly as valid for every comparison check "
    "except the operational one."
)


def sha256_of(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks so a large model is not held twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Provenance:
    """Where the champion came from — recorded in the report, verbatim.

    A comparison record read a year later has to say *which* incumbent the
    challenger was measured against. A name is not enough; a hash is.
    """

    source: str
    loader: str | None = None
    path: str | None = None
    sha256: str | None = None
    sha256_pinned: bool = False
    size_bytes: int | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value not in (None, {})}

    def describe(self) -> str:
        if self.path:
            digest = f", sha256 {self.sha256[:12]}…" if self.sha256 else ""
            pinned = ", pinned" if self.sha256_pinned else ""
            return f"{Path(self.path).name} ({self.loader}{digest}{pinned})"
        return self.source


class Champion:
    """The incumbent model, scored on the same rows as the challenger.

    Args:
        model: Anything `ModelAdapter` can call — `.predict()`, a callable.
        predict_fn: `fn(DataFrame) -> array`, as on `StructuredGateContext`.
        predict_proba_fn: `fn(DataFrame) -> array` of class probabilities.
        predictions: The champion's predictions **for the rows of
            `context.X`, in the same form as `context.y_pred`**: positive-class
            probabilities if the challenger's are, labels if they are labels,
            an `(n, n_classes)` matrix for a multiclass probability matrix.
            Use this when the champion cannot be loaded in the challenger's
            environment.
        features: The columns the champion was fitted on. Usually read from
            `feature_names_in_`; supply it when the model does not record
            them. A challenger usually adds features, so `context.X` carries
            the union and each model receives its own.
        decision_threshold: The cut-off the champion is *operated* at, if it
            differs from the challenger's. Churn and label metrics compare
            each model at its real threshold rather than a shared 0.5 that
            neither used.
        latencies_ms: The champion's benchmark latencies, for the
            operational comparison.
        cost_per_inference: The champion's cost per inference.
        name: How the champion is named in the report.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        predict_fn: Callable[[pd.DataFrame], Any] | None = None,
        predict_proba_fn: Callable[[pd.DataFrame], Any] | None = None,
        predictions: Any = None,
        features: list[str] | None = None,
        decision_threshold: float | None = None,
        latencies_ms: list[float] | None = None,
        cost_per_inference: float | None = None,
        name: str = "champion",
        provenance: Provenance | None = None,
    ):
        sources = [s for s in (model, predict_fn, predictions) if s is not None]
        if not sources:
            raise GateConfigurationError(
                "a Champion needs a model, a predict_fn, or predictions — or use "
                "Champion.from_file(path) / Champion.from_loader('pkg.module:factory')"
            )
        if predictions is not None and (model is not None or predict_fn is not None):
            raise GateConfigurationError(
                "pass either predictions or a model/predict_fn to Champion, not both — "
                "with both, it is ambiguous which one the comparison should trust"
            )
        if decision_threshold is not None and not 0.0 <= float(decision_threshold) <= 1.0:
            raise GateConfigurationError(
                f"Champion decision_threshold={decision_threshold!r} is not a probability"
            )

        self.model = model
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.predictions = None if predictions is None else np.asarray(predictions)
        self.features = list(features) if features is not None else None
        self.decision_threshold = decision_threshold
        self.latencies_ms = latencies_ms
        self.cost_per_inference = cost_per_inference
        self.name = name
        self.provenance = provenance or Provenance(
            source="predictions" if predictions is not None else _describe_object(model, predict_fn)
        )
        self._cache: tuple[int, Any] | None = None

    # ------------------------------------------------------------------ build

    @classmethod
    def from_file(cls, path: str | Path, *, sha256: str | None = None, **kwargs: Any) -> Champion:
        """Loads a champion from a file, choosing the loader by extension.

        `sha256` pins the artifact. It is checked **before** anything is
        deserialised, so a mismatched pickle is never executed.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in TORCH_SUFFIXES or path.is_dir():
            raise GateConfigurationError(
                f"{path.name}: PyTorch checkpoints and SavedModel directories are not loaded "
                "directly. A .pt file is either a state_dict, which needs the class that "
                "defines the network, or TorchScript, whose tensor dtype and device are "
                "yours to choose — a guess here would produce plausible numbers rather than "
                "an error. Write a three-line factory that loads it and returns a "
                "fn(DataFrame) -> array, and pass Champion.from_loader('pkg.module:factory')."
            )
        if suffix not in _SUPPORTED:
            raise GateConfigurationError(
                f"{path.name}: no loader for {suffix or 'a file with no extension'} — "
                f"from_file reads {', '.join(_SUPPORTED)}. For anything else, use "
                "Champion.from_loader('pkg.module:factory')."
            )
        if not path.is_file():
            raise GateConfigurationError(f"champion file not found: {path}")

        digest = sha256_of(path)
        if sha256 is not None and digest != sha256.strip().lower():
            raise GateConfigurationError(
                f"{path.name} does not match its pinned sha256: expected {sha256.strip()}, "
                f"found {digest}. Nothing was loaded — a pinned artifact that has changed is "
                "not the champion that was approved, and unpickling unknown bytes runs code."
            )

        model, loader, predict_fn, predict_proba_fn, versions = _load(path, suffix)
        provenance = Provenance(
            source="file",
            loader=loader,
            path=str(path),
            sha256=digest,
            sha256_pinned=sha256 is not None,
            size_bytes=path.stat().st_size,
            versions=versions,
        )
        logger.info("loaded champion %s", provenance.describe())
        return cls(
            model,
            predict_fn=predict_fn,
            predict_proba_fn=predict_proba_fn,
            provenance=provenance,
            **kwargs,
        )

    @classmethod
    def from_loader(cls, spec: str, **kwargs: Any) -> Champion:
        """Calls a `"package.module:factory"` that returns a model or a callable.

        The route for PyTorch, a SavedModel, a model registry, or anything a
        file extension cannot describe. Your factory does the framework import,
        so this package needs none.
        """
        module_name, _, attr = spec.partition(":")
        if not module_name or not attr:
            raise GateConfigurationError(
                f"champion loader {spec!r} must look like 'package.module:factory'"
            )
        try:
            factory = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise GateConfigurationError(
                f"could not import champion loader {spec!r}: {exc}"
            ) from exc
        loaded = factory()
        if not (callable(loaded) or hasattr(loaded, "predict")):
            raise GateConfigurationError(
                f"champion loader {spec!r} returned a {type(loaded).__name__}, which is "
                "neither callable nor has .predict()"
            )
        provenance = Provenance(source="loader", loader=spec)
        return cls(loaded, provenance=provenance, **kwargs)

    # -------------------------------------------------------------- inspection

    @property
    def predictions_only(self) -> bool:
        return self.predictions is not None

    @property
    def adapter(self) -> ModelAdapter:
        return ModelAdapter(
            model=self.model, predict_fn=self.predict_fn, predict_proba_fn=self.predict_proba_fn
        )

    def describe(self) -> dict[str, Any]:
        """The champion as recorded in the report."""
        described = {"name": self.name, **self.provenance.as_metadata()}
        if self.decision_threshold is not None:
            described["decision_threshold"] = self.decision_threshold
        if self.features is not None:
            described["features"] = list(self.features)
        return described

    def __repr__(self) -> str:
        return f"Champion({self.name!r}, {self.provenance.describe()})"

    # ------------------------------------------------------------------ scoring

    def frame_for(self, X: pd.DataFrame) -> pd.DataFrame:
        """The columns of `X` this champion was fitted on, in its order."""
        wanted = self.features
        if wanted is None:
            names = getattr(self.model, "feature_names_in_", None)
            wanted = [str(n) for n in names] if names is not None else None
        if wanted is None:
            return X
        missing = [c for c in wanted if c not in X.columns]
        if missing:
            raise GateConfigurationError(
                f"the champion expects column(s) {missing} that context.X does not have. "
                "X must carry the union of both models' features — each model receives its "
                "own columns."
            )
        return X[wanted]


def _describe_object(model: Any, predict_fn: Any) -> str:
    if predict_fn is not None:
        return "predict_fn"
    return f"model ({type(model).__name__})"


def _load(path: Path, suffix: str):
    """`(model, loader name, predict_fn, predict_proba_fn, versions)`."""
    if suffix in JOBLIB_SUFFIXES:
        model, loader = _load_pickled(path, suffix)
        return model, loader, None, None, _versions_of(model)
    if suffix in SKOPS_SUFFIXES:
        model = _load_skops(path)
        return model, "skops", None, None, _versions_of(model)
    if suffix in ONNX_SUFFIXES:
        predict_fn, predict_proba_fn, versions = _load_onnx(path)
        return None, "onnxruntime", predict_fn, predict_proba_fn, versions
    model = _load_keras(path)
    return model, "keras", None, None, _versions_of(model)


def _strict(load: Callable[[], Any], path: Path) -> Any:
    """Runs `load`, turning a version warning into an error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load()
    for warning in caught:
        text = f"{type(warning.message).__name__}: {warning.message}"
        if _VERSION_WARNING.search(type(warning.message).__name__) or _VERSION_WARNING.search(
            str(warning.message)
        ):
            raise GateConfigurationError(
                f"{path.name} loaded with a version warning — {text}. A model unpickled "
                "under a different library version may load and behave differently, and a "
                f"comparison against the wrong champion is worse than none. {_PREDICTIONS_FALLBACK}"
            )
        warnings.warn_explicit(
            warning.message, warning.category, warning.filename, warning.lineno or 0
        )
    return model


def _load_pickled(path: Path, suffix: str) -> tuple[Any, str]:
    try:
        import joblib
    except ImportError:
        if suffix == ".joblib":
            raise GateConfigurationError(
                f"{path.name} is a joblib file and joblib is not installed — "
                "pip install joblib (it ships with the [structured] extra)"
            ) from None
        joblib = None

    def load() -> Any:
        if joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)  # documented above: loading a pickle runs code

    try:
        return _strict(load, path), "joblib" if joblib is not None else "pickle"
    except GateConfigurationError:
        raise
    except Exception as exc:
        raise GateConfigurationError(
            f"{path.name} could not be unpickled in this environment ({type(exc).__name__}: "
            f"{exc}). {_PREDICTIONS_FALLBACK}"
        ) from exc


def _load_skops(path: Path) -> Any:
    try:
        sio = importlib.import_module("skops.io")
    except ImportError:
        raise GateConfigurationError(
            f"{path.name} is a skops file and skops is not installed — pip install skops"
        ) from None
    untrusted = list(sio.get_untrusted_types(file=str(path)))
    if untrusted:
        # skops' safety guarantee is that it refuses types nobody vouched for.
        # Passing `trusted=` here on the user's behalf would throw that away.
        raise GateConfigurationError(
            f"{path.name} contains types skops does not trust by default: {untrusted}. "
            "Inspect them, then load the file yourself with "
            "skops.io.load(path, trusted=[...]) and pass Champion(model=...)."
        )
    return _strict(lambda: sio.load(str(path)), path)


def _load_keras(path: Path) -> Any:
    for module_name in ("keras", "tensorflow.keras"):
        try:
            keras = importlib.import_module(module_name)
        except ImportError:
            continue

        def load(keras: Any = keras) -> Any:
            return keras.models.load_model(str(path))

        return _strict(load, path)
    raise GateConfigurationError(
        f"{path.name} is a Keras model and neither keras nor tensorflow is installed"
    )


#: ONNX tensor element types, mapped to numpy. The model declares its own input
#: type, so this is a lookup, not a guess.
_ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(string)": np.object_,
    "tensor(bool)": np.bool_,
}


def _load_onnx(path: Path):
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        raise GateConfigurationError(
            f"{path.name} is an ONNX model and onnxruntime is not installed — "
            "pip install onnxruntime"
        ) from None
    session = _strict(lambda: ort.InferenceSession(str(path)), path)
    inputs = session.get_inputs()
    outputs = [o.name for o in session.get_outputs()]

    def feed(X: pd.DataFrame) -> dict[str, np.ndarray]:
        if len(inputs) == 1:
            spec = inputs[0]
            dtype = _ONNX_DTYPES.get(spec.type)
            if dtype is None:
                raise GateConfigurationError(
                    f"{path.name} declares input type {spec.type}, which this loader does "
                    "not map — use Champion.from_loader with your own feed"
                )
            return {spec.name: X.to_numpy(dtype=dtype)}
        # One input per column, the layout skl2onnx emits for a mixed frame.
        missing = [spec.name for spec in inputs if spec.name not in X.columns]
        if missing:
            raise GateConfigurationError(
                f"{path.name} expects inputs {missing} that context.X does not have"
            )
        return {
            spec.name: X[[spec.name]].to_numpy(dtype=_ONNX_DTYPES.get(spec.type, np.float32))
            for spec in inputs
        }

    def run(X: pd.DataFrame) -> list[Any]:
        return session.run(outputs, feed(X))

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        return np.asarray(run(X)[0])

    predict_proba_fn = None
    if len(outputs) > 1:

        def predict_proba_fn(X: pd.DataFrame) -> np.ndarray:
            return _onnx_probabilities(run(X)[1])

    versions = {"onnxruntime": str(getattr(ort, "__version__", "unknown"))}
    return predict_fn, predict_proba_fn, versions


def _onnx_probabilities(raw: Any) -> np.ndarray:
    """skl2onnx emits probabilities either as an array or, through its ZipMap
    operator, as one `{class: probability}` dict per row. Both become an
    `(n, n_classes)` matrix with columns in sorted class order — the order
    scikit-learn's own `predict_proba` uses."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        classes = sorted(raw[0])
        return np.array([[row[c] for c in classes] for row in raw], dtype=float)
    return np.asarray(raw, dtype=float)


def _versions_of(model: Any) -> dict[str, str]:
    """Framework versions an artifact records about itself, where it does.

    scikit-learn stamps `_sklearn_version` into every pickled estimator — the
    version that *saved* it, which is the one worth recording.
    """
    versions: dict[str, str] = {}
    saved = getattr(model, "_sklearn_version", None)
    if saved:
        versions["scikit-learn (saved with)"] = str(saved)
    return versions


__all__ = ["Champion", "Provenance", "sha256_of"]
