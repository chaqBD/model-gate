# Replacing a model

The gate answers *is this model good enough?* A replacement raises a second
question, and governance usually cares about it more: **is it better than the
model already in production, and what changes for the people it scores?**

A challenger can clear every absolute threshold and still be worse than the
champion it replaces. Supply the champion and the gate measures that too.

```python
from bdp_model_gate import Champion, ModelGate, StructuredGateContext

champion = Champion.from_file("models/scorecard_v3.joblib", sha256="9f2c…")

context = StructuredGateContext(
    model=challenger,  # the new model, gated in full as before
    X=X_val,
    y_true=y_val,
    y_pred=challenger.predict_proba(X_val)[:, 1],
    protected_df=protected_val,
    champion=champion,  # adds the comparison checks
)

report = ModelGate().run(context)
print(report.summary())
```

```text
Gate status: NEEDS_REVIEW (3120ms, binary)
  roc_auc: 0.8396
  vs champion: roc_auc 0.8396 against 0.8063, Δ +0.0333 [+0.0182, +0.0477] -> improved
  validation: 0 flag(s)
  comparison: 2 flag(s)
  ...
```

Here the challenger is measurably better — and the two comparison findings,
both non-blocking, are a calibration guardrail and a fairness gap that might
have widened. Better on the headline metric is where the review starts.

Exit codes and `gate_status` do not change. The comparison adds results — it
does not add a new kind of verdict.

## The challenger still runs the full gate

The comparison is added on top and never replaces anything. A challenger that
beats a bad champion is still `BLOCKED` if it fails fairness, validation or
compliance on its own terms: *better than what we had* is not a pass. The
reverse holds too — a challenger that passes every absolute check but is
materially worse than the champion is `BLOCKED` by the comparison.

## Both models, the same rows

The champion's score in last year's model card is not a baseline. It was
measured on different data, and comparing across datasets measures the data.
So the champion is **re-scored on this validation set**, and every interval is
on the **difference** between the two models, from one paired bootstrap: each
resample scores both models on the same rows.

Checking whether two separate intervals overlap is the tempting alternative,
and it is wrong. Two models scored on the same rows make correlated errors, so
their difference is far more precise than either score alone — overlapping
intervals would call a real improvement `UNCERTAIN` almost every time.

Δ is signed so that **positive always means the challenger is better**, using
each metric's own direction: an RMSE that fell and an AUC that rose both read
as a positive Δ.

## The decision rule

Non-inferiority, read off the interval the same three ways as every other
interval in the library:

| Interval of Δ | Outcome | Flag |
|---|---|---|
| entirely above `min_improvement` | improved | `OK` |
| entirely above `−margin` | non-inferior | `OK` |
| straddles `−margin` | cannot tell | `UNCERTAIN` — non-blocking, `on_uncertain` applies |
| entirely below `−margin` | worse | `CHALLENGER_WORSE` — **blocks** |

The default margin is **1% of the champion's score**. It is unit-free, which no
absolute default could be: 0.01 means something for AUC and nothing for RMSE on
claim costs. Set `config.comparison.margin` for an absolute one.

### Must it be better, or just not worse?

`require="non_inferior"` is the default: the challenger must not be materially
worse. Retraining on fresh data, a cheaper model, a simpler one, or one that
drops a contested feature are all legitimate reasons to replace a model that is
not measurably better, and requiring improvement would block every one of them.

If your change process says a replacement must earn its risk:

```python
config.comparison.require = "superior"
```

Now a non-inferior challenger is `CHALLENGER_NOT_BETTER`, and blocks.

## Who it is better for

| Check | Asks | Blocks |
|---|---|---|
| `challenger_guardrails` | did a secondary metric regress? | on a clear regression |
| `subgroup_regression` | better on average, and worse for someone? | no |
| `fairness_delta` | did a disparity widen? | no |
| `decision_churn` | how many decisions change, and how many were right before? | no |
| `operational_delta` | is it slower or dearer to run? | no |

The last four route to a person. A segment that got worse, a disparity that
widened inside its absolute limit, or a book where one decision in ten flips
may each be entirely acceptable — but someone should decide that knowingly.

**Guardrails** catch the challenger that gains a point of AUC by wrecking its
calibration. Give calibration measures an absolute margin:

```python
config.comparison.guardrails = {"brier": 0.01, "ece": 0.02}
```

**Decision churn** splits every changed decision into a *fix* (champion wrong,
challenger right) and a *break* (the reverse). Breaks are the people for whom
the "better" model is worse; a large break rate under a small net gain is a
model that is different rather than better.

For a pricing model, [`prediction_dislocation`](reference/checks.md#prediction_dislocation)
reads the champion's predictions automatically when no `baseline_pred` is given.

## Supplying the champion

| Source | Use it when |
|---|---|
| `Champion.from_file(path)` | the champion is a file this environment can load |
| `Champion.from_loader("pkg.mod:factory")` | PyTorch, a SavedModel, a registry — anything a file extension cannot describe |
| `Champion(model=estimator)` | it is already in memory |
| `Champion(predict_fn=fn)` | it is a remote endpoint or a closure |
| `Champion(predictions=y_old)` | it cannot be loaded here — score it in its own environment |

`from_file` chooses the loader by extension, importing each framework only when
its file is asked for:

| Extension | Loader | Notes |
|---|---|---|
| `.joblib`, `.pkl`, `.pickle` | joblib (stdlib `pickle` without it) | what `--model` has always used |
| `.skops` | `skops.io.load` | the safe format — untrusted types are refused, not trusted for you |
| `.onnx` | `onnxruntime` | the input dtype is declared by the model, so nothing is guessed |
| `.keras`, `.h5` | `keras.models.load_model` | |
| `.pt`, `.pth`, a SavedModel directory | **refused** | see below |

**PyTorch gets a loader, not a guess.** A `.pt` file is either a `state_dict`,
which cannot be loaded without the class that defines the network, or
TorchScript, whose tensor dtype, device and batching are yours to choose. This
library never converts a DataFrame into a tensor — a wrong guess produces
plausible numbers rather than an error. Write the three lines:

```python
# mypkg/serving.py
def load_v3():
    net = torch.jit.load("v3.pt")
    net.eval()
    return lambda df: net(torch.tensor(df.values).float()).detach().numpy()
```

```python
champion = Champion.from_loader("mypkg.serving:load_v3")
```

### An old pickle may not load cleanly

The champion was saved under last year's scikit-learn; the challenger's
environment has this year's. Unpickling across versions either fails or,
worse, **warns and loads a subtly different model**. A comparison against the
wrong champion is worse than none, so a version warning while loading is an
error here, and the message names the way out: score the champion in its own
environment, then pass the result.

```python
champion = Champion(predictions=scored_in_old_env)  # same rows as X_val, same form as y_pred
```

Predictions are exactly as valid for every comparison check except
`operational_delta`.

### Pin the artifact

Loading a pickle runs code. The champion comes from a registry you control, so
that is acceptable — but it should be visible and checkable.

```python
Champion.from_file("v3.joblib", sha256="9f2c…")
```

The hash is checked **before** anything is deserialised: a mismatch refuses to
load at all rather than running unknown bytes and reporting the problem after.
The report records the loader, the file's hash and size, whether the hash was
pinned, and the scikit-learn version the model was saved with — so *the model
we compared against is the one in production* is something the record proves.

### Different features, different thresholds

A challenger usually adds features. `X` carries the union, and each model
receives its own columns — the champion's are read from `feature_names_in_`,
or from `Champion(features=[...])` when the model does not record them.

The champion may have been operated at a cut-off of 0.42.
`Champion(decision_threshold=0.42)` means churn and label metrics compare each
model at its real threshold, not at a shared 0.5 that neither used.

## Gate the champion too

```python
config.comparison.gate_champion = True
report = ModelGate(checks=default_structured_checks(config), config=config).run(context)
report.champion_report.gate_status  # e.g. "BLOCKED"
```

Runs the absolute checks against the champion as well, and puts both verdicts
side by side. **Informational only — it never affects the status.** Its point
is the finding *"the incumbent would itself be blocked today"*, which belongs
in front of anyone arguing the challenger is too risky to replace it. Off by
default because it roughly doubles the run time, and unavailable for a champion
supplied as predictions, which cannot be perturbed or explained.

## From the command line

```bash
bdp-model-gate --model scorecard_v4.joblib \
  --champion scorecard_v3.joblib --champion-sha256 9f2c… \
  --data validation.csv --target-col label --output gate_report.json
```

`--champion-col old_score` takes the champion's predictions from a column of
`--data` instead, and drops it from the features. See
[Command line](reference/cli.md#champion).

See [notebook 09](examples/09_champion_challenger.ipynb) for the whole
walkthrough, including a challenger that improves AUC while breaking decisions
for one segment.
