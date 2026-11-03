# LMI CCM for the 9D quadrotor

This directory owns the right-invariant metric `W(x) = J(x) Q J(x).T`, its
semidefinite optimization, and the saved model used by `CRaSafeControls-review`.

| File | Purpose |
| --- | --- |
| `structured_metric.py` | NumPy metric evaluation and exact derivatives |
| `train_structured_metric.py` | Sampled C1 LMI constraints; CVXPY/CLARABEL optimization |
| `metrics/invariant_metric.npz` | Existing optimized CCM; unchanged |
| `metrics/training_reference.npz` | Small frozen target/domain for reproducing the existing optimization objective |
| `tests/test_metric.py` | SPD, derivatives, input-flow invariance, and model Jacobian checks |

Inference needs only NumPy. Training imports the shared 9D dynamics and CCM
conditions from `CRaSafeControls-review/systems/quadrotor_9d/ccm_conditions.py`.
It does not import NeuralCCM or load a neural network: the training reference
stores the exact zero-state target and original domain previously read from
the baseline neural checkpoint, with that checkpoint's SHA256 for provenance.

From `CRaSafeControls-review`:

```bash
python -m pip install -r systems/quadrotor_9d/LMICCM/requirements.txt
python -m systems.quadrotor_9d.LMICCM.train_structured_metric --output outputs/lmi_training
python -m systems.quadrotor_9d.verify_ccm --metric lmi --output outputs/lmi_check
python -m unittest discover -s systems/quadrotor_9d/LMICCM/tests -v
```

Use `--checkpoint outputs/lmi_training/metric.npz` to run a newly optimized
model. Training keeps the original domain/rate options and writes its metric,
sampled constraints, and summary to a new output directory.

Positive definiteness and full C2=0 follow from the invariant structure on
regular Euler coordinates. C1 is checked on finite samples and by numerical
search; it is not a whole-domain certificate. The metric does not depend on
the uncertainty coefficients `b`, so its parameter derivatives are zero.

Domain metadata uses `x_min`/`x_max` and `b_min`/`b_max` for the five
physical coefficients `b`. Legacy `a_min`/`a_max` keys are normalized on load;
new exports use `b_min`/`b_max`. Bundled NPZ files are preserved.
