# Neural CCM for the 9D quadrotor

This folder contains one trained metric, its NumPy runtime, and the code that
trains it from random initialization. `CRaSafeControls-review/systems/quadrotor_9d/ccm.py` loads
`metrics/metric.npz` for `--metric neural` and for the default neural selection.

| File | Role |
| --- | --- |
| `train_neural_metric.py` | Random initialization, sampled C1/C2 losses, Adam training and NPZ export |
| `neural_metric.py` | NumPy evaluation of W and exact state/parameter derivatives |
| `metrics/metric.npz` | The current model, domain, rate, training provenance and selected validation results |
| `__init__.py` | Public runtime class and default-model path |
| `tests/test_training.py` | Initialization, gradients, shared conditions, training/export and CLI checks |
| `requirements.txt` | Training and shared-dynamics dependencies; runtime inference needs only NumPy |
| `.gitignore` | Exclude caches and local output folders |
| `README.md` | File map, training instructions and model scope |

The network maps six velocity/attitude states and five physical coefficients
`b = Theta @ a_hat` through `11 -> 128(tanh) -> 128(tanh) -> 81`. Its outputs form
a 9x9 matrix L, and `W(x,b) = L.T @ L + 0.03 I`. Position does not enter the
network. The runtime provides W, dW/dx and dW/db without importing PyTorch.

Training follows the metric architecture, eigenvalue losses and projected C2
penalty in the supplied `Trial/NeuralCCM.zip` (`main_adaptive.py` and the payload
planar-style model/config). It uses the shared estimated dynamics
`F(x,b) = f_nominal(x) + Psi(x)b` in `CRaSafeControls-review`; the deployed
controller remains CRaCCM. There is no pretrained-weight or anchor-model input.

## Train and use a new model

Run from `CRaSafeControls-review`:

```bash
python -m pip install -r systems/quadrotor_9d/NeuralCCM/requirements.txt
python -m systems.quadrotor_9d.NeuralCCM.train_neural_metric --output outputs/neural_training
python -m systems.quadrotor_9d.verify_ccm --checkpoint outputs/neural_training/metric.npz --output outputs/neural_check
python simulations/quadrotor_9d/run_example.py --checkpoint outputs/neural_training/metric.npz --output outputs/neural_run
python -m unittest discover -s systems/quadrotor_9d/NeuralCCM/tests -v
```

Default training uses 60 epochs, learning rate 0.001 with decay every 20 epochs,
16384 random training samples, 2048 training corners, 4096 held-out random
validation samples, batch size 256, and seed 20261031. Loss weights are
`C1=100`, `projected C2=1`, and `W upper bound=1`. Epoch zero is diagnostic only.

The independent domain defaults are position +/-5, velocity +/-2, roll/pitch
+/-0.35 rad, yaw +/-0.4 rad, and physical coefficients
`b_min = [-.72, -.72, -.66, -.2, -.2]`, `b_max = -b_min`.
Use `--rate`, `--w-lb`, `--w-ub`, `--margin`, `--hidden`, the domain limits, and
`--device cpu|cuda|auto` to adjust the run. Defaults are rate=0.5, w_lb=0.03,
w_ub=10 and margin=0.05.

When an independent search finds a failing point, its report can be supplied
to a fresh run using `--counterexamples path/to/verification/summary.json`.
That point and nearby samples become training data, so subsequent independent
checks must use a fresh seed. The historical counterexample file and the two
superseded model files have been removed.

## Current training record

The bundled model was trained for 60 epochs from random weights using the
defaults above plus a counterexample discovered after an initial 15-epoch run.
Full training history and the exact source used are recorded at
`outputs/neural_from_scratch_robust/`.
Independent verification is recorded at
`outputs/neural_from_scratch_robust_check/`.

Training writes `metric.npz`, `summary.json`, `history.json` and `train_source.py`
to a new output directory, plus the source JSON when counterexamples are used.
Only the selected current NPZ is bundled here to avoid duplicating run logs.

W is positive definite by construction. C1 is evaluated on finite samples;
projected C2 and the W upper bound are soft penalties. These checks do not
certify the whole domain or the separate true-plant disturbances. Domain keys
are `b_min`/`b_max`; the runtime can also read legacy checkpoint bound names.
