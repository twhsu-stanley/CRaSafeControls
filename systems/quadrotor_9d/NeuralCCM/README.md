# Neural CCM for the 9D quadrotor

This folder contains the trained neural metric and its NumPy runtime.
`systems/quadrotor_9d/ccm.py` loads `metrics/metric.npz` by default, or a custom
trained neural NPZ supplied with `--checkpoint`.

| File | Role |
| --- | --- |
| `neural_metric.py` | NumPy evaluation of W and exact state/parameter derivatives |
| `metrics/metric.npz` | Fixed network weights, domain, rate, provenance and validation results |
| `requirements.txt` | NumPy dependency for metric evaluation |
| `.gitignore` | Exclude caches and local output folders |
| `README.md` | Runtime usage and model scope |

The network maps six velocity/attitude states and five physical coefficients
`b = Theta @ a_hat` through `11 -> 128(tanh) -> 128(tanh) -> 81`. Its outputs form
a 9x9 matrix L, and `W(x,b) = L.T @ L + 0.03 I`. Position does not enter the
network. The runtime provides W, dW/dx and dW/db using only NumPy.

## Use the saved metric

```bash
python -m pip install -r systems/quadrotor_9d/NeuralCCM/requirements.txt
```

```python
from systems.quadrotor_9d.ccm import load_metric

metric = load_metric()
b = Theta @ a_hat
W = metric.W(x, b)
dW_dx, dW_db = metric.derivatives(x, b)
```

The controller uses `M = W^{-1}`. The quadrotor adapter applies the chain rule
through `b = Theta @ a_hat` to obtain derivatives with respect to `a_hat`.
The metric weights remain fixed while the controller updates Theta and a_hat.

Full simulations additionally require `scipy matplotlib casadi qpsolvers quadprog`.
Run from the repository root:

```bash
python simulations/quadrotor_9d/run_example.py --output outputs/quad9d_neural
```

The bundled checkpoint has rate 0.5 and domain limits position +/-5, velocity
+/-2, roll/pitch +/-0.35 rad, yaw +/-0.4 rad, and physical coefficients
`b_min = [-.72, -.72, -.66, -.2, -.2]`, `b_max = -b_min`.

W is positive definite by construction. The stored contraction and
input-direction residual checks cover finite samples and do not certify the
whole domain or the separate true-plant disturbances. Domain keys are
`b_min`/`b_max`; the runtime can also read legacy checkpoint bound names.
