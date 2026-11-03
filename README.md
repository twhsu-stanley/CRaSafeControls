## Conformally Robust Adaptive Safe Control (work in progress...)

9D quadrotor: [code and reference-generation guide](simulations/quadrotor_9d/README.md).
CCM inference, offline training and saved metrics are bundled in
[NeuralCCM](systems/quadrotor_9d/NeuralCCM/README.md) and
[LMICCM](systems/quadrotor_9d/LMICCM/README.md).

```text
CRaSafeControls-review/
  systems/quadrotor_9d/
    NeuralCCM/  # Neural metric runtime, training from scratch, saved NPZ
    LMICCM/     # LMI metric runtime, optimization, saved NPZ/reference
    ccm.py      # Select and load a bundled metric
    ccm_conditions.py
    verify_ccm.py
  simulations/quadrotor_9d/
```

`systems/quadrotor_9d/ccm.py` loads the selected implementation without Torch or CVXPY.
The repository contains all runtime code and saved metrics; no sibling folders,
editable install or manual PYTHONPATH is needed. Shared CCM calculations are in
`systems/quadrotor_9d/ccm_conditions.py`, so both trainers use the shared
estimated 9D model F(x,b) = f_nominal(x) + Psi(x)b.

Run from this directory. The saved CCMs are ready for simulation; see the
[9D guide](simulations/quadrotor_9d/README.md#run) for dependencies, PowerShell
setup, replay and a short smoke run.

```powershell
python simulations/quadrotor_9d/run_example.py --metric neural --output outputs/neural_run
python simulations/quadrotor_9d/run_example.py --metric lmi --output outputs/lmi_run
python -m systems.quadrotor_9d.verify_ccm --metric lmi --output outputs/lmi_check
```

`--checkpoint path/to/metric.npz` selects a custom model instead of `--metric`.
The default is `systems/quadrotor_9d/NeuralCCM/metrics/metric.npz`, with provenance
stored in the NPZ. Simulation initializes Theta uniformly in a fixed box, collects
and learns one interval at a time, and initializes CP from recent pretraining scores.
Wind changes at observation boundaries: a B-interval excitation cycle during
pretraining and a fixed five-interval cycle during online simulation.
The saved CCM weights stay fixed while Theta and the adaptive parameter estimates evolve.
The default online mission lasts 40 s. Its ascending figure eight keeps a 12 s
horizontal lap and climbs at 0.03 m/s; extending the run does not slow the path.
Before it starts, 64 PD-driven observation intervals (128 s) prepare Theta and CP.

For offline neural metric training and regression checks:

```powershell
python -m systems.quadrotor_9d.NeuralCCM.train_neural_metric --output outputs/neural_from_scratch
python -m unittest discover -s tests -v
python -m unittest discover -s systems/quadrotor_9d/NeuralCCM/tests -v
python -m unittest discover -s systems/quadrotor_9d/LMICCM/tests -v
```
