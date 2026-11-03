# 9D Quadrotor

Compare four controllers using the shared CE-MPC, CRaCCM and OLACP implementations.
Simulation learns the uncertainty representation with a fixed offline CCM from
[NeuralCCM](../../systems/quadrotor_9d/NeuralCCM/README.md) or
[LMICCM](../../systems/quadrotor_9d/LMICCM/README.md).

## Files

| File | Role |
| --- | --- |
| [run_example.py](run_example.py) | Entry point, configuration, metric selection and four-arm comparison. |
| [learning.py](learning.py) | Streaming pretraining, interval fits, Theta updates, CP scores and adaptation gains. |
| [reference.py](reference.py) | Ascending figure-eight guide, custom waypoint splines and MPC configuration. |
| [simulation.py](simulation.py) | MPC replanning, CRaCCM feedback and coupled plant/reference integration. |
| [results.py](results.py) | Diagnostics, saved arrays and plots. |
| [quadrotor_9d.py](../../systems/quadrotor_9d/quadrotor_9d.py) | Dynamics, uncertainty features and metric adapter. |
| [nonlinear_uncertainty.py](../../systems/quadrotor_9d/nonlinear_uncertainty.py) | Physical plant disturbance. |
| [ccm.py](../../systems/quadrotor_9d/ccm.py) | Load the saved NeuralCCM or LMICCM metric. |
| [ccm_conditions.py](../../systems/quadrotor_9d/ccm_conditions.py) | Shared estimated model and CCM checks for offline training/verification. |
| [verify_ccm.py](../../systems/quadrotor_9d/verify_ccm.py) | Sample CCM conditions and optionally search for C1 violations. |

## Model and learning

- State: `[px, py, pz, vx, vy, vz, roll, pitch, yaw]`, with inertial velocities.
  Input: `[thrust_acceleration, omega_x, omega_y, omega_z]`.
  Body rates are inputs; rotor and torque dynamics are omitted.
- Estimated uncertainty: `Y_Theta(x) a = Psi(x) Theta a`, with
  `Psi: 9x5`, `Theta: 5x4`, `a: 4`, and physical coefficients `b = Theta a: 5`.
  The true plant separately adds quadratic drag, payload, gusts and rate biases.
- Theta entries are sampled uniformly from a fixed box: rows 1-3 use
  `[-.15, .15]`, rows 4-5 use `[-.04, .04]`. The same box constrains every update.
  `--theta-seed 42` is independent of the observation `--seed`.

Pretraining follows the planar example:

1. Collect one interval of physical observations under nominal PD feedback.
2. Fit bounded `a_k` and save the maximum residual norm using that interval's Theta.
3. Update Theta every B intervals, then collect the next interval.
4. Compute the initial CP quantile from recent scores and copy the learner into each arm.

Defaults are 64 pretraining intervals and a 32-score CP window. Scores retain their
historical Theta; delta stays at 0.1 during pretraining. The same learner's scores,
Theta and update counters continue into online control. `--calibration-intervals`
sets the score-window size. Pretraining must fill this window and complete B-interval blocks.

Pretraining is a separate 128 s PD-driven simulation at the default settings,
with 16 Theta updates. Each online arm then starts its own clock at zero using
that learned Theta, the latest 32 CP scores and the resulting fixed Gamma.
The plant and reference restart at `--initial-reference`; `a_hat` and `rho` start
at zero. Loading the offline CCM is separate: its weights stay fixed throughout.
Without `--source`, every command collects fresh pretraining data before the mission.

Wind is held constant inside each observation interval and changes at its boundary.
Pretraining uses a sinusoidal sequence of B wind vectors, so one wind cycle spans
one Theta update block (8 s by default). Online simulation repeats the five vectors
in `ONLINE_WIND_PATTERN`, giving a 10 s cycle at the default 2 s interval length.
Changing B changes the pretraining cycle and Theta update period; the online
pattern still has five intervals. Every controller case receives the same wind.
The adaptive estimate `a_hat` continues evolving throughout ODE integration.

Online control uses `position cost guide -> MPC inputs -> reference ODE`.
The default spatial guide is an ascending figure eight:
`p(t) = p(0) + [0.45 sin(2 pi t/12), 0.30 sin(4 pi t/12), 0.03 t]` in meters.
Each horizontal lap takes 12 s and rises 0.36 m; a 40 s mission rises 1.2 m.
Increasing `--sim-time` adds laps without slowing the motion. Position samples
are recorded every observation interval; the MPC evaluates the smooth analytic
guide between them. Custom `--waypoints` and point targets use a natural cubic spline.
Both guide types hold the final position beyond the mission end.
MPC freezes estimates during each solve. The reference ODE uses live `a_hat` and is
integrated jointly with the plant and adaptation states. Interval-end Theta/CP
updates precede the next MPC replan. Gamma is computed once from pretrained Theta:
`gain * inv(Theta.T @ Theta + regularization * I)`, plus a payload-direction boost.
This gain controls `a_hat` adaptation; `--learning-rate` controls Theta learning.

The four arms cover all combinations of CP and continuous `a_hat/rho` adaptation.
Theta/OLACP updates remain enabled in all arms by default. `--no-online-theta`
freezes Theta after pretraining. MPC bounds apply to reference inputs `u_d`;
the final CRaCCM input `u` is not saturated.

## Run

Run from `CRaSafeControls-review`; both metric packages and their saved models
are included under `systems/quadrotor_9d`.
Runtime packages: `numpy scipy matplotlib casadi qpsolvers quadprog`.
Each output directory must be new or empty. PowerShell examples:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MPLBACKEND = "Agg"

# Default: neural CCM, fresh Theta pretraining, four 40 s controller arms.
python simulations/quadrotor_9d/run_example.py --output outputs/quad9d_neural

# Use the saved LMI CCM, or select a custom NPZ with --checkpoint.
python simulations/quadrotor_9d/run_example.py --metric lmi --output outputs/quad9d_lmi

# Independently check a saved metric.
python -m systems.quadrotor_9d.verify_ccm --metric neural --output outputs/quad9d_metric_check

# Replay saved observations and reconstruct the same learner state.
python simulations/quadrotor_9d/run_example.py --source outputs/quad9d_neural/pretraining --output outputs/quad9d_replay
```

Replay requires matching metric, `--dt`, `--interval-steps`, `--residual-scale`
and the recorded periodic pretraining wind schedule.
Counts, B, learning rate and seeds come from the saved pretraining configuration.
Archives from the previous random-wind pretraining protocol require a fresh run
without `--source`.

To amplify unstructured disturbances, use `--residual-scale`. It scales additive
gusts and time/state-varying rate biases while keeping drag/payload coefficients
and the prescribed wind sequence fixed.
Collect fresh pretraining data at the same scale as the online experiment:

```powershell
python simulations/quadrotor_9d/run_example.py --metric neural --residual-scale 20 --output outputs/quad9d_neural_residual20 --quiet
```

For new 40 s comparisons with ascending figure-eight motion, use empty outputs:

```powershell
python simulations/quadrotor_9d/run_example.py --residual-scale 1 --output outputs/wind40_residual1
python simulations/quadrotor_9d/run_example.py --residual-scale 20 --output outputs/wind40_residual20
```

For a short point-tracking check:

```powershell
python simulations/quadrotor_9d/run_example.py --output outputs/quad9d_smoke --sim-time 1 --dt .05 --interval-steps 10 --pretrain-intervals 32 --calibration-intervals 30 --representation-period 2 --trajectory point --goal 0 0 .01 --quiet
```

## Defaults and outputs

| Setting | Default |
| --- | --- |
| Simulation / control step | 40 s / 0.02 s |
| Spatial guide | 12 s per horizontal lap; 0.03 m/s climb |
| Observation / waypoint interval | 100 steps = 2 s |
| Theta update period / learning rate | B = 4 intervals / 1e-4 |
| Pretraining / online wind cycle | B intervals = 8 s / 5 intervals = 10 s |
| MPC input hold / horizon / replan period | 0.02 s / 0.8 s / 0.4 s |
| Initial reference | Nine zeros; spatial waypoints in `reference.py` |

Theta updates at 8, 16, 24 and 32 s affect subsequent control in the default
40 s mission. The update at 40 s is recorded but has no remaining control interval.

Use `--waypoints positions.json` for K+1 position triples, where
`K = sim_time / (interval_steps * dt)`. The first position must match
`--initial-reference`, which specifies the full nine-state initial reference.
Use `--help` for all options and [methods.tex](methods.tex) for the equations.

Results include per-arm NPZ histories, `summary.json`, `intervals.json`,
`calibration.json`, and `comparison.png`, `inputs.png`, `reference.png`, `wind.png`.
`pretraining/` stores observations, Theta/score history, initial Theta, bounds and seeds.
Reports include both wind schedules; per-arm NPZ files include the active `wind` vector.
In histories, `target` is the position cost guide and `x_d` is the integrated reference.

```powershell
python -m unittest discover -s tests -p "test_quadrotor_9d.py" -v
python -m unittest discover -s tests -p "test_quadrotor_reference.py" -v
```
