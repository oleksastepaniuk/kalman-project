# kalman-project
Demonstration of the Kalman filter usage

## Environment setup
Sync environment with development and notebook extras
```
uv sync --extra dev --extra notebooks
```
Install pre-commit hooks
```
uv run pre-commit install
```

## 1d motion
`run_1d.py` models the one-dimensional movement of an object with commanded acceleration input.
To run the simulation:
```
uv run python examples/run_1d.py
```
**Configurable parameters**

You can adjust these directly in `run_1d.py`:
- Number of steps (n_steps) - default 40
- Process noise / acceleration std, for instance wind, (accel_std) - default 0.05
- Measurement noise (meas_std) - default 3.0
- Commanded acceleration (accel_cmds), list of floats controlling the input acceleration
