# GEMINI.md

This project, `sensmaps`, is an interactive Python/Tkinter GUI for exploring 2D sensitivity maps (the Jacobian) of detected optical signals in diffuse optical spectroscopy and imaging. It is a direct port of the MATLAB `SensitivityCompendium`.

## Project Overview

- **Purpose**: Visualization and computation of sensitivity to absorption changes ($\mathcal{S} = \partial Y / \partial \mu_a$).
- **Status**: v1 implements CW single-distance intensity (`CW_SD_I`) under diffusion theory. Includes advanced features like optode visualization and perturbation kernel overrides.
- **Core Technologies**: Python 3.11+, NumPy, SciPy, Matplotlib, Tkinter.
- **Architecture**: Strictly bottom-up four-layer design:
  1. `physics.py` (NumPy): Low-level analytical formulas ported from MATLAB.
  2. `compute.py` (Dispatcher): High-level logic for grid building and computation dispatch.
  3. `views.py` (Matplotlib): Plotting logic and colormap generation.
  4. `gui.py` (Tkinter): Interactive GUI handling user input and session state.

## Building and Running

### Setup
```bash
# Development install
pip install -e .[dev]
```

### Running the Application
```bash
# Run the GUI
sensmaps

# Alternative entry point
python -m sensmaps

# Smoke test (used in CI)
sensmaps --smoke-test
```

### Running Tests
```bash
# Run full suite
pytest

# Test numerical fidelity against MATLAB fixtures
pytest tests/test_physics.py
pytest tests/test_compute.py
```

## UI and Computation Features

- **Auto-Compute on Open**: Automatically loads the last session or defaults and performs an initial computation upon launch.
- **Enter to Recalculate**: Pressing `Enter` in any input field triggers a recalculation.
- **Boundary Enforcement**: Sensitivity is explicitly set to zero for voxels with $z < 0$ (above the medium surface).
- **Perturbation Override**: By default, perturbation size tracks the voxel size (`dr`). An "Override" checkbox allows custom kernel dimensions.
- **Optode Visualization**: Sources are marked with red downward triangles (`v`); detectors are marked with blue upward triangles (`^`).
- **Visual Cues**:
    - The **Recalculate** and **Revert** buttons are highlighted (bolded) when changes are pending.
    - A status indicator shows `● click recalculate` (orange) when the form is modified or `● image updated` (green) when in sync.

## Development Conventions

### MATLAB-Fidelity Principle
- **Naming**: Use snake_case names that mirror the original MATLAB symbols (e.g., `makeS` → `make_s`).
- **Formulas**: Preserve the mathematical structure and line-by-line logic of the MATLAB source.
- **Verification**: All ports must be numerically equivalent to MATLAB reference values stored in `tests/fixtures/*.mat`.
- **Tolerances**: `rtol=1e-10` for direct formulas; `rtol=1e-8` for FFT-convolved arrays.

### Architectural Rules
- **Layer Integrity**: No layer may import from a layer above it.
- **No Circular Imports**: Maintain a strict one-way dependency flow.
- **Stateless Physics**: Keep `physics.py` functions pure and NumPy-centric.
- **Persistent State**: The GUI saves session state to `last_session.json` on exit and reloads it on launch.

### Tool-Specific Guidance
- **Tkinter**: Requires system-level dependencies on Linux (`python3-tk` or `python3-tkinter`).
- **Numerical Ports**: Use `scipy.io.loadmat(squeeze_me=True)` for fixtures, but note the "re-inflation" logic in `tests/conftest.py` for correct array dimensionality.
- **Matplotlib**: Uses raw strings (`r"..."`) for LaTeX titles to avoid escape sequence warnings.

## Key Files
- `src/sensmaps/physics.py`: The "ground truth" implementation of diffusion theory formulas.
- `src/sensmaps/compute.py`: Orchestrates grid-building, dispatch, and $z \ge 0$ enforcement.
- `src/sensmaps/views.py`: Pure matplotlib logic for slicing, optode plotting, and dynamic labels.
- `src/sensmaps/gui.py`: Manages the interactive state, button styles, and Enter-key bindings.
- `CLAUDE.md`: Contains detailed internal guidance for developers.
- `docs/superpowers/`: Contains original design specs and implementation plans.
