# sensmaps

Interactive Python/Tkinter GUI for exploring 2D slice maps of the Jacobian

$$\mathcal{S}(\vec{r}) = \frac{\partial Y / \partial \mu_a(\vec{r})}{\partial Y / \partial \mu_{a,\text{homogeneous}}}$$

of a detected optical signal `Y` with respect to local absorption changes — "sensitivity to absorption change" maps for diffuse optical spectroscopy and imaging.

This is the Python companion to the MATLAB [`SensitivityCompendium`](https://github.com/DOIT-Lab/DOIT-Public/tree/main/SensitivityCompendium) that accompanies:

> G. Blaney, A. Sassaroli, and S. Fantini, *Spatial sensitivity to absorption changes for various near-infrared spectroscopy methods: A compendium review*, J. Innov. Opt. Health Sci. **17**(04), 2430001 (2024). [DOI](https://doi.org/10.1142/S1793545824300015)

## Status

v1.1 — CW (`SD`/`SS`/`DS` × `I`) and FD (`SD`/`SS`/`DS` × `I`/`P`) under
diffusion theory. Tkinter GUI, save PNG/PDF + `.npz`, multi-row optode
entry, modulation frequency control. Time-domain types in v1.2/v1.3;
Monte Carlo backend in v3.

## Install

### venv (zero prerequisites beyond Python 3.11+)

```bash
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
python -m venv .venv
source .venv/bin/activate
pip install -e .
sensmaps                     # or: python -m sensmaps
```

> **Linux note:** `tkinter` is not always bundled with the system Python. On Fedora install `python3-tkinter`; on Debian/Ubuntu install `python3-tk`.

### conda

```bash
conda create -n sensmaps python=3.11
conda activate sensmaps
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
pip install -e .
sensmaps
```

### uv

```bash
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
uv venv
source .venv/bin/activate
uv pip install -e .
sensmaps
```

## Using the GUI

| Control                  | Class      | What it does                                       |
|--------------------------|------------|----------------------------------------------------|
| Type                     | expensive  | Measurement type (v1.1: 9 CW + FD combos)          |
| rs, rd                   | expensive  | Source / detector coordinates [mm]; multi-row via `;` |
| n_in, n_out              | expensive  | Index of refraction inside / outside               |
| musp, mua                | expensive  | Reduced scattering / absorption [1/mm]             |
| xl, yl, zl, dr           | expensive  | Voxel-grid limits and resolution [mm]              |
| fmod                     | expensive  | Modulation frequency [MHz]; greyed out unless type is `FD_*` |
| pert                     | cheap      | Perturbation box size [mm]                         |
| slice axis, value        | cheap      | Which 2D slice to display                          |
| quantiles                | cheap      | Color-limit quantiles (lo, hi)                     |

**Multi-optode formats:** `SS` accepts `(1 src, 2 dets)` *or* `(2 srcs, 1 det)`;
`DS` requires `(2 srcs, 2 dets)`. Enter additional rows separated by `;` —
e.g. `0 0 0; 30 0 0` for two sources.

- **cheap** params redraw the plot live.
- **expensive** params mark the form dirty; click **Recalculate** to rerun the physics.
- **Revert** restores the form to the inputs that produced the currently-displayed map.
- **Save Figure…** writes PNG / PDF / SVG (by extension).
- **Save Data…** writes an `.npz` containing `S`, `Svox`, grid axes, and all inputs.

The last session is saved to `./last_session.json` on exit and reloaded on launch. Delete that file to reset to defaults.

## Development

```bash
pip install -e .[dev]
pytest
```

Regression tests compare the Python port to MATLAB reference values stored as `.mat` fixtures in `tests/fixtures/`. To regenerate them, run `tests/fixtures/generate_fixtures.m` in MATLAB.

## License

TBD — will be added before public release.

## Authorship

Giles Blaney, with development assistance from Claude Code.
