# sensmaps v1.4 — Third-angle projection (3-slice view)

**Status:** ready for implementation
**Author:** Giles + Claude (Opus 4.7)
**Date:** 2026-05-02
**Branch:** `v1.4-dev` (direct branch, no worktree)

Combined spec + plan. v1.4 is GUI-only — no physics or compute changes. The
single existing slice display gets a sibling 4-panel layout that the user
toggles between.

---

## 1. Scope

Add a **third-angle projection** mode that displays:
- Top-left: x–y plane at z = `slice_value_z`
- Top-right: 3D context view — three slice planes embedded in 3D space
  (no iso-surface in v1.4; deferred to a later release to avoid the
  `scikit-image` marching-cubes dependency).
- Bottom-left: x–z plane at y = `slice_value_y`
- Bottom-right: y–z plane at x = `slice_value_x`

The four panels share a single colorbar and color-limit pair (built once
from the full Svox via the existing `make_color_limits`). Optode markers
appear in each 2D panel where the slice axis matches; depth-axis panels
have y inverted (NIRS convention, already done by `render_slice`).

A radio control toggles **Single slice ↔ Three views**. Single-slice
remains the default (preserves v1.3 behavior).

**Out of scope** (later releases):
- Iso-surface in the 3D panel (needs marching cubes / skimage).
- SNR thresholding / noise model — v1.5.
- Saving the 3-view as a single PDF — `Save Figure` already serializes
  the active matplotlib `Figure`, so this falls out for free.

---

## 2. Views layer (`views.py`)

### 2.1 New: `render_three_view(fig, S, params, slice_xyz, ...) -> dict`

Build the 2×2 layout on a caller-provided `Figure`. Returns the four `Axes`
in a dict (`"xy"`, `"3d"`, `"xz"`, `"yz"`) so the caller can attach the
shared colorbar.

Signature:
```python
def render_three_view(
    fig: Figure,
    S: np.ndarray,
    params: GridParams,
    slice_xyz: tuple[float, float, float],   # (x, y, z) slice values [mm]
    *,
    rs=None, rd=None, pert=(1.0, 1.0, 1.0),
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Axes]
```

Implementation:
1. Compute color limits once via `make_color_limits(S, quantiles)`.
2. Build a 2×2 grid via `fig.subplots(2, 2)`. Replace `(0, 1)` with a
   3D Axes (`fig.add_subplot(2, 2, 2, projection="3d")`).
3. For each 2D panel, call existing `slice_s` + `render_slice`, passing
   the precomputed `clim`/`cmap` to keep all three on the same scale.
4. For the 3D panel: render three colored quads using `plot_surface`,
   one per slice plane. Each quad is sampled from `S` at the slice value;
   colors come from the shared colormap. Optode markers at `rs`/`rd` are
   drawn with `scatter`.

### 2.2 Refactor: `render_slice` accepts external `clim`/`cmap`

`render_slice(ax, plane, plot_params, *, clim=None, cmap=None, ...)` — when
both are supplied, skip recomputing color limits and reuse for shared
colorbars across panels.

### 2.3 Tests (`tests/test_views.py`)

- `test_render_three_view_builds_axes` — create a tiny dummy `S`, call
  `render_three_view` on a headless `Figure`, assert four `Axes` present
  and the 3D one has `name == "3d"`.
- `test_render_three_view_shares_colorbar_limits` — assert all four panels
  use the same `clim` (read off the `images`/`Poly3DCollection`).

Headless: tests use `matplotlib.use("Agg")`.

---

## 3. GUI layer (`gui.py`)

### 3.1 New form fields

```python
# PARAM_CLASS additions (all cheap — re-render only):
"view_mode":     "cheap",   # "single" or "three"
"slice_value_x": "cheap",
"slice_value_y": "cheap",
"slice_value_z": "cheap",
```

`_DEFAULTS`:
- `"view_mode": "single"`
- `"slice_value_x": 12.5`  (midpoint between default optodes)
- `"slice_value_y": 0.0`
- `"slice_value_z": 10.0`  (typical depth where bulk-sensitivity peaks)

The legacy `"slice_value"` key remains for single-mode and shares the
slice-axis convention (controlled by the existing `slice_axis` combobox).

### 3.2 Form widget: view-mode radio

A two-button radio group ("Single slice" / "Three views") sits above the
slice controls. State-driven enables:
- **Single mode**: `slice_axis` combobox + `slice_value` entry enabled;
  `slice_value_x/y/z` disabled.
- **Three-views mode**: `slice_value_x/y/z` enabled;
  `slice_axis` + `slice_value` disabled.

### 3.3 PlotCanvas refactor

Replace single-Axes layout with a swappable layout:

```python
class PlotCanvas:
    def show_single(self, S, params, axis, value, ...): ...
    def show_three(self, S, params, slice_xyz, ...): ...
```

Each method clears the figure and rebuilds the appropriate Axes
configuration. The `FigureCanvasTkAgg` is preserved across switches.

### 3.4 MainWindow.recalculate / cheap-change dispatch

`recalculate` and `_apply_cheap_change` choose between `show_single` and
`show_three` based on `values["view_mode"]`.

### 3.5 Tests (`tests/test_gui.py`)

- `test_view_mode_radio_state_matrix` — toggle to "three" and assert
  single-mode controls disabled, 3-mode controls enabled, vice versa.
- `test_three_view_recalculate_uses_three_slice_values` — monkey-patch
  `PlotCanvas.show_three` to capture args; assert the three slice values
  reach the canvas.

---

## 4. Documentation

- `README.md`: bump status block to v1.4; mention 3-slice view; remove
  v1.4 entry from Roadmap; controls table gains `view_mode` and three
  slice-value rows.
- `CLAUDE.md`: short note that `views.py` now has both single-slice and
  3-slice rendering paths, and the GUI dispatches per `view_mode`.
- `pyproject.toml` + `__init__.py`: bump to `1.4.0`.

---

## 5. Atomic task list (TDD; ~6 tasks)

1. **Views: `render_three_view`** — refactor `render_slice` to accept
   external `clim`/`cmap`; implement `render_three_view` with 3 slice
   panels + 3D slice-plane embedding. Tests in `test_views.py`.
   *Checkpoint: code-reviewer on views.*
2. **GUI form fields** — add `view_mode` radio + three slice-value
   entries; PARAM_CLASS / `_DEFAULTS`. Widget enable/disable matrix
   tests pass.
3. **PlotCanvas refactor** — `show_single` / `show_three` API, layout
   swap clears + rebuilds Axes. Headless smoke test for both paths.
4. **MainWindow wiring** — `recalculate` and `_apply_cheap_change`
   pick the right canvas method per `view_mode`. Dispatch test passes.
5. **Docs** — README, CLAUDE, version bump 1.4.0.
6. **Release** — merge PR, tag `v1.4.0`, push tag.

Each task = single commit on `v1.4-dev`. Smoke test (`pytest`,
`sensmaps --smoke-test`) green at every commit.

---

## 6. Open questions / risks

- **3D panel rendering cost.** `plot_surface` on three slice planes is
  cheap (a few hundred quads each) and should not noticeably slow GUI
  recalculate. If it does, fall back to `imshow` projections onto the
  3D axes' walls (matplotlib's `contourf` with `zdir`).
- **Color-limit choice for 3D panel.** The 3D slice-plane colors must
  use the same `clim` as the 2D panels — drive both from the single
  pre-computed pair. The `Poly3DCollection`'s `set_array` + `set_clim`
  is the matplotlib-native way.
- **Optode markers in 3D.** Source = down-triangle, detector = up-triangle
  in the existing 2D `render_slice`. In 3D use `scatter(marker='v'/'^',
  s=80)` at each optode coordinate.
