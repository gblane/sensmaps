# Review of `efd8572 gem vibe`

Reviewed 2026-04-25. The commit added optode visualization, a pert-override
checkbox, Enter-to-recalc binding, auto-compute on launch, prominent
dirty/sync indicator, units in labels, dynamic colorbar text, a `z<0` mask
in the physics dispatcher, and `GEMINI.md`.

All 27 tests passed, but a manual review and live probes surfaced the items
below.

## Findings

### Real bugs

**#1 — `pert_override` not in `PARAM_CLASS` dirties the form on toggle.** *(fix)*

`gui.py:54` lists every input parameter except `pert_override`.
`MainWindow._on_param_changed` falls back to `"expensive"` for unknown
names, so toggling the checkbox calls `_set_dirty(True)` even though the
form's pert value is unchanged (or, when syncing back to `dr`, is handled
on a separate cheap-path notification). Live probe confirmed
`is_dirty=True` after both enabling and disabling the checkbox.
**Fix:** add `"pert_override": "cheap"` to `PARAM_CLASS`.

**#2 — Optode markers drawn regardless of distance from slice plane.** *(fix)*

`render_slice` plots every source/detector at its (h, v) projection without
checking whether the optode actually intersects the slice plane. Live
probe placed a source at `y=-10 mm` and verified it appeared on a slice at
`y=+10 mm`, 20 mm out of plane. Misleading.
**Fix:** require `PlotParams` to know the slice axis and value; filter
optodes whose slice-axis coordinate is more than ~dr/2 from `slice_value`.

**#3 — Dead `label=` kwargs on the scatter calls.** *(fix)*

`label='Sources'` / `label='Detectors'` are set on the scatter calls but
no `ax.legend()` is ever invoked, so the labels never appear. Either add a
legend or drop the labels.
**Fix:** drop the labels (the marker shapes/colors are self-evident; a
legend would clutter the figure).

### MATLAB-fidelity divergence

**#4 — `compute.py` zeroes voxels with `z < 0`.** *(keep as-is)*

MATLAB `makeS.m` only does `l(isnan(l))=0`; it has no `z<0` mask. The
default `zl=[0, 25]` never exercises the new mask, so the regression test
still passes, but a user who sets `zl=[-5, 25]` will diverge from MATLAB.
The mask is physically correct (voxels above the surface aren't part of
the medium) — keeping as-is per the project owner's call.

### Minor issues

**#5 — `render_slice` axis-index lookup is string-based.** *(fix)*

`get_indices` does `'x' in label.lower()`, which only works because labels
are exactly `"$x$ (mm)"` etc. Fragile.
**Fix:** record axis names (`'x'`, `'y'`, `'z'`) on `PlotParams` from
`slice_s` and read them directly. Sets up #2 cleanly.

**#6 — Inconsistent S typography between title and colorbar.** *(fix)*

Title uses `\mathcal{S}` (script S); colorbar uses plain `"S to a..."`.
**Fix:** use `\mathcal{S}` in the colorbar label too.

**#7 — `MainWindow.__init__` is fragile if `last_session.json` is bad.**
*(fix, but keep auto-recalc on launch)*

`load_session` only `try`s the JSON decode, not the `set_values()` call —
so a session file with a non-numeric `dr` will raise out of the
constructor (because `f"{'foo':g}"` throws). Auto-recalc on launch is
desired behavior; the fix is just to widen the try/except.
**Fix:** move `set_values(values)` inside the existing `try/except` in
`load_session`.

**#8 — `_apply_cheap_change` mutates `self._cache.S` and `self._cache.pert`,
duplicating kernel construction from `make_s_full`.** *(fix)*

`gui.py:418–423` is a verbatim copy of `compute.py:192–194`. Mutating the
dataclass works but is a side effect on a cached result.
**Fix:** extract `compute.apply_pert_kernel(Svox, pert, dr)` and call it
from both sites. Replace the in-place mutation with `dataclasses.replace`
so the cache is updated atomically. (Same finding as the prior `/simplify`
pass.)

**#9 — `set_values` silently ignores unknown keys.** *(fix)*

The new optional-key behavior is needed for partial dicts (used by the
new pert-syncing test), but it now hides typos: `set_values({"slcie_axis":
"z"})` is a silent no-op.
**Fix:** raise `ValueError` on unknown keys; still permit missing keys.

### Documentation

**#10 — `GEMINI.md` adds a Gemini-CLI counterpart to `CLAUDE.md`.**
*(don't keep updated)*

GEMINI.md duplicates much of CLAUDE.md content; project owner has decided
to keep only CLAUDE.md current going forward.

**#11 — `claude-resume.txt` has the same line twice.** *(keep as-is)*

Probably a stray `>>` append. Cosmetic; project owner has decided to
leave as-is.

## What's good

- The pert override / sync logic in `_on_var_changed` is solid (verified
  via the new test and a manual probe — including the revert path, which
  correctly re-convolves Svox back to the saved kernel).
- The `render_slice` contour-level filter
  (`[v for v in levels if clim[0] <= v <= clim[1]]`) silences spurious
  matplotlib warnings cleanly.
- The dirty-state UX (orange "● click recalculate" + bold buttons →
  green "● image updated") is a clear improvement over the prior
  text-only indicator.
