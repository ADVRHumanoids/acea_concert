#!/usr/bin/env python3
"""Deterministic RGB seam detector (Variant A) -- numpy + scipy only, NO depth.

Detects a thin DARK circumferential weld seam on a pipe from a single RGB image,
robust to shadows, without depth. The seam is perpendicular to the pipe axis, so
with the pipe axis horizontal in the image the seam is a (near-)VERTICAL line --
the orientation is known, so no general line detector is needed.

Pipeline (all deterministic):
  1. luminance
  2. black top-hat with a HORIZONTAL structuring element: isolates thin dark
     vertical structures and REMOVES smooth/broad shadow gradients (a shadow is
     wider than the SE, so morphological closing fills it and it cancels).
  3. vertical-coherence smoothing: a real seam is consistent over many rows.
  4. per-column profile inside the bright pipe band.
  5. gate on the peak column: a-contrario-style significance (MAD z-score) +
     thinness (seam is narrow) + vertical span (covers the arc) + not at border
     (pipe-end / silhouette).

Self-test:  python3 acea_seam_detector.py   (synthetic seam / shadow / band / blob)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

DEFAULT_PARAMS: dict[str, Any] = {
    "tophat_se_len_px": 21,         # horizontal SE > seam width; removes broader shadows
    "vert_smooth_sigma_px": 9.0,    # accumulate a coherent vertical line
    "pipe_bright_frac": 0.55,       # a row is "pipe" if row-mean brightness >= frac*max
    "min_significance_z": 5.0,      # peak vs background (robust MAD z-score)
    "max_seam_width_px": 14.0,      # the seam is thin (gap6 ~ up to ~12 px)
    "min_vertical_run_px": 100,     # longest CONTIGUOUS vertical run of the seam (px);
                                    # tuned for the operational close/aligned regime
                                    # (clear seam run 80-200 px). Lower (~70) raises recall
                                    # on borderline views but also plain false positives.
    "resp_span_thresh_frac": 0.40,  # fraction of the local peak to count a row as "on"
    "border_margin_px": 15,         # reject candidates at the image border (Isaac renderer
                                    # produces a dark-clamp artefact at x=W-1 that mimics
                                    # a seam; 15 px margin safely excludes it)
    "neighborhood_px": 2,           # half-width of columns averaged for the vertical span
    "orientation_search_deg": 0.0,  # residual rotation search around pipe_axis_angle_deg
    "orientation_search_step_deg": 2.0,
}


@dataclass
class SeamDetection:
    accepted: bool
    x_px: int
    significance: float
    seam_width_px: float
    vertical_span_frac: float
    reason: str
    vertical_run_px: int = 0
    pipe_band: tuple[int, int] = (0, 0)
    orientation_deg: float = 0.0
    line_xyxy: tuple[int, int, int, int] | None = None
    rotated_column_x_px: int | None = None
    column_profile: np.ndarray | None = field(default=None, repr=False)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float64)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _pipe_band(gray: np.ndarray, frac: float) -> tuple[int, int]:
    """Vertical extent of the bright pipe surface (contiguous bright rows)."""
    row_mean = ndi.uniform_filter1d(gray.mean(axis=1), size=7)
    thr = frac * float(row_mean.max())
    rows = np.where(row_mean >= thr)[0]
    if rows.size < 5:
        return 0, gray.shape[0]
    # largest contiguous run
    splits = np.where(np.diff(rows) > 3)[0]
    segs = np.split(rows, splits + 1)
    seg = max(segs, key=len)
    return int(seg.min()), int(seg.max()) + 1


def _black_tophat_vertical(gray: np.ndarray, se_len: int) -> np.ndarray:
    """Closing with a horizontal line SE fills thin DARK vertical features; subtract
    -> high response on thin vertical dark lines, ~0 on broad shadows."""
    fp = np.ones((1, int(se_len)), dtype=bool)
    closed = ndi.grey_closing(gray, footprint=fp)
    resp = closed - gray
    resp[resp < 0] = 0.0
    return resp


def _detect_seam_on_gray(gray: np.ndarray, p: dict[str, Any], orientation_deg: float) -> SeamDetection:
    h, w = gray.shape
    r0, r1 = _pipe_band(gray, p["pipe_bright_frac"])

    resp = _black_tophat_vertical(gray, p["tophat_se_len_px"])
    respv = ndi.gaussian_filter(resp, sigma=(float(p["vert_smooth_sigma_px"]), 0.0))
    col = respv[r0:r1, :].mean(axis=0)

    med = float(np.median(col))
    mad = float(np.median(np.abs(col - med))) + 1e-9
    z = (col - med) / (1.4826 * mad)
    x = int(np.argmax(col))
    zx = float(z[x])

    # thinness: full width at half maximum of the column peak
    peak = float(col[x])
    half = 0.5 * peak
    lo = x
    while lo > 0 and col[lo - 1] >= half:
        lo -= 1
    hi = x
    while hi < w - 1 and col[hi + 1] >= half:
        hi += 1
    width = float(hi - lo + 1)

    # vertical coherence: longest CONTIGUOUS run of strong response (px) at the
    # peak column, over the FULL image height -> band-independent. A real seam is a
    # long connected vertical line; a blob/speck/shadow-edge is short.
    nb = int(p["neighborhood_px"])
    colresp = resp[:, max(0, x - nb):min(w, x + nb + 1)].mean(axis=1)
    cmax = float(colresp.max())
    on = colresp >= (p["resp_span_thresh_frac"] * cmax if cmax > 0 else np.inf)
    run = run_best = 0
    for v in on:
        run = run + 1 if v else 0
        run_best = max(run_best, run)
    vertical_run_px = int(run_best)
    span_frac = float(on[r0:r1].mean()) if r1 > r0 else 0.0

    reasons = []
    if zx < p["min_significance_z"]:
        reasons.append(f"low_z={zx:.1f}")
    if width > p["max_seam_width_px"]:
        reasons.append(f"too_wide={width:.0f}px")
    if vertical_run_px < p["min_vertical_run_px"]:
        reasons.append(f"short_run={vertical_run_px}px")
    if x < p["border_margin_px"] or x > w - p["border_margin_px"]:
        reasons.append("at_border")
    accepted = not reasons
    return SeamDetection(accepted, x, zx, width, span_frac,
                         "ok" if accepted else ";".join(reasons),
                         vertical_run_px=vertical_run_px, pipe_band=(r0, r1),
                         orientation_deg=float(orientation_deg),
                         column_profile=col)


def _candidate_score(d: SeamDetection) -> float:
    accepted_bonus = 1_000_000.0 if d.accepted else 0.0
    return accepted_bonus + 1000.0 * float(d.significance) + float(d.vertical_run_px) - 10.0 * float(d.seam_width_px)


def detect_seam(rgb: np.ndarray, params: dict | None = None,
                pipe_axis_angle_deg: float = 0.0) -> SeamDetection:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    gray0 = _luminance(rgb)

    search = max(0.0, float(p.get("orientation_search_deg", 0.0)))
    step = max(0.5, float(p.get("orientation_search_step_deg", 2.0)))
    if search <= 0.0:
        angles = [float(pipe_axis_angle_deg)]
    else:
        offsets = np.arange(-search, search + 0.5 * step, step)
        angles = [float(pipe_axis_angle_deg + off) for off in offsets]
        if not any(abs(a - pipe_axis_angle_deg) < 1e-6 for a in angles):
            angles.append(float(pipe_axis_angle_deg))

    best: SeamDetection | None = None
    for angle in angles:
        if abs(angle) > 1.0:
            gray = ndi.rotate(gray0, angle, reshape=False, order=1, mode="nearest")
        else:
            gray = gray0
        candidate = _detect_seam_on_gray(gray, p, angle)
        if best is None or _candidate_score(candidate) > _candidate_score(best):
            best = candidate
    assert best is not None
    if abs(best.orientation_deg) > 1e-6:
        best.reason = best.reason if best.accepted else f"{best.reason};angle={best.orientation_deg:.1f}deg"
    return best


# --------------------------------------------------------------------------- #
def _synthetic_test() -> int:
    rng = np.random.default_rng(0)
    H, W = 480, 640
    base = np.full((H, W), 180.0)  # bright pipe surface

    def noisy(img):
        return np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255)

    PASS, FAIL, res = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m", []

    def check(n, ok, d=""):
        res.append(ok)
        print(f"  [{PASS if ok else FAIL}] {n}{('  -> ' + d) if d else ''}")

    # 1) real seam: thin dark vertical line spanning the pipe
    seam = base.copy()
    seam[80:400, 317:321] -= 70.0
    d = detect_seam(np.stack([noisy(seam)] * 3, -1))
    check("seam detected near x=319", d.accepted and abs(d.x_px - 319) <= 4,
          f"x={d.x_px} z={d.significance:.1f} span={d.vertical_span_frac:.2f} w={d.seam_width_px:.0f}")

    # 2) smooth shadow gradient (no seam) -> rejected (top-hat removes broad shadow)
    grad = base.copy()
    grad += np.linspace(-60, 60, W)[None, :]
    d = detect_seam(np.stack([noisy(grad)] * 3, -1))
    check("smooth shadow gradient rejected", not d.accepted, d.reason)

    # 3) broad dark band (wider than the SE) -> rejected
    band = base.copy()
    band[:, 300:340] -= 60.0
    d = detect_seam(np.stack([noisy(band)] * 3, -1))
    check("broad dark band rejected (too wide)", not d.accepted, f"{d.reason} w={d.seam_width_px:.0f}")

    # 4) small dark blob (bracket-like, no vertical span) -> rejected
    blob = base.copy()
    blob[230:260, 315:323] -= 70.0
    d = detect_seam(np.stack([noisy(blob)] * 3, -1))
    check("small blob rejected (low vertical span)", not d.accepted, d.reason)

    # 5) flat pipe, no feature -> rejected
    d = detect_seam(np.stack([noisy(base)] * 3, -1))
    check("flat pipe rejected (no significant peak)", not d.accepted, d.reason)

    ok = sum(res)
    print(f"\nRESULT: {ok}/{len(res)} synthetic checks passed")
    return 0 if ok == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(_synthetic_test())
