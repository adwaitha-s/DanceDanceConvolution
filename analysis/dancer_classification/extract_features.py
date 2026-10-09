"""
Dancer vs. non-dancer: window-level pose features from one DanceDanceConvolution run.

Dataset : runs/20261007-133746 -- 3 dancers standing in a row, a group of seated
          spectators at the right edge of the frame; RTMW pose, 30 fps, 31 s.
          The tracker found 18 persistent "people" (spectators fragment into many tracks).
Outcome : is_dancer (1/0). Ground truth = the tracks vetted as dancers in the app: tracks 1, 2, 3
          (found by reproducing that run's deviation.csv to 5e-5). Every other track
          is a spectator. Confirmed by eye on the overlay video. DANCER_TRACKS below is
          specific to this run; change it to use another run.
Sample  : one tracked person x one 1.5 s window (hop 0.5 s), only where the pose is
          usable (wrists/shoulders/hips visible in >= 80% of frames).

Every feature is computed from the pose signal only (never from the label or the
deviation score), grouped as the lecture suggests:
    time domain, frequency domain, time-frequency, domain knowledge, human/context.

    python analysis/dancer_classification/extract_features.py [--run-dir runs/<timestamp>]   # writes features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, stft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, so the project's tracker/normalizer import as `ddc`
from ddc import io as ddc_io  # noqa: E402
from ddc.normalize import normalize  # noqa: E402
from ddc.skeleton import J  # noqa: E402
from ddc.tracking import stabilize, to_dense  # noqa: E402

WIN_S, HOP_S, MIN_VALID = 1.5, 0.5, 0.8
DANCER_TRACKS = {1, 2, 3}


def fill(x):
    """Linear-interpolate NaNs along axis 0 (edges held); all-NaN columns -> 0 (windows are masked later)."""
    out = pd.DataFrame(x.reshape(len(x), -1)).interpolate(limit_direction="both").to_numpy()
    return np.nan_to_num(out).reshape(x.shape)


def deriv(x, fps, order):
    return savgol_filter(x, 7, 3, deriv=order, delta=1 / fps, axis=0)


def spectrum(sig, fps, nfft=256):
    """One-sided power spectrum of a detrended window (Hann), 0.5-8 Hz only."""
    s = (sig - sig.mean()) * np.hanning(len(sig))
    p = np.abs(np.fft.rfft(s, nfft)) ** 2 + 1e-12
    f = np.fft.rfftfreq(nfft, 1 / fps)
    m = (f >= 0.5) & (f <= 8)
    return f[m], p[m]


def build(run_dir):
    det = ddc_io.load_jsonl(run_dir / "tracking.jsonl")
    ids, n_tracks = stabilize(det.kps)
    dense = to_dense(det.kps, ids, n_tracks)
    fps = 1 / np.median(np.diff(det.t))
    W, H = int(round(WIN_S * fps)), int(round(HOP_S * fps))
    rows = []

    for trk in range(n_tracks):
        raw = dense[:, trk]                                       # (F,17,3) pixels
        norm, origin, scale, _ = normalize(dense[:, [trk]])
        nxy, org, sc = norm[:, 0, :, :2], origin[:, 0], scale[:, 0]   # body frame, in torso lengths
        need = [J[n] for n in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        present = np.isfinite(nxy[:, need, 0]).all(1) & np.isfinite(sc)
        conf = np.nan_to_num(raw[:, :, 2])

        xy, org_f, sc_f = fill(nxy), fill(org), fill(sc[:, None])[:, 0]
        sc_f = np.where(sc_f > 1, sc_f, np.nan)                    # guard against degenerate scales
        vel, jerk = deriv(xy, fps, 1), deriv(xy, fps, 3)           # torso lengths / s, / s^3
        speed = lambda *names: np.mean([np.linalg.norm(vel[:, J[n]], axis=1) for n in names], axis=0)
        wrist_speed, ankle_speed = speed("left_wrist", "right_wrist"), speed("left_ankle", "right_ankle")
        wrist_jerk = np.mean([np.linalg.norm(jerk[:, J[n]], axis=1) ** 2 for n in ("left_wrist", "right_wrist")], 0)
        sho_c = (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]]) / 2
        wrist_c = (xy[:, J["left_wrist"]] + xy[:, J["right_wrist"]]) / 2
        wrist_h = -(wrist_c[:, 1] - sho_c[:, 1])                   # +ve = hands above shoulders (image y points down)
        reach = np.mean([np.linalg.norm(xy[:, J[w]] - xy[:, J[s]], axis=1)
                         for w, s in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder"))], 0)
        leg_ext = np.mean([xy[:, J[a], 1] - xy[:, J[h], 1]
                           for a, h in (("left_ankle", "left_hip"), ("right_ankle", "right_hip"))], 0)
        knee_ext = np.mean([xy[:, J[k], 1] - xy[:, J[h], 1]
                            for k, h in (("left_knee", "left_hip"), ("right_knee", "right_hip"))], 0)
        travel = np.linalg.norm(deriv(org_f, fps, 1), axis=1) / sc_f
        tv = fill((raw[:, J["left_shoulder"], :2] + raw[:, J["right_shoulder"], :2]) / 2
                  - (raw[:, J["left_hip"], :2] + raw[:, J["right_hip"], :2]) / 2)
        lean = np.degrees(np.abs(np.arctan2(tv[:, 0], -tv[:, 1])))  # 0 = upright
        asym = np.abs(np.linalg.norm(vel[:, J["left_wrist"]], axis=1) - np.linalg.norm(vel[:, J["right_wrist"]], axis=1))

        # time-frequency: STFT (0.5 s Hann, 80% overlap) of wrist height over the whole track
        fz, tz, Z = stft(wrist_h, fps, nperseg=15, noverlap=12, nfft=128, boundary=None, padded=False)
        P = np.abs(Z) ** 2
        band = (fz >= 0.5) & (fz <= 8)
        fz, P = fz[band], P[band] + 1e-12
        centroid = (fz[:, None] * P).sum(0) / P.sum(0)
        Pn = P / P.sum(0)
        flux = np.r_[0, np.linalg.norm(np.diff(Pn, axis=1), axis=0)]

        for a in range(0, len(det.t) - W + 1, H):
            sl = slice(a, a + W)
            if present[sl].mean() < MIN_VALID:
                continue
            f_, p_ = spectrum(wrist_h[sl], fps)
            _, ps_ = spectrum(wrist_speed[sl], fps)
            pr = ps_ / ps_.sum()
            tm = (tz >= det.t[a]) & (tz <= det.t[a + W - 1])
            rows.append({
                "track": trk, "t_start": round(det.t[a], 2),
                # --- time domain
                "wrist_speed_mean": wrist_speed[sl].mean(),
                "wrist_jerk_rms": np.sqrt(wrist_jerk[sl].mean()),
                "ankle_speed_mean": ankle_speed[sl].mean(),
                "body_sway_std": np.std(org_f[sl, 0] / sc_f[sl]),
                # --- frequency domain
                "wrist_dom_freq_hz": f_[np.argmax(p_)],
                "beat_band_frac": p_[(f_ >= 1.5) & (f_ <= 3.0)].sum() / p_.sum(),
                "wrist_spec_entropy": -(pr * np.log(pr)).sum() / np.log(len(pr)),
                # --- time-frequency
                "stft_centroid_hz": centroid[tm].mean(),
                "stft_flux": flux[tm].mean(),
                # --- domain knowledge (standing vs. seated body, how the limbs move)
                "arm_reach": reach[sl].mean(),
                "wrist_height": wrist_h[sl].mean(),
                "leg_extension": leg_ext[sl].mean(),
                "knee_extension": knee_ext[sl].mean(),
                "lr_wrist_asym": asym[sl].mean(),
                "torso_lean_deg": lean[sl].mean(),
                "travel_speed": travel[sl].mean(),
                # --- human / context
                "visible_joint_frac": (conf[sl] >= 0.3).mean(),
                "mean_kp_conf": conf[sl].mean(),
                "torso_px": np.nanmean(sc[sl]),
                "x_pos_px": np.nanmean(org[sl, 0]),
                "y_pos_px": np.nanmean(org[sl, 1]),
                # --- outcome
                "is_dancer": int(trk in DANCER_TRACKS),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(HERE.parents[1] / "runs" / "20261007-133746"),
                    help="a runs/<timestamp> folder containing tracking.jsonl")
    df = build(Path(ap.parse_args().run_dir))
    df.to_csv(HERE / "features.csv", index=False, float_format="%.5f")
    print(df.shape, "->", HERE / "features.csv", " NaNs:", int(df.isna().sum().sum()))
    print("windows per class:", df.is_dancer.value_counts().to_dict(), " tracks per class:",
          df.groupby("is_dancer").track.nunique().to_dict())
    print(df.groupby("is_dancer").mean(numeric_only=True).drop(columns=["track", "t_start"]).T.round(3).to_string())
