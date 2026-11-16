"""Head pose from a calibrated stereo pair, by visual odometry.

For a keyframe, corners in the left view are matched to the right view and triangulated. The corners are then followed
through the next left frames, and each frame's camera pose is solved from the keyframe's 3D points and where they are
seen now (PnP with RANSAC). A new keyframe starts when too few points are left or the view has moved on. Hands and
forearms are masked, because they move with the wearer and not with the room.

The pose of a frame is "left camera → world": X_world = R X_camera + t. The world frame is the left camera at the first
frame (x right, y down, z forward). With an accelerometer, `level` turns it so that y points along gravity.

Old keyframes are kept. When the view comes back near one of them, the pose is solved against that keyframe again
instead of a new one (`revisits`), which stops the error from adding up while the head stays in one work area. There
is no global optimisation: errors still add up at every new keyframe, so the pose drifts when the wearer moves on.
While the head is still the keyframe is kept and the pose does not drift. Where tracking fails the pose is held and
marked (`status` 0); frames after such a break belong to a new `part`, and poses are only comparable within one part.

`fit_distortion` measures one radial distortion term from the footage itself, for video that is published as
undistorted but is not quite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .stereo import StereoRig, _rotate

LOST, SOLVED = 0, 1


@dataclass
class Settings:
    width: int = 960  # each view is resized to this width for tracking
    corners: int = 1200  # corners looked for in a keyframe
    min_points: int = 40  # triangulated corners a keyframe needs
    min_inliers: int = 25  # points that must agree with the solved pose
    refresh_share: float = 0.5  # new keyframe when fewer than this share of its points still agree
    refresh_shift: float = 0.10  # ... or when the points have moved by this share of the image width (median)
    max_depth: float = 20.0  # metres
    min_disparity: float = 2.0  # pixels at the tracking size; farther points have no usable depth
    max_epipolar: float = 3.0  # pixels at the tracking size; loose, because a wrong match is caught by the track-back check
    ransac_px: float = 2.0  # pixels at the tracking size
    revisit: bool = True  # solve against an old keyframe again when the view returns to it
    max_keyframes: int = 300  # old keyframes kept for revisits (about 0.6 MB each at width 960)


@dataclass
class HeadPose:
    fps: float
    R: np.ndarray  # (n, 3, 3) left camera → world
    t: np.ndarray  # (n, 3) position of the left camera in the world, metres
    status: np.ndarray  # (n,) 1 solved, 0 held from the last solved frame
    part: np.ndarray  # (n,) counts the breaks so far; poses are comparable within one part
    keyframe: np.ndarray  # (n,) frame index of the keyframe this frame was solved against (-1 if none)
    inliers: np.ndarray  # (n,) points that agree with the solved pose
    error: np.ndarray  # (n,) their median reprojection error, in pixels of the original video (nan if not solved)
    world: str = "left camera at the first frame"
    settings: dict | None = None
    checks: dict = field(default_factory=dict)  # for example {"gravity": ...} from `level`
    revisits: int = 0  # times an old keyframe was taken up again

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, fps=self.fps, R=self.R, t=self.t, status=self.status, part=self.part,
                            keyframe=self.keyframe, inliers=self.inliers, error=self.error, world=self.world,
                            settings=np.array(repr(self.settings or {})), checks=np.array(repr(self.checks)),
                            revisits=self.revisits)

    @classmethod
    def load(cls, path: str | Path) -> HeadPose:
        d = np.load(path, allow_pickle=False)
        import ast
        return cls(float(d["fps"]), d["R"], d["t"], d["status"], d["part"], d["keyframe"], d["inliers"], d["error"],
                   str(d["world"]), ast.literal_eval(str(d["settings"])), ast.literal_eval(str(d["checks"])),
                   int(d["revisits"]))

    def to_world(self, X: np.ndarray, i: int) -> np.ndarray:
        """Points (..., 3) in the left camera frame of frame i → world."""
        return np.einsum("ij,...j->...i", self.R[i], X) + self.t[i]

    def vector(self, i: int) -> np.ndarray:
        """x y z and the first two columns of R, the layout used for wrist poses."""
        return np.concatenate([self.t[i], self.R[i][:, 0], self.R[i][:, 1]])

    def path_length(self) -> float:
        """Metres travelled by the camera, over solved frames, after smoothing over a quarter of a second."""
        total = 0.0
        k = max(1, round(self.fps / 4))
        for p in np.unique(self.part):
            pts = self.t[(self.part == p) & (self.status == SOLVED)]
            if len(pts) > k:
                total += float(np.linalg.norm(np.diff(pts[::k], axis=0), axis=1).sum())
        return total

    def summary(self) -> dict:
        solved = self.status == SOLVED
        keyframes = len(np.unique(self.keyframe[self.keyframe >= 0]))
        return {"frames": int(len(solved)), "tracked": round(float(solved.mean()), 3) if len(solved) else None,
                "breaks": int(self.part.max()) if len(self.part) else 0, "keyframes": keyframes,
                "revisits": int(self.revisits),
                "frames_per_keyframe": round(float(solved.sum() / max(1, keyframes)), 1),
                "inliers_median": int(np.median(self.inliers[solved])) if solved.any() else None,
                "reprojection_px_median": round(float(np.nanmedian(self.error[solved])), 2) if solved.any() else None,
                "path_m": round(self.path_length(), 2), "world": self.world}


def hand_masks(kp2d: np.ndarray, width: int, height: int, spread: int = 2):
    """A function (frame, w, h) → mask that is 0 over the hands and the forearm below them, 255 elsewhere.

    `kp2d`: (n, 2 hands, 21, 3) left-view keypoints with a score, in pixels of a width × height image. The box of each
    seen hand is widened and extended to the bottom edge, where the forearm of a head-camera wearer enters the view.
    Detections from `spread` frames before and after are included, to cover frames where the detector missed."""
    kp2d = np.asarray(kp2d, float)

    def mask_for(i: int, w: int, h: int) -> np.ndarray:
        mask = np.full((h, w), 255, np.uint8)
        for j in range(max(0, i - spread), min(len(kp2d), i + spread + 1)):
            for hand in kp2d[j]:
                seen = np.isfinite(hand[:, 0]) & (hand[:, 2] > 0)
                if seen.sum() < 5:
                    continue
                lo, hi = hand[seen, :2].min(0), hand[seen, :2].max(0)
                grow = 0.35 * float((hi - lo).max()) + 0.02 * width
                x0, x1 = (lo[0] - grow) * w / width, (hi[0] + grow) * w / width
                y0 = (lo[1] - grow) * h / height
                mask[max(0, int(y0)):, max(0, int(x0)):max(0, int(np.ceil(x1)))] = 0
        return mask
    return mask_for


def _lk(a, b, pts: np.ndarray, guess: np.ndarray | None = None, levels: int = 4):
    import cv2
    if not len(pts):
        return np.zeros((0, 2)), np.zeros(0, bool)
    p0 = np.ascontiguousarray(pts, np.float32).reshape(-1, 1, 2)
    p1 = None if guess is None else np.ascontiguousarray(guess, np.float32).reshape(-1, 1, 2).copy()
    out, status, _ = cv2.calcOpticalFlowPyrLK(
        a, b, p0, p1, winSize=(21, 21), maxLevel=levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0)
    return out.reshape(-1, 2).astype(float), status.reshape(-1).astype(bool)


def _lk_checked(a, b, pts, guess=None, levels: int = 4, tolerance: float = 1.0):
    """Lucas-Kanade from a to b, kept only where tracking back from b lands within `tolerance` pixels of the start."""
    forward, ok = _lk(a, b, pts, guess, levels)
    back, ok_back = _lk(b, a, forward, pts, levels)
    return forward, ok & ok_back & (np.linalg.norm(back - pts, axis=1) < tolerance)


def _keyframe(left, right, rig: StereoRig, mask, s: Settings, depth: float = 1.5):
    """Corners of the left view that have a checked match in the right view: pixels (m, 2) and points (m, 3).
    `depth` is where the search in the right view starts (metres)."""
    import cv2
    found = cv2.goodFeaturesToTrack(left, s.corners, 0.005, max(6, left.shape[1] // 100), mask=mask, blockSize=7)
    if found is None:
        return np.zeros((0, 2)), np.zeros((0, 3))
    pts = found.reshape(-1, 2).astype(float)
    start = rig.project(np.column_stack([rig.left.rays(pts), np.ones(len(pts))]) * depth, 1)
    other, ok = _lk_checked(left, right, pts, start)
    X, epipolar, _ = rig.triangulate(pts, other)
    with np.errstate(all="ignore"):
        ok &= np.isfinite(X).all(1) & (epipolar < s.max_epipolar) & (X[:, 2] > 0.2) & (X[:, 2] < s.max_depth)
        ok &= rig.focal * rig.baseline / X[:, 2] >= s.min_disparity
    return pts[ok], X[ok]


def _solve(X: np.ndarray, uv: np.ndarray, rig: StereoRig, s: Settings):
    """Pose of the current left camera from keyframe points X and their pixels uv now: X_now = R X + t.
    Returns R, t, the indices that agree and their median reprojection error in pixels, or None."""
    import cv2
    xy = rig.left.rays(uv)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(X, np.float64), np.ascontiguousarray(xy, np.float64), np.eye(3), None,
            reprojectionError=s.ransac_px / rig.focal, iterationsCount=300, confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < s.min_inliers:
        return None
    inliers = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(X[inliers], np.float64),
                                      np.ascontiguousarray(xy[inliers], np.float64), np.eye(3), None, rvec, tvec)
    R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
    seen = _rotate(R, X[inliers]) + t
    error = np.linalg.norm(seen[:, :2] / seen[:, 2:3] - xy[inliers], axis=1) * rig.focal
    return R, t, inliers, float(np.median(error))


def _revisit(keys: list, current, left, mask, R_now, t_now, rig: StereoRig, s: Settings):
    """Try to solve this frame against an old keyframe that should see the same part of the scene.
    Returns (keyframe, ids, pixels, solution) for the best one that works, or None."""
    h, w = left.shape
    ranked = []
    for key in keys:
        if key is current or float(key["R"][:, 2] @ R_now[:, 2]) < 0.9:  # viewing directions more than 26° apart
            continue
        X = _rotate(R_now.T, _rotate(key["R"], key["X"]) + key["t"] - t_now)  # its points in this frame's camera
        front = X[:, 2] > 0.1
        uv = np.full((len(X), 2), -1.0)
        uv[front] = rig.left.pixels(X[front, :2] / X[front, 2:3])
        seen = front & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if mask is not None and seen.any():
            seen[seen] = mask[uv[seen, 1].astype(int), uv[seen, 0].astype(int)] > 0
        if seen.sum() >= max(s.min_points, 0.6 * key["n"]):
            ranked.append((int(seen.sum()), key, seen, uv))
    best = None
    for _, key, seen, uv in sorted(ranked, key=lambda r: -r[0])[:2]:
        ids = np.flatnonzero(seen)
        now, ok = _lk_checked(key["gray"], left, key["pts"][ids], uv[ids], levels=3, tolerance=1.5)
        if ok.sum() < s.min_inliers:
            continue
        solution = _solve(key["X"][ids[ok]], now[ok], rig, s)
        if solution and len(solution[2]) >= max(s.min_inliers + 5, s.refresh_share * key["n"]):
            if best is None or len(solution[2]) > len(best[3][2]):
                best = (key, ids[ok][solution[2]], now[ok][solution[2]], solution)
    return best


def track_frames(frames, rig: StereoRig, mask_for=None, settings: Settings | None = None, fps: float = 30.0,
                 progress: bool = False) -> HeadPose:
    """Visual odometry over `frames`, an iterable of (frame index, left RGB, right RGB) at one size. `rig` is the
    calibration at any size (it is scaled to the frames). `mask_for(i, w, h)` gives the usable area (see hand_masks)."""
    import cv2
    s = settings or Settings()
    out = {}
    key = None  # the current keyframe: gray, pts, X, R, t, index
    keys, revisits = [], 0
    ids = pts = prev = scaled = None
    R_now, t_now, part, scale = np.eye(3), np.zeros(3), 0, 1.0
    started = False
    for i, left_rgb, right_rgb in frames:
        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        h, w = left.shape
        if scaled is None:
            scaled = rig.scaled(w, h)
            scale = rig.width / w
        mask = mask_for(i, w, h) if mask_for else None
        solution, n_in, err = None, 0, np.nan
        if key is not None and len(ids) >= s.min_inliers:
            guess, ok = _lk(prev, left, pts)  # frame to frame: follows fast motion
            now, ok_key = _lk_checked(key["gray"], left, key["pts"][ids], guess, levels=2)  # against the keyframe: no drift
            ok &= ok_key & (np.linalg.norm(now - guess, axis=1) < 3.0)
            ok &= (now[:, 0] >= 0) & (now[:, 0] < w - 1) & (now[:, 1] >= 0) & (now[:, 1] < h - 1)
            if mask is not None and ok.any():
                inside = np.zeros(len(ok), bool)
                inside[ok] = mask[now[ok, 1].astype(int), now[ok, 0].astype(int)] > 0
                ok &= inside
            ids, pts = ids[ok], now[ok]
            if len(ids) >= s.min_inliers:
                solution = _solve(key["X"][ids], pts, scaled, s)
            if solution:
                R, t, inliers, err = solution
                ids, pts, n_in = ids[inliers], pts[inliers], len(inliers)
                R_now = key["R"] @ R.T
                t_now = key["t"] - R_now @ t
        solved = solution is not None
        refresh = not solved
        if solved:
            moved = float(np.median(np.linalg.norm(pts - key["pts"][ids], axis=1)))
            few = n_in < max(s.min_inliers + 5, s.refresh_share * key["n"])
            refresh = few or moved > s.refresh_shift * w
        if refresh and s.revisit and started and keys:
            old = _revisit(keys, key, left, mask, R_now, t_now, scaled, s)
            if old is not None and (not solved or len(old[1]) >= 1.2 * n_in):
                key, ids, pts, (R, t, _, err) = old
                R_now = key["R"] @ R.T
                t_now = key["t"] - R_now @ t
                n_in, solved, refresh = len(ids), True, False
                revisits += 1
        if refresh:
            depth = float(np.median(key["X"][:, 2])) if key is not None else 1.5
            new_pts, X = _keyframe(left, cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY), scaled, mask, s, depth)
            # a keyframe that is only short of points is replaced only by one with clearly more
            if len(new_pts) >= s.min_points and not (solved and few and moved <= s.refresh_shift * w
                                                     and len(new_pts) < 1.2 * n_in):
                if not solved and started:
                    part += 1  # the motion since the last solved frame is unknown
                    keys = []  # keyframes of an earlier part are in another frame of reference
                key = {"gray": left, "pts": new_pts, "X": X, "R": R_now.copy(), "t": t_now.copy(), "index": i,
                       "n": len(new_pts)}
                keys.append(key)
                if len(keys) > s.max_keyframes:
                    keys.pop(0)
                ids, pts = np.arange(len(new_pts)), new_pts
                if not solved:
                    solved, n_in, err = True, len(new_pts), 0.0  # the start of a part: its pose is the held one
                started = True
            elif not solved:
                key = None
        out[i] = (R_now.copy(), t_now.copy(), SOLVED if solved else LOST, part, key["index"] if key is not None else -1,
                  n_in, err * scale)
        prev = left
        if progress and i and i % 1000 == 0:
            print(f"  head pose: frame {i}", flush=True)
    n = max(out) + 1 if out else 0
    pose = HeadPose(fps, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.zeros(n, np.int8), np.zeros(n, np.int32),
                    np.full(n, -1, np.int32), np.zeros(n, np.int32), np.full(n, np.nan), settings=asdict(s),
                    revisits=revisits)
    last = None
    for i in range(n):
        if i in out:
            last = out[i]
            (pose.R[i], pose.t[i], pose.status[i], pose.part[i], pose.keyframe[i], pose.inliers[i], pose.error[i]) = last
        elif last is not None:  # a frame missing from the video: hold
            pose.R[i], pose.t[i], pose.part[i] = last[0], last[1], last[3]
    return pose


def track_head(video, rig: StereoRig, kp2d_left: np.ndarray | None = None, kp2d_size: tuple[int, int] | None = None,
               settings: Settings | None = None, progress: bool = True) -> HeadPose:
    """Head pose for every frame of a StereoVideo. `kp2d_left` (n, 2, 21, 3), in pixels of an image of `kp2d_size`,
    masks the hands; without it nothing is masked."""
    s = settings or Settings()
    mask_for = hand_masks(kp2d_left, *kp2d_size) if kp2d_left is not None else None
    return track_frames(video.frames(width=min(s.width, video.width)), rig, mask_for, s, video.fps, progress)


# ---------- gravity ----------

def accel_columns(names: list[str]) -> list[int] | None:
    """Positions of the accelerometer x, y, z among IMU column names (ax ay az, acc_x ..., linear_acceleration_x ...)."""
    found = []
    for axis in "xyz":
        hits = [k for k, n in enumerate(names) if n.lower() in (f"a{axis}", f"acc_{axis}", f"accel_{axis}",
                                                               f"linear_acceleration_{axis}")]
        if len(hits) != 1:
            return None
        found.append(hits[0])
    return found


def level(pose: HeadPose, t: np.ndarray, accel: np.ndarray, imu_to_camera: np.ndarray | None = None):
    """Turn the world frame so that y points along gravity, using an accelerometer.

    `t` (m,) seconds since the first frame and `accel` (m, 3) in m/s², in the IMU's axes; `imu_to_camera` is the
    rotation from those axes to the left camera's (identity if not given). At rest an accelerometer reads straight up,
    so each solved frame gives one estimate of "up" in the world frame; their median is taken, which averages out the
    head's own acceleration. The world keeps its origin, and z stays the first frame's viewing direction, levelled.

    Returns the levelled pose and a dict with the tilt check: the angle between the half-second estimates of "up"
    and the overall one. It is zero for a perfect pose and grows where the pose's roll or pitch has drifted."""
    Rc = np.eye(3) if imu_to_camera is None else np.asarray(imu_to_camera, float)[:3, :3]
    t, accel = np.asarray(t, float), np.asarray(accel, float)
    n = len(pose.status)
    ups = np.full((n, 3), np.nan)
    edges = np.searchsorted(t, (np.arange(n + 1) - 0.5) / pose.fps)
    for i in range(n):
        if pose.status[i] != SOLVED or edges[i + 1] <= edges[i]:
            continue
        a = Rc @ accel[edges[i]:edges[i + 1]].mean(0)
        size = np.linalg.norm(a)
        if 7.0 < size < 12.6:
            ups[i] = pose.R[i] @ (a / size)
    known = np.isfinite(ups[:, 0])
    if known.sum() < max(10, pose.fps):
        raise ValueError("too few frames with both a solved pose and accelerometer samples near 9.8 m/s²")
    up = np.median(ups[known], axis=0)
    up /= np.linalg.norm(up)
    y = -up
    z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    if np.linalg.norm(z) < 0.2:  # looking straight up or down at the first frame: use its x axis instead
        x = np.array([1.0, 0.0, 0.0]) - y[0] * y
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
    z /= np.linalg.norm(z)
    G = np.stack([np.cross(y, z), y, z])
    k = max(1, round(pose.fps / 2))
    tilts = []
    for a in range(0, n - k + 1, k):
        chunk = ups[a:a + k]
        chunk = chunk[np.isfinite(chunk[:, 0])]
        if len(chunk) >= k // 2:
            m = chunk.mean(0)
            tilts.append(np.degrees(np.arccos(np.clip(m @ up / np.linalg.norm(m), -1, 1))))
    check = {"frames_used": int(known.sum()),
             "tilt_deg_median": round(float(np.median(tilts)), 2) if tilts else None,
             "tilt_deg_p95": round(float(np.percentile(tilts, 95)), 2) if tilts else None,
             "first_camera_pitch_deg": round(float(np.degrees(np.arcsin(np.clip(-G[1, 2], -1, 1)))), 1)}
    levelled = replace(pose, R=np.einsum("ij,njk->nik", G, pose.R), t=np.einsum("ij,nj->ni", G, pose.t),
                       world="origin at the left camera of the first frame; y along gravity (down); z the first "
                             "frame's viewing direction, levelled", checks={**pose.checks, "gravity": check})
    return levelled, check


# ---------- distortion ----------

def with_radial(rig: StereoRig, k1: float) -> StereoRig:
    """`rig` with the radial term k1 on both lenses (pinhole model; the other distortion terms are kept)."""
    def lens(old):
        d = (tuple(old.distortion) + (0.0,) * 5)[:5]
        return replace(old, distortion=(float(k1), *d[1:]))
    return StereoRig(rig.width, rig.height, lens(rig.left), lens(rig.right), rig.R, rig.T, rig.calibrated_size)


def fit_distortion(frames_from, rig: StereoRig, mask_for=None, settings: Settings | None = None, fps: float = 30.0,
                   candidates=(-0.09, -0.075, -0.06, -0.045, -0.03, -0.015, 0.0, 0.015, 0.03)) -> dict:
    """One radial distortion term for both lenses, measured from the footage.

    For each candidate k1 the head is tracked over the same frames, and the median reprojection error of the scene
    points is noted: with the wrong lens model, points triangulated in one frame do not land where they are seen in
    the next. `frames_from()` must return a fresh iterable of (index, left RGB, right RGB) each time it is called
    (a stretch of 20 to 60 seconds with head motion is enough). Returns the k1 at the bottom of a parabola through the
    best candidate and its neighbours, and the whole curve. It refines a pinhole calibration; it does not replace one."""
    if rig.left.model != "pinhole":
        raise ValueError("fit_distortion works on the pinhole model")
    s = replace(settings or Settings(), revisit=False)
    curve = []
    for k1 in candidates:
        pose = track_frames(frames_from(), with_radial(rig, k1), mask_for, s, fps)
        solved = pose.status == SOLVED
        curve.append({"k1": k1, "reprojection_px": round(float(np.nanmedian(pose.error[solved])), 3) if solved.any() else None,
                      "keyframes": pose.summary()["keyframes"], "tracked": pose.summary()["tracked"]})
    usable = [c for c in curve if c["reprojection_px"] is not None]
    if len(usable) < 3:
        return {"k1": None, "curve": curve}
    b = min(range(len(usable)), key=lambda j: usable[j]["reprojection_px"])
    k1 = usable[b]["k1"]
    if 0 < b < len(usable) - 1:
        x = [usable[j]["k1"] for j in (b - 1, b, b + 1)]
        a, lin, _ = np.polyfit(x, [usable[j]["reprojection_px"] for j in (b - 1, b, b + 1)], 2)
        if a > 0:
            k1 = float(np.clip(-lin / (2 * a), x[0], x[2]))
    return {"k1": round(k1, 4), "at_edge_of_search": b in (0, len(usable) - 1), "curve": curve}


# ---------- comparison with another trajectory ----------

def compare(pose: HeadPose, R_ref: np.ndarray, t_ref: np.ndarray, seconds: tuple[float, ...] = (1.0, 10.0),
            ahead_m: float = 0.6) -> dict:
    """How this pose differs from a reference trajectory (camera → world, any world frame), frame by frame.

    Relative motion over windows is compared, which does not depend on the two world frames: for each window the
    camera's motion between its first and last frame is taken from both, and the difference is reported as a distance
    (mm) and an angle (degrees). `point_ahead` is the difference for a point `ahead_m` in front of the camera, where the
    hands are: a small turn and a small shift of the camera can cancel there, so it says more about hands placed in
    the world than the camera's position alone. `scale` is the median ratio of the distance moved in this pose to the distance moved in
    the reference, over windows where the reference moved more than 5 cm; it differs from 1 when the stereo baseline
    or the depth scale disagree."""
    n = min(len(pose.status), len(R_ref))
    ok = pose.status[:n] == SOLVED
    ahead = np.array([0.0, 0.0, ahead_m])
    out = {"frames": int(n), "compared": int(ok.sum())}
    for sec in seconds:
        k = max(1, round(sec * pose.fps))
        dist, ang, moved, ratio, near = [], [], [], [], []
        for a in range(0, n - k, max(1, k // 4)):
            b = a + k
            if not (ok[a] and ok[b]) or pose.part[a] != pose.part[b]:
                continue
            Ra = pose.R[a].T @ pose.R[b]
            ta = pose.R[a].T @ (pose.t[b] - pose.t[a])
            Rb = R_ref[a].T @ R_ref[b]
            tb = R_ref[a].T @ (t_ref[b] - t_ref[a])
            dist.append(np.linalg.norm(ta - tb) * 1000)
            near.append(np.linalg.norm(Ra @ ahead + ta - Rb @ ahead - tb) * 1000)
            ang.append(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
            moved.append(np.linalg.norm(tb) * 1000)
            if moved[-1] > 50:
                ratio.append(np.linalg.norm(ta) * 1000 / moved[-1])
        if dist:
            out[f"{sec:g}s"] = {"windows": len(dist), "point_ahead_mm_median": round(float(np.median(near)), 1),
                                "point_ahead_mm_p90": round(float(np.percentile(near, 90)), 1),
                                "position_mm_median": round(float(np.median(dist)), 1),
                                "position_mm_p90": round(float(np.percentile(dist, 90)), 1),
                                "rotation_deg_median": round(float(np.median(ang)), 2),
                                "rotation_deg_p90": round(float(np.percentile(ang, 90)), 2),
                                "reference_motion_mm_median": round(float(np.median(moved)), 1),
                                "scale": round(float(np.median(ratio)), 3) if len(ratio) >= 5 else None}
    return out
