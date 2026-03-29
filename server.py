import time
import threading
import subprocess
import numpy as np
import json
import yaml
import os
import cv2
from scipy.spatial import KDTree
from mcap_ros2.reader import read_ros2_messages

import foxglove
from foxglove.channels import CompressedImageChannel, CameraCalibrationChannel, SceneUpdateChannel, FrameTransformChannel, ImageAnnotationsChannel
from foxglove.messages import (
    CompressedImage, CameraCalibration, SceneUpdate, SceneEntity,
    CubePrimitive, LinePrimitive, LinePrimitiveLineType,
    Pose, Vector3, Quaternion, Color, Point3, Timestamp,
    FrameTransform, ImageAnnotations, PointsAnnotation, PointsAnnotationType, Point2,
)

# ---------------------------------------------------------------------------
# Camera extrinsics — camera_fl in base_link body frame
# Body frame: x forward, y left, z up (ROS REP-103)
# Camera frame: x right, y down, z forward (OpenCV convention)
# Tune these if the projection is off.
# ---------------------------------------------------------------------------
CAM_T_BODY = np.array([0.5, 0.3, 0.8])   # camera pos in body frame (fwd, left, up) metres

# Rotation: body → camera
#   body frame: x forward, y left, z up  (ROS REP-103)
#   camera frame: x right, y down, z forward  (OpenCV)
#     cam_x = -body_y,  cam_y = -body_z,  cam_z = body_x
_R_BASE = np.array([
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1,  0,  0],
], dtype=float)

# With _R_BASE only and camera at 0.8 m height, road points project naturally:
#   10 m ahead → v ≈ 683  (lower quarter)
#   30 m ahead → v ≈ 538  (just below centre)
#  100 m ahead → v ≈ 489  (near centre)
# No extra pitch needed — the height offset does the work.
# Increase PITCH_DEG slightly (positive = camera tilts down = road moves up in image).
_PITCH_DEG = 0.0
_YAW_DEG   = 0.0

def _Rx(deg):
    a = np.radians(deg)
    return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

CAM_R_BODY = _Rx(_PITCH_DEG) @ _Rz(_YAW_DEG) @ _R_BASE  # body → camera rotation


def rip_trajectory(mcap_path, topic):
    """Rips (x, y, z, yaw, v, gas, brake) from state_estimation topic."""
    print(f"Ripping trajectory from {mcap_path}...")
    traj = []
    try:
        with open(mcap_path, "rb") as f:
            for msg in read_ros2_messages(f):
                if msg.channel.topic == topic:
                    d = msg.ros_msg
                    traj.append([
                        float(d.x_m), float(d.y_m), float(d.z_m),
                        float(d.yaw_rad),
                        float(d.v_mps), float(d.gas), float(d.brake),
                    ])
        print(f"  {len(traj)} state messages.")
        return np.array(traj)
    except Exception as e:
        print(f"Error ripping trajectory: {e}")
        return None


def rip_camera_frames(mcap_path, topic):
    print(f"Ripping camera frames from {mcap_path}...")
    frames = []
    try:
        with open(mcap_path, "rb") as f:
            for msg in read_ros2_messages(f):
                if msg.channel.topic == topic:
                    frames.append(bytes(msg.ros_msg.data))
    except Exception as e:
        print(f"Error ripping camera frames: {e}")
    print(f"  {len(frames)} camera frames.")
    return frames


def load_calibration(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def make_timestamp(now_ns):
    return Timestamp(sec=int(now_ns // 1_000_000_000), nsec=int(now_ns % 1_000_000_000))


def project_trajectory_onto_frame(
    frame_bgr, world_pts, car_x, car_y, car_z, car_yaw, K, dist_coeffs,
    color=(0, 255, 0), thickness=3,
):
    """
    Projects world_pts (Nx3) onto the camera.
    If frame_bgr is None, returns the 2D points array for use as annotations.
    Otherwise draws a polyline on frame_bgr in-place and returns the frame.
    """
    if len(world_pts) == 0:
        return None if frame_bgr is None else frame_bgr

    cos_y, sin_y = np.cos(-car_yaw), np.sin(-car_yaw)
    dx = world_pts[:, 0] - car_x
    dy = world_pts[:, 1] - car_y
    dz = world_pts[:, 2] - car_z

    pts_body = np.column_stack([
        cos_y * dx - sin_y * dy,
        sin_y * dx + cos_y * dy,
        dz,
    ])

    pts_cam = (CAM_R_BODY @ (pts_body - CAM_T_BODY).T).T

    mask = pts_cam[:, 2] > 0.5
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return None if frame_bgr is None else frame_bgr

    pts_2d, _ = cv2.projectPoints(
        pts_cam.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K, dist_coeffs,
    )
    pts_2d = pts_2d.reshape(-1, 2)

    if frame_bgr is not None:
        h, w = frame_bgr.shape[:2]
        valid = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        pts_int = pts_2d[valid].astype(np.int32)
        if len(pts_int) >= 2:
            cv2.polylines(frame_bgr, [pts_int.reshape(-1, 1, 2)], isClosed=False,
                          color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return frame_bgr

    # Return raw 2D float points for annotation channel
    return pts_2d


def main():
    # -----------------------------------------------------------------------
    # Data Prep — trajectories
    # -----------------------------------------------------------------------
    FAST_NPY = "fast_laps_full.npy"
    GOOD_NPY  = "good_lap_full.npy"

    if not os.path.exists(FAST_NPY):
        data = rip_trajectory("hackathon_fast_laps.mcap", "/constructor0/state_estimation")
        if data is not None:
            np.save(FAST_NPY, data)

    if not os.path.exists(GOOD_NPY):
        data = rip_trajectory("hackathon_good_lap.mcap", "/constructor0/state_estimation")
        if data is not None:
            np.save(GOOD_NPY, data)

    try:
        fast_data = np.load(FAST_NPY)   # cols: x,y,z,yaw,v,gas,brake
        good_data = np.load(GOOD_NPY)
    except FileNotFoundError:
        print("ERROR: trajectory .npy files missing.")
        return

    good_tree = KDTree(good_data[:, :2])

    with open("yas_marina_bnd.json") as f:
        track_boundaries = json.load(f).get("boundaries", {})

    # -----------------------------------------------------------------------
    # Camera frames — from fast_laps
    # -----------------------------------------------------------------------
    camera_frames = rip_camera_frames(
        "hackathon_fast_laps.mcap",
        "/constructor0/sensor/camera_fl/compressed_image",
    )
    if not camera_frames:
        print("WARNING: no camera frames found in fast_laps MCAP.")

    # -----------------------------------------------------------------------
    # Camera calibration
    # -----------------------------------------------------------------------
    calib = load_calibration("intrinsics/camera_fl_info.yaml")
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.array(calib["distortion_coefficients"]["data"], dtype=np.float64)

    # -----------------------------------------------------------------------
    # Server
    # -----------------------------------------------------------------------
    foxglove.start_server(name="RacingCoach", host="0.0.0.0", port=8888)
    camera_chan      = CompressedImageChannel("/camera")
    camera_info_chan = CameraCalibrationChannel("/camera_info")
    scene_chan       = SceneUpdateChannel("/scene")
    tf_chan          = FrameTransformChannel("/tf")
    ann_chan         = ImageAnnotationsChannel("/camera/annotations")

    print("Server live at ws://localhost:8888")

    num_state  = len(fast_data)
    num_frames = len(camera_frames)

    # ── Precompute minimap ────────────────────────────────────────────────────
    MAP_SIZE   = 300   # pixels square (circle diameter)
    MAP_PAD    = 10    # padding inside the circle
    _all_xy    = np.vstack([good_data[:, :2], fast_data[:, :2]])
    _map_xmin, _map_ymin = _all_xy.min(axis=0)
    _map_xmax, _map_ymax = _all_xy.max(axis=0)
    _map_cx = (_map_xmin + _map_xmax) / 2.0   # world-space centre
    _map_cy = (_map_ymin + _map_ymax) / 2.0
    _map_r  = MAP_SIZE // 2 - MAP_PAD          # usable radius in pixels
    # Scale so the longest half-dimension fits within _map_r
    _half_w = (_map_xmax - _map_xmin) / 2.0
    _half_h = (_map_ymax - _map_ymin) / 2.0
    _map_scale = _map_r / max(_half_w, _half_h)

    def _world_to_map(x, y):
        px = int(MAP_SIZE // 2 + (x - _map_cx) * _map_scale)
        py = int(MAP_SIZE // 2 - (y - _map_cy) * _map_scale)  # flip y
        return px, py

    # Pre-draw static track lines onto a base minimap image
    _map_base = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    _gpts = np.array([_world_to_map(p[0], p[1]) for p in good_data[::4]], dtype=np.int32)
    cv2.polylines(_map_base, [_gpts], isClosed=True, color=(0, 180, 60), thickness=2, lineType=cv2.LINE_AA)
    _fpts = np.array([_world_to_map(p[0], p[1]) for p in fast_data[::4]], dtype=np.int32)
    cv2.polylines(_map_base, [_fpts], isClosed=True, color=(200, 100, 0), thickness=1, lineType=cv2.LINE_AA)

    # ── Precompute turn apexes from reference lap ─────────────────────────────
    def _angle_diff(a, b):
        d = (b - a + np.pi) % (2 * np.pi) - np.pi
        return d

    _gx, _gy = good_data[:, 0], good_data[:, 1]
    _gh = np.arctan2(np.gradient(_gy), np.gradient(_gx))          # heading at each point
    _gc = np.abs(_angle_diff(np.roll(_gh, 1), _gh))               # raw heading change per sample
    # Arc-length between samples for normalisation
    _gdx = np.diff(_gx, append=_gx[-1]); _gdy = np.diff(_gy, append=_gy[-1])
    _gds = np.hypot(_gdx, _gdy).clip(0.1)
    _gcurv = _gc / _gds                                            # curvature [rad/m]
    # Smooth curvature
    from scipy.ndimage import uniform_filter1d
    _gcurv_s = uniform_filter1d(_gcurv, size=15)
    # Cumulative distance along reference lap
    _gcum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(_gx), np.diff(_gy)))])

    # Find local maxima in smoothed curvature (turn apexes)
    from scipy.signal import find_peaks
    _apex_idxs, _ = find_peaks(_gcurv_s, height=0.004, distance=40)

    # Classify each apex and compute warning distance
    #   category: (label, colour_bgr, warn_dist_m, min_curv)
    _TURN_CATS = [
        ("HAIRPIN",   (0,  30, 255), 220, 0.045),
        ("SHARP",     (0,  80, 255), 170, 0.025),
        ("MEDIUM",    (0, 165, 255), 120, 0.012),
        ("BEND",      (0, 210, 255),  70, 0.004),
    ]
    def _classify_turn(curv):
        for label, color, warn_dist, min_c in _TURN_CATS:
            if curv >= min_c:
                return label, color, warn_dist
        return "BEND", (0, 210, 255), 70

    _turns = []   # list of dicts: idx, cum_dist, curv, label, color, warn_dist, ref_speed
    for idx in _apex_idxs:
        curv = float(_gcurv_s[idx])
        label, color, warn_dist = _classify_turn(curv)
        _turns.append({
            "idx":       int(idx),
            "cum_dist":  float(_gcum[idx]),
            "curv":      curv,
            "label":     label,
            "color":     color,
            "warn_dist": warn_dist,
            "ref_speed": float(good_data[idx, 4]),   # speed at apex in ref lap
        })
    print(f"Detected {len(_turns)} turns: "
          + ", ".join(f"{t['label']}@{t['cum_dist']:.0f}m" for t in _turns[:6]) + " ...")

    # Build a fast lookup: for a given track index, find the nearest upcoming turn
    _good_tree_1d = None   # will use linear search on cum_dist

    def _nearest_upcoming_turn(car_x, car_y, lookahead_m=300.0):
        """Return the closest upcoming turn within lookahead_m, or None."""
        # Find nearest point on reference lap
        _, near_idx = good_tree.query([car_x, car_y])
        car_dist = _gcum[near_idx]
        lap_len  = _gcum[-1]
        best = None
        best_ahead = float("inf")
        for t in _turns:
            if t["label"] != "SHARP":
                continue
            ahead = t["cum_dist"] - car_dist
            if ahead < 0:
                ahead += lap_len
            if ahead < lookahead_m and ahead < best_ahead:
                best_ahead = ahead
                best = (t, ahead)
        return best   # (turn_dict, dist_ahead_m) or None

    # ── Voice warning state ───────────────────────────────────────────────────
    _spoken_turns = set()   # turn idx already announced this lap
    _voice_lock   = threading.Lock()

    def _speak(text: str) -> None:
        def _run():
            with _voice_lock:
                subprocess.run(["spd-say", "-w", "-y", "female1", text], capture_output=True)
        threading.Thread(target=_run, daemon=True).start()

    # ── Brake EMA state ───────────────────────────────────────────────────────
    _brake_ema = 0.0
    _BRAKE_UP   = 0.35   # fast attack  (large value = snappier)
    _BRAKE_DOWN = 0.12   # slow release (small value = smoother fade)

    i = 0
    try:
        while True:
            now_ns = time.time_ns()
            ts = make_timestamp(now_ns)

            # Current car pose from fast_lap state estimation
            car_x, car_y, car_z, car_yaw = (
                fast_data[i, 0], fast_data[i, 1], fast_data[i, 2], fast_data[i, 3],
            )

            # Fresh minimap copy — only mark HAIRPIN and SHARP turns
            _map_base_frame = _map_base.copy()
            for t in _turns:
                if t["label"] == "SHARP":
                    tc = t["color"]
                    ap = _world_to_map(good_data[t["idx"], 0], good_data[t["idx"], 1])
                    cv2.circle(_map_base_frame, ap, 4, (int(tc[0]), int(tc[1]), int(tc[2])), -1)

            # ---------------------------------------------------------------
            # Camera image — annotated with good_lap projection
            # ---------------------------------------------------------------
            frame_idx = int(i / num_state * num_frames) % num_frames if num_frames else 0

            if camera_frames:
                raw_jpeg = camera_frames[frame_idx]
                frame = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)

                # Find good_lap points within 120 m, sorted by track order
                dists, idxs = good_tree.query(
                    [car_x, car_y], k=min(300, len(good_data))
                )
                in_range = idxs[dists < 120]
                in_range_sorted = np.sort(in_range)   # preserve track order for smooth polyline
                nearby = good_data[in_range_sorted]

                # Keep only points ahead of car (dot product with heading)
                fwd = np.array([np.cos(car_yaw), np.sin(car_yaw)])
                rel = nearby[:, :2] - np.array([car_x, car_y])
                ahead = rel @ fwd > 0
                nearby = nearby[ahead]

                pts3d = np.column_stack([
                    nearby[:, 0], nearby[:, 1],
                    np.full(len(nearby), car_z),
                ])

                # Project to 2D for annotations (don't bake into JPEG)
                pts2d = project_trajectory_onto_frame(
                    None, pts3d, car_x, car_y, car_z, car_yaw, K, dist,
                )

                # ── HUD overlay ──────────────────────────────────────────────
                speed_ms  = float(fast_data[i, 4])
                gas       = float(fast_data[i, 5])
                brake_raw = float(fast_data[i, 6])
                speed_kph = speed_ms * 3.6
                brake_raw_norm = min(1.0, max(0.0, brake_raw / 10000.0)) if brake_raw > 1.0 else min(1.0, max(0.0, brake_raw))
                # EMA smoothing — fast attack, slow release
                alpha = _BRAKE_UP if brake_raw_norm > _brake_ema else _BRAKE_DOWN
                _brake_ema = alpha * brake_raw_norm + (1.0 - alpha) * _brake_ema
                brake_norm = _brake_ema
                h, w = frame.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                bold = cv2.FONT_HERSHEY_DUPLEX

                # Compute turn state early so speedometer can use it
                result       = _nearest_upcoming_turn(car_x, car_y)
                in_warn_zone = (result and result[1] <= result[0]["warn_dist"]
                                and result[0]["label"] == "SHARP")

                # ── Digital speedometer — bottom right ───────────────────────
                spd_val  = int(speed_kph)
                spd_str  = f"{spd_val:3d}"   # right-aligned 3 digits
                # Panel dimensions
                sp_w, sp_h = 250, 90
                sp_x = w - sp_w - 10
                sp_y = h - sp_h - 10
                # Dark background with rounded feel
                ov_sp = frame.copy()
                cv2.rectangle(ov_sp, (sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h), (10, 10, 10), -1)
                cv2.addWeighted(ov_sp, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h), (50, 50, 50), 1)
                # "Ghost" digits — dim 888 behind to mimic 7-segment display
                (gw, _), _ = cv2.getTextSize("888", bold, 2.2, 3)
                gx = sp_x + sp_w - gw - 12
                cv2.putText(frame, "888", (gx, sp_y + sp_h - 14), bold, 2.2, (35, 35, 35), 3, cv2.LINE_AA)
                # Speed digit color stays white always
                spd_col = (255, 255, 255)
                cv2.putText(frame, spd_str, (gx, sp_y + sp_h - 14), bold, 2.2, spd_col, 3, cv2.LINE_AA)
                # km/h label — bottom-left of panel, always visible
                cv2.putText(frame, "km/h", (sp_x + 8, sp_y + sp_h - 10), font, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                # Dotted speed bar — 5-color gradient: blue→green→yellow→orange→red
                _spd_stops = [
                    (0.00, (255, 168, 58)),   # BGR: #3aa8ff blue
                    (0.25, (53, 255, 73)),    # BGR: #49ff35 green
                    (0.50, (38, 255, 243)),   # BGR: #f3ff26 yellow
                    (0.75, (34, 171, 255)),   # BGR: #ffab22 orange
                    (1.00, (32, 32, 253)),    # BGR: #fd2020 red
                ]
                def _lerp_color(stops, t):
                    t = max(0.0, min(1.0, t))
                    for i in range(len(stops) - 1):
                        t0, c0 = stops[i]
                        t1, c1 = stops[i + 1]
                        if t <= t1:
                            f = (t - t0) / (t1 - t0)
                            return tuple(int(c0[j] + f * (c1[j] - c0[j])) for j in range(3))
                    return stops[-1][1]
                max_spd = 280.0
                spd_frac = min(1.0, speed_kph / max_spd)
                bar_total = sp_w - 8
                seg_w, seg_gap = 8, 3
                seg_step = seg_w + seg_gap
                num_segs = bar_total // seg_step
                filled_segs = int(spd_frac * num_segs)
                bx0 = sp_x + 4
                by0, by1 = sp_y + sp_h - 6, sp_y + sp_h - 2
                for si in range(num_segs):
                    sx = bx0 + si * seg_step
                    if si < filled_segs:
                        seg_col = _lerp_color(_spd_stops, si / max(1, num_segs - 1))
                        cv2.rectangle(frame, (sx, by0), (sx + seg_w, by1), seg_col, -1)
                    else:
                        cv2.rectangle(frame, (sx, by0), (sx + seg_w, by1), (40, 40, 40), -1)

                # ── Top-right: symmetric throttle | turn warning | brake ───────
                panel_w, panel_h = 280, 160
                px = w - panel_w - 10
                py = 10
                ov2 = frame.copy()
                cv2.rectangle(ov2, (px, py), (px + panel_w, py + panel_h), (0,0,0), -1)
                cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)

                bar_w   = 34
                bar_h   = 100
                bar_top = py + 42
                gap     = 16

                # Turn warning label (centre of panel)

                # Voice warning — fire once per turn per lap
                if in_warn_zone:
                    t_key = result[0]["idx"]
                    if t_key not in _spoken_turns:
                        _spoken_turns.add(t_key)
                        label_v = result[0]["label"].lower()
                        dist_rounded = int(round(result[1] / 10) * 10)
                        _speak(f"{label_v} turn in {dist_rounded} meters")
                if in_warn_zone:
                    turn, dist_ahead = result
                    tc  = turn["color"]
                    lbl = f"{turn['label']}  {dist_ahead:.0f}m"
                    (lw, _), _ = cv2.getTextSize(lbl, bold, 0.7, 1)
                    cv2.putText(frame, lbl, (px + panel_w//2 - lw//2, py + 28),
                                bold, 0.7, (int(tc[0]), int(tc[1]), int(tc[2])), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "INPUTS", (px + panel_w//2 - 32, py + 26),
                                font, 0.55, (140,140,140), 1, cv2.LINE_AA)

                # Throttle bar — left, always green
                tx = px + gap
                cv2.rectangle(frame, (tx, bar_top), (tx + bar_w, bar_top + bar_h), (30,30,30), -1)
                fill_t = int(bar_h * min(1.0, max(0.0, gas)))
                cv2.rectangle(frame, (tx, bar_top + bar_h - fill_t), (tx + bar_w, bar_top + bar_h), (0, 210, 60), -1)
                cv2.rectangle(frame, (tx, bar_top), (tx + bar_w, bar_top + bar_h), (90,90,90), 1)
                cv2.putText(frame, "GAS", (tx + 4, bar_top + bar_h + 18), font, 0.55, (0,210,60), 1, cv2.LINE_AA)

                # Brake bar — right, red→orange→green
                bx = px + panel_w - gap - bar_w
                if brake_norm < 0.15:
                    bcol = (0, 40, 220)
                elif brake_norm < 0.5:
                    bcol = (0, 140, 255)
                else:
                    bcol = (0, 210, 60)
                cv2.rectangle(frame, (bx, bar_top), (bx + bar_w, bar_top + bar_h), (30,30,30), -1)
                fill_b = int(bar_h * brake_norm)
                cv2.rectangle(frame, (bx, bar_top + bar_h - fill_b), (bx + bar_w, bar_top + bar_h), bcol, -1)
                cv2.rectangle(frame, (bx, bar_top), (bx + bar_w, bar_top + bar_h), (90,90,90), 1)
                cv2.putText(frame, "BRAKE", (bx + 2, bar_top + bar_h + 18), font, 0.55,
                            (int(bcol[0]),int(bcol[1]),int(bcol[2])), 1, cv2.LINE_AA)

                # ── Minimap — bottom left (circular, rotates with car heading) ─
                mx, my = 14, h - MAP_SIZE - 14
                cx_m = cy_m = MAP_SIZE // 2
                r_m = MAP_SIZE // 2 - 2

                # Rotate map so car always points up, centred on car position
                cp = _world_to_map(car_x, car_y)
                rot_deg = 90.0 - np.degrees(car_yaw)
                # Rotate around the car's pixel position, then translate it to the image centre
                rot_mat = cv2.getRotationMatrix2D((float(cp[0]), float(cp[1])), rot_deg, 1.0)
                rot_mat[0, 2] += cx_m - cp[0]
                rot_mat[1, 2] += cy_m - cp[1]
                rotated = cv2.warpAffine(_map_base_frame, rot_mat, (MAP_SIZE, MAP_SIZE),
                                         flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

                # Car dot always at centre, heading arrow pointing up
                cv2.circle(rotated, (cx_m, cy_m), 5, (0, 220, 255), -1)
                cv2.line(rotated, (cx_m, cy_m), (cx_m, cy_m - 14), (0, 220, 255), 2)

                # Circular mask
                circ_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
                cv2.circle(circ_mask, (cx_m, cy_m), r_m, 255, -1)
                roi = frame[my:my + MAP_SIZE, mx:mx + MAP_SIZE].copy()
                blended = cv2.addWeighted(rotated, 0.82, roi, 0.18, 0)
                mask3 = circ_mask[:, :, np.newaxis] / 255.0
                composited = (blended * mask3 + roi * (1 - mask3)).astype(np.uint8)
                cv2.circle(composited, (cx_m, cy_m), r_m, (100, 100, 100), 2)
                frame[my:my + MAP_SIZE, mx:mx + MAP_SIZE] = composited

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buf.tobytes()

                # Publish as Foxglove image annotations clipped to image bounds
                if pts2d is not None and len(pts2d) >= 2:
                    img_h, img_w = frame.shape[:2]
                    valid = (
                        (pts2d[:, 0] >= 0) & (pts2d[:, 0] < img_w) &
                        (pts2d[:, 1] >= 0) & (pts2d[:, 1] < img_h)
                    )
                    pts2d_valid = pts2d[valid]
                    if len(pts2d_valid) >= 2:
                        # Remove points that fall over the minimap (bottom-left corner)
                        mm_x1, mm_y1 = 14, img_h - MAP_SIZE - 14
                        mm_x2, mm_y2 = mm_x1 + MAP_SIZE, mm_y1 + MAP_SIZE
                        not_over_map = ~(
                            (pts2d_valid[:, 0] >= mm_x1) & (pts2d_valid[:, 0] <= mm_x2) &
                            (pts2d_valid[:, 1] >= mm_y1) & (pts2d_valid[:, 1] <= mm_y2)
                        )
                        pts2d_valid = pts2d_valid[not_over_map]
                        # Dotted line: sample every 4th point so dots have visible gaps
                        dot_pts = pts2d_valid[::4]
                        # Dot color matches brake state: green → orange → red
                        if brake_norm < 0.15:
                            dot_r, dot_g, dot_b = 0.0, 1.0, 0.3   # green
                        elif brake_norm < 0.5:
                            dot_r, dot_g, dot_b = 1.0, 0.55, 0.0  # orange
                        else:
                            dot_r, dot_g, dot_b = 1.0, 0.1, 0.1   # red
                        ann_chan.log(ImageAnnotations(
                            circles=[],
                            texts=[],
                            points=[
                                PointsAnnotation(
                                    timestamp=ts,
                                    type=PointsAnnotationType.Points,
                                    points=[Point2(x=float(p[0]), y=float(p[1])) for p in dot_pts],
                                    outline_color=Color(r=dot_r, g=dot_g, b=dot_b, a=1.0),
                                    fill_color=Color(r=dot_r, g=dot_g, b=dot_b, a=1.0),
                                    outline_colors=[],
                                    thickness=14.0,
                                )
                            ],
                        ), log_time=now_ns)
            else:
                noise = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                cv2.putText(noise, "NO CAMERA", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                _, buf = cv2.imencode(".jpg", noise)
                img_bytes = buf.tobytes()

            camera_chan.log(
                CompressedImage(timestamp=ts, frame_id="camera_fl", data=img_bytes, format="jpeg"),
                log_time=now_ns,
            )

            # ---------------------------------------------------------------
            # Camera calibration
            # ---------------------------------------------------------------
            camera_info_chan.log(
                CameraCalibration(
                    timestamp=ts,
                    frame_id=calib["camera_name"],
                    width=calib["image_width"],
                    height=calib["image_height"],
                    distortion_model=calib["distortion_model"],
                    D=calib["distortion_coefficients"]["data"],
                    K=calib["camera_matrix"]["data"],
                    R=calib["rectification_matrix"]["data"],
                    P=calib["projection_matrix"]["data"],
                ),
                log_time=now_ns,
            )

            # ---------------------------------------------------------------
            # 3D Scene
            # ---------------------------------------------------------------
            entities = [
                SceneEntity(
                    id="fast_car", frame_id="map", timestamp=ts,
                    cubes=[CubePrimitive(
                        pose=Pose(
                            position=Vector3(x=car_x, y=car_y, z=car_z + 0.5),
                            orientation=Quaternion(w=np.cos(car_yaw/2), x=0, y=0, z=np.sin(car_yaw/2)),
                        ),
                        size=Vector3(x=4, y=2, z=1),
                        color=Color(r=1, g=0.6, b=0, a=1),
                    )],
                ),
            ]

            # Good lap line — green
            entities.append(SceneEntity(
                id="good_lap_line", frame_id="map",
                lines=[LinePrimitive(
                    type=LinePrimitiveLineType.LineStrip,
                    thickness=0.4,
                    color=Color(r=0, g=1, b=0.3, a=0.9),
                    points=[Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in good_data[::5]],
                )],
            ))

            # Fast lap line — blue
            entities.append(SceneEntity(
                id="fast_lap_line", frame_id="map",
                lines=[LinePrimitive(
                    type=LinePrimitiveLineType.LineStrip,
                    thickness=0.4,
                    color=Color(r=0.0, g=0.4, b=1.0, a=0.8),
                    points=[Point3(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.3) for p in fast_data[::5]],
                )],
            ))

            for name, pts in track_boundaries.items():
                entities.append(SceneEntity(
                    id=f"line_{name}", frame_id="map",
                    lines=[LinePrimitive(
                        type=LinePrimitiveLineType.LineStrip,
                        thickness=0.2,
                        color=Color(r=0.6, g=0.6, b=0.6, a=1),
                        points=[Point3(x=float(p[0]), y=float(p[1]), z=0) for p in pts],
                    )],
                ))

            scene_chan.log(SceneUpdate(entities=entities), log_time=now_ns)

            # Publish map as root frame
            tf_chan.log(FrameTransform(
                timestamp=ts,
                parent_frame_id="",
                child_frame_id="map",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ), log_time=now_ns)

            # Publish car frame (map → base_link) so 3D panel can follow
            half_yaw = float(fast_data[i, 3]) / 2.0
            tf_chan.log(FrameTransform(
                timestamp=ts,
                parent_frame_id="map",
                child_frame_id="base_link",
                translation=Vector3(x=float(fast_data[i, 0]), y=float(fast_data[i, 1]), z=float(fast_data[i, 2])),
                rotation=Quaternion(x=0.0, y=0.0, z=np.sin(half_yaw), w=np.cos(half_yaw)),
            ), log_time=now_ns)

            i += 1
            if i >= num_state:
                i = 0
                _spoken_turns.clear()   # new lap — allow warnings again
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
