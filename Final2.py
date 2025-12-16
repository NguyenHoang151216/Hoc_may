import os
import cv2
import json
import time
import argparse
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Missing ultralytics. Install with: pip install ultralytics") from e

CONFIG = {
    "yolo_weights": "D:\\Hocmay\\yolo_results_2\\arrow_detection\\weights\\best.pt",
    "yolo_weights1": "D:\\Hocmay\\yolo_results_3\\arrow_detection\\weights\\best.pt",
    "vehicle_classes": ["oto", "xemay"],
    "left_light_green_names": ["dentrai_xanh"],
    "left_light_red_names": ["dentrai_do"],
    "right_light_green_names": ["denphai_xanh"],
    "right_light_red_names": ["denphai_do"],
    "det_conf_threshold": 0.35,
    "light_debounce_frames": 2,
    "zone_polygon": [
        (350, 160), (610, 160), (700, 270), (250, 270)
    ],
    "track_state_expiry_frames": 250,
    "snapshot_margin": 0.2,
    "save_violation_images": True,
    "violation_image_prefix": "viol",
    "violation_json": "violations.json",
    "log_once_per_track": True,
    "viz_scale": 1.0,
    "show_debug_prints": True,
    "print_detections_first_n_frames": 5,
    "waiting_conf_threshold": 0.28,
    "require_center_well_inside": True,
    "well_inside_margin_px": 0,
}

Point = Tuple[float, float]

def point_in_polygon(pt: Point, polygon: List[Point]) -> bool:
    if pt is None:
        return False
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi)
        if intersect:
            inside = not inside
    return inside

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def seg_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on_seg(p,q,r):
        return min(p[0],r[0]) <= q[0] <= max(p[0],r[0]) and min(p[1],r[1]) <= q[1] <= max(p[1],r[1])
    o1 = orient(a1,a2,b1)
    o2 = orient(a1,a2,b2)
    o3 = orient(b1,b2,a1)
    o4 = orient(b1,b2,a2)
    if o1*o2 < 0 and o3*o4 < 0:
        return True
    if abs(o1) < 1e-9 and on_seg(a1,b1,a2): return True
    if abs(o2) < 1e-9 and on_seg(a1,b2,a2): return True
    if abs(o3) < 1e-9 and on_seg(b1,a1,b2): return True
    if abs(o4) < 1e-9 and on_seg(b1,a2,b2): return True
    return False

def center_of_bbox(bbox):
    return bbox_center(bbox)

def polygon_edges(polygon: List[Point]):
    return [(polygon[i], polygon[(i+1)%len(polygon)]) for i in range(len(polygon))]

def classify_polygon_edges(polygon: List[Point]):
    edges = polygon_edges(polygon)
    avg_y = [(a[0][1]+a[1][1])/2 for a in edges]
    avg_x = [(a[0][0]+a[1][0])/2 for a in edges]
    idx_bottom = int(np.argmax(avg_y))
    idx_top = int(np.argmin(avg_y))
    idx_left = int(np.argmin(avg_x))
    idx_right = int(np.argmax(avg_x))
    return {
        'bottom': edges[idx_bottom],
        'top': edges[idx_top],
        'left': edges[idx_left],
        'right': edges[idx_right],
        'all': edges
    }

def is_center_well_inside(center: Point, polygon: List[Point], margin_px: int = 10) -> bool:
    if center is None:
        return False
    if not point_in_polygon(center, polygon):
        return False
    def dist_point_to_seg(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return np.hypot(px-ax, py-ay)
        t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        projx = ax + t*dx
        projy = ay + t*dy
        return np.hypot(px-projx, py-projy)
    min_dist = min(dist_point_to_seg(center, e[0], e[1]) for e in polygon_edges(polygon))
    return min_dist >= margin_px

@dataclass
class TrackState:
    track_id: int
    waitingleft: bool = False
    waitingright: bool = False
    entry_left_is_red: bool = False
    entry_right_is_red: bool = False
    violation_logged: bool = False
    last_positions: deque = field(default_factory=lambda: deque(maxlen=12))
    last_seen_frame: int = 0

@dataclass
class LightState:
    left_red_count: int = 0
    left_green_count: int = 0
    right_red_count: int = 0
    right_green_count: int = 0
    left_is_red: bool = False
    right_is_red: bool = False

class DetectorTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_vehicle = YOLO(cfg["yolo_weights"]) if cfg.get("yolo_weights") else None
        self.model_light = YOLO(cfg["yolo_weights1"]) if cfg.get("yolo_weights1") else None
        try:
            if self.model_vehicle is not None:
                print("Vehicle model.names:", self.model_vehicle.names)
            if self.model_light is not None:
                print("Light model.names:", self.model_light.names)
        except Exception:
            print("Model.names: (unable to read names)")
        self.frame_index = 0

    def _extract_results(self, results, for_light: bool = False):
        outs = []
        if results is None:
            return outs
        if isinstance(results, (list, tuple)):
            res = results[0]
        else:
            res = results
        boxes = getattr(res, "boxes", None)
        names_map = getattr(res, "names", {}) or {}
        if boxes is None:
            return outs
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_idxs = boxes.cls.cpu().numpy().astype(int)
            ids = None
            if (not for_light) and hasattr(boxes, "id") and boxes.id is not None:
                try:
                    ids = boxes.id.cpu().numpy().astype(int)
                except Exception:
                    ids = None
            for i in range(len(xyxy)):
                box = xyxy[i]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                cname = names_map.get(int(cls_idxs[i]), str(int(cls_idxs[i]))).lower()
                outs.append({
                    "track_id": int(ids[i]) if ids is not None else None,
                    "class_name": cname,
                    "conf": float(confs[i]),
                    "bbox": [x1, y1, x2, y2],
                    "center": bbox_center([x1, y1, x2, y2]),
                    "is_light": for_light
                })
            return outs
        except Exception:
            try:
                for b in boxes:
                    try:
                        b_xy = b.xyxy[0].tolist()
                        box = [float(b_xy[0]), float(b_xy[1]), float(b_xy[2]), float(b_xy[3])]
                    except Exception:
                        continue
                    try:
                        conf = float(b.conf[0]) if hasattr(b.conf, "__len__") else float(b.conf)
                    except Exception:
                        conf = 0.0
                    try:
                        cls_idx = int(b.cls[0]) if hasattr(b.cls, "__len__") else int(b.cls)
                    except Exception:
                        cls_idx = -1
                    tid = None
                    if (not for_light) and hasattr(b, "id"):
                        try:
                            tid = int(b.id[0]) if hasattr(b.id, "__len__") else int(b.id)
                        except Exception:
                            tid = None
                    class_name = names_map.get(cls_idx, str(cls_idx)).lower()
                    outs.append({
                        "track_id": tid,
                        "class_name": class_name,
                        "conf": conf,
                        "bbox": box,
                        "center": center_of_bbox(box),
                        "is_light": for_light
                    })
                return outs
            except Exception:
                return outs

    def run_frame(self, frame):
        detections = []
        if self.model_vehicle is not None:
            try:
                results_vehicle = self.model_vehicle.track(source=frame, stream=False, persist=True, conf=self.cfg["det_conf_threshold"])                
            except Exception:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_vehicle = self.model_vehicle.track(source=img, stream=False, persist=True, conf=self.cfg["det_conf_threshold"])            
            dets_vehicle = self._extract_results(results_vehicle, for_light=False)
            detections.extend(dets_vehicle)

        if self.model_light is not None:
            try:
                try:
                    results_light = self.model_light(frame, stream=False, conf=self.cfg["det_conf_threshold"])
                except TypeError:
                    results_light = self.model_light.predict(frame, stream=False, conf=self.cfg["det_conf_threshold"]) 
            except Exception:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    results_light = self.model_light(img, stream=False, conf=self.cfg["det_conf_threshold"])                    
                except Exception:
                    try:
                        results_light = self.model_light.predict(img, stream=False, conf=self.cfg["det_conf_threshold"]) 
                    except Exception:
                        results_light = None
            dets_light = self._extract_results(results_light, for_light=True)
            detections.extend(dets_light)
        self.frame_index += 1
        return detections

class RuleEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.zone_polygon = cfg["zone_polygon"]
        self.edges = classify_polygon_edges(self.zone_polygon)
        self.track_states: Dict[int, TrackState] = {}
        self.light_state = LightState()
        self.frame_idx = 0

    def update_light_state_from_dets(self, detections: List[dict]):
        cfg = self.cfg
        left_red_seen = False
        left_green_seen = False
        right_red_seen = False
        right_green_seen = False
        for d in detections:
            cname = d["class_name"].lower()
            if any(sub in cname for sub in cfg["left_light_red_names"]):
                left_red_seen = True
            if any(sub in cname for sub in cfg["left_light_green_names"]):
                left_green_seen = True
            if any(sub in cname for sub in cfg["right_light_red_names"]):
                right_red_seen = True
            if any(sub in cname for sub in cfg["right_light_green_names"]):
                right_green_seen = True
        ls = self.light_state
        if left_red_seen:
            ls.left_red_count += 1
            ls.left_green_count = 0
        elif left_green_seen:
            ls.left_green_count += 1
            ls.left_red_count = 0
        else:
            ls.left_red_count = max(0, ls.left_red_count - 1)
            ls.left_green_count = max(0, ls.left_green_count - 1)
        if right_red_seen:
            ls.right_red_count += 1
            ls.right_green_count = 0
        elif right_green_seen:
            ls.right_green_count += 1
            ls.right_red_count = 0
        else:
            ls.right_red_count = max(0, ls.right_red_count - 1)
            ls.right_green_count = max(0, ls.right_green_count - 1)
        K = self.cfg["light_debounce_frames"]
        if ls.left_red_count >= K:
            ls.left_is_red = True
        elif ls.left_green_count >= K:
            ls.left_is_red = False
        if ls.right_red_count >= K:
            ls.right_is_red = True
        elif ls.right_green_count >= K:
            ls.right_is_red = False

    def ensure_track(self, track_id: int, frame_idx: int) -> TrackState:
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackState(track_id=track_id)
            if self.cfg.get("show_debug_prints"):
                print(f"[DEBUG] New track created: id={track_id} frame={frame_idx}")
        ts = self.track_states[track_id]
        ts.last_seen_frame = frame_idx
        return ts

    def cleanup_expired(self, frame_idx: int):
        expiry = self.cfg["track_state_expiry_frames"]
        to_remove = [tid for tid, s in self.track_states.items() if frame_idx - s.last_seen_frame > expiry]
        for tid in to_remove:
            if self.cfg.get("show_debug_prints"):
                print(f"[DEBUG] Removing expired track: {tid}")
            del self.track_states[tid]

    def process_frame(self, detections: List[dict], frame, frame_idx: int):
        violations = []
        self.frame_idx = frame_idx
        self.update_light_state_from_dets(detections)
        ls = self.light_state
        if self.cfg.get("show_debug_prints"):
            print(f"[LIGHT] frame={frame_idx} left_is_red={ls.left_is_red} right_is_red={ls.right_is_red} "
                  f"counts L(Red/Green)={ls.left_red_count}/{ls.left_green_count} R(Red/Green)={ls.right_red_count}/{ls.right_green_count}")
        vehicle_dets = []
        for d in detections:
            cname = d["class_name"].lower()
            if any(vc in cname for vc in self.cfg["vehicle_classes"]):
                vehicle_dets.append(d)
        for d in vehicle_dets:
            tid = d["track_id"]
            if tid is None:
                continue
            ts = self.ensure_track(tid, frame_idx)
            ts.last_positions.append(d["center"])
            prev_center = ts.last_positions[-2] if len(ts.last_positions) >= 2 else None
            cur_center = ts.last_positions[-1]
            prev_in = point_in_polygon(prev_center, self.zone_polygon) if prev_center is not None else False
            cur_in = point_in_polygon(cur_center, self.zone_polygon)
            if (not prev_in) and cur_in:
                should_assign = True
                if d.get("conf", 0.0) < max(self.cfg.get("waiting_conf_threshold", 0.30), self.cfg.get("det_conf_threshold", 0.35)):
                    should_assign = False
                    if self.cfg.get("show_debug_prints"):
                        print(f"[DEBUG] Skip assign waiting for id={tid} due to low conf {d.get('conf'):.2f}")
                well = False
                if not should_assign:
                    pass
                else:
                    if not self.cfg.get("require_center_well_inside", True):
                        well = True
                    else:
                        center_ok = is_center_well_inside(cur_center, self.zone_polygon, max(2, self.cfg.get("well_inside_margin_px", 2)))
                        x1c,y1c,x2c,y2c = d["bbox"]
                        corners = [(x1c,y1c),(x1c,y2c),(x2c,y1c),(x2c,y2c)]
                        corner_inside = any(point_in_polygon(c, self.zone_polygon) for c in corners)
                        moved_and_crossed = False
                        if prev_center is not None:
                            seg0, seg1 = prev_center, cur_center
                            for e in self.edges['all']:
                                if seg_intersect(seg0, seg1, e[0], e[1]):
                                    moved_and_crossed = True
                                    break
                        well = center_ok or corner_inside or moved_and_crossed
                    if not well:
                        should_assign = False
                        if self.cfg.get("show_debug_prints"):
                            print(f"[DEBUG] Skip assign waiting for id={tid} center/corners not well inside and not crossed: center={cur_center} bbox_corners={corners}")
                if should_assign:
                    ts.entry_left_is_red = ls.left_is_red
                    ts.entry_right_is_red = ls.right_is_red
                    if ls.left_is_red:
                        ts.waitingleft = True
                        if self.cfg.get("show_debug_prints"):
                            print(f"[DEBUG] Assigned WAIT_L to id={tid} at frame={frame_idx} (entry_left_is_red={ts.entry_left_is_red})")
                    if ls.right_is_red:
                        ts.waitingright = True
                        if self.cfg.get("show_debug_prints"):
                            print(f"[DEBUG] Assigned WAIT_R to id={tid} at frame={frame_idx} (entry_right_is_red={ts.entry_right_is_red})")
            if prev_center is not None:
                seg0, seg1 = prev_center, cur_center
                if seg_intersect(seg0, seg1, self.edges['left'][0], self.edges['left'][1]):
                    if ts.waitingleft and not ts.violation_logged:
                        if ts.entry_left_is_red:
                            rec = self._log_violation(ts, d, frame, frame_idx, reason="cross_left_while_waitingleft")
                            if rec:
                                violations.append(rec)
                        else:
                            ts.waitingleft = False
                            ts.entry_left_is_red = False
                            if self.cfg.get("show_debug_prints"):
                                print(f"[DEBUG] Cleared WAIT_L for id={tid} at frame={frame_idx} (left was green at entry)")
                if seg_intersect(seg0, seg1, self.edges['right'][0], self.edges['right'][1]):
                    continue
                if seg_intersect(seg0, seg1, self.edges['top'][0], self.edges['top'][1]):
                    if ts.waitingright and not ts.violation_logged:
                        if ts.entry_right_is_red:
                            rec = self._log_violation(ts, d, frame, frame_idx, reason="cross_top_while_waitingright")
                            if rec:
                                violations.append(rec)
                        else:
                            ts.waitingright = False
                            ts.entry_right_is_red = False
                            if self.cfg.get("show_debug_prints"):
                                print(f"[DEBUG] Cleared WAIT_R (top cross) for id={tid} at frame={frame_idx} (main was green at entry)")
                    if ts.waitingleft and not ts.violation_logged:
                        ts.waitingleft = False
                        ts.entry_left_is_red = False
                        if self.cfg.get("show_debug_prints"):
                            print(f"[DEBUG] Cleared WAIT_L (top cross) for id={tid} at frame={frame_idx} (vehicle went straight)")
            ts.last_seen_frame = frame_idx
        self.cleanup_expired(frame_idx)
        return violations

    def _log_violation(self, ts: TrackState, det: dict, frame, frame_idx: int, reason: str):
        if self.cfg["log_once_per_track"] and ts.violation_logged:
            return None
        ts.violation_logged = True
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1
        h = y2 - y1
        margin = self.cfg["snapshot_margin"]
        x1m = max(0, int(x1 - w * margin))
        y1m = max(0, int(y1 - h * margin))
        x2m = min(frame.shape[1], int(x2 + w * margin))
        y2m = min(frame.shape[0], int(y2 + h * margin))
        crop = frame[y1m:y2m, x1m:x2m].copy() if frame is not None else None
        ts_record = {
            "track_id": ts.track_id,
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "reason": reason,
            "bbox": [x1, y1, x2, y2],
            "crop_bbox": [x1m, y1m, x2m, y2m],
            "class_name": det["class_name"],
            "conf": det["conf"]
        }
        if self.cfg["save_violation_images"] and crop is not None:
            out_dir = os.path.join(self.cfg.get("output_dir", "./out"), "viol_images")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{self.cfg.get('violation_image_prefix','viol')}_t{ts.track_id}_f{frame_idx}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            ts_record["image_path"] = fpath
        if self.cfg.get("show_debug_prints"):
            print(f"[VIOL] id={ts.track_id} frame={frame_idx} reason={reason} saved={ts_record.get('image_path')}")
        return ts_record

# ---------------------------
# INTERACTIVE POLYGON DRAWING
# ---------------------------
class PolygonDrawer:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(self.display, (x, y), 5, (255, 255, 255), 1)
            if len(self.points) > 1:
                cv2.line(self.display, self.points[-2], self.points[-1], (0, 255, 0), 2)
            print(f"Point {len(self.points)} added: {(x, y)}")

    def draw_polygon(self):
        window_name = "Draw Polygon - Click to add points, SPACE to finish, R to reset"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "="*60)
        print("INTERACTIVE POLYGON DRAWING")
        print("="*60)
        print("Instructions:")
        print("  - Click to add polygon points (minimum 3 points)")
        print("  - Press SPACE when done")
        print("  - Press R to reset all points")
        print("  - Press Q to cancel and use default polygon")
        print("="*60 + "\n")

        while True:
            self.display = self.frame.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(self.display, pt, 5, (0, 255, 0), -1)
                cv2.circle(self.display, pt, 5, (255, 255, 255), 1)
                cv2.putText(self.display, str(i+1), (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for i in range(1, len(self.points)):
                cv2.line(self.display, self.points[i-1], self.points[i], (0, 255, 0), 2)
            
            info_text = f"Points: {len(self.points)} | Press SPACE to finish, R to reset, Q to cancel"
            cv2.putText(self.display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(50) & 0xFF

            if key == ord(' '):
                if len(self.points) >= 3:
                    cv2.line(self.display, self.points[-1], self.points[0], (0, 255, 0), 2)
                    print(f"\n✓ Polygon created with {len(self.points)} points!")
                    print(f"Polygon points: {self.points}\n")
                    cv2.imshow(window_name, self.display)
                    cv2.waitKey(1000)
                    break
                else:
                    print("✗ Need at least 3 points! Current: " + str(len(self.points)))

            elif key == ord('r') or key == ord('R'):
                self.points = []
                print("→ Reset: All points cleared")

            elif key == ord('q') or key == ord('Q'):
                self.points = []
                print("→ Cancelled! Using default polygon")
                break

        cv2.destroyWindow(window_name)
        return self.points

def np_int_points(poly):
    arr = np.array(poly, dtype=np.int32)
    return arr.reshape((-1,1,2))

def draw_polygon(img, poly, color=(0,255,255), thickness=2):
    pts = np_int_points(poly)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def get_box_color_and_thickness(d, engine: RuleEngine):
    normal_color = (255,255,255)
    light_color  = (0,200,255)
    waiting_color= (0,255,255)
    violation_color = (0,0,255)
    cname = d.get("class_name","").lower()
    tid = d.get("track_id", None)
    if any(vc in cname for vc in engine.cfg["vehicle_classes"]):
        if tid is not None and tid in engine.track_states:
            ts = engine.track_states[tid]
            if ts.violation_logged:
                return violation_color, 3
            if ts.waitingleft or ts.waitingright:
                return waiting_color, 2
        return normal_color, 1
    else:
        return light_color, 1

def process_video(input_video: str, output_dir: str, cfg_override: dict = None):
    cfg = dict(CONFIG)
    if cfg_override:
        cfg.update(cfg_override)
    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    dt = DetectorTracker(cfg)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from video")
    
    # Draw polygon interactively on first frame
    drawer = PolygonDrawer(first_frame)
    polygon_points = drawer.draw_polygon()
    
    # Update config with drawn polygon (or use default if cancelled)
    if polygon_points and len(polygon_points) >= 3:
        cfg["zone_polygon"] = polygon_points
        print(f"Using custom polygon with points: {polygon_points}")
    else:
        print(f"Using default polygon: {CONFIG['zone_polygon']}")
        cfg["zone_polygon"] = CONFIG["zone_polygon"]
    
    # Re-initialize RuleEngine with new polygon
    engine = RuleEngine(cfg)
    
    frame_idx = 0
    violations_all = []
    print_det_n = cfg.get("print_detections_first_n_frames", 5)

    # prepare video writer for visualization
    write_viz = True
    out_writer = None
    if write_viz:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        viz_path = os.path.join(output_dir, "out_viz.mp4")
        out_writer = cv2.VideoWriter(viz_path, fourcc, fps, (w, h))

    print("\nProcessing... (press 'q' in the display window to stop early)\n")
    try:
        # Process first frame
        detections = dt.run_frame(first_frame)
        if cfg.get("show_debug_prints") and frame_idx < print_det_n:
            print(f"[DETS] frame={frame_idx} ->", [(d['class_name'], round(d['conf'],2), d['track_id']) for d in detections])

        violations = engine.process_frame(detections, first_frame, frame_idx)
        for v in (violations or []):
            if v:
                violations_all.append(v)

        vis = first_frame.copy()
        draw_polygon(vis, cfg["zone_polygon"], color=(0,255,255), thickness=2)

        for d in detections:
            x1,y1,x2,y2 = map(int, d["bbox"])
            tid = d.get("track_id", None)
            cname = d.get("class_name","")
            color, thickness = get_box_color_and_thickness(d, engine)
            cv2.rectangle(vis, (x1,y1),(x2,y2), color, thickness)

        if out_writer is not None:
            out_writer.write(vis)

        frame_idx += 1
        
        # Process remaining frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = dt.run_frame(frame)
            if cfg.get("show_debug_prints") and frame_idx < print_det_n:
                print(f"[DETS] frame={frame_idx} ->", [(d['class_name'], round(d['conf'],2), d['track_id']) for d in detections])

            violations = engine.process_frame(detections, frame, frame_idx)
            for v in (violations or []):
                if v:
                    violations_all.append(v)

            vis = frame.copy()
            draw_polygon(vis, cfg["zone_polygon"], color=(0,255,255), thickness=2)

            for d in detections:
                x1,y1,x2,y2 = map(int, d["bbox"])
                tid = d.get("track_id", None)
                cname = d.get("class_name","")
                color, thickness = get_box_color_and_thickness(d, engine)
                cv2.rectangle(vis, (x1,y1),(x2,y2), color, thickness)
                label_parts = [cname]
                if tid is not None:
                    label_parts.append(f"ID{tid}")
                    ts = engine.track_states.get(tid)
                    if ts:
                        if ts.violation_logged:
                            label_parts.append("VIOL")
                        else:
                            waiting_flags = []
                            if ts.waitingleft:
                                waiting_flags.append("WAIT_L")
                            if ts.waitingright:
                                waiting_flags.append("WAIT_R")
                            if waiting_flags:
                                label_parts.append("&".join(waiting_flags))
                label = "|".join(label_parts)
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, max(0,y1-lh-8)), (x1 + lw + 6, y1), color, -1)
                cv2.putText(vis, label, (x1 + 3, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

                ts = engine.track_states.get(tid)
                if ts:
                    pts = list(ts.last_positions)
                    for i in range(1, len(pts)):
                        p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                        p2 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(vis, p1, p2, (255,255,0), 1)

            waiting_L_ids = [tid for tid,ts in engine.track_states.items() if ts.waitingleft]
            waiting_R_ids = [tid for tid,ts in engine.track_states.items() if ts.waitingright]
            viol_ids    = [tid for tid,ts in engine.track_states.items() if ts.violation_logged]
            cv2.putText(vis, f"WAIT_L: {waiting_L_ids}", (10, vis.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(vis, f"WAIT_R: {waiting_R_ids}", (10, vis.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(vis, f"VIOL: {viol_ids}", (10, vis.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            ls = engine.light_state
            left_status = "RED" if ls.left_is_red else "GREEN"
            right_status = "RED" if ls.right_is_red else "GREEN"
            tl_color = (0,0,255) if ls.left_is_red or ls.right_is_red else (0,255,0)
            cv2.putText(vis, f"L:{left_status}  R:{right_status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tl_color, 2)

            cv2.imshow("Traffic Violation System - Press q to quit", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested stop (q pressed).")
                break

            if out_writer is not None:
                out_writer.write(vis)

            frame_idx += 1

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()

    out_json = os.path.join(output_dir, cfg["violation_json"])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([v for v in violations_all if v is not None], f, ensure_ascii=False, indent=2)
    print(f"Done. Violations saved: {out_json}")
    return out_json

if __name__ == "__main__":
    input_video = r"D:\Hocmay\5.mp4"
    output_dir = r".\output"

    process_video(
        input_video=input_video,
        output_dir=output_dir,
        cfg_override={
            "yolo_weights": r"D:\\Hocmay\\yolo_results_2\\arrow_detection\\weights\\best.pt",
            "yolo_weights1": r"D:\\Hocmay\\yolo_results_3\\arrow_detection\\weights\\best.pt",
        }
    )
