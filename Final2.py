import os
import cv2
import json
import time
import argparse
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from tkinter import ttk  # For scrollbar support
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Missing ultralytics. Install with: pip install ultralytics") from e

CONFIG = {
    "yolo_weights": r".\phuongtien\arrow_detection\weights\best.pt",
    "yolo_weights1": r".\yolo_results_3\arrow_detection\weights\best.pt",
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
                            rec = self._log_violation(
                                ts,
                                d,
                                frame,
                                frame_idx,
                                reason="cross_left_while_waitingleft"
                            )

                            if rec:
                                violations.append(rec)

                                # ===== BÁO REALTIME CHO GUI =====
                                if hasattr(self, "on_violation") and self.on_violation:
                                    self.on_violation(rec)

                        else:
                            ts.waitingleft = False
                            ts.entry_left_is_red = False
                            if self.cfg.get("show_debug_prints"):
                                print(f"[DEBUG] Cleared WAIT_L for id={tid} at frame={frame_idx} (left was green at entry)")
                if seg_intersect(seg0, seg1, self.edges['right'][0], self.edges['right'][1]):
                    continue
                if seg_intersect(seg0, seg1, self.edges['top'][0], self.edges['top'][1]):
                    if ts.waitingright and not ts.violation_logged:
                        if ts.entry_left_is_red:
                            rec = self._log_violation(
                                ts,
                                d,
                                frame,
                                frame_idx,
                                reason="cross_left_while_waitingleft"
                            )

                            if rec:
                                violations.append(rec)

                                # ===== BÁO REALTIME CHO GUI =====
                                if hasattr(self, "on_violation") and self.on_violation:
                                    self.on_violation(rec)

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
        # ===== CHẶN GHI NHIỀU LẦN CHO CÙNG 1 TRACK =====
        if ts.violation_logged:
            return None

        # ===== ĐÁNH DẤU ĐÃ VI PHẠM NGAY LẬP TỨC =====
        ts.violation_logged = True

        # ===== TẠO RECORD VI PHẠM =====
        ts_record = {
            "track_id": ts.track_id,
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "reason": reason,
            "bbox": det["bbox"],
            "class_name": det["class_name"],
            "conf": det["conf"],
            "image_path": None
        }

        # ===== LƯU ẢNH VI PHẠM (CHỈ 1 LẦN) =====
        if self.cfg.get("save_violation_images", True):
            os.makedirs("output", exist_ok=True)

            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[y1:y2, x1:x2]

            filename = f"output/track_{ts.track_id}_{int(time.time())}.jpg"

            if crop.size > 0:
                cv2.imwrite(filename, crop)
                ts_record["image_path"] = filename

        if self.cfg.get("show_debug_prints"):
            print(f"[VIOL] id={ts.track_id} frame={frame_idx} reason={reason}")

        return ts_record




class TrafficViolationApp:



    def clear_polygon(self):
        # Xóa đường polygon
        for l in self.polygon_lines:
            self.video_canvas.delete(l)

        # Xóa chấm đỏ
        for d in self.polygon_dots:
            self.video_canvas.delete(d)

        self.polygon_lines.clear()
        self.polygon_dots.clear()
        self.polygon_points.clear()



    def canvas_to_frame(self, x, y):
        if self.current_frame is None:
            return None

        fh, fw = self.current_frame.shape[:2]
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()

        scale_x = fw / cw
        scale_y = fh / ch

        xf = int(x * scale_x)
        yf = int(y * scale_y)
        return (xf, yf)


    def frame_to_canvas(self, x, y):
        if self.current_frame is None:
            return None

        fh, fw = self.current_frame.shape[:2]
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()

        scale_x = cw / fw
        scale_y = ch / fh

        xc = int(x * scale_x)
        yc = int(y * scale_y)
        return (xc, yc)


    def on_canvas_click(self, event):
        if not self.drawing_enabled:
            return

        if len(self.polygon_points) >= 4:
            return

        # ===== 1. Lấy tọa độ click trên CANVAS =====
        x_canvas, y_canvas = event.x, event.y

        # ===== 2. Convert CANVAS → FRAME =====
        pt_frame = self.canvas_to_frame(x_canvas, y_canvas)
        if pt_frame is None:
            return

        # ===== 3. LƯU polygon theo FRAME (RẤT QUAN TRỌNG) =====
        self.polygon_points.append(pt_frame)

        # ===== 4. Convert ngược FRAME → CANVAS để VẼ =====
        pt_canvas = self.frame_to_canvas(pt_frame[0], pt_frame[1])
        x, y = pt_canvas

        # ===== 5. VẼ ĐIỂM =====
        r = 4
        dot = self.video_canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill="red", outline="white"
        )
        self.polygon_dots.append(dot)


        # ===== 6. VẼ CẠNH =====
        if len(self.polygon_points) > 1:
            x1f, y1f = self.polygon_points[-2]
            x1c, y1c = self.frame_to_canvas(x1f, y1f)

            line = self.video_canvas.create_line(
                x1c, y1c, x, y, fill="yellow", width=2
            )
            self.polygon_lines.append(line)

        # ===== 7. ĐÓNG POLYGON =====
        if len(self.polygon_points) == 4:
            x0f, y0f = self.polygon_points[0]
            x0c, y0c = self.frame_to_canvas(x0f, y0f)

            line = self.video_canvas.create_line(
                x, y, x0c, y0c, fill="yellow", width=2
            )
            self.polygon_lines.append(line)

    def _open_full_image(self, image_path):
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(image_path))

        img = Image.open(image_path)

        screen_w = win.winfo_screenwidth() - 100
        screen_h = win.winfo_screenheight() - 100

        w, h = img.size
        scale = min(screen_w / w, screen_h / h)

        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)

        lbl = tk.Label(win, image=tk_img)
        lbl.image = tk_img
        lbl.pack(expand=True)


    def enable_draw(self):
        self.clear_polygon()
        self.drawing_enabled = True
        self.btn_draw.pack(side=tk.LEFT, padx=5)
        self.btn_confirm.pack(side=tk.LEFT, padx=5)

    def on_polygon_confirmed(self):
        self.state = "POLYGON_CONFIRMED"
        self.drawing_enabled = False

        self.btn_draw.pack_forget()
        self.btn_confirm.pack_forget()
        self.btn_skip.pack_forget()

        self.process_button.config(state=tk.NORMAL)

    
    
    def draw_polygon_on_canvas(self, points):
        self.clear_polygon()
        self.polygon_points = points

        for i in range(len(points)):
            # ===== FRAME → CANVAS =====
            x1f, y1f = points[i]
            x2f, y2f = points[(i + 1) % len(points)]

            x1c, y1c = self.frame_to_canvas(x1f, y1f)
            x2c, y2c = self.frame_to_canvas(x2f, y2f)

            line = self.video_canvas.create_line(
                x1c, y1c, x2c, y2c,
                fill="yellow", width=2
            )
            self.polygon_lines.append(line)


    def _build_gallery_window(self, image_paths):
        win = tk.Toplevel(self.root)
        win.title("Ảnh vi phạm")
        win.geometry("800x600")

        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # giữ reference ảnh (RẤT QUAN TRỌNG)
        self.gallery_thumbs = []

        cols = 4
        size = (180, 120)

        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                img.thumbnail(size)
                tk_img = ImageTk.PhotoImage(img)

                lbl = tk.Label(scroll_frame, image=tk_img, cursor="hand2", bd=2, relief=tk.RIDGE)
                lbl.image = tk_img
                self.gallery_thumbs.append(tk_img)

                lbl.grid(row=idx // cols, column=idx % cols, padx=10, pady=10)
                lbl.bind("<Button-1>", lambda e, p=img_path: self._open_full_image(p))

            except Exception as e:
                print("Không load được ảnh:", img_path, e)


    def confirm_polygon(self):
        if len(self.polygon_points) != 4:
            messagebox.showerror("Lỗi", "Cần đúng 4 điểm")
            return

        from db.repository import PolygonRepository
        repo = PolygonRepository()
        repo.save_polygon(self.video_path, self.polygon_points)

        self.on_polygon_confirmed()


    def show_frame_on_canvas(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        img = img.resize((canvas_w, canvas_h))
        self.tk_img = ImageTk.PhotoImage(img)

        self.video_canvas.delete("all")
        self.canvas_image_id = self.video_canvas.create_image(
            0, 0, anchor=tk.NW, image=self.tk_img
        )


    def open_violation_gallery(self):
        output_dir = r".\output"

        if not os.path.exists(output_dir):
            messagebox.showerror("Lỗi", "Chưa có thư mục output")
            return

        image_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not image_files:
            messagebox.showinfo("Thông báo", "Chưa có ảnh vi phạm")
            return

        self._build_gallery_window(image_files)

    def show_polygon_options(self):
        self.btn_draw.pack(side=tk.LEFT, padx=5)
        self.btn_skip.pack(side=tk.LEFT, padx=5)

    def show_draw_button(self):
        self.btn_draw.pack(side=tk.LEFT, padx=5)

    def run_model(self):
        if self.state != "POLYGON_CONFIRMED":
            messagebox.showerror("Lỗi", "Chưa xác nhận polygon")
            return
        
        self.state = "PROCESSING"

        self.cap = cv2.VideoCapture(self.video_path)

        cfg = dict(CONFIG)
        cfg["zone_polygon"] = self.polygon_points

        self.detector = DetectorTracker(cfg)
        self.engine = RuleEngine(cfg)
        self.engine.on_violation = self._on_violation

        self.frame_idx = 0
        self.running = True

        self.process_next_frame()

    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Violation Detection")
        self.root.geometry("600x400")  # Adjusted size since image display is removed
        # Canvas hiển thị video
        self.video_canvas = tk.Canvas(root, width=800, height=450, bg="black")
        self.video_canvas.pack(padx=10, pady=10)

        
        self.canvas_image_id = None
        self.current_frame = None
        self.polygon_points = []
        self.polygon_lines = []
        self.polygon_dots = []   # <-- THÊM DÒNG NÀY

        self.has_polygon_db = False

        self.video_path = None
        self.violations = []  # To store detected violations

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)

        self.btn_draw = tk.Button(self.btn_frame, text="Vẽ polygon", command=self.enable_draw)
        self.btn_skip = tk.Button(
            self.btn_frame,
            text="Không vẽ lại",
            command=self.on_polygon_confirmed
        )

#
        self.video_canvas.bind("<Button-1>", self.on_canvas_click)
        self.drawing_enabled = False

        self.btn_confirm = tk.Button(
            self.btn_frame,
            text="Xác nhận polygon",
            command=self.confirm_polygon
        )

        self.cap = None
        self.detector = None
        self.engine = None
        self.frame_idx = 0
        self.running = False

        # UI Elements
        self.label = tk.Label(root, text="Traffic Violation Detection System", font=("Arial", 14))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Video", command=self.select_video, width=20)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(
            self.btn_frame,
            text="Process Video",
            state=tk.DISABLED,
            command=self.run_model
        )

        self.btn_open_images = tk.Button(
            self.btn_frame,
            text="Mở ảnh vi phạm",
            command=self.open_violation_gallery
        )
        self.btn_open_images.pack(side=tk.LEFT, padx=5)


        self.process_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Waiting for input", font=("Arial", 10))
        self.status_label.pack(pady=10)

        # Frame for violations list
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Listbox for violations
        self.violations_label = tk.Label(self.main_frame, text="Detected Violations", font=("Arial", 12))
        self.violations_label.pack(side=tk.LEFT, anchor="n", padx=10)

        self.violations_listbox = tk.Listbox(self.main_frame, width=40, height=20)
        self.violations_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Add a scrollbar to the listbox
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.violations_listbox.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.violations_listbox.config(yscrollcommand=self.scrollbar.set)
        # Make violations clickable: double-click to open saved image (if available)
        self.violations_listbox.bind("<Double-Button-1>", self.on_violation_double_click)

        # ===== STATE =====
        self.state = "INIT"   # INIT | VIDEO_SELECTED | POLYGON_CONFIRMED | PROCESSING

    def select_video(self):

        # ===== STOP ANY RUNNING VIDEO =====
        if self.running:
            self.running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # reset engine
        self.detector = None
        self.engine = None
        self.frame_idx = 0

        # reset UI state
        self.state = "INIT"
        self.process_button.config(state=tk.DISABLED)

        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )


        # ===== RESET FOR NEW VIDEO =====
        self.polygon_points = []
        self.polygon_lines = []
        self.clear_polygon()

        self.has_polygon_db = False
        self.state = "INIT"

        self.process_button.config(state=tk.DISABLED)

        if not file_path:
            return

        self.video_path = file_path


        # 1️⃣ Open video
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không đọc được frame đầu")
            return

        # 2️⃣ Hiển thị frame đầu lên Canvas
        self.current_frame = frame
        self.show_frame_on_canvas(frame)

        from db.repository import PolygonRepository
        repo = PolygonRepository()

        polygon = repo.get_polygon_by_video(self.video_path)

        self.state = "VIDEO_SELECTED"
        self.process_button.config(state=tk.DISABLED)
        self.btn_draw.pack_forget()
        self.btn_skip.pack_forget()

        if polygon:
            self.draw_polygon_on_canvas(polygon)

            answer = messagebox.askyesno(
                "Polygon đã tồn tại",
                "Video này đã có polygon.\nBạn có muốn vẽ lại polygon không?"
            )

            if answer:
                self.enable_draw()
            else:
                self.polygon_points = polygon
                self.on_polygon_confirmed()


        else:
            
            self.enable_draw()

            

    

    def _on_violation(self, viol: dict):
        # This may be called from the processing thread; schedule update on main thread
        try:
            self.root.after(0, lambda: self._add_violation(viol))
        except Exception:
            # fallback: try direct add if after fails
            try:
                self._add_violation(viol)
            except Exception:
                pass

    def _add_violation(self, viol: dict):
        # Append to internal list and listbox
        self.violations.append(viol)
        summary = f"ID: {viol.get('track_id')}, Reason: {viol.get('reason')}"
        self.violations_listbox.insert(tk.END, summary)
        # Optional: auto-scroll to the latest entry
        self.violations_listbox.yview_moveto(1.0)

    def on_violation_double_click(self, event):
        try:
            sel = self.violations_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            viol = self.violations[idx]
            img = viol.get("image_path")
            if not img:
                messagebox.showinfo("Thông báo", "Không có ảnh được lưu cho vi phạm này")
                return
            if not os.path.exists(img):
                messagebox.showerror("Lỗi", f"Ảnh không tìm thấy: {img}")
                return
            self._open_full_image(img)
        except Exception as e:
            print("Error opening violation image:", e)

    def process_next_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.running = False
            messagebox.showinfo("Done", "Video đã chạy xong")
            return

        detections = self.detector.run_frame(frame)
        self.engine.process_frame(detections, frame, self.frame_idx)

        vis = frame.copy()
        self.draw_polygon_cv(vis, self.polygon_points)

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            tid = d.get("track_id")

            # ===== DEFAULT =====
            color = (255, 255, 255)  # trắng
            thickness = 2

            # ===== LẤY TRẠNG THÁI TRACK =====
            if tid is not None and tid in self.engine.track_states:
                ts = self.engine.track_states[tid]

                if ts.violation_logged:
                    color = (0, 0, 255)      # đỏ
                    thickness = 3
                elif ts.waitingleft or ts.waitingright:
                    color = (0, 255, 255)    # vàng
                    thickness = 3

            # ===== VẼ BOX =====
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            label = d["class_name"]
            if tid is not None:
                label += f" ID:{tid}"

            cv2.putText(
                vis,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


        self.show_frame_on_canvas(vis)

        self.frame_idx += 1
        self.root.after(15, self.process_next_frame)

    def draw_polygon_cv(self, img, points):
        if len(points) < 3:
            return
        pts = np.array(points, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,255), 2)


     

    def update_violations_list(self, violations):
        """Update the violations listbox with detected violations."""
        self.violations = violations
        for violation in violations:
            summary = f"ID: {violation['track_id']}, Reason: {violation['reason']}"
            self.violations_listbox.insert(tk.END, summary)

    

    # # Remove the image saving functionality from the `_log_violation` method
    # def _log_violation(self, ts: TrackState, det: dict, frame, frame_idx: int, reason: str):
    #     if self.cfg["log_once_per_track"] and ts.violation_logged:
    #         return None
    #     ts.violation_logged = True
    #     ts_record = {
    #         "track_id": ts.track_id,
    #         "frame_idx": frame_idx,
    #         "timestamp": time.time(),
    #         "reason": reason,
    #         "bbox": det["bbox"],
    #         "class_name": det["class_name"],
    #         "conf": det["conf"]
    #     }
    #     if self.cfg.get("show_debug_prints"):
    #         print(f"[VIOL] id={ts.track_id} frame={frame_idx} reason={reason}")
    #     return ts_record

# Main function to start the app
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficViolationApp(root)
    root.mainloop()
