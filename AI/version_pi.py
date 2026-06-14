#!/usr/bin/env python3
"""
Smart Basket – ONNX Inference + Firebase Realtime Database (Raspberry Pi 5 + Pi Camera v2)
===========================================================================================
Uses picamera2 (libcamera backend) — requires Pi OS Bookworm

Synced with main version features:
- External classes.txt loading
- PRICE_MAP
- Advanced postprocess (supports 3 ONNX output formats)
- Vectorised NMS
- BasketStateTracker with confirm/hold stabilisation
- Firebase Realtime Database push (only on change)
"""

import argparse
import time
import signal
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import firebase_admin
from firebase_admin import credentials, db

from picamera2 import Picamera2
from libcamera import controls


# ──────────────────────────────────────────────
# DEFAULT CONFIG
# ──────────────────────────────────────────────
DEFAULT_INPUT_SIZE  = 640
DEFAULT_CONF_THRESH = 0.45
DEFAULT_IOU_THRESH  = 0.45
DEFAULT_THREADS     = 4

# Pi Camera v2 native resolution
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30

# Firebase / basket behaviour
DEFAULT_BASKET_ID = "BASKET_01"
CONFIRM_FRAMES    = 2   # product must appear in this many consecutive inference cycles
HOLD_FRAMES       = 3   # keep product alive for this many missed inference cycles


# ──────────────────────────────────────────────
# PRICE MAP  (EDIT THESE)
# ──────────────────────────────────────────────
PRICE_MAP = {
    "doritos sweet chili":  10.0,
    "indomie chicken curry": 10.0,
    "pepsi diet":           15.0,
    "tea el_arosa":         40.0,
}


# ──────────────────────────────────────────────
# GRACEFUL CTRL+C
# ──────────────────────────────────────────────
def _sigint(sig, frame):
    print("\n[INFO] Interrupted — shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint)


# ──────────────────────────────────────────────
# FIREBASE
# ──────────────────────────────────────────────
def init_firebase(service_account_path: str, database_url: str) -> None:
    if firebase_admin._apps:
        return
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred, {"databaseURL": database_url})
    print("[INFO] Firebase initialised")


def set_basket_connection_status(basket_id: str, connected: bool) -> None:
    ref = db.reference(f"baskets/{basket_id}")
    ref.update({"connected": connected, "updated_at": int(time.time())})


def clear_basket_in_firebase(basket_id: str) -> None:
    ref = db.reference(f"baskets/{basket_id}/current")
    ref.set({
        "basket_id":   basket_id,
        "connected":   False,
        "updated_at":  int(time.time()),
        "items":       {},
        "total_items": 0,
        "total_price": 0.0,
    })


# ──────────────────────────────────────────────
# CLASSES & COLOURS
# ──────────────────────────────────────────────
def load_classes(classes_path: str) -> list[str]:
    path = Path(classes_path)
    if not path.exists():
        raise FileNotFoundError(f"classes.txt not found: {path}")
    names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        raise ValueError("classes.txt is empty")
    print(f"[INFO] Classes loaded: {len(names)}")
    for i, n in enumerate(names):
        print(f"       {i}: {n}")
    return names


def make_colors(num_classes: int) -> list[tuple]:
    np.random.seed(42)
    c = np.random.randint(60, 255, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(map(int, row)) for row in c]


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
def load_model(model_path: str, num_threads: int = 4) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads  = num_threads
    opts.inter_op_num_threads  = 1
    opts.execution_mode        = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_profiling      = False

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"[INFO] Model  : {model_path}")
    print(f"[INFO] Input  : {inp.name}  {inp.shape}")
    print(f"[INFO] Output : {out.name}  {out.shape}")
    print(f"[INFO] Threads: {num_threads}")
    return session


def get_model_input_size(session: ort.InferenceSession, fallback: int) -> int:
    shape = session.get_inputs()[0].shape
    try:
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0:
            return h
    except (IndexError, TypeError):
        pass
    return fallback


def warmup(session: ort.InferenceSession, input_size: int) -> None:
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    name  = session.get_inputs()[0].name
    session.run(None, {name: dummy})
    print(f"[INFO] Warm-up done ({input_size}x{input_size})")


# ──────────────────────────────────────────────
# PRE-PROCESSING
# ──────────────────────────────────────────────
class Preprocessor:
    def __init__(self, input_size: int):
        self.size    = input_size
        self._canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        self._blob   = np.empty((1, 3, input_size, input_size), dtype=np.float32)

    def __call__(self, frame: np.ndarray):
        s  = self.size
        oh, ow = frame.shape[:2]

        scale = s / max(oh, ow)
        nw    = int(ow * scale)
        nh    = int(oh * scale)

        resized   = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_top   = (s - nh) // 2
        pad_left  = (s - nw) // 2

        canvas = self._canvas
        canvas[:] = 114
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        np.divide(rgb.transpose(2, 0, 1), 255.0, out=self._blob[0], casting="unsafe")

        return np.ascontiguousarray(self._blob), scale, pad_top, pad_left


# ──────────────────────────────────────────────
# NMS  (vectorised — same as main version)
# ──────────────────────────────────────────────
def nms_vectorised(boxes: np.ndarray, scores: np.ndarray,
                   class_ids: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    max_coord = boxes.max() + 1
    offsets   = class_ids.astype(np.float32) * max_coord
    shifted   = boxes + offsets[:, None]

    x1 = shifted[:, 0]; y1 = shifted[:, 1]
    x2 = shifted[:, 2]; y2 = shifted[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou   = inter / union
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return np.array(keep, dtype=np.int32)


# ──────────────────────────────────────────────
# POST-PROCESSING  (supports 3 ONNX output formats)
# ──────────────────────────────────────────────
def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = b.copy()
    out[:, 0] = b[:, 0] - b[:, 2] * 0.5
    out[:, 1] = b[:, 1] - b[:, 3] * 0.5
    out[:, 2] = b[:, 0] + b[:, 2] * 0.5
    out[:, 3] = b[:, 1] + b[:, 3] * 0.5
    return out


def undo_letterbox(boxes: np.ndarray, scale: float, pad_top: int, pad_left: int,
                   orig_h: int, orig_w: int) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top)  / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top)  / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


def postprocess(raw_output, class_names: list[str], scale: float,
                pad_top: int, pad_left: int,
                orig_h: int, orig_w: int, input_size: int,
                conf_thresh: float, iou_thresh: float) -> list[tuple]:

    pred = np.asarray(raw_output[0] if isinstance(raw_output, list) else raw_output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Unexpected output shape: {pred.shape}")

    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 300:
        pred = np.ascontiguousarray(pred.T)

    nc   = len(class_names)
    cols = pred.shape[1]

    # ── Format A: [x1,y1,x2,y2, conf, cls] ──────────
    if cols == 6:
        mask = pred[:, 4] >= conf_thresh
        pred = pred[mask]
        boxes_xyxy  = pred[:, :4].astype(np.float32)
        confidences = pred[:, 4].astype(np.float32)
        class_ids   = pred[:, 5].astype(np.int32)

    # ── Format B: [cx,cy,w,h, cls0…clsN] ────────────  (YOLOv8/v9/v10)
    elif cols == 4 + nc:
        scores_mat  = pred[:, 4:].astype(np.float32)
        class_ids   = np.argmax(scores_mat, axis=1).astype(np.int32)
        confidences = scores_mat[np.arange(len(class_ids)), class_ids]

        mask = confidences >= conf_thresh
        if not mask.any():
            return []

        pred        = pred[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]
        boxes_raw   = pred[:, :4].astype(np.float32)
        if boxes_raw.max() <= 2.0:
            boxes_raw *= input_size
        boxes_xyxy = xywh2xyxy(boxes_raw)

    # ── Format C: [cx,cy,w,h, obj, cls0…clsN] ────────  (YOLOv5/v7)
    elif cols == 5 + nc:
        obj_conf    = pred[:, 4].astype(np.float32)
        scores_mat  = pred[:, 5:].astype(np.float32)
        class_ids   = np.argmax(scores_mat, axis=1).astype(np.int32)
        confidences = obj_conf * scores_mat[np.arange(len(class_ids)), class_ids]

        mask = confidences >= conf_thresh
        if not mask.any():
            return []

        pred        = pred[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]
        boxes_raw   = pred[:, :4].astype(np.float32)
        if boxes_raw.max() <= 2.0:
            boxes_raw *= input_size
        boxes_xyxy = xywh2xyxy(boxes_raw)

    else:
        raise ValueError(f"Output cols={cols}, nc={nc}. Expected {4+nc}, {5+nc}, or 6.")

    if len(boxes_xyxy) == 0:
        return []

    if boxes_xyxy.max() <= 2.0:
        boxes_xyxy *= input_size

    boxes_xyxy = undo_letterbox(boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w)
    keep       = nms_vectorised(boxes_xyxy, confidences, class_ids, iou_thresh)

    results = []
    for i in keep:
        cls_id = int(class_ids[i])
        if 0 <= cls_id < nc:
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            results.append((x1, y1, x2, y2, float(confidences[i]), cls_id))

    return results


# ──────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────
def draw(frame: np.ndarray, detections: list, class_names: list,
         colors: list, fps: float) -> np.ndarray:
    counts = {}
    fw = frame.shape[1]

    for x1, y1, x2, y2, conf, cls_id in detections:
        name  = class_names[cls_id]
        color = colors[cls_id % len(colors)]
        counts[name] = counts.get(name, 0) + 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        yt = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, yt), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    total    = sum(counts.values())
    panel_h  = 35 + 25 * max(1, len(counts))
    cv2.rectangle(frame, (5, 5), (320, panel_h), (20, 20, 20), -1)
    cv2.putText(frame, f"Items: {total}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    y = 58
    for name, cnt in counts.items():
        cv2.putText(frame, f"{name}: {cnt}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 25

    # FPS — top-right
    cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# FIREBASE CART STATE TRACKER
# ──────────────────────────────────────────────
class BasketStateTracker:
    """Stabilises detections before writing them to Firebase."""

    def __init__(self, class_names: list[str], basket_id: str):
        self.class_names    = class_names
        self.basket_id      = basket_id
        self.states         = {}
        self.last_signature = None

    def update(self, detections: list[tuple]) -> dict:
        current_counts = {}
        current_confs  = {}

        for _, _, _, _, conf, cls_id in detections:
            name = self.class_names[cls_id]
            current_counts[name] = current_counts.get(name, 0) + 1
            current_confs.setdefault(name, []).append(float(conf))

        seen_now  = set(current_counts.keys())
        all_known = set(self.states.keys()) | seen_now

        for name in all_known:
            if name in seen_now:
                qty      = current_counts[name]
                avg_conf = sum(current_confs[name]) / len(current_confs[name])
                prev     = self.states.get(name, {
                    "quantity": 0, "seen_streak": 0,
                    "miss_streak": 0, "stable": False, "confidence": 0.0,
                })
                prev["quantity"]    = qty
                prev["confidence"]  = avg_conf
                prev["seen_streak"] += 1
                prev["miss_streak"] = 0
                if prev["seen_streak"] >= CONFIRM_FRAMES:
                    prev["stable"] = True
                self.states[name] = prev
            else:
                prev = self.states.get(name)
                if prev is None:
                    continue
                prev["miss_streak"]  += 1
                prev["seen_streak"]   = 0
                if prev["miss_streak"] >= HOLD_FRAMES:
                    del self.states[name]
                else:
                    self.states[name] = prev

        return self._build_payload()

    def _build_payload(self) -> dict:
        items       = {}
        total_items = 0
        total_price = 0.0

        for name, state in self.states.items():
            if not state.get("stable", False):
                continue
            qty   = int(state["quantity"])
            conf  = float(state["confidence"])
            price = float(PRICE_MAP.get(name, 0.0))
            items[name] = {
                "name":       name,
                "quantity":   qty,
                "price":      price,
                "confidence": round(conf, 3),
            }
            total_items += qty
            total_price += qty * price

        return {
            "basket_id":   self.basket_id,
            "connected":   True,
            "updated_at":  int(time.time()),
            "items":       items,
            "total_items": total_items,
            "total_price": round(total_price, 2),
        }

    def push_if_changed(self, payload: dict) -> None:
        signature = (
            tuple(sorted((k, v["quantity"]) for k, v in payload["items"].items())),
            payload["total_items"],
            payload["total_price"],
        )
        if signature == self.last_signature:
            return
        self.last_signature = signature
        ref = db.reference(f"baskets/{self.basket_id}/current")
        ref.set(payload)
        print(f"[FIREBASE] Updated basket: items={payload['total_items']} "
              f"total={payload['total_price']}")


# ──────────────────────────────────────────────
# INFERENCE HELPER
# ──────────────────────────────────────────────
def run_inference(session: ort.InferenceSession, frame: np.ndarray,
                  preprocessor: Preprocessor, class_names: list[str],
                  input_size: int, conf_thresh: float,
                  iou_thresh: float) -> list[tuple]:
    oh, ow = frame.shape[:2]
    blob, scale, pad_top, pad_left = preprocessor(frame)
    inp_name = session.get_inputs()[0].name
    outputs  = session.run(None, {inp_name: blob})

    return postprocess(
        raw_output  = outputs[0],
        class_names = class_names,
        scale       = scale,
        pad_top     = pad_top,
        pad_left    = pad_left,
        orig_h      = oh,
        orig_w      = ow,
        input_size  = input_size,
        conf_thresh = conf_thresh,
        iou_thresh  = iou_thresh,
    )


# ──────────────────────────────────────────────
# CAMERA SETUP  (Pi Camera v2 via picamera2)
# ──────────────────────────────────────────────
def create_camera() -> Picamera2:
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={
            "size":   (CAM_WIDTH, CAM_HEIGHT),
            "format": "BGR888",          # OpenCV-native, no channel swap needed
        },
        controls={
            "FrameRate":          CAM_FPS,
            "AfMode":             controls.AfModeEnum.Continuous,
            "AeEnable":           True,
            "AwbEnable":          True,
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
        },
        buffer_count=4
    )
    cam.configure(config)
    return cam


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def run(session: ort.InferenceSession, class_names: list[str], colors: list,
        preprocessor: Preprocessor, input_size: int,
        conf_thresh: float, iou_thresh: float,
        headless: bool, save: bool, snapshot: bool,
        basket_id: str, use_firebase: bool):

    cam = create_camera()
    cam.start()
    time.sleep(0.5)                    # let AE/AWB settle

    tracker = BasketStateTracker(class_names, basket_id)

    if use_firebase:
        set_basket_connection_status(basket_id, True)
        clear_basket_in_firebase(basket_id)

    writer  = None
    fps_q   = deque(maxlen=30)
    frame_n = 0

    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output_picam.mp4", fourcc, CAM_FPS,
                                 (CAM_WIDTH, CAM_HEIGHT))
        print("[INFO] Saving to output_picam.mp4")

    if not headless:
        cv2.namedWindow("Smart Basket – Pi Cam v2", cv2.WINDOW_NORMAL)

    print(
        f"[INFO] Running | Size: {input_size} | Conf: {conf_thresh} | "
        f"IoU: {iou_thresh} | Basket: {basket_id} | Q = quit"
    )

    try:
        while True:
            t0    = time.perf_counter()
            frame = cam.capture_array("main")   # BGR888 already
            oh, ow = frame.shape[:2]

            try:
                last_dets = run_inference(
                    session, frame, preprocessor, class_names,
                    input_size, conf_thresh, iou_thresh,
                )

                if use_firebase:
                    payload = tracker.update(last_dets)
                    tracker.push_if_changed(payload)

            except Exception as exc:
                print(f"[WARN] Inference error: {exc}")
                last_dets = []

            t1 = time.perf_counter()
            fps_q.append(1.0 / max(t1 - t0, 1e-6))
            avg_fps = sum(fps_q) / len(fps_q)

            result = draw(frame.copy(), last_dets, class_names, colors, avg_fps)

            # print detections every 30 frames
            if frame_n % 30 == 0 and last_dets:
                print(f"\n[Frame {frame_n}]  FPS={avg_fps:.1f}")
                for d in last_dets:
                    name = class_names[d[5]] if d[5] < len(class_names) else f"cls{d[5]}"
                    print(f"  {name:<25} conf={d[4]:.3f}  "
                          f"box=({d[0]},{d[1]},{d[2]},{d[3]})")

            if writer:
                writer.write(result)

            # snapshot mode: save one frame and exit
            if snapshot:
                out = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(out, result)
                print(f"[INFO] Snapshot saved → {out}")
                break

            if not headless:
                cv2.imshow("Smart Basket – Pi Cam v2", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    snap = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap, result)
                    print(f"[INFO] Snapshot saved → {snap}")
            elif frame_n % 30 == 0:
                print(f"[INFO] Frame {frame_n:6d} | FPS {avg_fps:.1f} | Dets {len(last_dets)}")

            frame_n += 1

    finally:
        cam.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped.")

        if use_firebase:
            clear_basket_in_firebase(basket_id)
            set_basket_connection_status(basket_id, False)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Smart Basket – ONNX + Firebase (Raspberry Pi 5)"
    )
    ap.add_argument("--model",           required=True,                      help="Path to best.onnx")
    ap.add_argument("--classes",         default=None,                       help="classes.txt (default: same dir as model)")
    ap.add_argument("--conf",            type=float, default=DEFAULT_CONF_THRESH)
    ap.add_argument("--iou",             type=float, default=DEFAULT_IOU_THRESH)
    ap.add_argument("--imgsz",           type=int,   default=DEFAULT_INPUT_SIZE)
    ap.add_argument("--threads",         type=int,   default=DEFAULT_THREADS)
    ap.add_argument("--basket-id",       default=DEFAULT_BASKET_ID)
    ap.add_argument("--firebase",        action="store_true",                help="Enable Firebase Realtime Database updates")
    ap.add_argument("--service-account", default="serviceAccountKey.json",   help="Path to Firebase service account JSON")
    ap.add_argument("--database-url",    default="",                         help="Firebase Realtime Database URL")
    ap.add_argument("--no-show",         action="store_true",                help="Headless mode (no display window)")
    ap.add_argument("--save",            action="store_true",                help="Save output video")
    ap.add_argument("--snapshot",        action="store_true",                help="Capture one frame and exit")
    args = ap.parse_args()

    model_path  = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    classes_path = args.classes or str(model_path.parent / "classes.txt")
    class_names  = load_classes(classes_path)
    colors       = make_colors(len(class_names))

    if args.firebase:
        if not args.database_url:
            raise ValueError("When using --firebase you must pass --database-url")
        init_firebase(args.service_account, args.database_url)

    session    = load_model(str(model_path), num_threads=args.threads)
    input_size = get_model_input_size(session, args.imgsz)

    if input_size != args.imgsz:
        print(f"[INFO] Model input size: {input_size} (overrides --imgsz {args.imgsz})")

    warmup(session, input_size)
    preprocessor = Preprocessor(input_size)

    run(
        session      = session,
        class_names  = class_names,
        colors       = colors,
        preprocessor = preprocessor,
        input_size   = input_size,
        conf_thresh  = args.conf,
        iou_thresh   = args.iou,
        headless     = args.no_show,
        save         = args.save,
        snapshot     = args.snapshot,
        basket_id    = args.basket_id,
        use_firebase = args.firebase,
    )


if __name__ == "__main__":
    main()