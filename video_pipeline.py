"""معالجة الفيديو: تتبع مفاصل الجسم (MediaPipe) وتتبع الثقل."""
import cv2
import numpy as np

KEYS = ["nose", "ls", "rs", "le", "re", "lw", "rw", "lh", "rh", "lk", "rk", "la", "ra"]
MP_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
IDX = {k: i for i, k in enumerate(KEYS)}
BONES = [("ls", "rs"), ("lh", "rh"), ("ls", "lh"), ("rs", "rh"), ("ls", "le"), ("le", "lw"),
         ("rs", "re"), ("re", "rw"), ("lh", "lk"), ("lk", "la"), ("rh", "rk"), ("rk", "ra")]


def resize_frame(frame, max_w=720):
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)
    return frame


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), n, (n / fps if fps else 0.0)


def read_frame_at(path, t, max_w=720):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    return resize_frame(f, max_w)


class ShotTracker:
    """تتبع الثقل: CSRT إن توفر وإلا مطابقة القالب."""

    def __init__(self, frame, xy, box=40):
        self.box = int(box)
        self.pos = np.array(xy, float)
        self.kind = None
        H, W = frame.shape[:2]
        x = int(max(0, min(W - self.box - 1, xy[0] - self.box / 2)))
        y = int(max(0, min(H - self.box - 1, xy[1] - self.box / 2)))
        self.tpl = frame[y:y + self.box, x:x + self.box].copy()
        for fac in (lambda: cv2.TrackerCSRT_create(), lambda: cv2.legacy.TrackerCSRT_create()):
            try:
                t = fac()
                t.init(frame, (x, y, self.box, self.box))
                self.t = t
                self.kind = "csrt"
                break
            except Exception:
                continue
        if self.kind is None:
            self.kind = "tm"

    def update(self, frame):
        if self.kind == "csrt":
            ok, b = self.t.update(frame)
            if ok:
                self.pos = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
                return self.pos.copy(), True
            return self.pos.copy(), False
        r = int(self.box * 2.5)
        H, W = frame.shape[:2]
        x0, y0 = int(max(0, self.pos[0] - r)), int(max(0, self.pos[1] - r))
        x1, y1 = int(min(W, self.pos[0] + r)), int(min(H, self.pos[1] + r))
        win = frame[y0:y1, x0:x1]
        th, tw = self.tpl.shape[:2]
        if win.shape[0] <= th or win.shape[1] <= tw:
            return self.pos.copy(), False
        res = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        self.pos = np.array([x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2])
        return self.pos.copy(), mx > 0.4


def process_video(path, shot_xy=None, box=40, t0=0.0, t1=15.0, max_w=720,
                  complexity=1, progress=None):
    """يقرأ الفيديو مرة واحدة: مفاصل الجسم + (اختيارياً) تتبع الثقل + تخزين الإطارات JPEG."""
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, end = int(t0 * fps), int(t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=complexity,
                                  smooth_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    frames, P, S, OK = [], [], [], []
    tracker = None
    total = max(1, end - start)
    size = None
    for i in range(total):
        ok, f = cap.read()
        if not ok:
            break
        f = resize_frame(f, max_w)
        h, w = f.shape[:2]
        size = (w, h)
        if shot_xy is not None:
            if tracker is None:
                tracker = ShotTracker(f, shot_xy, box)
                s, sok = np.array(shot_xy, float), True
            else:
                s, sok = tracker.update(f)
        else:
            s, sok = np.array([np.nan, np.nan]), False
        res = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        arr = np.full((len(KEYS), 3), np.nan)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            for j, m in enumerate(MP_IDX):
                arr[j] = [lm[m].x * w, lm[m].y * h, lm[m].visibility]
        P.append(arr)
        S.append(s)
        OK.append(sok)
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if progress and i % 5 == 0:
            progress(min(1.0, (i + 1) / total))
    cap.release()
    pose.close()
    if progress:
        progress(1.0)
    return dict(fps=float(fps), frames=frames, pose=np.array(P), shot=np.array(S),
                shot_ok=np.array(OK), size=size, tracker=(tracker.kind if tracker else None))
