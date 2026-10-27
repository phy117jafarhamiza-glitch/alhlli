"""قاعدة بيانات بسيطة (JSON) للاعبات والمحاولات والمراجع."""
import datetime
import io
import json
import os
import uuid

import numpy as np
import pandas as pd

from biomech import DEFAULT_REF, ANGLE_LABELS, EVENT_LABELS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "db.json")


def _default():
    return {"players": {}, "reference": [dict(r) for r in DEFAULT_REF]}


def load():
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        db.setdefault("players", {})
        db.setdefault("reference", [dict(r) for r in DEFAULT_REF])
        return db
    except Exception:
        return _default()


def _enc(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dumps(db):
    return json.dumps(db, ensure_ascii=False, default=_enc)


def save(db):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        f.write(dumps(db))


def new_id():
    return uuid.uuid4().hex[:8]


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def best_trial(player):
    tr = [t for t in player.get("trials", []) if t.get("distance")]
    return max(tr, key=lambda t: t["distance"]) if tr else None


def get_model(player):
    mid = player.get("model_trial_id")
    for t in player.get("trials", []):
        if t["id"] == mid:
            return t
    return best_trial(player)


def to_excel(db):
    rows_p, rows_t, rows_e = [], [], []
    for pid, p in db["players"].items():
        rows_p.append({"الاسم": p["name"], "الطول (سم)": p.get("height"), "الوزن (كغم)": p.get("weight"),
                       "طول الذراع (سم)": p.get("arm"), "طول الرجل (سم)": p.get("leg")})
        for t in p.get("trials", []):
            sm = t.get("views", {}).get("side", {}).get("metrics", {})
            rows_t.append({"اللاعبة": p["name"], "التاريخ": t["date"], "الأسلوب": t["technique"],
                           "نوع المحاولة": t["test_type"], "المسافة (م)": t.get("distance"),
                           "سرعة الإطلاق (م/ث)": sm.get("speed"), "زاوية الإطلاق": sm.get("release_angle"),
                           "الزاوية المثلى": sm.get("optimal_angle"), "ارتفاع الإطلاق (م)": sm.get("height"),
                           "عدد الأخطاء": len(t.get("errors", []))})
            for e in t.get("errors", []):
                rows_e.append({"اللاعبة": p["name"], "التاريخ": t["date"], "الكاميرا": e["view"],
                               "المرحلة": e["event_label"], "الزاوية": e["angle_label"], "القيمة": e["value"],
                               "المستهدف": e["target"], "الانحراف": e["dev"], "الشدة": e["severity"],
                               "المصدر": e["kind_label"], "التصحيح": e["text"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows_p).to_excel(w, "اللاعبات", index=False)
        pd.DataFrame(rows_t).to_excel(w, "المحاولات", index=False)
        pd.DataFrame(rows_e).to_excel(w, "الأخطاء", index=False)
    return buf.getvalue()
