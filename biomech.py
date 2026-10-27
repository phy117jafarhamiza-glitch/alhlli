"""الحسابات البايوميكانيكية: الزوايا، الأحداث، متغيرات الإطلاق، المراجع، الأخطاء والتصحيح."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from video_pipeline import IDX

G = 9.81
EVENTS = ["start", "mid", "power", "release"]
EVENT_LABELS = {
    "glide": {"start": "بداية الحركة (وضع الانطلاق)", "mid": "منتصف الانزلاق",
              "power": "وضع القوة", "release": "لحظة الإطلاق"},
    "rotational": {"start": "بداية الحركة (وضع الانطلاق)", "mid": "منتصف الدوران",
                   "power": "وضع القوة", "release": "لحظة الإطلاق"},
}
TECHNIQUES = {"glide": "الأسلوب الانزلاقي (أوبراين)", "rotational": "الأسلوب الدوراني"}

ANGLE_LABELS = {
    "knee_rear": "ركبة الرجل الخلفية (الدافعة)",
    "knee_front": "ركبة الرجل الأمامية",
    "hip_rear": "زاوية الورك (الخلفي)",
    "hip_front": "زاوية الورك (الأمامي)",
    "trunk_incl": "ميلان الجذع عن العمودي",
    "shoulder_ang": "زاوية الكتف (رفع الذراع)",
    "elbow_ang": "زاوية مرفق الرمي",
    "shoulder_tilt": "ميل خط الكتفين",
    "hip_tilt": "ميل خط الحوض",
    "trunk_lateral": "الميلان الجانبي للجذع",
    "shoulder_rot": "دوران الكتفين (تقريبي)",
    "hip_rot": "دوران الحوض (تقريبي)",
    "xfactor": "الفصل بين الكتفين والحوض (تقريبي)",
}
SIDE_ANGLES = ["knee_rear", "knee_front", "hip_rear", "hip_front", "trunk_incl",
               "shoulder_ang", "elbow_ang"]
FRONT_ANGLES = ["shoulder_tilt", "hip_tilt", "trunk_lateral", "shoulder_rot", "hip_rot", "xfactor"]
VIEW_ANGLES = {"side": SIDE_ANGLES, "front": FRONT_ANGLES}
VIEW_LABELS = {"side": "الكاميرا الجانبية", "front": "الكاميرا الأمامية/الخلفية"}


# ---------------------------------------------------------------- هندسة
def angle3(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def smooth_pose(P, fps):
    """P: (N,13,3) -> (N,13,2) مصفّاة ومملوءة الفجوات."""
    N = P.shape[0]
    out = P[:, :, :2].copy()
    vis = P[:, :, 2]
    out[(vis < 0.3) | np.isnan(vis)] = np.nan
    t = np.arange(N)
    win = int(max(5, round(fps / 6)))
    win += (win % 2 == 0)
    for j in range(out.shape[1]):
        for c in range(2):
            y = out[:, j, c]
            m = ~np.isnan(y)
            if m.sum() < 4:
                continue
            y = np.interp(t, t[m], y[m])
            if N > win:
                y = savgol_filter(y, win, 2)
            out[:, j, c] = y
    return out


def smooth_shot(S, ok, fps):
    N = len(S)
    t = np.arange(N)
    out = S.copy().astype(float)
    good = ok & ~np.isnan(S[:, 0])
    if good.sum() < 4:
        return out
    win = int(max(5, round(fps / 8)))
    win += (win % 2 == 0)
    for c in range(2):
        y = np.interp(t, t[good], S[good, c])
        if N > win:
            y = savgol_filter(y, win, 2)
        out[:, c] = y
    return out


def flip_x(Ps, direction, width):
    Q = Ps.copy()
    if direction == "left":
        Q[:, :, 0] = width - Q[:, :, 0]
    return Q


def _g(Q, k):
    return Q[:, IDX[k], :]


def angles_side(Q, d):
    o = "l" if d == "r" else "r"
    sh, el, wr, hp, kn, an = (_g(Q, d + s) for s in ("s", "e", "w", "h", "k", "a"))
    sho, hpo, kno, ano = _g(Q, o + "s"), _g(Q, o + "h"), _g(Q, o + "k"), _g(Q, o + "a")
    out = {
        "knee_rear": angle3(hp, kn, an),
        "knee_front": angle3(hpo, kno, ano),
        "hip_rear": angle3(sh, hp, kn),
        "hip_front": angle3(sho, hpo, kno),
        "shoulder_ang": angle3(el, sh, hp),
        "elbow_ang": angle3(sh, el, wr),
    }
    v = (sh + sho) / 2 - (hp + hpo) / 2
    out["trunk_incl"] = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
    return out


def angles_front(Q, d):
    o = "l" if d == "r" else "r"
    shd, sho, hpd, hpo = _g(Q, d + "s"), _g(Q, o + "s"), _g(Q, d + "h"), _g(Q, o + "h")
    out = {}
    out["shoulder_tilt"] = np.degrees(np.arctan2(shd[:, 1] - sho[:, 1], np.abs(shd[:, 0] - sho[:, 0]) + 1e-6))
    out["hip_tilt"] = np.degrees(np.arctan2(hpd[:, 1] - hpo[:, 1], np.abs(hpd[:, 0] - hpo[:, 0]) + 1e-6))
    v = (shd + sho) / 2 - (hpd + hpo) / 2
    sign = np.sign(shd[:, 0] - sho[:, 0])
    out["trunk_lateral"] = np.degrees(np.arctan2(v[:, 0] * sign, -v[:, 1]))
    wsh, whp = np.abs(shd[:, 0] - sho[:, 0]), np.abs(hpd[:, 0] - hpo[:, 0])
    with np.errstate(all="ignore"):
        out["shoulder_rot"] = np.degrees(np.arccos(np.clip(wsh / (np.nanpercentile(wsh, 95) + 1e-6), 0, 1)))
        out["hip_rot"] = np.degrees(np.arccos(np.clip(whp / (np.nanpercentile(whp, 95) + 1e-6), 0, 1)))
    out["xfactor"] = out["shoulder_rot"] - out["hip_rot"]
    return out


# ---------------------------------------------------------------- الأحداث
def detect_release(Ps, shot, d, fps):
    """لحظة خروج الثقل من اليد: ابتعاد الثقل عن رسغ الرمي نسبةً لطول الساعد."""
    n = len(Ps)
    wr, el = Ps[:, IDX[d + "w"]], Ps[:, IDX[d + "e"]]
    fore = np.nanmedian(np.linalg.norm(wr - el, axis=1))
    if not np.isfinite(fore) or fore < 1:
        fore = 40.0
    dist = np.linalg.norm(shot - wr, axis=1) / fore
    base = np.nanmedian(dist[:max(3, n // 4)])
    thr = base + 0.5
    for i in range(3, n - 2):
        if dist[i] > thr and dist[i + 1] > thr and dist[i + 1] >= dist[i]:
            j = i
            # الرجوع إلى آخر إطار كان فيه الثقل ما زال في اليد (قبل بدء الابتعاد السريع)
            while j > 1 and (dist[j] - dist[j - 1]) > 0.08:
                j -= 1
            return int(max(j, 1))
    sp = np.linalg.norm(np.diff(wr, axis=0), axis=1)
    return int(np.nanargmax(sp)) if len(sp) else n - 1


def suggest_events(knee_rear, rel):
    rel = int(max(rel, 3))
    lo = int(rel * 0.35)
    seg = knee_rear[lo:rel]
    if len(seg) > 2 and not np.all(np.isnan(seg)):
        power = lo + int(np.nanargmin(seg))
    else:
        power = int(rel * 0.7)
    power = min(max(power, 2), rel - 1)
    return {"start": 0, "mid": power // 2, "power": power, "release": rel}


# ---------------------------------------------------------------- المعايرة والفيزياء
def ppm_from_points(p1, p2, real_m):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]) / real_m)


def ppm_from_height(Ps, height_m):
    k = min(10, len(Ps))
    nose = np.nanmedian(Ps[:k, IDX["nose"], 1])
    ank = np.nanmedian((Ps[:k, IDX["la"], 1] + Ps[:k, IDX["ra"], 1]) / 2)
    px = abs(ank - nose)
    return float(px / (0.90 * height_m)) if px > 10 else None


def ground_level(Ps, upto):
    a = np.nanmax(np.stack([Ps[:upto + 1, IDX["la"], 1], Ps[:upto + 1, IDX["ra"], 1]]), axis=0)
    return float(np.nanpercentile(a, 95))


def optimal_angle(v, h):
    return float(np.degrees(np.arctan(v / np.sqrt(v * v + 2 * G * max(h, 0)))))


def throw_range(v, theta_deg, h):
    th = np.radians(theta_deg)
    return float(v * np.cos(th) / G * (v * np.sin(th) + np.sqrt((v * np.sin(th)) ** 2 + 2 * G * max(h, 0))))


def shot_metrics(res, rel):
    """سرعة وزاوية وارتفاع الإطلاق (من حركة الثقل بعد الخروج من اليد مباشرة)."""
    shot, fps, ppm = res["shot"], res["fps"], res.get("ppm")
    n = len(shot)
    sgn = -1.0 if res["direction"] == "left" else 1.0
    a, b = rel + 2, min(rel + 5, n - 1)  # نبدأ بعد إطار الإطلاق لتجنب إطار يكون فيه الثقل ما زال في اليد
    if b - a < 2:
        a, b = max(rel - 3, 0), rel
    t = np.arange(a, b + 1) / fps
    out = {"release_frame": int(rel), "ppm": ppm}
    if len(t) < 3 or np.any(np.isnan(shot[a:b + 1])):
        return out
    vx = np.polyfit(t, shot[a:b + 1, 0], 1)[0] * sgn
    vy = -np.polyfit(t, shot[a:b + 1, 1], 1)[0]
    ang = float(np.degrees(np.arctan2(vy, vx)))
    out["release_angle"] = ang
    if ppm:
        v = float(np.hypot(vx, vy) / ppm)
        h = float((res["ground_y"] - shot[rel, 1]) / ppm)
        out.update(speed=v, height=h, optimal_angle=optimal_angle(v, h),
                   range_actual_angle=throw_range(v, ang, h) if 0 < ang < 90 else None,
                   range_optimal=throw_range(v, optimal_angle(v, h), h))
    return out


def trajectory(res, start, rel):
    """مسار الثقل (م أو بكسل) من البداية حتى الإطلاق وما بعده."""
    shot, ppm = res["shot"], res.get("ppm")
    sgn = -1.0 if res["direction"] == "left" else 1.0
    s = 1.0 / ppm if ppm else 1.0
    x = (shot[:, 0] - shot[start, 0]) * sgn * s
    y = (res["ground_y"] - shot[:, 1]) * s if ppm else -(shot[:, 1] - shot[start, 1])
    return x, y


def resample(y, a, b, m=101):
    a, b = int(a), int(max(b, a + 1))
    seg = np.asarray(y[a:b + 1], float)
    return np.interp(np.linspace(0, len(seg) - 1, m), np.arange(len(seg)), seg)


def event_values(angles, events):
    return {k: {e: (float(v[events[e]]) if np.isfinite(v[events[e]]) else None) for e in EVENTS}
            for k, v in angles.items()}


# ---------------------------------------------------------------- نتيجة التحليل
def build_result(raw, view, hand, direction, technique, ppm=None, athlete_height=None):
    fps = raw["fps"]
    w = raw["size"][0]
    Ps = smooth_pose(raw["pose"], fps)
    Q = flip_x(Ps, direction, w) if view == "side" else Ps
    ang_side = angles_side(Q, hand)
    angles = ang_side if view == "side" else angles_front(Q, hand)
    res = dict(view=view, fps=fps, n=len(Ps), size=raw["size"], hand=hand, direction=direction,
               technique=technique, Ps=Ps, angles=angles, frames=raw["frames"], ppm=ppm)
    if view == "side":
        shot = smooth_shot(raw["shot"], raw["shot_ok"], fps)
        rel = detect_release(Ps, shot, hand, fps)
        if ppm is None and athlete_height:
            res["ppm"] = ppm_from_height(Ps, athlete_height)
        res["shot"] = shot
        res["ground_y"] = ground_level(Ps, rel)
        res["tracking_lost_ratio"] = float(1 - np.mean(raw["shot_ok"]))
    else:
        rel = int(np.nanargmax(np.abs(np.nan_to_num(np.gradient(Ps[:, IDX[hand + "w"], 1])))))
        rel = max(rel, 3)
    res["events_auto"] = suggest_events(ang_side["knee_rear"], rel)
    return res


def normalized_curves(res, events):
    return {k: resample(v, events["start"], events["release"]).tolist() for k, v in res["angles"].items()}


# ---------------------------------------------------------------- المراجع والتقييم
def _r(view, event, angle, lo, hi, tech="both"):
    return dict(technique=tech, view=view, event=event, angle=angle, lo=lo, hi=hi)


DEFAULT_REF = [
    _r("side", "power", "knee_rear", 100, 135), _r("side", "power", "knee_front", 140, 170),
    _r("side", "power", "trunk_incl", -35, -5), _r("side", "power", "shoulder_ang", 70, 110),
    _r("side", "power", "elbow_ang", 40, 90),
    _r("side", "release", "knee_rear", 150, 180), _r("side", "release", "knee_front", 150, 180),
    _r("side", "release", "trunk_incl", -5, 25), _r("side", "release", "shoulder_ang", 90, 125),
    _r("side", "release", "elbow_ang", 150, 180),
]

CORR = {
    ("knee_rear", "power", "low"): "ركبة الرجل الخلفية مثنية أكثر من اللازم في وضع القوة؛ قلّل عمق الجلوس قليلاً مع ثبات الجذع حتى لا يطول زمن الدفع.",
    ("knee_rear", "power", "high"): "ركبة الرجل الخلفية شبه ممدودة في وضع القوة فلا تُخزَّن قوة كافية؛ زد ثني الركبة وانقل الوزن فوق القدم الخلفية.",
    ("knee_rear", "release", "low"): "لم تكتمل مدّ الرجل الخلفية عند الإطلاق؛ ركّز على الدفع الكامل بالمد المتسلسل (كاحل ← ركبة ← ورك) قبل خروج الثقل.",
    ("knee_rear", "release", "high"): "فرط في مد الركبة الخلفية؛ حافظ على مد كامل دون قفل مفرط للمفصل.",
    ("knee_front", "power", "low"): "ركبة الرجل الأمامية مثنية كثيراً عند وضع القوة فيضعف الحاجز؛ ثبّت القدم الأمامية وقلّل الانحناء.",
    ("knee_front", "power", "high"): "الرجل الأمامية ممدودة مبكراً؛ احتفظ بثني خفيف ثم مدّها عند الدفع.",
    ("knee_front", "release", "low"): "الرجل الأمامية تنثني لحظة الإطلاق (انهيار الحاجز)؛ قوِّ الرجل الأمامية وتدرّب على التوقف بمد كامل.",
    ("knee_front", "release", "high"): "فرط مد في الركبة الأمامية؛ حافظ على مد كامل دون قفل.",
    ("trunk_incl", "power", "low"): "الجذع مائل للخلف أكثر من المطلوب في وضع القوة؛ قلّل الميل الخلفي ليبقى الثقل فوق القدم الخلفية بتوازن.",
    ("trunk_incl", "power", "high"): "الجذع منتصب أو أمامي مبكراً في وضع القوة؛ حافظ على ميل الجذع للخلف لتحقيق مسار دفع أطول.",
    ("trunk_incl", "release", "low"): "الجذع ما زال مائلاً للخلف عند الإطلاق؛ أكمل رفع الجذع مع مد الرجلين لتوجيه القوة للأمام والأعلى.",
    ("trunk_incl", "release", "high"): "الجذع مائل للأمام كثيراً عند الإطلاق فينخفض ارتفاع الإطلاق؛ أبقِ الجذع قريباً من العمودي حتى نهاية الدفع.",
    ("shoulder_ang", "power", "low"): "الذراع قريبة من الجسم في وضع القوة؛ ارفع المرفق قليلاً.",
    ("shoulder_ang", "power", "high"): "المرفق مرتفع/مبتعد أكثر من اللازم في وضع القوة؛ قرّبه قليلاً.",
    ("shoulder_ang", "release", "low"): "المرفق منخفض عند الإطلاق؛ ارفع مرفق الرمي إلى مستوى الكتف أو أعلى.",
    ("shoulder_ang", "release", "high"): "الذراع مرتفعة/مفتوحة أكثر من اللازم؛ وجّه الدفع في خط الثقل نحو القطاع.",
    ("elbow_ang", "power", "low"): "ثني المرفق أكبر من المطلوب في وضع القوة؛ حافظ على الثقل ملاصقاً للرقبة دون ضغط زائد.",
    ("elbow_ang", "power", "high"): "المرفق أكثر امتداداً من المطلوب في وضع القوة؛ اثنِ المرفق وأبقِ الثقل ملاصقاً للرقبة.",
    ("elbow_ang", "release", "low"): "الذراع لم تمتد كاملاً عند الإطلاق؛ أكمل مد الذراع مع دفع الثقل (سوط الأصابع).",
    ("elbow_ang", "release", "high"): "فرط في مد المرفق عند الإطلاق؛ تجنّب القفل المفاجئ للمفصل.",
}


def severity(dev):
    a = abs(dev)
    return "خفيف" if a < 5 else ("متوسط" if a < 12 else "كبير")


def evaluate(event_vals, ref_rows, technique, view, model_vals=None, tol=8.0):
    """يرجع قائمة بالأخطاء: مقارنة بالمرجع، ومقارنة بنموذج اللاعبة نفسها."""
    ref = pd.DataFrame(ref_rows) if ref_rows else pd.DataFrame(columns=["technique", "view", "event", "angle", "lo", "hi"])
    errs = []
    for ang, evs in event_vals.items():
        for ev, val in evs.items():
            if val is None:
                continue
            rows = ref[(ref.view == view) & (ref.event == ev) & (ref.angle == ang)
                       & (ref.technique.isin([technique, "both"]))]
            if len(rows):
                r = rows.iloc[0]
                lo, hi = r["lo"], r["hi"]
                if pd.notna(lo) and val < lo:
                    errs.append(_err(view, ev, ang, val, "ref", "low", val - lo, f"{lo:.0f}–{hi:.0f}", technique))
                elif pd.notna(hi) and val > hi:
                    errs.append(_err(view, ev, ang, val, "ref", "high", val - hi, f"{lo:.0f}–{hi:.0f}", technique))
            if model_vals:
                mv = model_vals.get(ang, {}).get(ev)
                if mv is not None and abs(val - mv) > tol:
                    dev = val - mv
                    errs.append(_err(view, ev, ang, val, "model", "high" if dev > 0 else "low", dev,
                                     f"{mv:.0f}±{tol:.0f}", technique))
    return errs


def _err(view, ev, ang, val, kind, direction, dev, target, technique):
    label = ANGLE_LABELS[ang]
    evl = EVENT_LABELS[technique][ev]
    if kind == "ref":
        text = CORR.get((ang, ev, direction),
                        f"{label} {'أقل' if direction == 'low' else 'أعلى'} من النطاق المرجعي بمقدار {abs(dev):.0f}°؛ عدّلها.")
    else:
        text = (f"تختلف {label} عن أفضل رمية للاعبة (النموذج) بمقدار {abs(dev):.0f}° "
                f"({'أعلى' if dev > 0 else 'أقل'})؛ حاولي {'تقليلها' if dev > 0 else 'زيادتها'} بنفس المقدار لتقريبها من نموذجك.")
    return dict(view=view, event=ev, event_label=evl, angle=ang, angle_label=label,
                value=round(float(val), 1), kind=kind, kind_label="مرجعي" if kind == "ref" else "نموذج اللاعبة",
                target=target, dev=round(float(dev), 1), severity=severity(dev), text=text)
