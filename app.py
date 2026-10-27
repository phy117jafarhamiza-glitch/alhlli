import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import biomech as bm
import storage
from video_pipeline import BONES, IDX, process_video, read_frame_at, video_info

st.set_page_config(page_title="محلل المسار الحركي - دفع الثقل", page_icon="🏋️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp {direction: rtl; text-align: right; font-family: 'Cairo', sans-serif;}
.stPlotlyChart, [data-testid="stDataFrame"], [data-testid="stSlider"], .stNumberInput input {direction: ltr; text-align: left;}
h1, h2, h3 {font-family: 'Cairo', sans-serif;}
.card {background:#f6f8fb; border-radius:10px; padding:10px 14px; margin-bottom:8px; border-right:4px solid #1f6feb;}
.err-كبير {border-right-color:#d73a49;} .err-متوسط {border-right-color:#f0883e;} .err-خفيف {border-right-color:#dbab09;}
</style>
""", unsafe_allow_html=True)

if "db" not in st.session_state:
    st.session_state.db = storage.load()
db = st.session_state.db


def commit():
    storage.save(db)


# ------------------------------------------------------------------ أدوات مساعدة
def save_temp(upload, vk):
    sig = (upload.name, upload.size)
    cur = st.session_state.get(f"tmp_{vk}")
    if cur and cur[0] == sig and os.path.exists(cur[1]):
        return cur[1]
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.getbuffer())
    st.session_state[f"tmp_{vk}"] = (sig, f.name)
    return f.name


def click_image(img_bgr, key, show_w=700):
    from streamlit_image_coordinates import streamlit_image_coordinates
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    val = streamlit_image_coordinates(Image.fromarray(rgb), key=key, width=show_w)
    if val:
        sx = rgb.shape[1] / val["width"]
        return val, (val["x"] * sx, val["y"] * sx)
    return None, None


def setup_view(vk, title, is_side):
    st.subheader(title)
    up = st.file_uploader("ارفع فيديو (من الهاتف يمكنك اختيار «تصوير فيديو» مباشرة)",
                          type=["mp4", "mov", "avi", "mkv", "m4v"], key=f"up_{vk}")
    if not up:
        return None
    path = save_temp(up, vk)
    fps, n, dur = video_info(path)
    st.caption(f"معدل الإطارات: {fps:.0f} إطار/ث — المدة: {dur:.1f} ث"
               + ("  ⚠️ يُفضّل 60 إطار/ث أو أكثر لدقة سرعة الإطلاق" if fps < 50 else ""))
    c1, c2 = st.columns(2)
    t0 = c1.number_input("بداية التحليل (ث)", 0.0, max(dur - 0.5, 0.0), 0.0, 0.1, key=f"t0_{vk}")
    t1 = c2.number_input("نهاية التحليل (ث)", 0.5, max(dur, 0.6), float(min(dur, t0 + 12)), 0.1, key=f"t1_{vk}")
    frame = read_frame_at(path, t0)
    if frame is None:
        st.error("تعذّر قراءة الفيديو.")
        return None
    cfg = dict(path=path, fps=fps, t0=t0, t1=min(t1, t0 + 20), frame_shape=frame.shape)
    if not is_side:
        cfg["camera"] = st.radio("موضع الكاميرا", ["أمامية", "خلفية"], horizontal=True, key=f"cam_{vk}")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="أول إطار", width=500)
        return cfg

    cfg["direction"] = "right" if st.radio("اتجاه الرمي في الصورة", ["نحو اليمين", "نحو اليسار"],
                                           horizontal=True, key=f"dir_{vk}") == "نحو اليمين" else "left"
    sig = (up.name, up.size, t0)
    if st.session_state.get(f"sig_{vk}") != sig:
        st.session_state[f"sig_{vk}"] = sig
        st.session_state[f"pts_{vk}"] = {"shot": None, "c1": None, "c2": None}
        st.session_state[f"last_{vk}"] = None
    pts = st.session_state[f"pts_{vk}"]
    mode = st.radio("ما الذي تنقر عليه الآن في الصورة؟",
                    ["① مركز الثقل", "② نقطة المعايرة الأولى", "③ نقطة المعايرة الثانية"],
                    horizontal=True, key=f"mode_{vk}")
    cfg["box"] = st.slider("حجم صندوق تتبع الثقل (بكسل)", 20, 90, 40, key=f"box_{vk}")
    disp = frame.copy()
    if pts["shot"]:
        x, y = map(int, pts["shot"])
        b = cfg["box"] // 2
        cv2.rectangle(disp, (x - b, y - b), (x + b, y + b), (0, 0, 255), 2)
    for k in ("c1", "c2"):
        if pts[k]:
            cv2.circle(disp, tuple(map(int, pts[k])), 6, (0, 200, 0), -1)
    if pts["c1"] and pts["c2"]:
        cv2.line(disp, tuple(map(int, pts["c1"])), tuple(map(int, pts["c2"])), (0, 200, 0), 2)
    val, xy = click_image(disp, f"img_{vk}")
    if val and val != st.session_state.get(f"last_{vk}"):
        st.session_state[f"last_{vk}"] = val
        pts[{"①": "shot", "②": "c1", "③": "c2"}[mode[0]]] = xy
        st.rerun()
    cal = st.radio("طريقة المعايرة (تحويل البكسل إلى متر)",
                   ["نقطتان بمسافة معروفة (الأدق)", "من طول اللاعبة (تقريبية)"], key=f"cal_{vk}")
    cfg["cal_method"] = "points" if cal.startswith("نقطتان") else "height"
    if cfg["cal_method"] == "points":
        cfg["cal_m"] = st.number_input("المسافة الحقيقية بين النقطتين (م) — قطر دائرة الرمي = 2.135",
                                       0.1, 20.0, 2.135, 0.005, key=f"calm_{vk}")
        cfg["ppm"] = bm.ppm_from_points(pts["c1"], pts["c2"], cfg["cal_m"]) if pts["c1"] and pts["c2"] else None
    cfg["shot_xy"] = pts["shot"]
    cfg["ready"] = bool(pts["shot"]) and (cfg["cal_method"] == "height" or cfg.get("ppm"))
    return cfg


# ------------------------------------------------------------------ رسم
def render_frame(res, idx):
    img = cv2.imdecode(np.frombuffer(res["frames"][idx], np.uint8), cv2.IMREAD_COLOR)
    P = res["Ps"][idx]
    d = res["hand"]
    for a, b in BONES:
        pa, pb = P[IDX[a]], P[IDX[b]]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        col = (0, 140, 255) if (a[0] == d and b[0] == d) else (255, 160, 0)
        cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), col, 3)
    for j in range(len(P)):
        if not np.any(np.isnan(P[j])):
            cv2.circle(img, tuple(map(int, P[j])), 4, (255, 255, 255), -1)
    if res["view"] == "side":
        for ang, joint in (("knee_rear", d + "k"), ("knee_front", ("l" if d == "r" else "r") + "k"), ("elbow_ang", d + "e")):
            v, p = res["angles"][ang][idx], P[IDX[joint]]
            if np.isfinite(v) and not np.any(np.isnan(p)):
                cv2.putText(img, f"{v:.0f}", (int(p[0]) + 8, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if "shot" in res and not np.any(np.isnan(res["shot"][idx])):
            cv2.circle(img, tuple(map(int, res["shot"][idx])), 10, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fig_traj(res, ev, model_traj=None):
    rel = ev["release"]
    x, y = bm.trajectory(res, ev["start"], rel)
    unit = "م" if res.get("ppm") else "بكسل"
    fig = go.Figure()
    s = ev["start"]
    fig.add_trace(go.Scatter(x=x[s:rel + 1], y=y[s:rel + 1], mode="lines+markers", name="مسار الثقل (داخل اليد)",
                             line=dict(color="#1f6feb", width=3), marker=dict(size=5, color=np.arange(rel - s + 1), colorscale="Viridis")))
    fig.add_trace(go.Scatter(x=x[rel:], y=y[rel:], mode="lines+markers", name="بعد الإطلاق",
                             line=dict(color="#999", dash="dot")))
    fig.add_trace(go.Scatter(x=[x[rel]], y=[y[rel]], mode="markers+text", text=["لحظة الإطلاق"], textposition="top center",
                             marker=dict(size=14, color="#d73a49", symbol="star"), name="الإطلاق"))
    if model_traj:
        fig.add_trace(go.Scatter(x=model_traj["x"], y=model_traj["y"], mode="lines", name="نموذج اللاعبة",
                                 line=dict(color="#2ea043", dash="dash")))
    fig.update_layout(title="مسار الثقل من البداية حتى الإطلاق", xaxis_title=f"المسافة الأفقية ({unit})",
                      yaxis_title=f"الارتفاع ({unit})", height=450, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fig_angle(res, ang, ev, model_curve=None):
    curve = bm.resample(res["angles"][ang], ev["start"], ev["release"])
    xs = np.linspace(0, 100, 101)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", name="الرمية الحالية", line=dict(color="#1f6feb", width=3)))
    if model_curve:
        fig.add_trace(go.Scatter(x=xs, y=model_curve, mode="lines", name="نموذج اللاعبة", line=dict(color="#2ea043", dash="dash")))
    span = max(ev["release"] - ev["start"], 1)
    for k in bm.EVENTS:
        p = 100 * (ev[k] - ev["start"]) / span
        fig.add_vline(x=p, line=dict(color="#aaa", dash="dot"))
        fig.add_annotation(x=p, y=1.02, yref="paper", text=bm.EVENT_LABELS[res["technique"]][k], showarrow=False, font=dict(size=10))
    fig.update_layout(title=bm.ANGLE_LABELS[ang], xaxis_title="% من زمن الحركة (من البداية حتى الإطلاق)",
                      yaxis_title="درجة", height=330, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ------------------------------------------------------------------ الصفحات
def page_players():
    st.header("👤 اللاعبات والقياسات الجسمية")
    with st.form("addp", clear_on_submit=True):
        c = st.columns(5)
        name = c[0].text_input("الاسم")
        h = c[1].number_input("الطول (سم)", 100.0, 220.0, 165.0)
        w = c[2].number_input("الوزن (كغم)", 30.0, 150.0, 60.0)
        arm = c[3].number_input("طول الذراع (سم)", 40.0, 100.0, 70.0)
        leg = c[4].number_input("طول الرجل (سم)", 60.0, 130.0, 90.0)
        notes = st.text_input("ملاحظات")
        if st.form_submit_button("➕ إضافة لاعبة") and name.strip():
            db["players"][storage.new_id()] = dict(name=name.strip(), height=h, weight=w, arm=arm, leg=leg,
                                                   notes=notes, trials=[], model_trial_id=None)
            commit()
            st.rerun()
    if db["players"]:
        df = pd.DataFrame([{"الاسم": p["name"], "الطول": p["height"], "الوزن": p["weight"], "الذراع": p["arm"],
                            "الرجل": p["leg"], "عدد المحاولات": len(p["trials"])} for p in db["players"].values()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        pid = st.selectbox("حذف لاعبة", [""] + list(db["players"]), format_func=lambda i: db["players"][i]["name"] if i else "—")
        if pid and st.button("🗑️ تأكيد الحذف"):
            del db["players"][pid]
            commit()
            st.rerun()


def pick_player(key):
    if not db["players"]:
        st.warning("أضيفي لاعبة أولاً من صفحة «اللاعبات».")
        return None
    pid = st.selectbox("اللاعبة", list(db["players"]), format_func=lambda i: db["players"][i]["name"], key=key)
    return pid


def page_analysis():
    st.header("🎥 تحليل رمية جديدة")
    pid = pick_player("pl_an")
    if not pid:
        return
    player = db["players"][pid]
    c = st.columns(4)
    technique = c[0].radio("أسلوب الدفع", list(bm.TECHNIQUES), format_func=bm.TECHNIQUES.get)
    hand = "r" if c[1].radio("يد الرمي", ["اليمنى", "اليسرى"]) == "اليمنى" else "l"
    test_type = c[2].selectbox("نوع المحاولة", ["تدريب", "اختبار قبلي", "اختبار بعدي"])
    distance = c[3].number_input("المسافة المنجزة (م) — اتركيها 0 إن لم تُقَس", 0.0, 30.0, 0.0, 0.01)
    cx = st.columns(2)
    complexity = cx[0].selectbox("دقة تتبع الجسم", [1, 2, 0], format_func={0: "سريعة", 1: "متوسطة (افتراضي)", 2: "عالية (أبطأ)"}.get)
    max_w = cx[1].selectbox("عرض الإطار للمعالجة", [720, 960, 540], index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        side = setup_view("side", "📷 الكاميرا الجانبية (مسار الثقل + زوايا المستوى السهمي)", True)
    with col_b:
        front = setup_view("front", "📷 الكاميرا الأمامية/الخلفية (ميل الكتفين والحوض والدوران)", False)

    ok_side = side is not None and side.get("ready")
    if side is not None and not ok_side:
        st.info("للتحليل الجانبي: انقري على مركز الثقل في أول إطار، وحدّدي نقطتي المعايرة (أو اختاري المعايرة من الطول).")
    if st.button("▶️ بدء التحليل", type="primary", disabled=not (ok_side or front)):
        views = {}
        bar = st.progress(0.0, text="جارٍ التحليل...")
        for vk, cfg in (("side", side if ok_side else None), ("front", front)):
            if cfg is None:
                continue
            bar.progress(0.0, text=f"معالجة {bm.VIEW_LABELS[vk]} ...")
            raw = process_video(cfg["path"], cfg.get("shot_xy") if vk == "side" else None, cfg.get("box", 40),
                                cfg["t0"], cfg["t1"], max_w, complexity, lambda p: bar.progress(p))
            if len(raw["pose"]) < 10:
                st.error(f"فيديو {bm.VIEW_LABELS[vk]} قصير جداً أو لم يُقرأ.")
                continue
            if np.all(np.isnan(raw["pose"][:, :, 0])):
                st.error(f"لم يُكتشف جسم اللاعبة في {bm.VIEW_LABELS[vk]}؛ تأكدي من ظهور الجسم كاملاً وإضاءة جيدة.")
                continue
            views[vk] = bm.build_result(raw, vk, hand, cfg.get("direction", "right"), technique,
                                        ppm=cfg.get("ppm"), athlete_height=player["height"] / 100)
        bar.empty()
        if views:
            st.session_state["an"] = dict(run=storage.new_id(), pid=pid, technique=technique, hand=hand,
                                          test_type=test_type, distance=distance, views=views)
    an = st.session_state.get("an")
    if an and an["pid"] == pid:
        show_results(an, player)


def show_results(an, player):
    st.divider()
    st.header("📊 النتائج")
    technique, run = an["technique"], an["run"]
    model_trial = storage.get_model(player)
    tol = st.slider("حدّ التسامح عند المقارنة بنموذج اللاعبة (±درجة)", 3, 20, 8)
    all_errors, saved_views = [], {}
    tabs_v = st.tabs([bm.VIEW_LABELS[v] for v in an["views"]])
    for tab, (vk, res) in zip(tabs_v, an["views"].items()):
        with tab:
            st.markdown("**حدّدي إطارات مراحل الحركة** (مقترحة تلقائياً — تحقّقي منها بالمعاينة وعدّليها):")
            cols = st.columns(4)
            ev = {}
            for c, k in zip(cols, bm.EVENTS):
                ev[k] = c.slider(bm.EVENT_LABELS[technique][k], 0, res["n"] - 1, int(res["events_auto"][k]), key=f"ev_{vk}_{k}_{run}")
            if not (ev["start"] < ev["mid"] <= ev["power"] < ev["release"]):
                st.warning("ترتيب المراحل غير منطقي؛ يجب: البداية < المنتصف ≤ وضع القوة < الإطلاق.")
                continue
            mt = (model_trial or {}).get("views", {}).get(vk)
            t_traj, t_ang, t_err, t_frames = st.tabs(["🎯 المسار" if vk == "side" else "—", "📐 الزوايا", "⚠️ الأخطاء والتصحيح", "🖼️ إطارات رئيسية"])
            evals = bm.event_values(res["angles"], ev)
            metrics = {}
            with t_traj:
                if vk != "side":
                    st.info("مسار الثقل يُستخرج من الكاميرا الجانبية.")
                else:
                    metrics = bm.shot_metrics(res, ev["release"])
                    st.plotly_chart(fig_traj(res, ev, mt["traj"] if mt else None), use_container_width=True)
                    m = st.columns(5)
                    if "speed" in metrics:
                        m[0].metric("سرعة الإطلاق (م/ث)", f"{metrics['speed']:.1f}")
                        m[1].metric("زاوية الإطلاق", f"{metrics['release_angle']:.1f}°")
                        m[2].metric("الزاوية المثلى لجسمها", f"{metrics['optimal_angle']:.1f}°")
                        m[3].metric("ارتفاع الإطلاق (م)", f"{metrics['height']:.2f}")
                        m[4].metric("المدى النظري (م)", f"{metrics['range_optimal']:.2f}")
                        st.caption("الزاوية المثلى تُحسب فيزيائياً من سرعة وارتفاع إطلاق هذه اللاعبة نفسها. السرعة تقديرية (كاميرا واحدة، "
                                   "ودقتها تزداد مع 60 إطار/ث فأكثر). المدى النظري يهمل مقاومة الهواء وإزاحة الإطلاق أمام الدائرة.")
                    elif "release_angle" in metrics:
                        m[1].metric("زاوية الإطلاق", f"{metrics['release_angle']:.1f}°")
                        st.warning("لا توجد معايرة؛ السرعة والارتفاع غير متاحين.")
                    else:
                        st.warning("لم يتوفر تتبع كافٍ للثقل بعد الإطلاق لحساب السرعة والزاوية.")
                    if res.get("tracking_lost_ratio", 0) > 0.2:
                        st.warning(f"فقد التتبع الثقل في {res['tracking_lost_ratio']*100:.0f}% من الإطارات؛ جرّبي حجم صندوق مختلف أو إضاءة/خلفية أوضح.")
                    st.caption("تحقّقي أن لحظة الإطلاق صحيحة بمعاينتها في تبويب «إطارات رئيسية» وعدّلي شريحة الإطلاق عند الحاجة.")
            with t_ang:
                sel = st.multiselect("الزوايا المعروضة", bm.VIEW_ANGLES[vk], default=bm.VIEW_ANGLES[vk][:4],
                                     format_func=bm.ANGLE_LABELS.get, key=f"sel_{vk}_{run}")
                for a in sel:
                    mc = mt["curves"].get(a) if mt else None
                    st.plotly_chart(fig_angle(res, a, ev, mc), use_container_width=True)
                rows = []
                for a, evs in evals.items():
                    r = {"الزاوية": bm.ANGLE_LABELS[a]}
                    for k in bm.EVENTS:
                        v = evs[k]
                        r[bm.EVENT_LABELS[technique][k]] = None if v is None else round(v, 1)
                    rows.append(r)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            mv = (mt or {}).get("event_vals")
            errs = bm.evaluate(evals, db["reference"], technique, vk, mv, tol)
            all_errors += errs
            with t_err:
                if not errs:
                    st.success("لا أخطاء ضمن النطاقات المرجعية/النموذج المتاحة. (إن لم تتوفر نطاقات لهذه الكاميرا فاعتمدي نموذج اللاعبة أو أضيفي مراجع من صفحة «المراجع».)")
                for e in sorted(errs, key=lambda e: -abs(e["dev"])):
                    st.markdown(f"<div class='card err-{e['severity']}'><b>{e['angle_label']}</b> — {e['event_label']} "
                                f"({e['kind_label']}، شدة {e['severity']})<br>"
                                f"القيمة {e['value']}° | المستهدف {e['target']} | الانحراف {e['dev']:+}°<br>{e['text']}</div>",
                                unsafe_allow_html=True)
            with t_frames:
                fc = st.columns(4)
                for c, k in zip(fc, bm.EVENTS):
                    c.image(render_frame(res, ev[k]), caption=bm.EVENT_LABELS[technique][k], use_container_width=True)
            saved = dict(event_vals=evals, curves=bm.normalized_curves(res, ev), events=ev, fps=res["fps"], metrics=metrics)
            if vk == "side":
                x, y = bm.trajectory(res, ev["start"], ev["release"])
                saved["traj"] = dict(x=bm.resample(x, ev["start"], ev["release"]).tolist(),
                                     y=bm.resample(y, ev["start"], ev["release"]).tolist())
            saved_views[vk] = saved
    st.divider()
    note = st.text_input("ملاحظات المحاولة")
    if st.button("💾 حفظ المحاولة في سجل اللاعبة", type="primary"):
        player["trials"].append(dict(id=storage.new_id(), date=storage.now(), technique=bm.TECHNIQUES[technique],
                                     technique_key=technique, test_type=an["test_type"], distance=an["distance"] or None,
                                     hand=an["hand"], notes=note, views=saved_views, errors=all_errors))
        commit()
        st.success("تم حفظ المحاولة.")


def page_models():
    st.header("🧬 النموذج الشخصي للاعبة")
    pid = pick_player("pl_mod")
    if not pid:
        return
    p = db["players"][pid]
    if not p["trials"]:
        st.info("لا توجد محاولات محفوظة بعد.")
        return
    st.caption("النموذج = أفضل محاولة للاعبة نفسها (الأبعد مسافة افتراضياً)، وتُقارَن به بقية رمياتها تلقائياً. يمكنك اعتماد محاولة أخرى.")
    cur = storage.get_model(p)
    ids = [t["id"] for t in p["trials"]]
    fmt = lambda i: next(f"{t['date']} | {t['test_type']} | {t.get('distance') or '—'} م" for t in p["trials"] if t["id"] == i)
    sel = st.selectbox("المحاولة المعتمدة كنموذج", ids, index=ids.index(cur["id"]) if cur else 0, format_func=fmt)
    if st.button("اعتماد كنموذج"):
        p["model_trial_id"] = sel
        commit()
        st.success("تم الاعتماد.")
    m = next(t for t in p["trials"] if t["id"] == sel)
    other = st.selectbox("قارنِي بمحاولة", [i for i in ids if i != sel] or ids, format_func=fmt)
    o = next(t for t in p["trials"] if t["id"] == other)
    for vk, mv in m["views"].items():
        ov = o["views"].get(vk)
        if not ov:
            continue
        st.subheader(bm.VIEW_LABELS[vk])
        a = st.selectbox("الزاوية", list(mv["curves"]), format_func=bm.ANGLE_LABELS.get, key=f"mc_{vk}")
        xs = np.linspace(0, 100, 101)
        f = go.Figure([go.Scatter(x=xs, y=mv["curves"][a], name="النموذج", line=dict(color="#2ea043", width=3)),
                       go.Scatter(x=xs, y=ov["curves"][a], name="المحاولة", line=dict(color="#1f6feb", dash="dash"))])
        f.update_layout(height=320, xaxis_title="% من زمن الحركة", yaxis_title="درجة", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(f, use_container_width=True)


def page_records():
    st.header("📁 السجل والتقدم")
    pid = pick_player("pl_rec")
    if not pid:
        return
    p = db["players"][pid]
    if not p["trials"]:
        st.info("لا توجد محاولات.")
    else:
        rows = []
        for t in p["trials"]:
            sm = t["views"].get("side", {}).get("metrics", {})
            rows.append({"التاريخ": t["date"], "النوع": t["test_type"], "الأسلوب": t["technique"], "المسافة": t.get("distance"),
                         "السرعة": sm.get("speed"), "زاوية الإطلاق": sm.get("release_angle"),
                         "الزاوية المثلى": sm.get("optimal_angle"), "الأخطاء": len(t.get("errors", []))})
        df = pd.DataFrame(rows)
        st.dataframe(df.round(2), use_container_width=True, hide_index=True)
        if len(df) > 1:
            f = go.Figure([go.Scatter(x=df["التاريخ"], y=df["المسافة"], name="المسافة (م)", mode="lines+markers"),
                           go.Bar(x=df["التاريخ"], y=df["الأخطاء"], name="عدد الأخطاء", opacity=0.4, yaxis="y2")])
            f.update_layout(yaxis2=dict(overlaying="y", side="right"), height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(f, use_container_width=True)
        errs = pd.DataFrame([{**e, "التاريخ": t["date"]} for t in p["trials"] for e in t.get("errors", [])])
        if len(errs):
            st.subheader("الأخطاء المتكررة")
            cnt = errs.groupby(["angle_label", "event_label"]).size().reset_index(name="التكرار").sort_values("التكرار", ascending=False)
            cnt.columns = ["الزاوية", "المرحلة", "التكرار"]
            st.dataframe(cnt, use_container_width=True, hide_index=True)
        pre, post = df[df["النوع"] == "اختبار قبلي"], df[df["النوع"] == "اختبار بعدي"]
        if len(pre) and len(post):
            st.subheader("مقارنة القبلي والبعدي (متوسط)")
            cmp = pd.DataFrame({"قبلي": pre.mean(numeric_only=True), "بعدي": post.mean(numeric_only=True)}).round(2)
            st.dataframe(cmp, use_container_width=True)
    st.download_button("⬇️ تصدير كل البيانات (Excel)", storage.to_excel(db), "shotput_report.xlsx")


def page_reference():
    st.header("⚙️ النطاقات المرجعية للزوايا")
    st.warning("القيم الافتراضية **مبدئية تقريبية** وُضعت كنقطة بداية فقط. اعتمدي نطاقات مصادر رسالتك (الأدبيات) بتعديل الجدول. "
               "لم تُوضع قيم افتراضية للكاميرا الأمامية/الخلفية؛ أضيفيها هنا أو اعتمدي نموذج اللاعبة.")
    df = pd.DataFrame(db["reference"])
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, column_config={
        "technique": st.column_config.SelectboxColumn("الأسلوب", options=["both", "glide", "rotational"]),
        "view": st.column_config.SelectboxColumn("الكاميرا", options=["side", "front"]),
        "event": st.column_config.SelectboxColumn("المرحلة", options=bm.EVENTS),
        "angle": st.column_config.SelectboxColumn("الزاوية", options=list(bm.ANGLE_LABELS)),
        "lo": st.column_config.NumberColumn("الحد الأدنى"), "hi": st.column_config.NumberColumn("الحد الأعلى")})
    c = st.columns(2)
    if c[0].button("💾 حفظ المراجع"):
        db["reference"] = edited.dropna(subset=["angle"]).to_dict("records")
        commit()
        st.success("تم الحفظ.")
    if c[1].button("↩️ استعادة الافتراضي"):
        db["reference"] = [dict(r) for r in bm.DEFAULT_REF]
        commit()
        st.rerun()
    st.markdown("**الرموز:** knee_rear/front = ركبة الخلفية/الأمامية، trunk_incl = ميلان الجذع (سالب = للخلف)، "
                "shoulder_ang = زاوية الكتف، elbow_ang = المرفق، shoulder_tilt/hip_tilt = ميل خط الكتفين/الحوض.")


# ------------------------------------------------------------------ التنقل
st.sidebar.title("🏋️ محلل دفع الثقل")
page = st.sidebar.radio("القائمة", ["🎥 تحليل رمية", "👤 اللاعبات", "🧬 النموذج الشخصي", "📁 السجل والتقدم", "⚙️ المراجع"])
st.sidebar.divider()
st.sidebar.caption("⚠️ على Streamlit Cloud قد تُمسح البيانات عند إعادة التشغيل؛ احتفظي بنسخة احتياطية:")
st.sidebar.download_button("⬇️ نسخة احتياطية (JSON)", storage.dumps(db), "backup.json")
up = st.sidebar.file_uploader("استعادة نسخة احتياطية", type="json")
if up and st.sidebar.button("استعادة"):
    import json
    st.session_state.db = json.loads(up.getvalue().decode("utf-8"))
    storage.save(st.session_state.db)
    st.rerun()
st.sidebar.caption("المطوّر: م. جعفر حمزة كاظم")

{"🎥 تحليل رمية": page_analysis, "👤 اللاعبات": page_players, "🧬 النموذج الشخصي": page_models,
 "📁 السجل والتقدم": page_records, "⚙️ المراجع": page_reference}[page]()
