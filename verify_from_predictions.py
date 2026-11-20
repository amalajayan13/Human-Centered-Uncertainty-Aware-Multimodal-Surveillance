"""Independent re-computation of Tables 4, 6 and the window composition from the
exported event-level predictions of the locked run, using only the definitions
stated in the manuscript (Sections 3.3-3.6). No model is refitted."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = sys.argv[1]  # path to the exported tables/ directory
P = T + "/predictions/"
GRID = np.r_[np.linspace(0, 1, 101), np.nextafter(1.0, 2.0)]

def amb(p): return 1 - 2 * np.abs(p - 0.5)

def ba_fpr(y, s, t):
    pr = s >= t; y = y.astype(bool)
    tpr = (pr & y).sum() / y.sum(); fpr = (pr & ~y).sum() / (~y).sum()
    return (tpr + 1 - fpr) / 2, fpr

def select(y, s):
    keep = np.isfinite(s); y, s = y[keep], s[keep]
    best = None
    for t in GRID:
        ba, fpr = ba_fpr(y, s, t)
        key = (ba, -fpr, -abs(t - 0.5), t)
        if best is None or key > best: best = key
    return best[-1]

# ---- Branch metrics (Table 4) from stored test predictions -------------
rows = []
for b, m, fs in [("cyber","RandomForest","full"),("cyber","LogisticRegression","full"),
                 ("cyber","CalibratedLogisticRegression","full"),("cyber","ExtraTrees","full"),
                 ("video","ExtraTrees","full"),("video","LogisticRegression","full"),("video","RandomForest","full"),
                 ("video","ExtraTrees","ablated"),("video","LogisticRegression","ablated"),("video","RandomForest","ablated")]:
    d = pd.read_csv(f"{P}{b}_{fs}_{m}_test.csv.gz")
    t = d.threshold.iloc[0]; y = d.label.to_numpy(); s = d.probability.to_numpy()
    ba, fpr = ba_fpr(y, s, t)
    rows.append(dict(branch=b, set=fs, model=m, tau=t, bal_acc=ba, roc_auc=roc_auc_score(y, s), fpr=fpr))
branch = pd.DataFrame(rows)
ref = pd.concat([pd.read_csv(T+"/cyber_summary_ci.csv"), pd.read_csv(T+"/video_summary_ci.csv")])
mrg = branch.merge(ref[["branch","feature_set","model","bal_acc","roc_auc","fpr"]],
                   left_on=["branch","set","model"], right_on=["branch","feature_set","model"], suffixes=("","_ref"))
print("Branch metrics, max |difference| vs exported tables:",
      float(np.abs(mrg[["bal_acc","roc_auc","fpr"]].to_numpy() - mrg[["bal_acc_ref","roc_auc_ref","fpr_ref"]].to_numpy()).max()))

# ---- Window construction (Section 3.4) ---------------------------------
ORIGIN = pd.Timestamp("2024-01-01")
def windows(role, width=120):
    c = pd.read_csv(f"{P}cyber_events_{role}.csv.gz", parse_dates=["timestamp"])
    v = pd.read_csv(f"{P}video_events_{role}.csv.gz", parse_dates=["timestamp"])
    out = []
    for name, e in (("c", c), ("v", v)):
        e = e.assign(w=((e.timestamp - ORIGIN).dt.total_seconds() // width).astype(int), u=amb(e.probability))
        out.append(e.groupby("w").agg(**{f"p_{name}":("probability","mean"), f"u_{name}":("u","mean"),
                                         f"n_{name}":("probability","size"), f"y_{name}":("label","max")}))
    w = out[0].join(out[1], how="outer").sort_index()
    for k in ("n_c","n_v","y_c","y_v"): w[k] = w[k].fillna(0).astype(int)
    w["or_"] = np.maximum(w.y_c, w.y_v); w["and_"] = np.minimum(w.y_c, w.y_v)
    return w

def scores(w, ctl):
    pres = np.c_[w.n_c > 0, w.n_v > 0]
    p = np.where(pres, w[["p_c","p_v"]].to_numpy(), 0.0); u = np.where(pres, w[["u_c","u_v"]].to_numpy(), 1.0)
    def wf(cap=1.0, k=0.0, fixed=None):
        conf = np.minimum(1 - u, cap) if fixed is None else np.tile(fixed, (len(w), 1)).astype(float)
        conf = np.where(pres, conf, 0.0); tot = conf.sum(1, keepdims=True)
        eq = pres / np.maximum(pres.sum(1, keepdims=True), 1)
        wt = np.where(tot > 1e-15, conf / np.where(tot > 0, tot, 1), eq)
        return np.clip((wt * p).sum(1) - k * (wt * u).sum(1), 0, 1)
    def ds(disc):
        r = np.where(pres, (1 - u) if disc else 1.0, 0.0); a, b, h = r * p, r * (1 - p), 1 - r
        A = a[:,0]*a[:,1] + a[:,0]*h[:,1] + h[:,0]*a[:,1]; B = b[:,0]*b[:,1] + b[:,0]*h[:,1] + h[:,0]*b[:,1]; H = h[:,0]*h[:,1]
        tot = A + B + H; return np.where(tot > 1e-12, (A + H/2) / np.where(tot > 0, tot, 1), 0.5)
    pat = (pres[:,0].astype(int) * 2 + pres[:,1].astype(int))
    return {"CyberOnly": np.where(pres[:,0], p[:,0], ctl["prior"]), "VideoOnly": np.where(pres[:,1], p[:,1], ctl["prior"]),
            "AvailabilityOnly": np.array([ctl["avail"][k] for k in pat]),
            "Mean": wf(fixed=[.5,.5]), "Maximum": np.max(np.where(pres, p, -np.inf), 1),
            "FixedCyber75": wf(fixed=[.75,.25]), "FixedCyber25": wf(fixed=[.25,.75]),
            "ConfidenceUncapped": wf(), "ConfidenceCapped": wf(cap=.8), "ConfidencePenalty": wf(k=.3),
            "DempsterShafer": ds(False), "DiscountedDempsterShafer": ds(True), "UVIF": wf(cap=.8, k=.3)}

va, te = windows("threshold_selection"), windows("test")
print("Test windows:", len(te), "| dual:", int(((te.n_c>0)&(te.n_v>0)).sum()), "| surveillance-only:",
      int(((te.n_c==0)&(te.n_v>0)).sum()), "| cyber-only:", int(((te.n_c>0)&(te.n_v==0)).sum()),
      "| surveillance events per window:", sorted(te.n_v.unique()), "| cyber per dual window:", sorted(te.n_c[te.n_c>0].unique()))
print("Threshold-selection windows:", len(va), "| dual:", int(((va.n_c>0)&(va.n_v>0)).sum()))

ref = pd.read_csv(T + "/fusion_baselines_raw.csv").query("subset == 'all_observed'")
worst = 0
for tgt, col in (("target_or","or_"),("target_and","and_")):
    y = va[col].to_numpy(); prior = y.mean()
    pat = ((va.n_c>0).astype(int)*2 + (va.n_v>0).astype(int)).to_numpy()
    ctl = {"prior": prior, "avail": {k: (y[pat==k].sum() + prior) / ((pat==k).sum() + 1) for k in range(4)}}
    sv, st = scores(va, ctl), scores(te, ctl)
    for m in sv:
        th = select(y, sv[m]); ba, fpr = ba_fpr(te[col].to_numpy(), st[m], th)
        r = ref[(ref.target==tgt) & (ref.method==m)].iloc[0]
        worst = max(worst, abs(ba - r.bal_acc), abs(fpr - r.fpr), abs(th - r.threshold))
        if m == "UVIF": print(f"{tgt}: UVIF theta={th:.2f} BA={ba:.4f} (exported {r.bal_acc:.4f}) FPR={fpr:.4f}")
print("Fusion table (13 non-constant rules x 2 targets), max |difference| in BA/FPR/threshold:", worst)
