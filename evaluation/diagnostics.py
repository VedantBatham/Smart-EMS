import os, sys, json, logging, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from stable_baselines3 import PPO
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from preprocessing.load_data import load_or_fetch_opsd_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAG_DIR = "results/diagnostics"
os.makedirs(DIAG_DIR, exist_ok=True)

ALGORITHMS = ["ppo"]
MODEL_PATHS = {
    "ppo": "models/ppo/ppo_smart_ems.zip",
}
MODEL_CLASSES = {"ppo": PPO}
NORM_LOAD, NORM_SOLAR, NORM_TARIFF = 5.0, 4.0, 12.0

def get_test_df():
    df = load_or_fetch_opsd_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    logger.info(f"Test partition: {len(test_df)} rows")
    return test_df

def verify_load():
    logger.info("=" * 60)
    logger.info("STEP 1 -- Model Loading Verification")
    logger.info("=" * 60)
    models, report = {}, {}
    for algo in ALGORITHMS:
        path = MODEL_PATHS[algo]
        cls = MODEL_CLASSES[algo]
        logger.info(f"  Loading {algo.upper()} | path={path} | class={cls.__name__}")
        if not os.path.exists(path):
            logger.error(f"    MISSING: {path}")
            report[algo] = {"status": "MISSING", "path": path, "size_bytes": 0}
            continue
        size = os.path.getsize(path)
        try:
            m = cls.load(path)
            ph = None
            try:
                params = list(m.policy.parameters())
                if params:
                    p0 = params[0].detach().cpu().numpy().flatten()[:10]
                    ph = float(np.sum(p0))
            except Exception:
                pass
            models[algo] = m
            report[algo] = {"status": "OK", "path": path, "size_bytes": size, "class": cls.__name__, "param_fingerprint": ph}
            logger.info(f"    OK  size={size:,} B  fingerprint={ph:.6f}" if ph is not None else f"    OK  size={size:,} B")
        except Exception as e:
            report[algo] = {"status": f"ERROR:{e}", "path": path, "size_bytes": size}
            logger.error(f"    ERROR: {e}")
    fps = {a: report[a].get("param_fingerprint") for a in ALGORITHMS if a in report and report[a].get("param_fingerprint") is not None}
    for (a1,f1),(a2,f2) in itertools.combinations(fps.items(), 2):
        d = abs(f1 - f2)
        if d < 1e-6:
            logger.warning(f"  !!IDENTICAL fingerprints: {a1} vs {a2} -- possible shared model!!")
        else:
            logger.info(f"  {a1} vs {a2} fingerprint diff = {d:.6f}")
    return models, report

def make_battery(soc=0.50):
    return Battery(capacity_kwh=5.0, initial_soc=soc, min_soc=0.20, max_soc=0.90, max_charge_kw=1.5, max_discharge_kw=1.5, charge_efficiency=0.95, discharge_efficiency=0.95, time_step_hours=0.25)

def build_obs(row, battery, tariff_model):
    load_kw = float(row.get("load_kw", 1.0))
    solar_kw = float(row.get("solar_kw", 0.0))
    hour = int(row.get("hour", 0))
    minute = int(row.get("minute", 0))
    tariff = float(row.get("current_tariff", tariff_model.get_tariff(hour, minute)))
    nl = np.clip(load_kw/NORM_LOAD, 0, 1)
    ns = np.clip(solar_kw/NORM_SOLAR, 0, 1)
    nt = np.clip(tariff/NORM_TARIFF, 0, 1)
    sh = np.sin(2*np.pi*(hour+minute/60)/24)
    ch = np.cos(2*np.pi*(hour+minute/60)/24)
    if "future_load_1h_avg" in row and not pd.isna(row.get("future_load_1h_avg", float("nan"))):
        fl1=float(row["future_load_1h_avg"]); fs1=float(row["future_solar_1h_avg"])
        fl2=float(row["future_load_2h_avg"]); fs2=float(row["future_solar_2h_avg"])
    else:
        fl1=load_kw; fs1=solar_kw; fl2=load_kw; fs2=solar_kw
    nht = tariff_model.get_next_hour_max_tariff(hour, minute)
    ft = tariff_model.get_future_tariffs(hour, minute, horizon_steps=8)
    n2ht = max(ft) if ft else nht
    return np.array([nl, ns, battery.soc, nt, np.clip(fl1/NORM_LOAD,0,1), np.clip(fs1/NORM_SOLAR,0,1), np.clip(nht/NORM_TARIFF,0,1), sh, ch, np.clip(fl2/NORM_LOAD,0,1), np.clip(fs2/NORM_SOLAR,0,1), np.clip(n2ht/NORM_TARIFF,0,1)], dtype=np.float32)

def to_cont(algo, raw):
    return float(np.atleast_1d(raw)[0])

def act2batt(cont, battery, solar, load, tariff):
    if cont > 0.01:
        md = battery.get_max_feasible_discharge_kw()
        nd = max(0.0, load - solar)
        if tariff >= 5.0:
            sc = min(1.0, max(0.6, cont*2.0))
            req = sc * md
        else:
            req = cont * battery.max_discharge_kw
        ex = min(req, md, nd)
        cr = "none" if abs(ex - req) < 0.01 else "clipped_discharge"
    elif cont < -0.01 and tariff <= 4.0:
        mc = battery.get_max_feasible_charge_kw()
        ch = min(abs(cont)*battery.max_charge_kw, mc)
        req = -ch; ex = -ch; cr = "none"
    else:
        req = 0.0; ex = 0.0; cr = "idle_or_hightar"
    return req, ex, cr

def run_trajectory(models, test_df, tariff_model):
    logger.info("=" * 60)
    logger.info("STEP 2-10 -- Trajectory Diagnostic (96 steps = 1 day)")
    logger.info("=" * 60)
    day = test_df.iloc[:96].reset_index(drop=True)
    batts = {a: make_battery(0.50) for a in ALGORITHMS}
    obs0 = {a: [] for a in ALGORITHMS}
    cont_acts = {a: [] for a in ALGORITHMS}
    exec_kws = {a: [] for a in ALGORITHMS}
    socs = {a: [] for a in ALGORITHMS}
    grids = {a: [] for a in ALGORITHMS}
    costs = {a: [] for a in ALGORITHMS}
    clips = {a: 0 for a in ALGORITHMS}
    records = []
    for t, row in day.iterrows():
        solar = float(row["solar_kw"]); load = float(row["load_kw"])
        hour = int(row["hour"]); minute = int(row["minute"])
        ts = str(row.get("timestamp", f"step_{t:03d}"))
        tariff = float(row.get("current_tariff", tariff_model.get_tariff(hour, minute)))
        ovr = (solar - load) > 0.0
        rec = {"timestamp": ts, "solar_kw": solar, "load_kw": load, "tariff": tariff, "solar_override": ovr}
        for algo in ALGORITHMS:
            b = batts[algo]
            obs = build_obs(row, b, tariff_model)
            if t < 10:
                obs0[algo].append(obs.copy())
            if algo not in models:
                continue
            if ovr:
                ex_s = solar - load; mc = b.get_max_feasible_charge_kw(); ch = min(ex_s, mc)
                cont = -1.0; req_kw = -ch; ex_kw = -ch; cr = "solar_override"
            else:
                raw, _ = models[algo].predict(obs, deterministic=True)
                cont = to_cont(algo, raw)
                req_kw, ex_kw, cr = act2batt(cont, b, solar, load, tariff)
                if cr == "clipped_discharge":
                    clips[algo] += 1
            act_kw, soc_end, _ = b.step(ex_kw)
            grid_kw = max(0.0, load - solar - act_kw)
            pc = costs[algo][-1] if costs[algo] else 0.0
            costs[algo].append(pc + grid_kw*tariff*0.25)
            cont_acts[algo].append(cont)
            exec_kws[algo].append(act_kw)
            socs[algo].append(soc_end)
            grids[algo].append(grid_kw)
            rec[f"{algo}_raw"] = round(cont,4)
            rec[f"{algo}_req_kw"] = round(req_kw,4)
            rec[f"{algo}_exec_kw"] = round(act_kw,4)
            rec[f"{algo}_soc"] = round(soc_end,4)
            rec[f"{algo}_grid_kw"] = round(grid_kw,4)
            rec[f"{algo}_clip"] = cr
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DIAG_DIR, "model_action_comparison.csv"), index=False)
    logger.info(f"Saved model_action_comparison.csv ({len(df)} rows)")
    logger.info("\n  First 30 steps -- Executed kW (PPO / SAC / Solar?)")
    for i in range(min(30, len(df))):
        r = df.iloc[i]
        logger.info(f"  t={i:3d}  ppo={r.get('ppo_exec_kw',0):7.3f}  sac={r.get('sac_exec_kw',0):7.3f}  solar={'YES' if r.get('solar_override') else 'no'}")
    logger.info("\n  First 30 steps -- Raw continuous actions [-1,1]")
    for i in range(min(30, len(df))):
        r = df.iloc[i]
        logger.info(f"  t={i:3d}  ppo={r.get('ppo_raw',0):7.3f}  sac={r.get('sac_raw',0):7.3f}")
    return df, obs0, clips, cont_acts, exec_kws, socs, grids, costs

def calc_diffs(cont_acts, exec_kws, n):
    logger.info("=" * 60)
    logger.info("STEP 10 -- Action Differences")
    logger.info("=" * 60)
    diffs = {}
    for a1, a2 in itertools.combinations(ALGORITHMS, 2):
        c1 = np.array(cont_acts.get(a1, [0]*n)); c2 = np.array(cont_acts.get(a2, [0]*n))
        e1 = np.array(exec_kws.get(a1, [0]*n));  e2 = np.array(exec_kws.get(a2, [0]*n))
        mad_c = float(np.mean(np.abs(c1-c2)))
        mad_e = float(np.mean(np.abs(e1-e2)))
        pct = float(np.mean(np.abs(e1-e2)<0.05)*100)
        key = f"{a1}_vs_{a2}"
        diffs[key] = {"mad_raw": round(mad_c,4), "mad_exec_kw": round(mad_e,4), "pct_same": round(pct,1)}
        logger.info(f"  {a1.upper()} vs {a2.upper()}: MAD_raw={mad_c:.4f}  MAD_exec={mad_e:.4f}  pct_same_exec={pct:.1f}%")
    return diffs

def single_state(models, tariff_model):
    logger.info("=" * 60)
    logger.info("STEP 13 -- Controlled Single-State Policy Tests")
    logger.info("=" * 60)
    cases = {
        "A_high_tariff_high_soc": {"load_kw":2.5,"solar_kw":0.2,"soc":0.80,"hour":19,"minute":0,"tariff":9.0,"expect":"DISCHARGE"},
        "B_low_tariff_med_soc":   {"load_kw":1.5,"solar_kw":0.3,"soc":0.50,"hour":2,"minute":0,"tariff":3.0,"expect":"CHARGE"},
        "C_solar_surplus":        {"load_kw":0.8,"solar_kw":2.5,"soc":0.55,"hour":12,"minute":0,"tariff":5.0,"expect":"CHARGE_SOLAR"},
        "D_near_min_soc":         {"load_kw":2.0,"solar_kw":0.1,"soc":0.22,"hour":20,"minute":0,"tariff":10.0,"expect":"IDLE"},
    }
    results = {}
    for cn, cfg in cases.items():
        logger.info(f"\n  Case {cn} (expect={cfg['expect']})")
        row = pd.Series({"load_kw":cfg["load_kw"],"solar_kw":cfg["solar_kw"],"hour":cfg["hour"],"minute":cfg["minute"],"current_tariff":cfg["tariff"],"future_load_1h_avg":cfg["load_kw"],"future_solar_1h_avg":cfg["solar_kw"],"future_load_2h_avg":cfg["load_kw"],"future_solar_2h_avg":cfg["solar_kw"]})
        cr = {}
        for algo in ALGORITHMS:
            if algo not in models:
                continue
            b = make_battery(cfg["soc"])
            obs = build_obs(row, b, tariff_model)
            raw, _ = models[algo].predict(obs, deterministic=True)
            cont = to_cont(algo, raw)
            req, ex, clip = act2batt(cont, b, cfg["solar_kw"], cfg["load_kw"], cfg["tariff"])
            cr[algo] = {"raw": round(cont,4), "req_kw": round(req,4), "exec_kw": round(ex,4), "clip": clip}
            logger.info(f"    {algo.upper()}: raw={cont:7.3f}  req={req:7.3f} kW  exec={ex:7.3f} kW  ({clip})")
        results[cn] = cr
    return results

def make_plots(df, socs, grids, costs):
    colors = {"ppo":"#38bdf8","sac":"#f59e0b"}
    n = len(df); x = np.arange(n)
    def styled_fig():
        fig, ax = plt.subplots(figsize=(14,4))
        fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#334155")
        return fig, ax
    # Battery power
    fig, ax = styled_fig()
    for a in ALGORITHMS:
        col = f"{a}_exec_kw"
        if col in df.columns:
            ax.plot(x, df[col].values, label=a.upper(), color=colors[a], lw=1.5, alpha=0.9)
    ax.axhline(0, color="#475569", lw=0.8, ls="--")
    ax.set_title("Battery Power (kW): +Discharge / -Charge", color="white", fontsize=12)
    ax.set_xlabel("Timestep (15 min)", color="#94a3b8"); ax.set_ylabel("kW", color="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"action_comparison.png"),dpi=120); plt.close()
    # SOC
    fig, ax = styled_fig()
    for a in ALGORITHMS:
        if socs.get(a):
            ax.plot(x[:len(socs[a])], np.array(socs[a])*100, label=a.upper(), color=colors[a], lw=1.5, alpha=0.9)
    ax.axhline(20, color="#ef4444", lw=1, ls=":"); ax.axhline(90, color="#34d399", lw=1, ls=":")
    ax.set_title("Battery SOC Trajectories (%)", color="white", fontsize=12)
    ax.set_xlabel("Timestep (15 min)", color="#94a3b8"); ax.set_ylabel("SOC %", color="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"soc_comparison.png"),dpi=120); plt.close()
    # Grid
    fig, ax = styled_fig()
    for a in ALGORITHMS:
        if grids.get(a):
            ax.plot(x[:len(grids[a])], grids[a], label=a.upper(), color=colors[a], lw=1.5, alpha=0.9)
    ax.set_title("Grid Import (kW)", color="white", fontsize=12)
    ax.set_xlabel("Timestep (15 min)", color="#94a3b8"); ax.set_ylabel("kW", color="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"grid_comparison.png"),dpi=120); plt.close()
    # Cumulative cost
    fig, ax = styled_fig()
    for a in ALGORITHMS:
        if costs.get(a):
            ax.plot(x[:len(costs[a])], costs[a], label=a.upper(), color=colors[a], lw=1.8, alpha=0.9)
    ax.set_title("Cumulative Electricity Cost (Rs.)", color="white", fontsize=12)
    ax.set_xlabel("Timestep (15 min)", color="#94a3b8"); ax.set_ylabel("Rs.", color="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"cumulative_cost_comparison.png"),dpi=120); plt.close()
    logger.info("4 plots saved to results/diagnostics/")

def build_report(load_report, diffs, clips, ovr_pct, obs_rep, ss, n):
    mad_es = [v["mad_exec_kw"] for v in diffs.values()]
    pct_ss = [v["pct_same"] for v in diffs.values()]
    mean_mad = float(np.mean(mad_es)) if mad_es else 0
    mean_pct = float(np.mean(pct_ss)) if pct_ss else 100
    any_clip = any(clips.get(a,0)/max(1,n)*100 > 30 for a in ALGORITHMS)
    if ovr_pct > 40 and mean_mad < 0.05:
        conc = "3. SOLAR OVERRIDE DOMINATES -- Solar priority logic bypasses RL for >40% of timesteps, masking policy differences."
    elif any_clip and mean_mad < 0.05:
        conc = "2. ACTION CLIPPING SIMILARITY -- High clipping rate collapses action diversity."
    elif mean_mad < 0.03:
        conc = "1. IMPLEMENTATION ISSUE -- Raw actions nearly identical. Possible shared weights or model reuse."
    elif mean_mad > 0.10 and mean_pct < 60:
        conc = "4. GENUINE POLICY DIVERSITY -- Models produce clearly different actions; metric similarity driven by physical constraints."
    else:
        conc = "5. INCONCLUSIVE -- Partial differences detected. Physical environment constraints partially mask policy diversity."
    lines = [
        "# RL Policy Differentiation Diagnostic Report\n",
        "## 1. Model Loading\n",
        "| Algorithm | Status | Size (bytes) | Param Fingerprint |",
        "|-----------|--------|-------------|-------------------|",
    ]
    for a in ALGORITHMS:
        r = load_report.get(a, {})
        lines.append(f"| {a.upper()} | {r.get('status','N/A')} | {r.get('size_bytes',0):,} | {r.get('param_fingerprint','N/A')} |")
    lines += [
        f"\n## 2. Solar Priority Override\n",
        f"**{ovr_pct:.1f}%** of timesteps were overridden by solar-priority logic (solar > load), bypassing the RL model.\n",
        "> If this exceeds 40%, policy comparisons are partially masked.\n",
        "## 3. Clipping Analysis\n",
        "| Algorithm | Clipped Steps | % Clipped |",
        "|-----------|--------------|-----------|",
    ]
    for a in ALGORITHMS:
        c = clips.get(a, 0)
        lines.append(f"| {a.upper()} | {c} | {c/max(1,n)*100:.1f}% |")
    lines += [
        "\n## 4. Mean Absolute Action Differences\n",
        "| Pair | MAD Raw Action | MAD Exec kW | % Same Executed (±0.05 kW) |",
        "|------|---------------|-------------|----------------------------|",
    ]
    for k, v in diffs.items():
        lines.append(f"| {k} | {v['mad_raw']:.4f} | {v['mad_exec_kw']:.4f} | {v['pct_same']:.1f}% |")
    lines += [
        f"\n**Mean MAD exec (all pairs):** {mean_mad:.4f} kW  |  **Mean % same:** {mean_pct:.1f}%\n",
        "## 5. Observation Consistency (Step 0)\n",
    ]
    for k, v in obs_rep.items():
        lines.append(f"- `{k}`: {v:.6f}" if isinstance(v, float) else f"- `{k}`: N/A")
    lines.append("\n## 6. Controlled Single-State Tests\n")
    for cn, cr in ss.items():
        lines.append(f"### {cn}")
        lines.append("| Algorithm | Raw Action | Req kW | Exec kW | Clip Reason |")
        lines.append("|-----------|-----------|--------|---------|-------------|")
        for a in ALGORITHMS:
            r = cr.get(a, {})
            lines.append(f"| {a.upper()} | {r.get('raw','N/A')} | {r.get('req_kw','N/A')} | {r.get('exec_kw','N/A')} | {r.get('clip','N/A')} |")
        lines.append("")
    lines += [
        "\n---\n",
        "## CONCLUSION\n",
        f"**Solar override:** {ovr_pct:.1f}%  |  **Mean MAD exec:** {mean_mad:.4f} kW  |  **Mean % same:** {mean_pct:.1f}%",
        f"\n> **{conc}**\n",
        "## Generated Artifacts\n",
        "| File | Description |",
        "|------|-------------|",
        "| `model_action_comparison.csv` | Per-timestep actions for all 4 models |",
        "| `action_difference_metrics.json` | Numeric MAD, clipping, obs metrics |",
        "| `action_comparison.png` | Battery power trajectories |",
        "| `soc_comparison.png` | SOC trajectories |",
        "| `grid_comparison.png` | Grid import trajectories |",
        "| `cumulative_cost_comparison.png` | Cumulative cost trajectories |",
    ]
    return "\n".join(lines)

def main():
    test_df = get_test_df()
    tariff_model = TariffModel(config_path="config.yaml")
    models, load_report = verify_load()
    df, obs0, clips, cont_acts, exec_kws, socs, grids, costs = run_trajectory(models, test_df, tariff_model)
    n = len(df)
    diffs = calc_diffs(cont_acts, exec_kws, n)
    ovr_pct = float(df["solar_override"].mean() * 100)
    obs_rep = {}
    ref = obs0.get(ALGORITHMS[0], [None])[0]
    for a in ALGORITHMS[1:]:
        cmp = obs0.get(a, [None])[0]
        if ref is not None and cmp is not None:
            obs_rep[f"ppo_vs_{a}_obs0_maxdiff"] = float(np.max(np.abs(ref - cmp)))
        else:
            obs_rep[f"ppo_vs_{a}_obs0_maxdiff"] = None
    ss = single_state(models, tariff_model)
    make_plots(df, socs, grids, costs)
    payload = {
        "model_loading": load_report,
        "action_differences": diffs,
        "clipping": {a: {"count": clips.get(a,0), "pct": round(clips.get(a,0)/max(1,n)*100,2)} for a in ALGORITHMS},
        "obs_consistency": obs_rep,
        "single_state_tests": ss,
        "solar_override_pct": ovr_pct,
        "n_diagnostic_steps": n,
    }
    with open(os.path.join(DIAG_DIR, "action_difference_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved action_difference_metrics.json")
    report = build_report(load_report, diffs, clips, ovr_pct, obs_rep, ss, n)
    with open(os.path.join(DIAG_DIR, "diagnostic_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved diagnostic_report.md")
    with open(os.path.join(DIAG_DIR, "algorithm_policy_audit.md"), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved algorithm_policy_audit.md")
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info(f"Solar override pct : {ovr_pct:.1f}%")
    if diffs:
        logger.info(f"Mean MAD exec kW   : {np.mean([v['mad_exec_kw'] for v in diffs.values()]):.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
