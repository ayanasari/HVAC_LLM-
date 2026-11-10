#!/usr/bin/env python3
"""
run_simulation.py — Run the distributed LLM HVAC controller against baselines.

Usage:
    python run_simulation.py                      # offline mock LLM, 48 h, all plots
    python run_simulation.py --backend anthropic  # real Claude API (needs ANTHROPIC_API_KEY)
    python run_simulation.py --hours 24 --profile mild --verbose

Outputs (results/):
    kpis.json            KPI table for every controller
    audit_log.jsonl      every LLM decision with reasoning + shield verdict
    bus_log.json         every message on the bus
    fig_*.png            temperature, power, cost plots
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hvac_llm.baseline import FixedSetpointController, ScheduledSetpointController
from hvac_llm.config import SimulationConfig, TariffConfig, default_building
from hvac_llm.environment import DemandResponseEvent
from hvac_llm.llm_backend import make_backend
from hvac_llm.simulation import run_baseline, run_llm_system


def plot_results(results: dict, building, out: Path) -> None:
    zone_ids = [z.zone_id for z in building.zones]
    zc = {z.zone_id: z for z in building.zones}
    llm_rec = results["llm_multi_agent"]["records"]
    sch_rec = results["scheduled_setpoint"]["records"]
    t = np.array([r.t_hours for r in llm_rec])

    # ---- Fig 1: zone temperatures, LLM vs scheduled -------------------
    fig, axes = plt.subplots(len(zone_ids), 1, figsize=(13, 2.6 * len(zone_ids)), sharex=True)
    for ax, z in zip(axes, zone_ids):
        occ = np.array([r.zone_occupants[z] > 0 for r in llm_rec])
        ax.fill_between(t, zc[z].comfort_low_C, zc[z].comfort_high_C, where=occ,
                        color="green", alpha=0.12, label="comfort band (occupied)")
        ax.plot(t, [r.zone_temps[z] for r in sch_rec], color="gray", lw=1.2, label="Scheduled BMS temp")
        ax.plot(t, [r.zone_temps[z] for r in llm_rec], color="tab:blue", lw=1.6, label="LLM agent temp")
        ax.step(t, [r.zone_cooling_sp[z] for r in llm_rec], color="tab:orange", lw=1,
                ls="--", where="post", label="LLM cooling setpoint")
        ax.step(t, [r.zone_heating_sp[z] for r in llm_rec], color="tab:red", lw=0.8,
                ls=":", where="post", label="LLM heating setpoint")
        ax.set_ylabel(f"{z}\n{zc[z].name}\n°C", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=5)
    axes[-1].set_xlabel("time [h]")
    _shade_peak(axes, llm_rec)
    fig.suptitle("Zone temperatures: LLM multi-agent vs scheduled BMS (red = ON-PEAK tariff)")
    fig.tight_layout()
    fig.savefig(out / "fig_zone_temperatures.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: total power + demand limit ------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for name, color in [("fixed_setpoint", "lightgray"), ("scheduled_setpoint", "gray"),
                        ("llm_multi_agent", "tab:blue")]:
        rec = results[name]["records"]
        ax.plot(t, [r.total_power_W / 1000 for r in rec], color=color, lw=1.5, label=name)
    ax.plot(t, [r.demand_limit_W / 1000 for r in llm_rec], "r--", lw=1.2, label="demand limit")
    ax2 = ax.twinx()
    ax2.plot(t, [r.outdoor_C for r in llm_rec], color="tab:red", alpha=0.4, lw=1, label="outdoor °C")
    ax2.set_ylabel("outdoor °C", color="tab:red")
    ax.set_ylabel("HVAC electrical power [kW]")
    ax.set_xlabel("time [h]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    _shade_peak([ax], llm_rec)
    fig.suptitle("Building HVAC demand: coordinator keeps the LLM system under the limit")
    fig.tight_layout()
    fig.savefig(out / "fig_total_power.png", dpi=130)
    plt.close(fig)

    # ---- Fig 3: KPI bar chart ----------------------------------------------
    names = ["fixed_setpoint", "scheduled_setpoint", "llm_multi_agent"]
    kp = {n: results[n]["kpis"] for n in names}
    metrics = [("energy_kWh", "Energy [kWh]"), ("total_cost", "Total cost [$]"),
               ("peak_demand_kW", "Peak demand [kW]"), ("discomfort_degree_h", "Discomfort [°C·h]")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (key, label) in zip(axes, metrics):
        vals = [getattr(kp[n], key) for n in names]
        bars = ax.bar(["Fixed", "Scheduled", "LLM agents"], vals,
                      color=["lightgray", "gray", "tab:blue"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("KPI comparison (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "fig_kpis.png", dpi=130)
    plt.close(fig)


def _shade_peak(axes, rec):
    t = [r.t_hours for r in rec]
    peak = [r.price >= 0.35 for r in rec]
    for ax in axes:
        start = None
        for i, p in enumerate(peak):
            if p and start is None:
                start = t[i]
            if (not p or i == len(peak) - 1) and start is not None:
                ax.axvspan(start, t[i], color="red", alpha=0.07)
                start = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "anthropic"])
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--profile", default="hot_summer", choices=["hot_summer", "mild", "cold_winter"])
    ap.add_argument("--no-dr", action="store_true", help="disable the demand-response event")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    building = default_building()
    tariff = TariffConfig()
    sim = SimulationConfig(horizon_hours=args.hours, outdoor_profile=args.profile)
    # A utility demand-response event on each weekday afternoon: cap HVAC at 9 kW 15:00-17:00
    dr = [] if args.no_dr else [DemandResponseEvent(15.0, 17.0, 9_000.0,
                                                    "Utility demand-response event")]
    backend = make_backend(args.backend)
    print(f"LLM backend: {backend.name}   horizon: {args.hours} h   profile: {args.profile}")
    print("=" * 78)

    results = {}
    for ctrl in (FixedSetpointController(building), ScheduledSetpointController(building)):
        kpis, rec = run_baseline(ctrl, building, tariff, sim, dr)
        results[ctrl.name] = {"kpis": kpis, "records": rec}
        print(f"[baseline] {ctrl.name:20s} done")

    kpis, rec, audit, agents, coord = asyncio.run(
        run_llm_system(building, tariff, sim, backend, dr, out / "audit_log.jsonl", args.verbose))
    results["llm_multi_agent"] = {"kpis": kpis, "records": rec}
    print(f"[llm]      llm_multi_agent      done  ({kpis.llm_calls} LLM calls, "
          f"{kpis.shield_clips} shield clips, {kpis.llm_fallbacks} fallbacks)")

    # ---- KPI table ----------------------------------------------------------
    print("\n" + "=" * 78)
    hdr = f"{'controller':20s} {'kWh':>8s} {'cost $':>8s} {'peak kW':>8s} {'discomf °C·h':>13s} {'viol %':>7s} {'>limit':>7s}"
    print(hdr); print("-" * len(hdr))
    for n, r in results.items():
        k = r["kpis"]
        print(f"{n:20s} {k.energy_kWh:8.1f} {k.total_cost:8.2f} {k.peak_demand_kW:8.2f} "
              f"{k.discomfort_degree_h:13.2f} {k.comfort_violation_pct:7.1f} {k.demand_limit_exceed_steps:7d}")
    base = results["scheduled_setpoint"]["kpis"]
    llm = results["llm_multi_agent"]["kpis"]
    print("-" * len(hdr))
    print(f"LLM vs scheduled BMS:  energy {100*(llm.energy_kWh/base.energy_kWh-1):+.1f}%   "
          f"cost {100*(llm.total_cost/base.total_cost-1):+.1f}%   "
          f"peak {100*(llm.peak_demand_kW/base.peak_demand_kW-1):+.1f}%   "
          f"discomfort {llm.discomfort_degree_h:.2f} vs {base.discomfort_degree_h:.2f} °C·h")

    # ---- persist ------------------------------------------------------------
    (out / "kpis.json").write_text(json.dumps({n: r["kpis"].as_dict() for n, r in results.items()}, indent=2))
    plot_results(results, building, out)

    # ---- show a few LLM reasoning samples -----------------------------------
    print("\nSample LLM decisions from the audit log:")
    shown = 0
    for rec_ in audit.records:
        if rec_["agent"] != "coordinator" and shown < 4 and rec_["t_hours"] in (7.5, 13.0, 14.5, 19.0):
            print(f"  [{rec_['agent']} @ t={rec_['t_hours']}h] heat/cool={rec_['heating_setpoint_C']}/{rec_['cooling_setpoint_C']} "
                  f"({rec_['source']})\n      → {rec_['reasoning']}")
            shown += 1
    print(f"\nFull audit log: {out/'audit_log.jsonl'}   figures: {out}/fig_*.png")


if __name__ == "__main__":
    main()
