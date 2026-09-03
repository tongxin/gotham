"""Replay measured benchmark records through the L1 core and report errors.

Error definition (signed, fraction):
    error = (predicted - measured) / measured
so a positive error means the simulator over-predicts the metric (or the
measured value is better than predicted); a negative error means the
simulator under-predicts (predicts faster/less time than measured).

Each record in validation_data.py is run in two operating-point modes:
  spec       computeScale=1.0, bandwidthScale=1.0 (classic upper bound)
  realistic  computeScale/bandwidthScale from data.REALISTIC

CLI:
    python3 -m simulator.validate [--out benchmarks/validation_report.json]
"""

import argparse
import datetime
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import core, data, validation_data  # noqa: E402


def _record_cfg(rec, phase):
    return {
        "phase": phase,
        "B": int(rec["B"]),
        "S": int(rec["S"]),
        "gpus": int(rec["gpus"]),
        "precision": rec["precision"],
        "kvPrecision": rec["kvPrecision"],
    }


def _decode_result(rec, model, gpu, scale_pair):
    compute_eff, bw_eff = scale_pair
    cfg = _record_cfg(rec, "decode")
    cfg["computeScale"] = compute_eff
    cfg["bandwidthScale"] = bw_eff
    sim = core.simulate(model, gpu, cfg)
    ph = sim["decode"]
    if ph is None:
        raise RuntimeError(f"decode phase missing for {rec['id']}")
    return {
        "cfg": cfg,
        "stepTimeMs": ph["time"] * 1e3,
        "interTokenLatencyMs": ph["time"] * 1e3,
        "throughputTokPerSec": ph["throughput"],
        "bytesPerStep": ph["bytes"],
        "wBytes": sim["wBytes"],
        "decodeWBytes": sim["decodeWBytes"],
        "decodeStreamedB": sim["decodeStreamedB"],
        "bound": ph["bound"],
        "utilization": ph["utilization"],
    }


def _composite_result(rec, model, gpu, scale_pair):
    compute_eff, bw_eff = scale_pair
    waves = math.ceil(int(rec["N"]) / int(rec["B"]))
    pre_cfg = _record_cfg(rec, "prefill")
    pre_cfg["computeScale"] = compute_eff
    pre_cfg["bandwidthScale"] = bw_eff
    dec_cfg = _record_cfg(rec, "decode")
    dec_cfg["computeScale"] = compute_eff
    dec_cfg["bandwidthScale"] = bw_eff
    pre = core.simulate(model, gpu, pre_cfg)["prefill"]
    dec = core.simulate(model, gpu, dec_cfg)["decode"]
    if pre is None or dec is None:
        raise RuntimeError(f"phase missing for {rec['id']}")
    n_out = int(rec["O"])
    total_s = waves * (pre["time"] + n_out * dec["time"])
    total_tokens = int(rec["N"]) * (int(rec["S"]) + n_out)
    return {
        "waves": waves,
        "totalDurationS": total_s,
        "throughputTokPerSec": total_tokens / total_s,
        "outputTokPerSec": int(rec["N"]) * n_out / total_s,
        "prefillTimeS": pre["time"],
        "decodeStepMs": dec["time"] * 1e3,
        "prefillBound": pre["bound"],
        "decodeBound": dec["bound"],
    }


def _implied_efficiency(rec, bytes_per_step, gpu):
    """Aggregate HBM bandwidth fraction implied by the measured step time."""
    if rec["metric"] != "interTokenLatencyMs" or rec["comparable"] != "decode":
        return None
    bw = float(gpu["bandwidth_GBps"]) * 1e9 * int(rec["gpus"])
    return bytes_per_step / (rec["measured"] / 1e3) / bw


def predict(rec, model, gpu, scale_pair):
    """Run one record in one operating-point mode."""
    if rec["comparable"] == "decode":
        result = _decode_result(rec, model, gpu, scale_pair)
        primary = result["interTokenLatencyMs"]
    elif rec["comparable"] == "composite_serial":
        result = _composite_result(rec, model, gpu, scale_pair)
        primary = result["totalDurationS"]
    else:  # reference rows: no L1 prediction
        return None
    return {"result": result, "primary": primary}


def run_modes(rec, model, gpu):
    realistic = data.REALISTIC
    pairs = {
        "spec": (1.0, 1.0),
        "realistic": (
            float(realistic["computeEfficiency"]),
            float(realistic["bandwidthEfficiency"]),
        ),
    }
    implied = None
    if rec["comparable"] == "decode":
        dec = _decode_result(rec, model, gpu, (1.0, 1.0))
        implied = _implied_efficiency(rec, dec["bytesPerStep"], gpu)
    row = {
        "id": rec["id"],
        "source": rec["source"],
        "url": rec["url"],
        "published": rec.get("published"),
        "accessed": rec.get("accessed"),
        "model": rec["model"],
        "gpu": rec["gpu"],
        "precision": rec["precision"],
        "kvPrecision": rec["kvPrecision"],
        "phase": rec["phase"],
        "B": rec["B"],
        "S": rec["S"],
        "gpus": rec["gpus"],
        "engine": rec.get("engine"),
        "engineVersion": rec.get("engineVersion"),
        "metric": rec["metric"],
        "measured": rec["measured"],
        "measuredThroughputTokPerSec": rec.get("measuredThroughputTokPerSec"),
        "measuredOutputTokPerSec": rec.get("measuredOutputTokPerSec"),
        "notes": rec.get("notes"),
        "comparable": rec["comparable"],
        "impliedBandwidthEfficiency": implied,
        "modes": {},
    }
    for mode, pair in pairs.items():
        pred = predict(rec, model, gpu, pair)
        if pred is None:
            row["modes"][mode] = None
            continue
        error = (pred["primary"] - rec["measured"]) / rec["measured"]
        row["modes"][mode] = {
            "predicted": pred["primary"],
            "signedErrorPct": error * 100.0,
            "absErrorPct": abs(error) * 100.0,
            "bound": pred["result"].get("bound") or pred["result"].get("decodeBound"),
            **pred["result"],
        }
    return row


def summarize(rows):
    out = {"modes": {}, "compositeSerial": [], "reference": []}
    for mode in ("spec", "realistic"):
        errs = []
        for r in rows:
            if r["comparable"] != "decode":
                continue
            m = r["modes"].get(mode)
            if m and m.get("signedErrorPct") is not None:
                errs.append(m["signedErrorPct"])
        n = len(errs)
        out["modes"][mode] = {
            "rows": n,
            "mapePct": sum(abs(e) for e in errs) / n if n else None,
            "signedMeanPct": sum(errs) / n if n else None,
            "maxAbsPct": max((abs(e) for e in errs), default=None),
        }
    for r in rows:
        if r["comparable"] == "composite_serial":
            out["compositeSerial"].append(
                {
                    "id": r["id"],
                    "measured": r["measured"],
                    **{
                        mode: ({"signedErrorPct": r["modes"][mode]["signedErrorPct"],
                               "predicted": r["modes"][mode]["predicted"]}
                              if r["modes"][mode] else None)
                        for mode in ("spec", "realistic")
                    },
                }
            )
        elif r["comparable"] == "reference":
            out["reference"].append({"id": r["id"], "measured": r["measured"]})
    return out


def run_all():
    rows = []
    for rec in validation_data.BENCHMARKS:
        model = data.get_model(rec["model"])
        gpu = data.get_gpu(rec["gpu"])
        rows.append(run_modes(rec, model, gpu))
    return rows, summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "benchmarks", "validation_report.json"),
    )
    parser.add_argument("--print", action="store_true", help="print a readable summary")
    args = parser.parse_args(argv)

    rows, summary = run_all()
    report = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "coreVersion": core.version(),
        "realistic": data.REALISTIC,
        "errorDefinition": (
            "signed error = (predicted - measured) / measured; "
            "positive = simulator over-predicts the metric"
        ),
        "aggregate": summary,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    if args.print:
        print(json.dumps(summary, indent=2))
    else:
        print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
