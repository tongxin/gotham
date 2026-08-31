"""Command-line access to the C++ core."""

import argparse
import json

from . import core, data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Gotham roofline simulator from the CLI")
    parser.add_argument("--model", required=True, help="model id, e.g. llama3-8b")
    parser.add_argument("--gpu", default="h100", help="gpu id, e.g. h100")
    parser.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", choices=list(core.PRECISION_ENUM), default="fp16")
    parser.add_argument("--kv-precision", choices=list(core.KV_ENUM), default="fp16")
    parser.add_argument("--sweep", action="store_true", help="include decode batch-size sweep")
    args = parser.parse_args(argv)

    model = data.get_model(args.model)
    gpu = data.get_gpu(args.gpu)
    cfg = {
        "phase": args.phase,
        "B": args.batch,
        "S": args.seq,
        "gpus": args.gpus,
        "precision": args.precision,
        "kvPrecision": args.kv_precision,
        "computeScale": 1.0,
        "bandwidthScale": 1.0,
    }
    out = {
        "model": model,
        "gpu": gpu,
        "cfg": cfg,
        "sim": core.simulate(model, gpu, cfg),
        "memory": core.memory_footprint(model, cfg),
    }
    if args.sweep:
        out["sweep"] = core.decode_sweep(model, gpu, cfg)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
