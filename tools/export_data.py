"""Export the simulator catalog for static (WASM) deployment.

Writes:
  data.json          catalog consumed by the WASM build at runtime
  js/data_static.js  same catalog embedded as window.GOTHAM_CATALOG
                     (works when the page is opened via file://)
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator import data  # noqa: E402


def main():
    catalog = data.catalog()
    with open(os.path.join(ROOT, "data.json"), "w") as fh:
        json.dump(catalog, fh, indent=1)
    with open(os.path.join(ROOT, "js", "data_static.js"), "w") as fh:
        fh.write("window.GOTHAM_CATALOG = ")
        json.dump(catalog, fh)
        fh.write(";\n")
    print("wrote data.json and js/data_static.js")


if __name__ == "__main__":
    main()
