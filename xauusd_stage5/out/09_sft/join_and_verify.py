"""Rebuild the full SFT files from the split gzip parts and verify them.
    python join_and_verify.py          # writes aurora_forex_combined_{train,val}.jsonl next to this script
"""
import glob, gzip, hashlib, json, os, shutil, sys
here = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(here, "SFT_MANIFEST.json")))
for name, meta in man["files"].items():
    parts = sorted(glob.glob(os.path.join(here, name + ".gz.part*")))
    if not parts:
        sys.exit(f"no parts for {name}")
    tmp = os.path.join(here, name + ".gz")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    h, n = hashlib.sha256(), 0
    with gzip.open(tmp, "rb") as g, open(os.path.join(here, name), "wb") as o:
        for chunk in iter(lambda: g.read(1 << 20), b""):
            h.update(chunk); o.write(chunk); n += chunk.count(b"\n")
    os.remove(tmp)
    ok = h.hexdigest() == meta["sha256"] and n == meta["rows"]
    print(name, "OK" if ok else "MISMATCH", n, "rows")
    if not ok:
        sys.exit(1)
