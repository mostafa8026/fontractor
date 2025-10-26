#!/usr/bin/env python3
import argparse
import os
import time
import random
import requests
import urllib.parse

"""
sample-picsum-dataset.py

Simple script to generate a small picsum.photos-backed dataset by downloading random images
at multiple sizes using the Picsum service (no API key required).

Files are saved as: <out_dir>/<query>/<size>/<query>_<idx>_<size>.jpg

Usage:
    python sample-picsum-dataset.py --queries "nature,sky" --sizes 1920x1080 800x600 --per 3 --out backgrounds
"""

USER_AGENT = "sample-picsum-dataset/1.0 (+https://github.com/)"

def download_image(url, dest_path, timeout=30):
    headers = {"User-Agent": USER_AGENT}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"ERROR downloading {url} -> {dest_path}: {e}")
        return False

def build_source_url(query, size, sig=None):
    # picsum.photos random or seeded by path:
    # - Random: https://picsum.photos/<width>/<height>
    # - Seeded: https://picsum.photos/seed/<seed>/<width>/<height>
    try:
        width, height = size.lower().split("x")
        width = int(width)
        height = int(height)
    except Exception:
        # fallback to raw string if parsing fails
        return f"https://picsum.photos/{size}"

    if sig is not None:
        seed = f"{query}-{sig}" if query else str(sig)
        seed = urllib.parse.quote_plus(seed)
        return f"https://picsum.photos/seed/{seed}/{width}/{height}"
    else:
        return f"https://picsum.photos/{width}/{height}"

def main():
    p = argparse.ArgumentParser(description="Download random picsum.photos images in multiple sizes.")
    p.add_argument("--queries", type=str, default="background,texture,nature",
                   help="Comma-separated list of queries (used only for folder/seed names, picsum does not support search)")
    p.add_argument("--sizes", nargs="+", default=["1920x1080", "1280x720", "800x600", "400x300"],
                   help="List of sizes WIDTHxHEIGHT (default common sizes)")
    p.add_argument("--per", type=int, default=5, help="Images per query per size (default 5)")
    p.add_argument("--out", type=str, default="backgrounds", help="Output directory (default ./backgrounds)")
    p.add_argument("--delay", type=float, default=1.0, help="Delay seconds between downloads (default 1.0)")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sig values")
    args = p.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    sizes = args.sizes
    per = max(1, args.per)
    out_dir = args.out
    delay = max(0.0, args.delay)

    rng = random.Random(args.seed)

    total = len(queries) * len(sizes) * per
    count = 0
    print(f"Starting download: {total} images -> {out_dir}")

    for q in queries:
        for size in sizes:
            for i in range(per):
                sig = rng.randint(0, 10**9) if args.seed is not None else None
                url = build_source_url(q, size, sig=sig)
                safe_q = q.replace(" ", "_")
                filename = f"{safe_q}_{i+1}_{size}.jpg"
                dest_folder = os.path.join(out_dir, safe_q, size)
                dest_path = os.path.join(dest_folder, filename)
                success = download_image(url, dest_path)
                if success:
                    count += 1
                    print(f"[{count}/{total}] Saved: {dest_path}")
                else:
                    print(f"[{count}/{total}] Failed: {url}")
                time.sleep(delay)

    print(f"Done. {count} images downloaded to '{out_dir}'.")

if __name__ == "__main__":
    main()