#!/usr/bin/env python3
"""
Generate gt_path.txt for QUSR training.
Usage: python scripts/get_path.py --folder /path/to/GT/images --output preset/gt_path.txt
"""

import os
import argparse


def write_image_paths(folder_path, txt_path, extensions=('.png', '.jpg', '.jpeg')):
    paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(extensions):
                paths.append(os.path.join(root, f))
    paths.sort()
    os.makedirs(os.path.dirname(txt_path) or '.', exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(paths))
    print(f"Wrote {len(paths)} paths to {txt_path}")
    return len(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", required=True, help="Folder containing GT images")
    parser.add_argument("--output", "-o", default="preset/gt_path.txt", help="Output txt path")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"], help="File extensions")
    args = parser.parse_args()
    write_image_paths(args.folder, args.output, tuple(args.extensions))


if __name__ == "__main__":
    main()
