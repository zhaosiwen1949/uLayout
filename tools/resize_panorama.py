#!/usr/bin/env python3
"""Resize panorama images to 1024x512 (equirectangular) for uLayout inference.

Usage:
    python tools/resize_panorama.py <input_dir> [-o <output_dir>]

- <input_dir>: folder containing the source images (.jpg/.png/.jpeg/.bmp).
- -o/--output: output folder. Defaults to "<input_dir>/../img" (an "img" folder
  next to the input folder), which matches the layout expected by infer.py.
"""
import argparse
import os
import sys

from PIL import Image

TARGET_W, TARGET_H = 1024, 512
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def resize_dir(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        sys.exit(f'[resize_panorama] input dir not found: {input_dir}')

    os.makedirs(output_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(input_dir)
                    if f.lower().endswith(EXTS))
    if not fnames:
        sys.exit(f'[resize_panorama] no images found in: {input_dir}')

    for fname in fnames:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        with Image.open(src) as im:
            im = im.convert('RGB').resize((TARGET_W, TARGET_H), Image.BICUBIC)
            im.save(dst)
        print(f'[resize_panorama] {src} -> {dst} ({TARGET_W}x{TARGET_H})')

    print(f'[resize_panorama] done: {len(fnames)} image(s) -> {output_dir}')


def default_output_dir(input_dir):
    # "<input_dir>/../img" : an "img" folder next to the input folder.
    parent = os.path.dirname(os.path.abspath(input_dir.rstrip(os.sep)))
    return os.path.join(parent, 'img')


def main():
    parser = argparse.ArgumentParser(
        description='Resize panorama images to %dx%d.' % (TARGET_W, TARGET_H))
    parser.add_argument('input_dir',
                        help='folder containing the source images')
    parser.add_argument('-o', '--output', default=None,
                        help='output folder (default: <input_dir>/../img)')
    args = parser.parse_args()

    output_dir = args.output or default_output_dir(args.input_dir)
    resize_dir(args.input_dir, output_dir)


if __name__ == '__main__':
    main()
