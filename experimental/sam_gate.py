"""EXPERIMENTAL: gate BiRefNet's alpha with SAM 2.1's tracked mask, then composite on the studio backdrop.

alpha = birefnet_alpha * soft(dilate(sam_mask)): SAM decides *what* is subject, BiRefNet keeps fine edges.
Only as good as the tracking; see sam_track.py for its known failure.

  python experimental/sam_gate.py out_alpha.mov /tmp/sam_work/masks.npy --studio gated.mp4 --audio in.mov
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import composite, decoder, encoder, probe, studio_backdrop, to_rgba  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("alpha_input")
p.add_argument("masks")
p.add_argument("--studio", required=True)
p.add_argument("--alpha")
p.add_argument("--audio")
p.add_argument("--dilate", type=int, default=15, help="px of slack around SAM's coarser mask")
p.add_argument("--feather", type=int, default=9)
args = p.parse_args()

w, h, fps = probe(args.alpha_input)
audio = args.audio or args.alpha_input
masks = np.load(args.masks, mmap_mode="r")
dec = decoder(args.alpha_input, "rgba")
enc_s = encoder(args.studio, w, h, fps, audio, "h264")
enc_a = encoder(args.alpha, w, h, fps, audio, "alpha") if args.alpha else None
backdrop = studio_backdrop(h, w)

n, fsz = 0, w * h * 4
while n < len(masks):
    raw = dec.stdout.read(fsz)
    if len(raw) < fsz:
        break
    f = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(1, h, w, 4).copy()).cuda()
    f = f.permute(0, 3, 1, 2).float() / 255
    x, a = f[:, :3], f[:, 3:]
    m = torch.from_numpy(np.unpackbits(masks[n], axis=1)[:, :w].copy()).cuda().float().view(1, 1, h, w)
    m = F.max_pool2d(m, 2 * args.dilate + 1, 1, args.dilate)
    m = F.avg_pool2d(m, 2 * args.feather + 1, 1, args.feather, count_include_pad=False)
    a = a * m
    enc_s.stdin.write(composite(x, a, backdrop).tobytes())
    if enc_a:
        enc_a.stdin.write(to_rgba(x, a).tobytes())
    n += 1
    if n % 600 == 0:
        print(f"{n} frames", flush=True)
dec.kill()
for e in (enc_s, enc_a):
    if e:
        e.stdin.close()
        e.wait()
print(f"done {n} frames", flush=True)
