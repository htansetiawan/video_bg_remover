"""Remove a video's background with BiRefNet.

Writes ProRes 4444 with alpha (for an editor) and, optionally, an H.264 composite on a studio backdrop.
Frames stream ffmpeg -> GPU -> ffmpeg in memory; nothing is written to disk per frame. Audio is copied.

  python remove_bg.py in.mov --alpha out_alpha.mov --studio out_studio.mp4
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation

from common import composite, decoder, encoder, probe, studio_backdrop, to_rgba

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("input")
p.add_argument("--alpha", required=True, help="output .mov (ProRes 4444 with alpha)")
p.add_argument("--studio", help="optional output .mp4 composited on the studio backdrop")
p.add_argument("--model", default="ZhengPeng7/BiRefNet",
               help="HF model id; ZhengPeng7/BiRefNet_HR infers at 2048px (only worth it for >1K sources)")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--ema", type=float, default=0.25, help="weight of the previous frame's alpha, against flicker")
p.add_argument("--limit", type=int, default=0, help="process only the first N frames (for tests)")
args = p.parse_args()

w, h, fps = probe(args.input)
res = (2048, 2048) if "HR" in args.model else (1024, 1024)
print(f"{w}x{h} @ {fps}, inference at {res[0]}px", flush=True)

dec = decoder(args.input, "rgb24")
enc_a = encoder(args.alpha, w, h, fps, args.input, "alpha", args.limit)
enc_s = encoder(args.studio, w, h, fps, args.input, "h264", args.limit) if args.studio else None

torch.set_float32_matmul_precision("high")
model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
backdrop = studio_backdrop(h, w)
prev = None
n = 0


def flush(frames):
    global prev, n
    x = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float() / 255
    inp = ((F.interpolate(x, res, mode="bilinear", align_corners=False) - mean) / std).half()
    with torch.no_grad():
        a = model(inp)[-1].sigmoid().float()
    a = F.interpolate(a, (h, w), mode="bilinear", align_corners=False)
    for i in range(a.shape[0]):
        if prev is not None:
            a[i:i + 1] = (1 - args.ema) * a[i:i + 1] + args.ema * prev
        prev = a[i:i + 1]
    enc_a.stdin.write(to_rgba(x, a).tobytes())
    if enc_s:
        enc_s.stdin.write(composite(x, a, backdrop).tobytes())
    n += len(frames)
    if n % 240 < len(frames):
        print(f"{n} frames", flush=True)


buf, fsz = [], w * h * 3
while True:
    raw = dec.stdout.read(fsz)
    if len(raw) < fsz or (args.limit and n + len(buf) >= args.limit):
        break
    buf.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    if len(buf) == args.batch:
        flush(buf)
        buf = []
if buf:
    flush(buf)
dec.kill()
for e in (enc_a, enc_s):
    if e:
        e.stdin.close()
        e.wait()
print(f"done {n} frames", flush=True)
