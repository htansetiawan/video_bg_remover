"""Re-composite an alpha video (from remove_bg.py) onto the studio backdrop. No model inference.

--band START END fades the backdrop back to the original frame over that height range (fractions, top=0).
Use it when the model flickers on background near the bottom edge (e.g. a bed that keeps popping in):
inside the band the leaked pixels sit over the real background, so the flicker disappears.

  python composite.py out_alpha.mov --studio studio.mp4 --band 0.80 0.90 --audio in.mov
"""
import argparse

import numpy as np
import torch

from common import composite, decoder, encoder, hex_rgb, probe, studio_backdrop, to_rgba

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("alpha_input", help="ProRes 4444 with alpha from remove_bg.py")
p.add_argument("--studio", required=True, help="output .mp4")
p.add_argument("--alpha", help="optional output .mov with the band made opaque")
p.add_argument("--band", type=float, nargs=2, metavar=("START", "END"))
p.add_argument("--inner", default="#f6f1e8", help="backdrop colour behind the subject")
p.add_argument("--outer", default="#dacfbe", help="backdrop colour at the edges")
p.add_argument("--audio", help="file to take audio from (default: the alpha input)")
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()

w, h, fps = probe(args.alpha_input)
audio = args.audio or args.alpha_input
dec = decoder(args.alpha_input, "rgba")
enc_s = encoder(args.studio, w, h, fps, audio, "h264", args.limit)
enc_a = encoder(args.alpha, w, h, fps, audio, "alpha", args.limit) if args.alpha else None

backdrop = studio_backdrop(h, w, inner=hex_rgb(args.inner), outer=hex_rgb(args.outer))
ramp = None
if args.band:
    yy = torch.linspace(0, 1, h, device="cuda").view(1, 1, h, 1).expand(1, 1, h, w)
    ramp = ((yy - args.band[0]) / (args.band[1] - args.band[0])).clamp(0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep

n, fsz = 0, w * h * 4
while True:
    raw = dec.stdout.read(fsz)
    if len(raw) < fsz or (args.limit and n >= args.limit):
        break
    f = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(1, h, w, 4).copy()).cuda()
    f = f.permute(0, 3, 1, 2).float() / 255
    x, a = f[:, :3], f[:, 3:]
    if ramp is not None:
        a = a + (1 - a) * ramp  # inside the band the original frame is opaque
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
