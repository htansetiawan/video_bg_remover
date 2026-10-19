"""EXPERIMENTAL: track the subject with SAM 2.1 so a gate can drop background BiRefNet flickers on.

Seeded from remove_bg.py's alpha on frame 0; tracks in chunks (SAM 2 holds every frame of a clip in memory),
seeding each chunk with the previous chunk's last mask. Writes masks.npy: (N, H, ceil(W/8)) packed bits.

Known failure: with a single seed it lost a violin that turned from upright (frame 0) to horizontal close-ups.
Re-prompting at keyframes would be needed for subjects whose appearance changes a lot.

  python experimental/sam_track.py in.mov out_alpha.mov --ckpt sam2.1_hiera_large.pt --work /tmp/sam_work
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import probe  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("input")
p.add_argument("alpha_input", help="alpha .mov from remove_bg.py, used to seed frame 0")
p.add_argument("--ckpt", required=True)
p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
p.add_argument("--work", required=True, help="scratch dir for JPEG frames and masks.npy")
p.add_argument("--chunk", type=int, default=500)
args = p.parse_args()

w, h, _ = probe(args.input)
frames = f"{args.work}/frames"
if not os.path.isdir(frames):
    os.makedirs(frames)
    subprocess.check_call(["ffmpeg", "-v", "error", "-i", args.input, "-q:v", "2", "-start_number", "0",
                           f"{frames}/%05d.jpg"])
names = sorted(os.listdir(frames))
N = len(names)
print(f"{N} frames", flush=True)

raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", args.alpha_input, "-frames:v", "1",
                               "-vf", "alphaextract", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
seed = np.frombuffer(raw, np.uint8).reshape(h, w) > 127

predictor = build_sam2_video_predictor(args.config, args.ckpt)
out = np.zeros((N, h, (w + 7) // 8), np.uint8)
start = 0
while start < N - 1:
    end = min(start + args.chunk, N)
    cdir = f"{args.work}/chunk"
    os.makedirs(cdir, exist_ok=True)
    for f in os.listdir(cdir):
        os.remove(f"{cdir}/{f}")
    for i in range(start, end):
        os.symlink(f"{os.path.abspath(frames)}/{names[i]}", f"{cdir}/{i - start:05d}.jpg")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=cdir)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
        for fi, _, logits in predictor.propagate_in_video(state):
            m = (logits[0, 0] > 0).cpu().numpy()
            out[start + fi] = np.packbits(m, axis=1)
            last = m
    predictor.reset_state(state)
    del state
    torch.cuda.empty_cache()
    seed = last
    print(f"tracked {end}/{N}", flush=True)
    start = end - 1

np.save(f"{args.work}/masks.npy", out)
print("done", flush=True)
