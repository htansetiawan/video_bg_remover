"""Shared helpers: probing, ffmpeg encoders, and the studio backdrop."""
import json
import subprocess
from fractions import Fraction

import torch


def probe(src):
    """Return (width, height, fps) as displayed, i.e. after the container's rotation is applied."""
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_side_data=rotation", "-of", "json", src]))
    st = info["streams"][0]
    w, h = st["width"], st["height"]
    rot = next((abs(int(d["rotation"])) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    if rot in (90, 270):
        w, h = h, w
    return w, h, st["r_frame_rate"]


def decoder(src, pix_fmt):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                            stdout=subprocess.PIPE)


def encoder(out, w, h, fps, audio_src, kind, limit=0):
    """Raw frames on stdin -> `out`, copying audio from `audio_src`.

    kind="alpha": RGBA in, ProRes 4444 with alpha out. kind="h264": RGB in, H.264 out.
    """
    in_fmt, codec = {
        "alpha": ("rgba", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]),
        "h264": ("rgb24", ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]),
    }[kind]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-r", fps,
           "-i", "-", "-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    if limit:
        cmd += ["-frames:v", str(limit), "-t", f"{limit / Fraction(fps):.3f}"]
    return subprocess.Popen(cmd + codec + [out], stdin=subprocess.PIPE)


def studio_backdrop(h, w, device="cuda", inner=(246, 241, 232), outer=(218, 207, 190)):
    """Warm off-white studio sweep: brighter behind the subject, gently darker toward the edges. (1,3,H,W), 0-255."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device),
                            indexing="ij")
    d = (((xx - 0.5) / 0.75) ** 2 + ((yy - 0.42) / 0.85) ** 2).sqrt().clamp(0, 1).view(1, 1, h, w)
    inner = torch.tensor(inner, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    outer = torch.tensor(outer, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return inner + (outer - inner) * d.pow(1.6)


def composite(x, a, backdrop):
    """x: (B,3,H,W) in 0-1, a: (B,1,H,W) alpha. Returns uint8 (B,H,W,3) numpy, dithered against banding."""
    bg = backdrop + torch.rand_like(backdrop.expand(x.shape[0], -1, -1, -1)) - 0.5
    return (x * 255 * a + bg * (1 - a)).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def to_rgba(x, a):
    return torch.cat([x, a], 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def hex_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
