# video_bg_remover

Remove a video's background with [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT) and put the subject on a
studio backdrop. It keeps people *and* what they hold (instruments, props), which person-only matting models drop.
Frames stream ffmpeg → GPU → ffmpeg in memory, and audio is copied through.

## Requirements

- An NVIDIA GPU. Tested on an RTX PRO 5000 Blackwell (48 GB). The default torch wheels need a recent driver
  (580+ for CUDA 13).
- `ffmpeg` built with `prores_ks` and `libx264`
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
git clone https://github.com/htansetiawan/video_bg_remover && cd video_bg_remover
uv sync

# segment: writes a transparent ProRes 4444 .mov plus a studio-backdrop .mp4
uv run remove_bg.py input.mov --alpha out_alpha.mov --studio out_studio.mp4

# re-composite without re-running the model (new colours, bottom band, ...)
uv run composite.py out_alpha.mov --studio out_band.mp4 --band 0.80 0.90 --audio input.mov
```

BiRefNet's weights download from Hugging Face on first run (`trust_remote_code=True`). A 540×960 clip runs at
about 15 fps.

## Options

| Script | Flag | Default | What it does |
|---|---|---|---|
| `remove_bg.py` | `--alpha` | required | ProRes 4444 with alpha, for an editor (Final Cut, Premiere, Resolve) |
| | `--studio` | none | H.264 composited on the studio backdrop |
| | `--model` | `ZhengPeng7/BiRefNet` | `ZhengPeng7/BiRefNet_HR` infers at 2048px; only worth it for sources above about 1K |
| | `--ema` | `0.25` | blends each frame's mask with the previous one to reduce flicker |
| | `--limit` | `0` (all) | process only the first N frames; use this to test |
| `composite.py` | `--band START END` | none | fades back to the original frame over that height range (0 = top) |
| | `--inner` / `--outer` | `#f6f1e8` / `#dacfbe` | backdrop colour at the centre and at the edges |
| | `--alpha` | none | also write ProRes with the band made opaque |
| | `--audio` | alpha input | where to take the audio from |

## Edge flicker

BiRefNet looks at each frame on its own. Background that looks like the subject and touches it, such as a bed
under the player's arm, gets included on some frames and not others. `--band 0.80 0.90` shows the original room
in the bottom of the frame, so any leak sits over the real background and can't be seen. Colour-keying the leak
fails when it shares the skin's hue.

## For coding agents

To rerun on a new video:

1. `uv sync`, then run `ffprobe input.mov` to check resolution, rotation and frame count.
2. Test on a short piece first: `uv run remove_bg.py input.mov --alpha t_alpha.mov --studio t.mp4 --limit 150`.
3. Grab frames to look at:
   `ffmpeg -ss 2 -i t.mp4 -frames:v 1 check.png`. Check thin objects (bows, hair, fingers) and the frame edges.
4. Run the whole video. It is long, so use `tmux new -d -s bg '…'` so it survives the session ending.
5. If background flickers near one edge, re-run only `composite.py` with `--band`. It needs no GPU inference and
   takes a few minutes. Don't re-run `remove_bg.py`.

`common.py` holds the shared ffmpeg encoders and the backdrop. Outputs go wherever you point them, and
`.gitignore` excludes media files.

## Experimental: SAM 2.1 gating

```bash
uv sync --extra sam
curl -LO https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
uv run experimental/sam_track.py input.mov out_alpha.mov --ckpt sam2.1_hiera_large.pt --work /tmp/sam_work
uv run experimental/sam_gate.py out_alpha.mov /tmp/sam_work/masks.npy --studio gated.mp4 --audio input.mov
```

This tracks the subject with SAM 2.1 and uses the tracked mask to gate BiRefNet. It did **not** work on the
footage it was tried on. Seeded once at frame 0, it lost a violin that went from upright to horizontal close-ups.
It would need prompts at keyframes.
