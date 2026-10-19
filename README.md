# video_bg_remover

Remove the background from a video with [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT) and put the
subject on a clean studio backdrop. Built for musician/tutorial footage where the subject is a person *and* an
object (a violin and bow), which person-only matting models such as RobustVideoMatting drop.

Frames are streamed ffmpeg → GPU → ffmpeg in memory, so there is no frame dump and reassembly. Audio is copied
through.

## Outputs

- **ProRes 4444 with alpha** (`.mov`), to put your own background under in an editor such as Final Cut Pro.
- **H.264 on a studio backdrop** (`.mp4`), a warm off-white sweep with a soft vignette (colours are configurable).

## Setup

Needs an NVIDIA GPU, ffmpeg with `prores_ks` and `libx264`, and Python 3.10+.

```bash
uv venv && uv pip install -r requirements.txt
```

The model weights download from Hugging Face on first run. BiRefNet loads with `trust_remote_code=True`.

## Usage

```bash
# 1. segment (the slow step, ~15 fps for 540x960 on an RTX PRO 5000)
python remove_bg.py in.mov --alpha out_alpha.mov --studio out_studio.mp4

# 2. optional: re-composite without re-running the model, e.g. other colours,
#    or keep the original room in the bottom band
python composite.py out_alpha.mov --studio out_band.mp4 --band 0.80 0.90 --audio in.mov
```

`--model ZhengPeng7/BiRefNet_HR` infers at 2048px. It only helps for sources larger than about 1K. On a
540×960 phone clip it was worse, keeping patches of a pink bedsheet that the standard model dropped.

## Flicker near the frame edge

BiRefNet works on each frame on its own. When background that looks like the subject touches it (a bed under
the player's arm, for example), the model includes it on some frames and not others, so it flickers.
`remove_bg.py` smooths the alpha lightly over time (`--ema`), but that cannot fix a whole patch switching on
and off.

`composite.py --band START END` fades the backdrop back to the original frame over that range of the frame
height (0 = top). Inside the band any leaked pixels sit over the real background, so the flicker can't be seen.

A colour key on the leaking background does not work when it shares the skin's hue. Here the pink bedsheet
measured 5–20° hue and the skin 5–25°, so keying the sheet would also remove fingers.

## Experimental: SAM 2.1 gating

`experimental/` tracks the subject with [SAM 2.1](https://github.com/facebookresearch/sam2) (Apache 2.0). The
tracked mask then gates BiRefNet's alpha: SAM decides what counts as the subject and BiRefNet supplies the edges.

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
curl -LO https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
python experimental/sam_track.py in.mov out_alpha.mov --ckpt sam2.1_hiera_large.pt --work /tmp/sam_work
python experimental/sam_gate.py out_alpha.mov /tmp/sam_work/masks.npy --studio gated.mp4 --audio in.mov
```

Status: this did not work on the footage it was tried on. Seeded once from frame 0, where the violin is held
upright, SAM lost it once the video moved to horizontal close-ups. It would need prompts at keyframes.
