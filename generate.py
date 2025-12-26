#!/usr/bin/env python3
import os, re, math, json, base64, random, subprocess, glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import yaml
import numpy as np
import requests
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from scipy.io.wavfile import write as write_wav, read as read_wav

# ----------------------------
# Utilities / IO
# ----------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp(x, a, b): 
    return a if x < a else b if x > b else x

def now_ms(seed=None):
    # deterministic-ish if seed passed
    if seed is None:
        import time
        return int(time.time() * 1000)
    return seed * 9973

# ----------------------------
# Ciphers (hidden meaning)
# ----------------------------
def rot13(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 65 <= o <= 90:
            out.append(chr(((o - 65 + 13) % 26) + 65))
        elif 97 <= o <= 122:
            out.append(chr(((o - 97 + 13) % 26) + 97))
        else:
            out.append(ch)
    return "".join(out)

def caesar(s: str, k: int = 7) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 65 <= o <= 90:
            out.append(chr(((o - 65 + k) % 26) + 65))
        elif 97 <= o <= 122:
            out.append(chr(((o - 97 + k) % 26) + 97))
        else:
            out.append(ch)
    return "".join(out)

def xor_b64(s: str, key: int = 73) -> str:
    b = bytes([c ^ key for c in s.encode("utf-8")])
    return base64.b64encode(b).decode("ascii")

def encrypt_words(words: List[str], cipher: str, rng: random.Random) -> str:
    if not words:
        return ""
    pick = rng.sample(words, k=min(len(words), rng.randint(1, min(4, len(words)))))
    joined = " ".join(pick)
    if cipher == "rot13":
        return rot13(joined)
    if cipher == "caesar":
        return caesar(joined, k=rng.choice([5,7,9,11]))
    if cipher == "xor_b64":
        return xor_b64(joined, key=rng.choice([31,73,101,149]))
    return joined

# ----------------------------
# Wikimedia image scraping (safe-ish, no API keys)
# ----------------------------
WIKI_API = "https://commons.wikimedia.org/w/api.php"

def wikimedia_search_image_urls(
    term: str,
    limit: int,
    rng: random.Random,
) -> List[str]:
    # search file pages
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": term + " filetype:bitmap",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    pages = (data.get("query", {}).get("pages", {}) or {}).values()
    urls = []
    for p in pages:
        ii = (p.get("imageinfo") or [])
        if not ii:
            continue
        info = ii[0]
        url = info.get("url")
        if url:
            urls.append(url)

    rng.shuffle(urls)
    return urls[:limit]

def download_images(
    urls: List[str],
    out_dir: str,
    min_width: int,
    max_download: int,
) -> List[str]:
    ensure_dir(out_dir)
    saved = []
    for i, url in enumerate(urls[:max_download]):
        try:
            fn = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("/")[-1])[:120]
            path = os.path.join(out_dir, f"{i:03d}_{fn}")
            if os.path.exists(path):
                saved.append(path)
                continue

            rr = requests.get(url, timeout=30)
            rr.raise_for_status()
            with open(path, "wb") as f:
                f.write(rr.content)

            # quick validate width
            im = Image.open(path)
            if im.size[0] < min_width:
                os.remove(path)
                continue

            saved.append(path)
        except Exception:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
    return saved

# ----------------------------
# Visual effects: VHS + PPT90s
# ----------------------------
@dataclass
class VidSpec:
    w: int
    h: int
    fps: int

def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = im.size
    s = max(w/iw, h/ih)
    nw, nh = int(iw*s), int(ih*s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0 = (nw - w)//2
    y0 = (nh - h)//2
    return im2.crop((x0, y0, x0+w, y0+h))

def np_cover(path: str, spec: VidSpec) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im = cover_resize(im, spec.w, spec.h)
    return np.array(im, dtype=np.uint8)

def precompute_masks(spec: VidSpec):
    W, H = spec.w, spec.h
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / np.sqrt(cx**2 + cy**2)
    v_mask = np.clip(1 - 0.65*(r**1.7), 0.30, 1.0).astype(np.float32)[...,None]
    scan = np.ones((H, W, 1), dtype=np.float32)
    scan[::2] = 0.84
    return v_mask, scan

def chroma_bleed(arr: np.ndarray, amt: int) -> np.ndarray:
    out = arr.copy()
    out[:,:,0] = np.roll(out[:,:,0], -amt, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  amt, axis=1)
    return out

def hslice_glitch(arr: np.ndarray, rng: random.Random, bands: int, maxshift: int) -> np.ndarray:
    out = arr.copy()
    H = out.shape[0]
    for _ in range(bands):
        y = rng.randint(0, H-18)
        hh = rng.randint(6, 24)
        shift = rng.randint(-maxshift, maxshift)
        out[y:y+hh] = np.roll(out[y:y+hh], shift, axis=1)
    return out

def noise_overlay(arr: np.ndarray, rng: random.Random, level: int) -> np.ndarray:
    n = rng.randint(0, level)
    if n <= 0:
        return arr
    nn = np.random.randint(-n, n+1, arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16)+nn, 0, 255).astype(np.uint8)

def tracking_line(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    out = arr.copy()
    H, W = out.shape[:2]
    y = rng.randint(int(H*0.45), H-14)
    hh = rng.randint(4, 10)
    out[y:y+hh] = np.roll(out[y:y+hh], rng.randint(-70, 70), axis=1)
    out[y:y+hh] = np.clip(out[y:y+hh].astype(np.int16) + rng.randint(35, 90), 0, 255).astype(np.uint8)
    return out

def redact_blocks(arr: np.ndarray, rng: random.Random, count: int) -> np.ndarray:
    out = arr.copy()
    H, W = out.shape[:2]
    for _ in range(count):
        ww = rng.randint(int(W*0.25), int(W*0.70))
        hh = rng.randint(16, 30)
        x = rng.randint(16, W-ww-16)
        y = rng.randint(60, H-60)
        out[y:y+hh, x:x+ww] = 0
    return out

def wave_warp_face(arr: np.ndarray, rng: random.Random, t: float) -> np.ndarray:
    out = arr.copy()
    H, W = out.shape[:2]
    for _ in range(4):
        x0 = rng.randint(0, W-140)
        ww = rng.randint(60, 160)
        shift = int(10*math.sin(t*2.0 + x0*0.02))
        out[:, x0:x0+ww] = np.roll(out[:, x0:x0+ww], shift, axis=0)
    y0 = int(H*0.30) + rng.randint(-8,8)
    out[y0:y0+34] = np.roll(out[y0:y0+34], rng.randint(-18,18), axis=1)
    return out

def alpha_over(bg_rgb: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    a = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    return np.clip(bg_rgb.astype(np.float32)*(1-a) + fg_rgba[:,:,:3].astype(np.float32)*a, 0, 255).astype(np.uint8)

def ppt90s_gradient(spec: VidSpec, rng: random.Random) -> np.ndarray:
    W, H = spec.w, spec.h
    c1 = np.array([rng.randint(30,255), rng.randint(30,255), rng.randint(30,255)], dtype=np.float32)
    c2 = np.array([rng.randint(30,255), rng.randint(30,255), rng.randint(30,255)], dtype=np.float32)
    gx = np.linspace(0, 1, W, dtype=np.float32)[None,:,None]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:,None,None]
    mix = np.clip(0.62*gx + 0.38*gy, 0, 1)
    return np.clip(c1*(1-mix) + c2*mix, 0, 255).astype(np.uint8)

def make_ui_layer(spec: VidSpec, title: str, body: str, theme: str, rng: random.Random, wordart: bool) -> np.ndarray:
    W, H = spec.w, spec.h
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer)
    try:
        fontT = ImageFont.truetype("DejaVuSans.ttf", 28)
        fontB = ImageFont.truetype("DejaVuSans.ttf", 19)
        fontM = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except:
        fontT = fontB = fontM = ImageFont.load_default()

    # loud header strip (90s)
    if theme == "fatal":
        hdr = (255, 50, 70, 235)
    elif theme in ("protocol","janedoe"):
        hdr = (255, 230, 60, 235)
    else:
        hdr = (70, 240, 255, 235)

    d.rectangle((0,0,W,62), fill=hdr)

    # wordart-ish title shadow
    if wordart:
        d.text((18, 18), title, fill=(0,0,0,130), font=fontT)
        d.text((16, 16), title, fill=(10,10,10,255), font=fontT)
    else:
        d.text((16, 16), title, fill=(10,10,10,255), font=fontT)

    # content panel
    px0, py0, px1, py1 = 22, 96, int(W*0.76), 96+260
    d.rounded_rectangle((px0,py0,px1,py1), radius=16, fill=(255,255,255,220), outline=(0,0,0,80), width=2)

    y = py0 + 16
    for line in body.split("\n"):
        d.text((px0+18, y), line, fill=(10,10,10,255), font=fontB)
        y += 28

    # footer
    d.rectangle((0, H-46, W, H), fill=(0,0,0,120))
    d.text((16, H-36), "TRAINING ARCHIVE // DO NOT DUPLICATE", fill=(255,255,255,255), font=fontM)

    if theme == "janedoe":
        d.rectangle((px0+18, py1-52, px1-18, py1-18), fill=(230,230,230,235))
        d.text((px0+24, py1-48), "FILE: JANE_DOE.TAPE // ACCESS: DENIED", fill=(0,0,0,255), font=fontM)

    if theme == "fatal":
        # BSOD block
        d.rectangle((0,62,W,H), fill=(20,60,175,210))
        d.text((20, 92), "A fatal exception has occurred.", fill=(255,255,255,255), font=fontM)
        d.text((20, 118), "The system has been halted to prevent memory loss.", fill=(255,255,255,255), font=fontM)
        d.text((20, 152), "ERROR: MNEMONIC_LEAK (0x0000F4)", fill=(255,255,255,255), font=fontM)
        d.text((20, 178), "Press ESC to continue (ESC is disabled).", fill=(255,255,255,255), font=fontM)

    return np.array(layer, dtype=np.uint8)

def timecode_overlay(arr: np.ndarray, spec: VidSpec, frame_idx: int) -> np.ndarray:
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except:
        font = ImageFont.load_default()
    secs = frame_idx / spec.fps
    hh = int(secs//3600); mm = int((secs%3600)//60); ss = int(secs%60); ff = int((secs - int(secs))*spec.fps)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}  CH03  SP"
    d.rectangle((10, spec.h-34, 290, spec.h-12), fill=(0,0,0))
    d.text((16, spec.h-32), tc, fill=(255,255,255), font=font)
    return np.array(im, dtype=np.uint8)

# ----------------------------
# Story generator
# ----------------------------
WELLNESS = [
    "WORKPLACE WELLNESS",
    "HAPPINESS HABITS",
    "PRODUCTIVITY TIP",
    "POSTURE & BREATHING",
    "MINDFUL ROUTINE",
]
WELLNESS_BODIES = [
    "• Hydrate every hour\n• Breathe in for 4, out for 6\n• Check posture\n• Smile (optional)",
    "• Short walk after lunch\n• Call a friend\n• Keep a simple journal\n• Sleep at the same time",
    "• One task at a time\n• Reduce distractions\n• Label your keys\n• Keep your desk tidy",
    "• Shoulders down\n• Jaw relaxed\n• Look at a distant point\n• Count slowly",
]

PROTOCOL_TITLES = [
    "MEMORY RETENTION",
    "LOSS PREVENTION",
    "IDENTIFICATION SAFETY",
    "RECOGNITION HAZARD",
]
PROTOCOL_BODIES = [
    "If a face feels wrong:\n1) Look away\n2) Do not describe it\n3) Touch an object you can name\n4) Leave immediately",
    "If you hear your voice:\nDo not answer.\nWrite your name once.\nCross it out twice.\nForget the rest.",
    "If the room repeats:\nDo not count doors.\nDo not check clocks.\nKeep moving.\nDo not look back.",
]
JANE_DOE = [
    "SUBJECT: JANE DOE\nSTATUS: [REDACTED]\nDO NOT ATTEMPT RECOGNITION",
    "IDENTITY: [REDACTED]\nLAST STABLE MEMORY: 03:17\nCOMPLIANCE: PARTIAL",
    "FILE MISSING\nNAME UNKNOWN\nIF YOU REMEMBER, STOP",
]

def build_slides(cfg: dict, rng: random.Random) -> List[dict]:
    sc = int(cfg["story"]["slide_count"])
    wellness_ratio = float(cfg["story"].get("include_wellness_ratio", 0.35))
    jane_freq = float(cfg["story"].get("jane_doe_frequency", 0.18))
    fatal_freq = float(cfg["story"].get("fatal_frequency", 0.12))
    kws = cfg["story"].get("keywords", []) or []
    enc_words = cfg["story"].get("encrypt_words", []) or []
    cipher = str(cfg["story"].get("cipher", "rot13"))

    slides = []
    # intro
    slides.append(dict(kind="bars", title="PLAYBACK", body="JOB TRAINING TAPE / 1993\nCHANNEL 03 // TRACKING: OK"))

    for _ in range(sc - 2):
        roll = rng.random()
        if roll < fatal_freq:
            slides.append(dict(kind="fatal", title="TRACKING LOST", body="FATAL ERROR: MNEMONIC_LEAK\nESC DISABLED\nDO NOT REWIND"))
            continue
        if roll < fatal_freq + jane_freq:
            slides.append(dict(kind="janedoe", title="JANE DOE INTERMISSION", body=rng.choice(JANE_DOE)))
            continue

        # wellness vs protocol
        if rng.random() < wellness_ratio:
            title = rng.choice(WELLNESS)
            body = rng.choice(WELLNESS_BODIES)
            # inject keyword as “too normal”
            if kws and rng.random() < 0.6:
                body += f"\n• {rng.choice(kws).upper()} (optional)"
            slides.append(dict(kind="wellness", title=title, body=body))
        else:
            title = rng.choice(PROTOCOL_TITLES)
            body = rng.choice(PROTOCOL_BODIES)
            # encrypted whisper line
            whisper = encrypt_words(enc_words, cipher, rng)
            if whisper:
                body += f"\n\n[{whisper}]"
            slides.append(dict(kind="protocol", title=title, body=body))

        # occasional full-face “recognition hazard”
        if rng.random() < 0.28:
            slides.append(dict(kind="facefull", title="IDENTIFICATION FAILURE", body="DO NOT ATTEMPT RECOGNITION\nDO NOT CONFIRM IT\nDO NOT DESCRIBE IT"))

    # outro
    slides.append(dict(kind="end", title="END OF MODULE", body="Thank you.\nDo not replay this tape.\nThe tape will replay you."))
    return slides

# ----------------------------
# Audio: TTS + VHS bed + pops
# ----------------------------
def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    y = x[::factor]
    y = np.repeat(y, factor)[:len(x)]
    return y

def mix_in(dst: np.ndarray, src: np.ndarray, sr: int, start_s: float, gain: float):
    start = int(start_s * sr)
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[:end-start] * gain

def gen_tts(text: str, wav_path: str, voice: str, speed: int, pitch: int, amp: int):
    subprocess.run(
        ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", wav_path, text],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ----------------------------
# Main generation
# ----------------------------
def main():
    # config path (optional)
    cfg_path = os.environ.get("CONFIG", "config.example.yaml")
    if os.path.exists("config.yaml"):
        cfg_path = "config.yaml"
    cfg = load_yaml(cfg_path)

    seed = int(cfg.get("seed", 1337))
    rng = random.Random(seed)

    out_mp4 = str(cfg.get("output", "out.mp4"))
    ensure_dir("cache/web")

    spec = VidSpec(
        w=int(cfg["format"]["width"]),
        h=int(cfg["format"]["height"]),
        fps=int(cfg["format"]["fps"]),
    )
    slide_sec = float(cfg["story"].get("slide_sec", 3.0))

    # --- collect images from repo ---
    local_imgs = sorted(glob.glob("assets/local/*"))
    local_faces = sorted(glob.glob("assets/faces/*"))

    # --- optional web scrape ---
    web_imgs = []
    if bool(cfg["sources"].get("use_web_scrape", True)):
        terms = cfg["sources"]["web"].get("search_terms", []) or []
        max_dl = int(cfg["sources"]["web"].get("max_download", 18))
        min_w = int(cfg["sources"]["web"].get("min_width", 600))
        urls = []
        for t in terms:
            urls.extend(wikimedia_search_image_urls(t, limit=max_dl, rng=rng))
        urls = list(dict.fromkeys(urls))  # unique preserving order
        rng.shuffle(urls)
        web_imgs = download_images(urls, "cache/web", min_width=min_w, max_download=max_dl)

    # fallback safety
    pool_imgs = [p for p in (local_imgs + web_imgs) if os.path.isfile(p)]
    pool_faces = [p for p in (local_faces + web_imgs) if os.path.isfile(p)]  # faces can come from web too
    if not pool_imgs:
        raise RuntimeError("No images found. Add images to assets/local/ or enable web scrape.")
    if not pool_faces:
        pool_faces = pool_imgs[:]  # worst case

    # build slide script
    slides = build_slides(cfg, rng)

    # precompute masks
    v_mask, scan = precompute_masks(spec)

    # schedule flashes
    frames_per_slide = int(slide_sec * spec.fps)
    total_frames = frames_per_slide * len(slides)
    flash_count = 10 if cfg["effects"].get("enable_flashes", True) else 0
    flash_frames = set()
    for _ in range(flash_count):
        flash_frames.add(rng.randint(int(0.05*total_frames), max(1, total_frames-2)))

    # build popups from uncorrelated objects
    popup_pool = []
    for _ in range(8):
        src = np_cover(rng.choice(pool_imgs), spec)
        x0 = rng.randint(0, spec.w-220) if spec.w > 220 else 0
        y0 = rng.randint(0, spec.h-220) if spec.h > 220 else 0
        crop = src[y0:y0+200, x0:x0+200]
        im = Image.fromarray(crop)
        if rng.random() < 0.6:
            im = ImageOps.grayscale(im)
            im = ImageOps.autocontrast(ImageOps.posterize(im, 3))
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
        else:
            im = ImageOps.autocontrast(im)
            im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
        popup_pool.append(np.array(im.resize((180,180), Image.Resampling.NEAREST).convert("RGB"), dtype=np.uint8))

    def stamp_popup(frame: np.ndarray, popup: np.ndarray) -> np.ndarray:
        out = frame.copy()
        ph, pw = popup.shape[:2]
        x = rng.randint(0, spec.w-pw) if spec.w > pw else 0
        y = rng.randint(70, spec.h-ph-70) if spec.h > ph+140 else 0
        out[y:y+ph, x:x+pw] = popup
        out[y:y+3, x:x+pw] = 0; out[y+ph-3:y+ph, x:x+pw] = 0
        out[y:y+ph, x:x+3] = 0; out[y:y+ph, x+pw-3:x+pw] = 0
        return out

    # --- render silent video ---
    silent_path = "cache/_silent.mp4"
    audio_path = "cache/_audio.wav"

    intensity = float(cfg["effects"].get("intensity", 0.85))
    enable_vhs = bool(cfg["effects"].get("enable_vhs", True))
    enable_glitch = bool(cfg["effects"].get("enable_glitch", True))
    enable_red = bool(cfg["effects"].get("enable_redactions", True))
    enable_popups = bool(cfg["effects"].get("enable_popups", True))
    ppt90s = bool(cfg["effects"].get("ppt90s_palette", True))
    wordart = bool(cfg["effects"].get("ppt90s_wordart", True))

    writer = imageio.get_writer(silent_path, fps=spec.fps, codec="libx264", bitrate="2500k")

    frame_idx = 0
    for si, s in enumerate(slides):
        kind = s["kind"]
        title = s["title"]
        body = s["body"]

        # choose backgrounds
        if ppt90s and kind in ("wellness","protocol","end","janedoe","facefull"):
            bg = ppt90s_gradient(spec, rng)
            # faintly blend a real image behind the gradient to make it “wrong”
            real = np_cover(rng.choice(pool_imgs), spec)
            bg = np.clip(bg.astype(np.float32)*0.70 + real.astype(np.float32)*0.30, 0, 255).astype(np.uint8)
        elif kind == "bars":
            bg = ppt90s_gradient(spec, rng) if ppt90s else np.zeros((spec.h, spec.w, 3), dtype=np.uint8)
        elif kind == "fatal":
            bg = np.full((spec.h, spec.w, 3), (20,60,170), dtype=np.uint8)
        else:
            bg = np_cover(rng.choice(pool_imgs), spec)

        theme = "normal"
        if kind in ("protocol","facefull"):
            theme = "protocol"
        if kind == "janedoe":
            theme = "janedoe"
        if kind == "fatal":
            theme = "fatal"

        ui = make_ui_layer(spec, title, body, theme=theme, rng=rng, wordart=wordart)

        # fullscreen face base for “uncanny”
        face_bg = None
        if kind in ("facefull","janedoe"):
            face_bg = np_cover(rng.choice(pool_faces), spec)

        for fi in range(frames_per_slide):
            t = frame_idx / spec.fps
            frame = bg

            # slight drift
            if fi % 4 == 0:
                frame = np.roll(frame, rng.randint(-2,2), axis=1)

            # face dominance (fully visible)
            if face_bg is not None:
                warped = wave_warp_face(face_bg, rng, t)
                alpha = 0.78 if kind == "janedoe" else 0.62
                frame = np.clip(frame.astype(np.float32)*(1-alpha) + warped.astype(np.float32)*alpha, 0, 255).astype(np.uint8)

            # overlay UI
            frame = alpha_over(frame, ui)

            # redactions
            if enable_red and kind in ("protocol","janedoe","facefull") and rng.random() < (0.25*intensity):
                frame = redact_blocks(frame, rng, count=rng.randint(1,3))

            # popups (uncorrelated objects)
            if enable_popups and rng.random() < (0.20*intensity):
                pop = rng.choice(popup_pool)
                if rng.random() < 0.28:
                    pop = 255 - pop
                frame = stamp_popup(frame, pop)

            # one-frame flashes
            if frame_idx in flash_frames:
                flash = np_cover(rng.choice(pool_faces), spec) if rng.random() < 0.65 else np_cover(rng.choice(pool_imgs), spec)
                if enable_glitch:
                    flash = hslice_glitch(chroma_bleed(noise_overlay(flash, rng, 35), amt=5), rng, bands=5, maxshift=80)
                frame = flash

            # VHS stack
            if enable_vhs:
                frame = chroma_bleed(frame, amt=3 if rng.random() < 0.75 else 5)
                if rng.random() < (0.35*intensity):
                    frame = tracking_line(frame, rng)
                if enable_glitch and rng.random() < (0.22*intensity):
                    frame = hslice_glitch(frame, rng, bands=3, maxshift=70)
                frame = noise_overlay(frame, rng, level=int(18*intensity))
                frame = (frame.astype(np.float32) * scan).astype(np.uint8)
                frame = (frame.astype(np.float32) * v_mask).astype(np.uint8)

            # edge transitions (extra harsh)
            if enable_glitch and (fi < 3 or fi > frames_per_slide-4):
                frame = hslice_glitch(frame, rng, bands=4, maxshift=90)
                if rng.random() < 0.45:
                    frame = 255 - frame

            frame = timecode_overlay(frame, spec, frame_idx)

            writer.append_data(frame)
            frame_idx += 1

    writer.close()

    # --- audio synthesis ---
    dur = frame_idx / spec.fps
    total_samples = int(dur * SR)
    t = np.linspace(0, dur, total_samples, False).astype(np.float32)

    harsh_bed = float(cfg["audio"].get("harsh_bed_level", 0.12))
    audio = (np.random.uniform(-1,1,total_samples).astype(np.float32) * harsh_bed * 0.6)
    audio += (0.03*np.sin(2*np.pi*55*t) + 0.018*np.sin(2*np.pi*110*t)).astype(np.float32)
    audio += (0.02*np.sin(2*np.pi*30*t)).astype(np.float32)
    audio *= (0.82 + 0.18*np.sin(2*np.pi*0.18*t)).astype(np.float32)

    # abrupt pops
    pop_count = int(cfg["audio"].get("pop_count", 20))
    for _ in range(pop_count):
        p0 = rng.randint(0, total_samples-1)
        span = rng.randint(int(0.01*SR), int(0.06*SR))
        p1 = min(total_samples, p0+span)
        burst = (np.random.uniform(-1,1,p1-p0).astype(np.float32) * rng.uniform(0.25, 0.55))
        audio[p0:p1] += burst

    # TTS narration (weird but intelligible)
    enable_tts = bool(cfg["audio"].get("enable_tts", True))
    voice = str(cfg["audio"].get("voice", "en-us"))
    speed = int(cfg["audio"].get("tts_speed", 152))
    pitch = int(cfg["audio"].get("tts_pitch", 32))
    amp = int(cfg["audio"].get("tts_amp", 170))
    weird_fx = bool(cfg["audio"].get("weird_voice_fx", True))

    if enable_tts:
        ensure_dir("cache/tts")
        # narration aligned per slide start
        slide_starts = [i*slide_sec for i in range(len(slides))]
        narrations = []
        for s in slides:
            k = s["kind"]
            if k == "bars":
                narrations.append("Playback. Job training tape. Tracking is stable.")
            elif k == "wellness":
                narrations.append(
                    "Workplace wellness. " +
                    re.sub(r"[•\[\]]", "", s["body"]).replace("\n", " ").strip() +
                    " Please continue normally."
                )
            elif k == "protocol":
                narrations.append(
                    "Memory retention. " +
                    re.sub(r"[\[\]]", "", s["body"]).replace("\n", " ").strip() +
                    " Do not describe what you notice."
                )
            elif k == "facefull":
                narrations.append("Identification failure. Do not attempt recognition. Do not confirm it.")
            elif k == "janedoe":
                narrations.append("Jane Doe intermission. Identity redacted. Do not attempt recognition.")
            elif k == "fatal":
                narrations.append("Tracking lost. Fatal error. Escape is disabled. Do not rewind.")
            else:
                narrations.append("End of module. Do not replay this tape. The tape will replay you.")

        for i, text in enumerate(narrations):
            wav_path = f"cache/tts/tts_{i:02d}.wav"
            gen_tts(text, wav_path, voice=voice, speed=speed, pitch=pitch, amp=amp)

            sr2, data = read_wav(wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            data /= (np.max(np.abs(data)) + 1e-6)

            # weird voice FX (still intelligible)
            if weird_fx:
                data = np.tanh(data * 1.8) * 0.72
                data = bitcrush(data, factor=rng.choice([4,5,6]))
                ring = np.sin(2*np.pi*220*np.linspace(0, len(data)/SR, len(data), False).astype(np.float32))
                data = (0.82*data + 0.18*data*ring).astype(np.float32)
                data += (np.random.uniform(-1,1,len(data)).astype(np.float32) * 0.006)
            else:
                data = np.tanh(data * 1.3) * 0.75

            mix_in(audio, data, SR, start_s=slide_starts[i] + 0.25, gain=0.95)

    audio /= (np.max(np.abs(audio)) + 1e-6)
    write_wav(audio_path, SR, (audio * 32767).astype(np.int16))

    # --- mux ---
    subprocess.run(
        ["ffmpeg","-y","-i", silent_path, "-i", audio_path, "-c:v","copy", "-c:a","aac", "-b:a","192k", "-shortest", out_mp4],
        check=True
    )

    print(f"OK: wrote {out_mp4}")

if __name__ == "__main__":
    main()

