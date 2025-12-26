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
        hdr = (70, 240, 255,
