"""
enhance_frames.py
Enhances video frames using Real-ESRGAN (AI upscaling) combined with
optional classical image processing (contrast/saturation boost, denoising,
sharpening).

Deliberately does NOT use face-specific generative restoration (GFPGAN/
CodeFormer) - those models can alter a real person's identity/features
since they generate plausible faces from a learned prior rather than
only enhancing real pixel data. For content involving real people,
non-generative enhancement is the safer, more faithful choice.

Uses RealESRNet_x4plus.pth (not RealESRGAN_x4plus.pth) - the GAN variant
was found to produce a "painted/smeared" artifact on busy/repetitive
textures (crowds, foliage). RealESRNet uses the same architecture but a
PSNR-oriented (non-GAN) loss, staying closer to real detail.

Contrast/saturation are OFF by default (opt-in via --contrast/--saturation).
Testing showed alpha=1.5 contrast clips bright skies to flat white, and
saturation=1.5 looks artificial/neon on well-lit footage - both are
scene-dependent and shouldn't be forced on by default. Sharpening (via
unsharp masking) is ON by default - it's what actually makes the
resolution increase perceptually visible, and doesn't have the same
clipping/oversaturation risk.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def build_upsampler(model_path: str, tile: int = 400) -> RealESRGANer:
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )


def increase_contrast(img: np.ndarray, alpha: float = 1.15) -> np.ndarray:
    """
    Increases contrast: alpha > 1 pushes dark pixels darker and bright
    pixels brighter. Keep alpha modest - values like 1.5 can clip
    already-bright areas (e.g. an overcast sky) to flat white.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def increase_saturation(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Makes colors more vivid/intense without shifting hue. Keep factor
    modest - values like 1.5 can look artificial/neon on well-lit footage.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255)
    hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Reduces film-grain/compression noise before sharpening. CPU-bound
    and slow on large upscaled frames - disabled by default."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5,
                                            templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Unsharp masking: blurs the image, then subtracts the blur from the
    original to emphasize edges. This is what makes an upscale actually
    look sharper, rather than just "softly bigger."
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def enhance_frames(input_dir: str, output_dir: str,
                    esrgan_model_path: str, tile: int = 400, outscale: int = 3,
                    contrast_boost: bool = False, contrast_amount: float = 1.15,
                    saturation_boost: bool = False, saturation_amount: float = 1.2,
                    denoise_frames: bool = False, sharpen_amount: float = 0.5) -> int:
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Real-ESRGAN onto GPU...")
    upsampler = build_upsampler(esrgan_model_path, tile=tile)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    print(f"Found {len(frame_files)} frames to enhance.")

    for filename in tqdm(frame_files, desc="Enhancing frames"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        # Step 1: AI upscale (general texture/detail improvement)
        upscaled, _ = upsampler.enhance(img, outscale=outscale)

        # Step 2: denoise before sharpening. Disabled by default (slow, CPU-bound).
        if denoise_frames:
            upscaled = denoise(upscaled)

        # Step 3: optional contrast/saturation punch-up (off by default -
        # see module docstring for why)
        if contrast_boost:
            upscaled = increase_contrast(upscaled, alpha=contrast_amount)
        if saturation_boost:
            upscaled = increase_saturation(upscaled, factor=saturation_amount)

        # Step 4: sharpen - this is what makes the upscale visibly sharper
        if sharpen_amount > 0:
            upscaled = sharpen(upscaled, amount=sharpen_amount)

        cv2.imwrite(output_path, upscaled)
        torch.cuda.empty_cache()

    print(f"Done. Enhanced {len(frame_files)} frames -> '{output_dir}'")
    return len(frame_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance video frames (Real-ESRGAN upscaling + optional color grading).")
    parser.add_argument("input_dir", help="Folder containing extracted frames")
    parser.add_argument("output_dir", help="Folder to save enhanced frames")
    parser.add_argument("--esrgan-model", default="models/RealESRNet_x4plus.pth")
    parser.add_argument("--tile", type=int, default=400)
    parser.add_argument("--outscale", type=int, default=3,
                         help="Upscale factor (e.g. 3 to go from 640x360 to 1920x1080). Default 3.")
    parser.add_argument("--contrast", action="store_true",
                         help="Enable contrast boost (experimental, tune per-video - can clip bright skies)")
    parser.add_argument("--contrast-amount", type=float, default=1.15,
                         help="Contrast multiplier if --contrast is set (default 1.15)")
    parser.add_argument("--saturation", action="store_true",
                         help="Enable saturation boost (experimental, tune per-video - can look artificial)")
    parser.add_argument("--saturation-amount", type=float, default=1.2,
                         help="Saturation multiplier if --saturation is set (default 1.2)")
    parser.add_argument("--denoise", action="store_true",
                         help="Enable denoising (slow, CPU-bound - off by default)")
    parser.add_argument("--sharpen-amount", type=float, default=0.5,
                         help="Sharpening strength, 0 to disable (default 0.5)")
    args = parser.parse_args()

    enhance_frames(args.input_dir, args.output_dir, args.esrgan_model, args.tile, args.outscale,
                   contrast_boost=args.contrast, contrast_amount=args.contrast_amount,
                   saturation_boost=args.saturation, saturation_amount=args.saturation_amount,
                   denoise_frames=args.denoise,
                   sharpen_amount=args.sharpen_amount)