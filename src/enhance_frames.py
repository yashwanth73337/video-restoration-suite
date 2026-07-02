"""
enhance_frames.py
Enhances video frames using Real-ESRGAN (AI upscaling) combined with
classical image processing (contrast/saturation boost, denoising, sharpening).

Deliberately does NOT use face-specific generative restoration (GFPGAN/
CodeFormer) - those models can alter a real person's identity/features
since they generate plausible faces from a learned prior rather than
only enhancing real pixel data. For content involving real people,
non-generative enhancement is the safer, more faithful choice.
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


def increase_contrast(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Increases contrast: alpha > 1 pushes dark pixels darker and bright
    pixels brighter (beta=0 means no flat brightness shift, so true
    blacks stay at 0).
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def increase_saturation(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Makes colors more vivid/intense without shifting hue - boosts only
    the Saturation channel in HSV color space.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255)
    hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Reduces film-grain/compression noise before sharpening."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5,
                                            templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Unsharp masking: blurs the image, then subtracts the blur from the
    original to emphasize edges. amount controls sharpening strength.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def enhance_frames(input_dir: str, output_dir: str,
                    esrgan_model_path: str, tile: int = 400,
                    contrast_boost: bool = True, contrast_amount: float = 1.5,
                    saturation_boost: bool = True, saturation_amount: float = 1.5,
                    denoise_frames: bool = True, sharpen_amount: float = 0.5) -> int:
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Real-ESRGAN onto GPU...")
    upsampler = build_upsampler(esrgan_model_path, tile=tile)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    print(f"Found {len(frame_files)} frames to enhance.")

    for filename in tqdm(frame_files, desc="Enhancing frames"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        # Step 1: AI upscale (general texture/detail improvement, 4x)
        upscaled, _ = upsampler.enhance(img, outscale=4)

        # Step 2: denoise before sharpening (order matters - sharpening
        # noisy footage amplifies the noise)
        if denoise_frames:
            upscaled = denoise(upscaled)

        # Step 3: punch up contrast (darker darks, brighter brights)
        # and saturation (more vivid colors), for a richer, more
        # "restored film" look rather than a flat/washed appearance
        if contrast_boost:
            upscaled = increase_contrast(upscaled, alpha=contrast_amount)
        if saturation_boost:
            upscaled = increase_saturation(upscaled, factor=saturation_amount)

        # Step 4: sharpen soft/blurry edges
        if sharpen_amount > 0:
            upscaled = sharpen(upscaled, amount=sharpen_amount)

        cv2.imwrite(output_path, upscaled)
        torch.cuda.empty_cache()

    print(f"Done. Enhanced {len(frame_files)} frames -> '{output_dir}'")
    return len(frame_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance video frames (Real-ESRGAN + classical restoration).")
    parser.add_argument("input_dir", help="Folder containing extracted frames")
    parser.add_argument("output_dir", help="Folder to save enhanced frames")
    parser.add_argument("--esrgan-model", default="models/RealESRGAN_x4plus.pth")
    parser.add_argument("--tile", type=int, default=400)
    parser.add_argument("--no-contrast", action="store_true", help="Disable contrast boost")
    parser.add_argument("--contrast-amount", type=float, default=1.5,
                         help="Contrast multiplier, 1.0=no change (default 1.5)")
    parser.add_argument("--no-saturation", action="store_true", help="Disable saturation boost")
    parser.add_argument("--saturation-amount", type=float, default=1.5,
                         help="Saturation multiplier, 1.0=no change (default 1.5)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--sharpen-amount", type=float, default=0.5,
                         help="Sharpening strength, 0 to disable (default 0.5)")
    args = parser.parse_args()

    enhance_frames(args.input_dir, args.output_dir, args.esrgan_model, args.tile,
                   contrast_boost=not args.no_contrast, contrast_amount=args.contrast_amount,
                   saturation_boost=not args.no_saturation, saturation_amount=args.saturation_amount,
                   denoise_frames=not args.no_denoise,
                   sharpen_amount=args.sharpen_amount)