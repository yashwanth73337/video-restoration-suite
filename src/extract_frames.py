"""
extract_frames.py
Extracts individual frames from a video file using ffmpeg,
saving them as lossless PNGs for the enhancement pipeline.
"""

import subprocess
import os
import argparse


def extract_frames(video_path: str, output_dir: str, fps: int = None) -> int:
    """
    Extract frames from a video into output_dir as PNG images.

    Args:
        video_path: path to the input video file.
        output_dir: folder where extracted frames will be saved.
        fps: if provided, extract at this frame rate instead of the
             video's native fps (useful to reduce frame count for
             faster testing).

    Returns:
        The number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%06d.png")

    # Build the ffmpeg command as a list of arguments.
    # We use a list (not a single string) so subprocess handles
    # spacing/escaping correctly, especially with Windows file paths.
    cmd = ["ffmpeg", "-i", video_path]

    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]

    cmd += [output_pattern, "-hide_banner", "-loglevel", "error"]

    print(f"Extracting frames from '{video_path}' -> '{output_dir}'")
    subprocess.run(cmd, check=True)

    frame_count = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(f"Done. Extracted {frame_count} frames.")
    return frame_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_dir", help="Folder to save extracted frames")
    parser.add_argument("--fps", type=int, default=None,
                         help="Optional: extract at this fps instead of native rate")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.fps)