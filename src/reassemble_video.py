"""
reassemble_video.py
Combines enhanced frames back into a video, reattaching the original
audio track from the source video.
"""

import subprocess
import os
import argparse


def reassemble_video(frames_dir: str, original_video_path: str,
                      output_path: str, fps: int = 12) -> None:
    """
    Combines frames into a video and merges in the original audio.

    Args:
        frames_dir: folder containing enhanced frame_%06d.png files.
        original_video_path: path to the original source video (for audio).
        output_path: where to save the final combined video.
        fps: frame rate the frames were extracted at (must match, or
             the video will play back at the wrong speed).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    temp_video_path = output_path.replace(".mp4", "_noaudio_temp.mp4")

    # Step 1: combine frames into a silent video.
    # -crf 16 -preset slow preserve the fine detail added during
    # enhancement - default compression (CRF 23) was found to blur it
    # away. Deflicker filter was tried and removed - it works by
    # blending neighboring frames, which introduces real motion blur
    # on anything moving in the shot.
    print("Combining frames into video...")
    cmd_frames_to_video = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        temp_video_path,
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd_frames_to_video, check=True)

    # Step 2: merge in the original audio
    print("Merging original audio...")
    cmd_merge_audio = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", original_video_path,
        "-c:v", "copy",       # don't re-encode video, just copy it (fast, no quality loss)
        "-c:a", "aac",        # re-encode audio to a standard, compatible format
        "-map", "0:v:0",      # take video from the first input (our enhanced video)
        "-map", "1:a:0",      # take audio from the second input (original source)
        "-shortest",          # trim to the shorter of the two streams
        output_path,
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd_merge_audio, check=True)

    # Clean up the intermediate silent video
    os.remove(temp_video_path)

    print(f"Done. Final video saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reassemble enhanced frames into a video with original audio.")
    parser.add_argument("frames_dir", help="Folder containing enhanced frames")
    parser.add_argument("original_video", help="Path to original video (for audio)")
    parser.add_argument("output_path", help="Path to save the final video")
    parser.add_argument("--fps", type=int, default=12,
                         help="Frame rate frames were extracted at (default 12)")
    args = parser.parse_args()

    reassemble_video(args.frames_dir, args.original_video, args.output_path, args.fps)