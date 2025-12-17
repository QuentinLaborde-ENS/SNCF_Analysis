# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Render side-by-side video (world + reference) with gaze overlays:
- Left  : world frame (downsampled), red dot = world_gazeX/Y
- Right : reference image (downsampled), blue dot = ref_gazeX/Y

Uses imageio + ffmpeg (H.264) for robust video writing on macOS.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import imageio 
import os 


os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"


def is_normalized(series: pd.Series, frac_threshold: float = 0.95) -> bool:
    """Heuristic: True if >= frac_threshold of values are in [0,1]."""
    s = series.dropna().to_numpy()
    if s.size == 0:
        return False
    frac = np.mean((s >= 0.0) & (s <= 1.0))
    return frac >= frac_threshold


def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and aggregate per frame (mean), index=worldFrame.
    Required columns:
      worldFrame, ref_gazeX, ref_gazeY, world_gazeX, world_gazeY
    """
    df = pd.read_csv(
        csv_path,
        usecols=["worldFrame", "ref_gazeX", "ref_gazeY", "world_gazeX", "world_gazeY"],
        dtype={
            "worldFrame": "int32",
            "ref_gazeX": "float32",
            "ref_gazeY": "float32",
            "world_gazeX": "float32",
            "world_gazeY": "float32",
        },
    )
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

    df = (
        df.groupby("worldFrame", as_index=True)
        .agg(
            ref_gazeX=("ref_gazeX", "mean"),
            ref_gazeY=("ref_gazeY", "mean"),
            world_gazeX=("world_gazeX", "mean"),
            world_gazeY=("world_gazeY", "mean"),
        )
        .sort_index()
    )
    return df


def open_video_reader(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    return cap


def draw_circle(img: np.ndarray, x: float, y: float, color=(0, 0, 255), radius: int = 5):
    """Draw circle if finite and inside image bounds."""
    if not np.isfinite(x) or not np.isfinite(y):
        return
    h, w = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        cv2.circle(img, (xi, yi), radius, color, -1)


def display_results(
    world_camera: Path,
    reference_image: Path,
    mapped_gaze_csv: Path,
    out_dir: Path,
    down_width: int = 600,
    down_height: int = 450,
    max_frames: int | None = None,
):
    print("OpenCV version:", cv2.__version__)

    # Ensure output dir is writable
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        test_f = out_dir / "_write_test.tmp"
        test_f.write_text("ok")
        test_f.unlink(missing_ok=True)
        print(f"Output directory is writable: {out_dir}")
    except Exception as e:
        raise PermissionError(f"Cannot write to {out_dir}: {e}")

    # Load CSV
    df = load_results_csv(mapped_gaze_csv)
    print(f"DataFrame loaded, frame index: {df.index.min()}..{df.index.max()}")
    print(
        "NaN counts:\n",
        df[["ref_gazeX", "ref_gazeY", "world_gazeX", "world_gazeY"]].isna().sum(),
    )

    # Detect normalization (0..1) vs pixels
    ref_norm = is_normalized(df["ref_gazeX"]) and is_normalized(df["ref_gazeY"])
    world_norm = is_normalized(df["world_gazeX"]) and is_normalized(df["world_gazeY"])
    print(f"Auto-detect normalization -> ref: {ref_norm}, world: {world_norm}")

    # Read reference image
    ref_bgr = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(f"Cannot read reference image: {reference_image}")
    ref_bgr = cv2.resize(ref_bgr, (down_width, down_height), interpolation=cv2.INTER_LINEAR)

    # Open video
    cap = open_video_reader(world_camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(fps) if fps and fps > 0 else 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w_in}x{h_in}, FPS={fps:.3f}, Frames={frame_count}")

    if frame_count == 0 or w_in == 0 or h_in == 0:
        cap.release()
        raise ValueError("Invalid video or empty file.")

    # Prepare imageio writer (H.264)
    out_w = down_width * 2
    out_h = down_height
    output_path = (out_dir / "video_rim.mp4")
    print(f"Writing to: {output_path.name} (H.264 via ffmpeg)")

    # imageio expects RGB frames
    writer = imageio.get_writer(
    str(output_path),
    fps=float(fps),
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    )

    # Iterate frames
    n_frames_target = min(frame_count, int(df.index.max()) + 1)
    if max_frames is not None:
        n_frames_target = min(n_frames_target, max_frames)
    print(f"Processing {n_frames_target} frames...")

    frame_number = 0
    try:
        while frame_number < n_frames_target:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"End of video at frame {frame_number}")
                break

            # Downscale world frame
            frame_resized = cv2.resize(frame, (down_width, down_height), interpolation=cv2.INTER_LINEAR)

            # Fetch gaze row for this frame
            if frame_number in df.index:
                row = df.loc[frame_number]
                # world coords
                if world_norm:
                    wx = float(row["world_gazeX"]) * down_width
                    wy = float(row["world_gazeY"]) * down_height
                else:
                    wx = float(row["world_gazeX"])/1600*down_width
                    wy = float(row["world_gazeY"])/1200*down_height
                # reference coords
                if ref_norm:
                    rx = float(row["ref_gazeX"]) * down_width
                    ry = float(row["ref_gazeY"]) * down_height
                else:
                    rx = float(row["ref_gazeX"])/4080*down_width
                    ry = float(row["ref_gazeY"])/3060*down_height
            else:
                wx = wy = rx = ry = np.nan

            # Combine side by side (BGR)
            combined_bgr = np.hstack((frame_resized, ref_bgr))

            # Draw dots (BGR colors)
            draw_circle(combined_bgr, wx, wy, color=(0, 0, 255), radius=5)  # red on world
            draw_circle(combined_bgr, (rx + down_width) if np.isfinite(rx) else rx, ry, color=(255, 0, 0), radius=5)  # blue on ref

            # imageio expects RGB
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            writer.append_data(combined_rgb)

            frame_number += 1
            if frame_number % 1000 == 0:
                print(f"  ... {frame_number}/{n_frames_target}")

    except Exception as e:
        print(f"Error during processing at frame {frame_number}: {e}")
    finally:
        cap.release()
        writer.close()

    # Verify output
    if output_path.exists():
        print(f"✔ Video saved: {output_path} ({output_path.stat().st_size} bytes)")
    else:
        print("✖ Error: output file not created.")


def parse_args():
    p = argparse.ArgumentParser(description="Render gaze mapping video (world + reference side-by-side) using imageio/ffmpeg.")
    p.add_argument("--world", required=True, help="Path to world video (e.g., world_video.mp4)")
    p.add_argument("--ref", required=True, help="Path to reference image (e.g., image_ref.jpg)")
    p.add_argument("--csv", required=True, help="Path to mapped gaze CSV (columns: worldFrame, ref_gazeX, ref_gazeY, world_gazeX, world_gazeY)")
    p.add_argument("--outdir", default="output", help="Output directory (default: output)")
    p.add_argument("--width", type=int, default=600, help="Downsampled width per panel (default: 600)")
    p.add_argument("--height", type=int, default=450, help="Downsampled height per panel (default: 450)")
    p.add_argument("--max-frames", type=int, default=None, help="Process at most this many frames (debug)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        display_results(
            world_camera=Path(args.world).resolve(),
            reference_image=Path(args.ref).resolve(),
            mapped_gaze_csv=Path(args.csv).resolve(),
            out_dir=Path(args.outdir).resolve(),
            down_width=args.width,
            down_height=args.height,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)




#python main.py --world data/world_video.mp4 --ref data/image_ref.jpg --csv data/mapped_gaze.csv 
 


