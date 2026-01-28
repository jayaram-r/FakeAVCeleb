import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


OUTPUT_DIR = "segmented_chunks"  # folder where chunks will be saved
SEGMENT_TIME = 3                 # chunk duration in seconds


def segment_video(video_path):
    # video_path = video_path.strip()
    if not os.path.exists(video_path):
        return f"{video_path},Skipped (file not found)"

    path_obj = Path(video_path)
    base_name = path_obj.stem
    extension = path_obj.suffix
    # example: segmented_chunks/example_chunk3sec_0001.mp4
    output_path = os.path.join(
        OUTPUT_DIR, f"{base_name}_chunk3sec_%04d{extension}"
    )

    # cmd = [
    #     "ffmpeg", "-y", "-i", video_path,
    #     "-f", "segment",
    #     "-segment_time", str(SEGMENT_TIME),
    #     "-reset_timestamps", "1",
    #     "-c", "copy",
    #     output_path
    # ]

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME),
        # Force a keyframe at every 3s mark for "sharp" cuts
        "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_TIME})",
        "-reset_timestamps", "1",
        "-c:v", "libx264",
        "-crf", "18",            # High quality, visually identical
        "-pix_fmt", "yuv420p",   # Maximum compatibility for all players/tools
        "-preset", "slow",       # Takes longer, but keeps more detail. "medium" is the default
        "-c:a", "copy",          # audio is untouched
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"{video_path},Success"
    except subprocess.CalledProcessError as e:
        return f"{video_path},Error processing: {e.stderr.decode()}"


def main():
    input_files = sys.argv[1]
    output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        n_workers = int(sys.argv[3])
    else:
        n_workers = 2

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir

    with open(input_files, "r") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    n_files = len(video_paths)
    print(f"Processing {n_files} videos using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(segment_video, video_paths),
                       total=n_files, desc="Chunking videos"))

    successes = sum(1 for r in results if "Success" in r)
    print(f"Successfully processed {successes}/{n_files} videos.")


if __name__ == "__main__":
    main()
