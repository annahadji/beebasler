"""Util functions for processing videos expected to be ~200 fps."""
import argparse
import datetime
import pathlib
import logging
import subprocess
import sys

import numpy as np

logger = logging.getLogger("Record")
logging.basicConfig(level=logging.INFO)

EXPECTED_DIFF = 0.01
EXPECTED_FPS = 200
SHORTEST_SEGMENT = 5 * EXPECTED_FPS


def consecutive(data: np.ndarray, stepsize: int = 1) -> list:
    """Returns a list of numpy arrays where each array contains consecutive numbers from the data."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def trim_invalid_fps_sections_of_video(video_name: pathlib.Path) -> int:
    """Trim the video to only include frames recorded in a valid fps.
    Returns number of splits video was divided into."""
    # Find timepoints where fps is valid
    timepoints = video_name.parent / f"{video_name.name}-timepoints.npz"
    timepoints = np.load(timepoints)["timepoints"]
    time_diffs = np.diff(timepoints)
    valid_indices = np.where(time_diffs < EXPECTED_DIFF)[0]  # May be non consecutive
    # Trim video into segments containing only valid indices
    split_indices = consecutive(valid_indices)  # (num_segments, ?)
    split_indices = [x for x in split_indices if len(x) > SHORTEST_SEGMENT]
    # Save video(s) with time intervals in name(s)
    for frame_indices in split_indices:
        segment_timepoints = timepoints[frame_indices]
        from_time = np.round(segment_timepoints[0], 3)
        to = np.round(segment_timepoints[-1], 3)
        label = f"{video_name.name}-{from_time}-{to}"
        (video_name.parent / "preprocessed").mkdir(parents=True, exist_ok=True)
        output_name = f"{video_name.parent / 'preprocessed' / label}.mkv"
        command = [
            "ffmpeg",
            "-i",
            f"{video_name.with_suffix('.mkv')}",
            "-ss",
            f"{str(datetime.timedelta(seconds=from_time))}",
            "-to",
            f"{str(datetime.timedelta(seconds=to))}",
            "-c",
            "copy",
            f"{output_name}",
        ]
        subprocess.run(command)
        logger.info(f"Trimmed & saved {output_name}.")
    return len(split_indices)


if __name__ == "__main__":
    # --------------
    # User inputs
    parser = argparse.ArgumentParser(description="Trim and tidy videos.")
    parser.add_argument(
        "data_dir",
        type=str,
        default="outputs/2023-06-13",
        help="Directory path containing videos to trim.",
    )
    args = vars(parser.parse_args())
    # -----
    # Trim videos: save consecutive valid frames as individual videos
    videos = list(pathlib.Path(args["data_dir"]).glob("*.mkv"))
    num_total_videos = 0
    for video in videos:
        logger.info(f"Trimming video {video}...")
        num_total_videos += trim_invalid_fps_sections_of_video(video.with_suffix(""))
    logger.info(f"Trimmed {len(videos)} videos. Resulted in {num_total_videos} videos.")
    # --------------
