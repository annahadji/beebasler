"""Record video from Basler camera using pypylon, and preview images on request
at localhost:8000. Tested on the acA1440-220um Basler camera running at 200 fps."""
import argparse
import datetime
import http.server
import logging
import multiprocessing as mp
import multiprocessing.connection as mp_conn
import pathlib
import json
import time
import subprocess
import sys

import imageio.v3 as iio
from imageio import get_writer
import matplotlib.pyplot as plt
import numpy as np
import pypylon.pylon as pylon

logger = logging.getLogger("Record")
logging.basicConfig(level=logging.INFO)


def convert_to_timepoints(ts: np.ndarray) -> np.ndarray:
    """Convert timesteps collected from Basler camera (array of nanosecond timestamps
    of camera) to time passed since recording began (in seconds)."""
    # Convert to delta time samples
    ts -= ts[0]
    # Convert nanosecond to seconds
    return ts / 1e9


def get_num_frames(filename: str) -> int:
    """Use ffprobe to get number of frames in video, given filename
    specifying path and name with extension."""
    check_frames_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        f"{filename}",
    ]
    output = subprocess.run(check_frames_command, check=True, capture_output=True)
    num_frames = int(output.stdout.decode("utf-8"))
    logger.info(f"Num frames in output video: {num_frames}.")
    return num_frames


def preview_server(server_conn: mp_conn.Connection) -> None:
    """Setup a process to handle previewing images from Basler camera when
    requested at localhost:8000."""

    class handler(http.server.BaseHTTPRequestHandler):
        """Process class for handling server requests."""

        def do_GET(self):
            server_conn.send_bytes(b"newframe")
            if server_conn.poll(2):  # Give recorder 2 s to respond
                # Receive image height and width (for reshaping), and actual image
                height = server_conn.recv_bytes()
                height = int.from_bytes(height, byteorder="little")
                width = server_conn.recv_bytes()
                width = int.from_bytes(width, byteorder="little")
                response = (
                    server_conn.recv_bytes()
                )  # Blocks until record server sends response
                np_img = np.frombuffer(response, dtype=np.uint8).reshape(
                    (height, width)
                )  # (H, W)
                png_encoded = iio.imwrite("<bytes>", np_img, extension=".png")
                self.send_response(200)
                # Send png image
                self.send_header("Content-type", "image/png")
                self.send_header("Content-length", len(png_encoded))
                self.end_headers()
                self.wfile.write(png_encoded)
            else:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Frame received timeout.")

    with http.server.HTTPServer(("", 8000), handler) as httpd:
        logger.info("Previews serving at port https://localhost:8000")
        httpd.serve_forever()


def record(args: dict, preview_conn: mp_conn.Connection) -> None:
    """Record using Basler camera configured by parameters in args."""
    # --------------
    # Initialise device
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    logger.info(f"Using device {devices[0].GetModelName()}.")
    cam = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
    cam.Open()  # Enable access to camera features
    cam.UserSetSelector = "Default"  # Always good to start from power-on state
    cam.UserSetLoad.Execute()
    cam.MaxNumBuffer = 5000
    logger.info(f"Cam max buffer number: {cam.MaxNumBuffer.Value}")
    logger.info(
        f"Max framerate in wakeup config: {round(cam.ResultingFrameRate.Value, 3)} fps."
    )
    logger.info(f"Start temp status: {cam.TemperatureState.Value}.\n")  # i.e. Ok.
    # -----
    # Set features of device
    if args["use_binning"]:  # Binning helps to increase frame rate
        logger.info(f"Horizontal and vertical binning to increase frame rate.")
        cam.BinningHorizontal = 2
        cam.BinningVertical = 2
    logger.info(f"Camera h x w: {cam.Height.Value, cam.Width.Value}.")
    args["camera_height"] = cam.Height.Value
    args["camera_width"] = cam.Width.Value
    cam.ExposureTime = args["exposure_time"]
    logger.info(f"Exposure time set to: {round(cam.ExposureTime.Value, 3)}.")
    # Sets upper limit for camera's fps (camera will not go above even if factors allow for higher fps)
    # If other factors limit fps below this, fps wont be affected by acquisition fps feature.
    cam.AcquisitionFrameRateEnable = True
    cam.AcquisitionFrameRate = args["fps"]
    logger.info(
        f"Acquisition framerate in use: {round(cam.AcquisitionFrameRate.Value, 3)} fps."
    )
    estimated_fps = cam.ResultingFrameRate.Value
    logger.info(f"Estimated resulting framerate: {round(estimated_fps, 3)} fps.\n")
    # --------------

    # --------------
    # Initialise ImageIO writer.
    pathlib.Path(args["file_out"]).parent.mkdir(parents=True, exist_ok=True)
    # Thanks to https://github.com/basler/pypylon/issues/113#issuecomment-1000461101.
    writer = get_writer(
        f"{args['file_out']}.mkv",  # mkv players often support H.264
        fps=estimated_fps,  # FPS is in units Hz; should be real-time.
        codec="libx264",  # When used properly, this is basically "PNG for video" (i.e. lossless)
        quality=None,  # disables variable compression
        macro_block_size=1,  # avoid ffmpeg resizing height 1080 -> 1088
        ffmpeg_params=[  # compatibility with older library versions
            "-preset",  # set to fast, faster, veryfast, superfast, ultrafast
            "fast",  # for higher speed but worse compression
            "-crf",  # quality; set to 0 for lossless, but keep in mind
            "24",  # that the camera probably adds static anyway
        ],
    )
    # --------------

    # --------------
    # Recording
    now = datetime.datetime.now()
    logger.info(f"Recording {args['record_n_seconds']} second video at {now}...")
    num_images_to_grab = int(args["record_n_seconds"] * estimated_fps)
    cam.StartGrabbingMax(num_images_to_grab, pylon.GrabStrategy_OneByOne)
    timestamps = []
    try:
        while cam.IsGrabbing():
            # Timeout 10 ms (0.01 second), returns if reached
            with cam.RetrieveResult(10, pylon.TimeoutHandling_ThrowException) as res:
                if res.GrabSucceeded():
                    img = res.GetArray()  # Mono img dims (H, W), e.g. (1080, 1440)
                    writer.append_data(img)
                    timestamps.append(res.TimeStamp)  # Nanosecond timestamp of camera
                    print(
                        len(timestamps),
                        "/",
                        num_images_to_grab,
                        file=sys.stderr,
                        end="\r",
                    )
                    res.Release()
                    # Check if there is a request to preview a frame
                    if preview_conn.poll():
                        request = preview_conn.recv_bytes()
                        if request == b"newframe":  # Received requset for new frame
                            height, width = img.shape
                            preview_conn.send_bytes(height.to_bytes(2, "little"))
                            preview_conn.send_bytes(width.to_bytes(2, "little"))
                            preview_conn.send_bytes(img.tobytes())
                            logger.info("Sent preview frame.")
                    # print(cam.SensorReadoutTime.Value)
                else:
                    raise RuntimeError("Grab failed.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user input. Ending recording...")
    time_elapsed_filming = datetime.datetime.now() - now  # NB this isnt resulting fps
    args["time_elapsed_filming_s"] = str(time_elapsed_filming)
    logger.info(f"Time elapsed: {time_elapsed_filming}.")
    cam.StopGrabbing()
    cam.Close()
    logger.info(f"Recorded output {args['file_out']} saved.")
    logger.info(f"Script finished at: {datetime.datetime.now()}.\n")
    # --------------

    # --------------
    # Save other artefacts
    timestamps = np.array(timestamps)  # (num_images_to_grab,)
    timepoints = convert_to_timepoints(timestamps)  # (num_images_to_grab,)
    np.savez(f"{args['file_out']}-timepoints", timepoints=timepoints)
    with open(f"{args['file_out']}-config.json", "w") as f:
        json.dump(args, f)
    # -----
    # Plot perfect timeline of frame grabbing vs what we actually got
    perfect_timeline = np.linspace(0, args["record_n_seconds"], num=num_images_to_grab)
    if perfect_timeline.shape[0] == timepoints.shape[0] == num_images_to_grab:
        plt.figure(dpi=150)
        plt.plot(perfect_timeline, perfect_timeline, label="Target")
        plt.plot(timepoints, perfect_timeline, "--", label="Actual")
        plt.legend()
        plt.xlabel("Actual time point (s)")
        plt.ylabel("Target time point of frame (s)")
        plt.savefig(f"{args['file_out']}-timepoints.png")
    else:
        logger.info(
            f"Timepoints {timepoints.shape} does not match expected {num_images_to_grab}. Timepoints not plotted."
        )
    # -----
    # Sanity check some differences between actual and expected timepoints
    # logger.info("Actual versus expected benchmark timepoints:")
    # logger.info(f"\t1 vs {np.round(timepoints[int(args['fps'])], 5)}")
    # logger.info(f"\t50 vs {np.round(timepoints[int(args['fps'] * 50.0)], 5)}")
    # logger.info(f"\t90 vs {np.round(timepoints[int(args['fps'] * 90.0)], 5)}")
    # --------------


if __name__ == "__main__":
    # --------------
    # User inputs
    parser = argparse.ArgumentParser(description="Record from Basler camera.")
    parser.add_argument(
        "--fps",
        type=float,
        default=200.0,
        help="Desired frame rate of camera in Hz. Defaults to 200 Hz.",
    )
    parser.add_argument(
        "--exposure_time",
        type=float,
        default=2000.0,
        help="Desired exposure time of camera. Defaults to 2000 microseconds.",
    )
    parser.add_argument(
        "--use_binning",
        action="store_true",
        help="Whether or not to reduce image resolution by binning.",
    )
    parser.add_argument(
        "--record_n_seconds",
        type=float,
        default=10.0,
        help="Desired length of time to record for in seconds. Defaults to 10 s.",
    )
    parser.add_argument(
        "--file_out",
        type=str,
        default=f"outputs/{time.strftime('%Y-%m-%d')}/recording-{time.strftime('%Y-%m-%d-%H%M')}",
        help="Name of recording. Appropriate extension will be added on save.",
    )
    args = vars(parser.parse_args())
    # -----
    # Setup and run servers
    server_conn, preview_conn = mp.Pipe()
    server_p = mp.Process(target=preview_server, args=(server_conn,))
    server_p.start()
    record(args, preview_conn)
    server_p.terminate()
    # --------------
