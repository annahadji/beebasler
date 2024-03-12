"""Record video from Basler camera using pypylon, and preview images on request
at localhost:8000. Tested on the acA1440-220um Basler camera running at 200 fps."""
import argparse
import base64
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
                png_encoded = base64.b64encode(
                    iio.imwrite("<bytes>", np_img, extension=".png")
                ).decode()
                self.send_response(200)
                # Embed img into html and add js to send coords of mouse click on img
                html = """
                <html>
                    <body>
                        <img id="myImage" style="max-width:100%;max-height:100%;height:auto" src="data:image/png;base64,{}">
                    </body>
                    <script>
                    document.getElementById('myImage').onclick = function(event) {{
                        var x = event.offsetX;
                        var y = event.offsetY;
                        var img = document.getElementById('myImage');
                        var width = img.width;
                        var height = img.height;
                        var xhr = new XMLHttpRequest();
                        xhr.open('POST', '/click');
                        xhr.setRequestHeader('Content-Type', 'application/json');
                        xhr.send(JSON.stringify({{x: x, y: y, w: width, h: height}}));
                    }}
                    </script>
                </html>""".format(
                    png_encoded
                )
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Frame received timeout.")

        def do_POST(self):
            if self.path == "/click":
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                x, y, w, h = data["x"], data["y"], data["w"], data["h"]
                logger.info(
                    f"Click registered by handler at ({x}, {y}) for img sized ({w} x {h})."
                )
                server_conn.send_bytes(b"click")
                server_conn.send_bytes(x.to_bytes(4, "little"))
                server_conn.send_bytes(y.to_bytes(4, "little"))
                server_conn.send_bytes(w.to_bytes(4, "little"))
                server_conn.send_bytes(h.to_bytes(4, "little"))
                self.send_response(200)
                self.end_headers()

    with http.server.HTTPServer(("", 8000), handler) as httpd:
        logger.info("Previews serving at port https://localhost:8000")
        httpd.serve_forever()


def adjust_offset(num, divisible_by: int = 4):
    """Return offset to be nearest divisible by 4."""
    return round(num / divisible_by) * divisible_by


def get_original_coords(
    downsized_coords, downsized_h, downsized_w, original_h, original_w
):
    logger.info("Getting original centre coords...")
    scale_x = original_w / downsized_w
    scale_y = original_h / downsized_h
    original_x = int(downsized_coords[0] * scale_x)
    original_y = int(downsized_coords[1] * scale_y)
    return (original_x, original_y)


def get_top_left_of_centred_coords(
    centred_coords, max_crop_h, max_crop_w, original_h, original_w
):
    logger.info("Shifting original coords...")
    x = centred_coords[0] - (max_crop_w // 2)
    y = centred_coords[1] - (max_crop_h // 2)
    # Check and shift for overlap with the borders of the image
    if x < 0:
        x += abs(x)
    if (x + max_crop_w) > original_w:
        x -= x + max_crop_w - original_w
    if y < 0:
        y += abs(y)
    if (y + max_crop_h) > original_h:
        y -= y + max_crop_w - original_h
    return (x, y)


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
    cam.MaxNumBuffer = 8000
    logger.info(f"Cam max buffer number: {cam.MaxNumBuffer.Value}.")
    cam.PixelFormat = "Mono8"
    logger.info(f"Pixel format set to: {cam.PixelFormat.Value}.")
    # cam.BalanceWhiteAuto = "Off"
    # logger.info(f"Balance white auto set to: {cam.BalanceWhiteAuto.Value}.")
    logger.info(
        f"Max framerate in wakeup config: {round(cam.ResultingFrameRate.Value, 3)} fps."
    )
    # logger.info(f"Start temp status: {cam.TemperatureState.Value}.\n")  # i.e. Ok.
    # -----
    # Set features of device
    if args["use_binning"]:  # Binning helps to increase frame rate
        logger.info(f"Horizontal and vertical binning to increase frame rate.")
        cam.BinningHorizontal = 2
        cam.BinningVertical = 2
    cam.ExposureTime = args["exposure_time"]
    logger.info(f"Exposure time set to: {round(cam.ExposureTime.Value, 3)}.")
    cam.Gain = args["gain"]
    logger.info(f"Gain set to: {cam.Gain.Value}.")
    # Set initial height and width of camera to be smaller
    # cam.Width = 5328
    # cam.OffsetY = 2204
    # cam.BslColorSpace = "Off"
    # -----
    # Optional, launch camera in pure preview
    cropping = True
    if args["crop"]:
        max_crop_height, max_crop_width = 800, 800
        while cropping is True:
            if preview_conn.poll():
                request = preview_conn.recv_bytes()
                if request == b"newframe":  # Received request for new frame
                    img = cam.GrabOne(100).GetArray()
                    original_h, original_w = img.shape
                    print(img.shape)
                    preview_conn.send_bytes(original_h.to_bytes(2, "little"))
                    preview_conn.send_bytes(original_w.to_bytes(2, "little"))
                    preview_conn.send_bytes(img.tobytes())
                    logger.info("Sent preview frame.")
                elif request == b"click":  # Received click coordinates
                    x = int.from_bytes(preview_conn.recv_bytes(), "little")
                    y = int.from_bytes(preview_conn.recv_bytes(), "little")
                    w = int.from_bytes(preview_conn.recv_bytes(), "little")
                    h = int.from_bytes(preview_conn.recv_bytes(), "little")
                    logger.info(f"Received click coords: ({x}, {y}).")
                    cam.AcquisitionStop.Execute()  # Cam stop
                    cam.TLParamsLocked = False  # Grab unlock
                    cam.Width = max_crop_width
                    cam.Height = max_crop_height
                    # Need to map the offset back to the original dims before css
                    # rescales the image
                    x, y = get_original_coords((x, y), h, w, original_h, original_w)
                    x, y = get_top_left_of_centred_coords(
                        (x, y), max_crop_height, max_crop_width, original_h, original_w
                    )
                    x = adjust_offset(x)
                    y = adjust_offset(y)
                    cam.OffsetX = x
                    cam.OffsetY = y
                    cam.TLParamsLocked = True  # Grab lock
                    cam.AcquisitionStart.Execute()  # Cam start
                    cropping = False
                    logger.info("New camera ROI being recorded.")
                    cam.GrabOne(100)

    logger.info(f"Camera h x w: {cam.Height.Value, cam.Width.Value}.")
    args["camera_height"] = cam.Height.Value
    args["camera_width"] = cam.Width.Value
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
            "0",  # that the camera probably adds static anyway
        ],
    )
    # --------------

    # --------------
    # Recording
    now = datetime.datetime.now()
    logger.info(f"Recording {args['record_n_seconds']} second video at {now}...")
    num_images_to_grab = round(args["record_n_seconds"] * estimated_fps)
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
                        if request == b"newframe":  # Received request for new frame
                            height, width = img.shape
                            preview_conn.send_bytes(height.to_bytes(2, "little"))
                            preview_conn.send_bytes(width.to_bytes(2, "little"))
                            preview_conn.send_bytes(img.tobytes())
                            logger.info("Sent preview frame.")
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
        "--gain",
        type=float,
        default=5,
        help="Desired gain of camera. Defaults to 5.",
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
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Wont start filming unless a crop selection has been selected from local host.",
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
