# 🎥 🐝 BeeBasler

Personal setup of recording honeybees using Basler's pypylon library. Used for orchestrating filming as well as previewing frames
from the camera as requested on localhost:8000.

## Getting started

Clone the repo, install the dependencies and run using `python3 record.py`. Whilst filming, you can
navigate to https://localhost:8000 and refresh to preview a current frame from camera.

There are a few parameters used to configure filming:

```
(.venv) [19:08:54] 🚀 beebasler $ python3 record.py --help
usage: record.py [-h] [--fps FPS] [--exposure_time EXPOSURE_TIME] [--use_binning] [--record_n_seconds RECORD_N_SECONDS] [--file_out FILE_OUT]

Record from Basler camera.

optional arguments:
  -h, --help            show this help message and exit
  --fps FPS             Desired frame rate of camera in Hz. Defaults to 200 Hz.
  --exposure_time EXPOSURE_TIME
                        Desired exposure time of camera. Defaults to 2000 microseconds.
  --use_binning         Whether or not to reduce image resolution by binning.
  --record_n_seconds RECORD_N_SECONDS
                        Desired length of time to record for in seconds. Defaults to 10 s.
  --file_out FILE_OUT   Name of recording. Appropriate extension will be added on save.
```
