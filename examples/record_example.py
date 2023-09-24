"""Example for recording compressed left, right and color video streams
"""

import argparse

import cv2
import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera, LiveView, MonoCamera
from depthai_lightning.nodes.output import EncodingConfig, MultiStreamRecorder
from depthai_lightning.utils import FPSCounter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="App for recording compressed camera streams"
    )
    parser.add_argument(
        "-p", "--preview", action="store_true", help="Show preview", default=True
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=["LOW", "MEDIUM", "HIGH", "BEST"],
        help="Qaulity of the stream compression",
        default="LOW",
    )
    args = parser.parse_args()

    preview = args.preview
    encoding_config = EncodingConfig[args.quality]

    # create pipeline manager
    pm = PipelineManager()

    # create left/right cameras
    left = MonoCamera(pm, dai.CameraBoardSocket.LEFT, 480)
    right = MonoCamera(pm, dai.CameraBoardSocket.RIGHT, 480)

    # also create color camera
    color = ColorCamera(pm)

    # Report encoding quality
    print("Encoding Quality Configuration:")
    print(f"{encoding_config}: {encoding_config.__repr__()}")
    # create recorder for camera streams
    rc = MultiStreamRecorder(
        pm,
        nodes={"left": left, "right": right, "color": color},
        quality=encoding_config,
    )
    if preview:
        # create color camera preview
        lv_preview = LiveView(pm, color, "preview")

    fpsCounter = FPSCounter()

    with pm:
        print(f"Start recording to {rc.path.absolute()} ...")
        fpsCounter.start()
        while True:
            # write streams to files
            rc.write()
            fpsCounter.tick()
            fpsCounter.publish()
            if preview:
                # show camera preview if requested
                lv_preview.show()

                if cv2.waitKey(1) == ord("q"):
                    break
