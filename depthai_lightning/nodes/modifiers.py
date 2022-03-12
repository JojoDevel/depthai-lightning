"""Modifiers to convert depthai blobs to cv image data"""

import cv2
import numba as nb
import numpy as np

# Packing scheme for RAW10 - MIPI CSI-2
# - 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
# - stored on 5 bytes (byte0..4) as:
# | byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
# |    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |

# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(
    nb.uint16[::1](nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True
)
def unpack_raw10(input_frame, out, expand16bit):
    lShift = 6 if expand16bit else 0

    # for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input_frame.size // 5):  # around  5ms per frame
        b4 = input_frame[i * 5 + 4]
        out[i * 4] = ((input_frame[i * 5] << 2) | (b4 & 0x3)) << lShift
        out[i * 4 + 1] = ((input_frame[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input_frame[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input_frame[i * 5 + 3] << 2) | (b4 >> 6)) << lShift

    return out


def preview_modifier(inFrame):
    """Convert preview data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        np.array: cv frame in bgr
    """
    return inFrame.getCvFrame()


def isp_modifier(inFrame):
    """Convert isp data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        _type_: cv frame in bgr
    """
    width, height = inFrame.getWidth(), inFrame.getHeight()
    payload = inFrame.getData()

    # isp specific handling
    shape = (height * 3 // 2, width)
    yuv420p = payload.reshape(shape).astype(np.uint8)
    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

    return bgr


def raw_modifier(inFrame):
    """Convert raw data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        _type_:
    """
    width, height = inFrame.getWidth(), inFrame.getHeight()
    payload = inFrame.getData()

    # Preallocate the output buffer
    unpacked = np.empty(payload.size * 4 // 5, dtype=np.uint16)
    # Full range for display, use bits [15:6] of the 16-bit pixels
    unpack_raw10(payload, unpacked, expand16bit=True)
    shape = (height, width)
    bayer = unpacked.reshape(shape).astype(np.uint16)
    # See this for the ordering, at the end of page:
    # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
    bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

    return bgr
