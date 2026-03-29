import os
import tempfile
from typing import Dict, Optional

import cv2
import numpy as np

from src.main import process_image


def decode_image_bytes(payload: bytes) -> Optional[np.ndarray]:
    if payload is None:
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def process_hardware_capture(
    normal_frame: np.ndarray,
    uv_frame: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
    debug: bool = False,
) -> Dict:
    if normal_frame is None:
        raise ValueError("normal_frame cannot be None")

    os.makedirs("data/processed", exist_ok=True)

    normal_tmp = tempfile.NamedTemporaryFile(prefix="hardware_normal_", suffix=".png", dir="data/processed", delete=False)
    uv_tmp = None
    try:
        normal_path = normal_tmp.name
        cv2.imwrite(normal_path, normal_frame)

        uv_path = None
        if uv_frame is not None:
            uv_tmp = tempfile.NamedTemporaryFile(prefix="hardware_uv_", suffix=".png", dir="data/processed", delete=False)
            uv_path = uv_tmp.name
            cv2.imwrite(uv_path, uv_frame)

        result = process_image(normal_path, uv_path, debug=debug)
        result["hardware_metadata"] = metadata or {}
        return result
    finally:
        normal_tmp.close()
        if uv_tmp is not None:
            uv_tmp.close()

        if os.path.exists(normal_tmp.name):
            os.remove(normal_tmp.name)
        if uv_tmp is not None and os.path.exists(uv_tmp.name):
            os.remove(uv_tmp.name)
