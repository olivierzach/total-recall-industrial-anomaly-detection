import json
from pathlib import Path

import numpy as np

from src.data.btad import _render_mask


def test_render_mask_smoke(tmp_path: Path):
    # Minimal Supervisely annotation with a 1x1 bitmap at origin.
    # We'll embed a tiny PNG (1x1 white) compressed with zlib and base64.
    import base64, zlib
    from PIL import Image
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(np.array([[255]], dtype=np.uint8)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    data = base64.b64encode(zlib.compress(png_bytes)).decode("ascii")

    ann = {
        "size": {"height": 4, "width": 5},
        "objects": [
            {
                "bitmap": {
                    "data": data,
                    "origin": [1, 2],
                }
            }
        ],
    }

    mask = _render_mask(ann)
    assert mask.shape == (4, 5)
    assert mask[1, 2] == 255
    assert mask.sum() == 255
