from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RawImage:
    # 歪み補正前の画像

    name: str
    data: np.ndarray
