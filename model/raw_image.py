from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RawImage:
    name: str
    data: np.ndarray
