from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Image:
    name: str
    data: np.ndarray
