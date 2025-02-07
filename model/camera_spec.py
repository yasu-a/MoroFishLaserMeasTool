import cv2


class CameraSpec:
    def __init__(self, width: int, height: int, fps: float):
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)

    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture) -> "CameraSpec":
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def __repr__(self):
        return f"CameraSpec(width={self.width}, height={self.height}, fps={self.fps})"

    def __hash__(self):
        return hash((self.width, self.height, self.fps))

    def __eq__(self, other):
        if isinstance(other, CameraSpec):
            return (
                    self.width == other.width
                    and self.height == other.height
                    and self.fps == other.fps
            )
        return NotImplemented

    def to_json(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            width=body["width"],
            height=body["height"],
            fps=body["fps"],
        )

    @classmethod
    def create_default(cls):
        return cls(
            width=640,
            height=480,
            fps=30.0,
        )
