from enum import IntEnum, auto

from model.camera_param import CameraParamProfile
from model.laser_detection import LaserDetectionProfile
from model.laser_param import LaserParamProfile
from model.video import Video
from scene.my_scene import MyScene


class State(IntEnum):
    FG_TRAINS_SELECTION = auto()
    FG_EXTRACT = auto()


class StitchingScene(MyScene):
    def __init__(
            self,
            video: Video,
            camera_param_profile: CameraParamProfile,
            laser_param_profile: LaserParamProfile,
            laser_detection_profile: LaserDetectionProfile,
    ):
        super().__init__()

        self._video = video
        self._camera_param_profile = camera_param_profile
        self._laser_param_profile = laser_param_profile
        self._laser_detection_profile = laser_detection_profile

    def update(self):
        pass
