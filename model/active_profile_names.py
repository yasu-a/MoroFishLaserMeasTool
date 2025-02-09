from dataclasses import dataclass


@dataclass(slots=True)
class ActiveProfileNames:
    distortion_profile_name: str | None
    camera_param_profile_name: str | None
    laser_param_profile_name: str | None
    laser_detection_profile_name: str | None

    def to_json(self):
        return {
            "distortion_profile_name": self.distortion_profile_name,
            "camera_param_profile_name": self.camera_param_profile_name,
            "laser_param_profile_name": self.laser_param_profile_name,
            "laser_detection_profile_name": self.laser_detection_profile_name,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            distortion_profile_name=body["distortion_profile_name"],
            camera_param_profile_name=body["camera_param_profile_name"],
            laser_param_profile_name=body["laser_param_profile_name"],
            laser_detection_profile_name=body["laser_detection_profile_name"],
        )

    @classmethod
    def create_default(cls):
        return cls(
            distortion_profile_name=None,
            camera_param_profile_name=None,
            laser_param_profile_name=None,
            laser_detection_profile_name=None,
        )
