from dataclasses import dataclass


@dataclass(slots=True)
class ActiveProfileNames:
    distortion_profile_name: str | None = None
    camera_profile_name: str | None = None
    laser_profile_name: str | None = None
