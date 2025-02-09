from dataclasses import dataclass

from model.active_profile_names import ActiveProfileNames
from model.camera_spec import CameraSpec


@dataclass(slots=True)
class GlobalConfig:
    camera_dev_id: int
    camera_spec: CameraSpec
    active_profile_names: ActiveProfileNames

    def to_json(self):
        return {
            "camera_dev_id": self.camera_dev_id,
            "camera_spec": self.camera_spec.to_json(),
            "active_profile_names": self.active_profile_names.to_json(),
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            camera_dev_id=body["camera_dev_id"],
            camera_spec=CameraSpec.from_json(body["camera_spec"]),
            active_profile_names=ActiveProfileNames.from_json(body["active_profile_names"]),
        )

    @classmethod
    def create_default(cls):
        return cls(
            camera_dev_id=0,
            camera_spec=CameraSpec.create_default(),
            active_profile_names=ActiveProfileNames.create_default(),
        )
