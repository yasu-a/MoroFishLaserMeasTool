from dataclasses import dataclass

from model.camera_spec import CameraSpec


@dataclass(slots=True)
class GlobalConfig:
    camera_dev_id: int
    camera_spec: CameraSpec

    def to_json(self):
        return {
            "camera_dev_id": self.camera_dev_id,
            "camera_spec": self.camera_spec.to_json(),
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            camera_dev_id=body["camera_dev_id"],
            camera_spec=CameraSpec.from_json(body["camera_spec"]),
        )

    @classmethod
    def create_default(cls):
        return cls(
            camera_dev_id=0,
            camera_spec=CameraSpec.create_default(),
        )
