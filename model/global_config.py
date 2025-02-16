from dataclasses import dataclass

import numpy as np

from model.active_profile_names import ActiveProfileNames
from model.camera_spec import CameraSpec


@dataclass(slots=True)
class ROI:
    screen_x_min: int
    screen_x_max: int
    screen_y_min: int
    screen_y_max: int
    world_z_min: int
    world_z_max: int
    world_x_min: int
    world_x_max: int
    world_y_min: int
    world_y_max: int

    def to_json(self):
        return {
            "screen_x_min": self.screen_x_min,
            "screen_x_max": self.screen_x_max,
            "screen_y_min": self.screen_y_min,
            "screen_y_max": self.screen_y_max,
            "world_z_min": self.world_z_min,
            "world_z_max": self.world_z_max,
            "world_x_min": self.world_x_min,
            "world_x_max": self.world_x_max,
            "world_y_min": self.world_y_min,
            "world_y_max": self.world_y_max,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            screen_x_min=body.get("screen_x_min"),
            screen_x_max=body.get("screen_x_max"),
            screen_y_min=body.get("screen_y_min"),
            screen_y_max=body.get("screen_y_max"),
            world_z_min=body.get("world_z_min"),
            world_z_max=body.get("world_z_max"),
            world_x_min=body.get("world_x_min"),
            world_x_max=body.get("world_x_max"),
            world_y_min=body.get("world_y_min"),
            world_y_max=body.get("world_y_max"),
        )

    @classmethod
    def create_default(cls):
        return cls(
            screen_x_min=0,
            screen_x_max=700,
            screen_y_min=0,
            screen_y_max=700,
            world_z_min=-150,
            world_z_max=+150,
            world_x_min=-400,
            world_x_max=+400,
            world_y_min=-400,
            world_y_max=+400,
        )

    def get_image_slice(self) -> tuple[slice, slice]:  # y-slice and x-slice
        return (
            slice(self.screen_y_min, self.screen_y_max),
            slice(self.screen_x_min, self.screen_x_max),
        )

    def screen_x_predicate(self, values: float | np.ndarray) -> bool | np.ndarray:
        return (self.screen_x_min <= values) & (values <= self.screen_x_max)

    def screen_y_predicate(self, values: float | np.ndarray) -> bool | np.ndarray:
        return (self.screen_y_min <= values) & (values <= self.screen_y_max)

    def world_x_predicate(self, values: float | np.ndarray) -> bool | np.ndarray:
        return (self.world_x_min <= values) & (values <= self.world_x_max)

    def world_y_predicate(self, values: float | np.ndarray) -> bool | np.ndarray:
        return (self.world_y_min <= values) & (values <= self.world_y_max)

    def world_z_predicate(self, values: float | np.ndarray) -> bool | np.ndarray:
        return (self.world_z_min <= values) & (values <= self.world_z_max)


@dataclass(slots=True)
class GlobalConfig:
    camera_dev_id: int
    camera_spec: CameraSpec
    active_profile_names: ActiveProfileNames
    roi: ROI

    def to_json(self):
        return {
            "camera_dev_id": self.camera_dev_id,
            "camera_spec": self.camera_spec.to_json(),
            "active_profile_names": self.active_profile_names.to_json(),
            "roi": self.roi.to_json(),
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            camera_dev_id=body["camera_dev_id"],
            camera_spec=CameraSpec.from_json(body["camera_spec"]),
            active_profile_names=ActiveProfileNames.from_json(body["active_profile_names"]),
            roi=ROI.from_json(body["roi"]),
        )

    @classmethod
    def create_default(cls):
        return cls(
            camera_dev_id=0,
            camera_spec=CameraSpec.create_default(),
            active_profile_names=ActiveProfileNames.create_default(),
            roi=ROI.create_default(),
        )
