import dataclasses
import einops
import numpy as np
from openpi import transforms
import openpi.models.model as model
import openpi.policies.libero_policy as libero_policy

@dataclasses.dataclass(frozen=True)
class RRLInputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        if "images/base_image" in data:
            data["base_image"] = data["images/base_image"]
        if "images/wrist_image" in data:
            data["wrist_image"] = data["images/wrist_image"]
        base_image = libero_policy._parse_image(data["base_image"])
        wrist_image = libero_policy._parse_image(data["wrist_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"] 
        if "action" in data:
            inputs["actions"] = data["action"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RRLOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:

        return {"actions": np.asarray(data["actions"][:, :10])}

