"""Deprecated shim: the splatter moved to ``nvs.render.splat`` (T09). Removed in T14."""

from nvs.render.splat import orbit_pose_legacy as orbit_pose
from nvs.render.splat import splat_legacy as splat

__all__ = ["orbit_pose", "splat"]
