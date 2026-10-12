"""First-person video → LeRobot datasets with hand keypoints, wrist actions and language."""
from .camera import Camera
from .hands import HandTrack, detect, end_effector, place_hand
from .segments import Segment, find_segments

__all__ = ["Camera", "HandTrack", "Segment", "detect", "end_effector", "find_segments", "place_hand"]
__version__ = "0.1.0"
