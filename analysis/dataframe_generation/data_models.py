import dataclasses


@dataclasses.dataclass
class LocaAccuModel:
    subject_id: str = None
    session_index: int = None
    plane: str = None
    setup: str = None
    block: int = None
    trial_index: int = None
    stim_type: str = None
    head_pose_offset_azi: float = None
    head_pose_offset_ele: float = None
    stim_filename: str = None
    stim_level: float = None
    speaker_id: int = None
    speaker_proc: str = None
    speaker_chan: str = None
    speaker_azi: float = None
    speaker_ele: float = None
    speaker_dist: float = None
    response_azi: float = None
    response_ele: float = None
    response_dist: float = None
    reaction_time: float = None


