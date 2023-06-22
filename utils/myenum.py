from enum import Enum


class Action(Enum):
    WALK = "walk"
    RUN = "run"
    DASH = "dash"
    WALK_BACK = "walk-back"
    WALK_RIGHT = "walk-right"
    WALK_LEFT = "walk-left"
    BOW = "bow"
    BYE = "bye"
    GUIDE = "guide"
    BYEBYE = "byebye"
    RESPOND = "respond"
    CALL = "call"
    KICK = "kick"
    SLASH = "slash"
    DANCE = "dance"

    @classmethod
    def get_all_action(cls) -> list[str]:
        return [action.value for action in cls]

    def __len__(self) -> int:
        return len(Action.__members__)


class Styles(Enum):
    ACTIVE = "active"
    NORMAL = "normal"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    PROUD = "proud"
    NOT_CONFIDENT = "not-confident"
    MASCULINITY = "masculinity"
    FEMININE = "feminine"
    CHILDISH = "childish"
    OLD = "old"
    TIRED = "tired"
    MUSICAL = "musical"
    GIANT = "giant"
    CHIMPIRA = "chimpira"

    @classmethod
    def get_all_styles(cls) -> list[str]:
        return [style.value for style in cls]

    def __len__(self) -> int:
        return len(Styles.__members__)


class DefaultColumns(Enum):
    JOINT_ROOT_X = 'joint_Root.x'
    JOINT_ROOT_Y = 'joint_Root.y'
    JOINT_ROOT_Z = 'joint_Root.z'
    HIPS_X = 'Hips.x'
    HIPS_Y = 'Hips.y'
    HIPS_Z = 'Hips.z'
    SPINE_X = 'Spine.x'
    SPINE_Y = 'Spine.y'
    SPINE_Z = 'Spine.z'
    CHEST_X = 'Chest.x'
    CHEST_Y = 'Chest.y'
    CHEST_Z = 'Chest.z'
    NECK_X = 'Neck.x'
    NECK_Y = 'Neck.y'
    NECK_Z = 'Neck.z'
    HEAD_X = 'Head.x'
    HEAD_Y = 'Head.y'
    HEAD_Z = 'Head.z'
    SHOULDER_L_X = 'Shoulder_L.x'
    SHOULDER_L_Y = 'Shoulder_L.y'
    SHOULDER_L_Z = 'Shoulder_L.z'
    UPPERARM_L_X = 'UpperArm_L.x'
    UPPERARM_L_Y = 'UpperArm_L.y'
    UPPERARM_L_Z = 'UpperArm_L.z'
    LOWERARM_L_X = 'LowerArm_L.x'
    LOWERARM_L_Y = 'LowerArm_L.y'
    LOWERARM_L_Z = 'LowerArm_L.z'
    HAND_L_X = 'Hand_L.x'
    HAND_L_Y = 'Hand_L.y'
    HAND_L_Z = 'Hand_L.z'
    SHOULDER_R_X = 'Shoulder_R.x'
    SHOULDER_R_Y = 'Shoulder_R.y'
    SHOULDER_R_Z = 'Shoulder_R.z'
    UPPERARM_R_X = 'UpperArm_R.x'
    UPPERARM_R_Y = 'UpperArm_R.y'
    UPPERARM_R_Z = 'UpperArm_R.z'
    LOWERARM_R_X = 'LowerArm_R.x'
    LOWERARM_R_Y = 'LowerArm_R.y'
    LOWERARM_R_Z = 'LowerArm_R.z'
    HAND_R_X = 'Hand_R.x'
    HAND_R_Y = 'Hand_R.y'
    HAND_R_Z = 'Hand_R.z'
    UPPERLEG_L_X = 'UpperLeg_L.x'
    UPPERLEG_L_Y = 'UpperLeg_L.y'
    UPPERLEG_L_Z = 'UpperLeg_L.z'
    LOWERLEG_L_X = 'LowerLeg_L.x'
    LOWERLEG_L_Y = 'LowerLeg_L.y'
    LOWERLEG_L_Z = 'LowerLeg_L.z'
    FOOT_L_X = 'Foot_L.x'
    FOOT_L_Y = 'Foot_L.y'
    FOOT_L_Z = 'Foot_L.z'
    TOES_L_X = 'Toes_L.x'
    TOES_L_Y = 'Toes_L.y'
    TOES_L_Z = 'Toes_L.z'
    UPPERLEG_R_X = 'UpperLeg_R.x'
    UPPERLEG_R_Y = 'UpperLeg_R.y'
    UPPERLEG_R_Z = 'UpperLeg_R.z'
    LOWERLEG_R_X = 'LowerLeg_R.x'
    LOWERLEG_R_Y = 'LowerLeg_R.y'
    LOWERLEG_R_Z = 'LowerLeg_R.z'
    FOOT_R_X = 'Foot_R.x'
    FOOT_R_Y = 'Foot_R.y'
    FOOT_R_Z = 'Foot_R.z'
    TOES_R_X = 'Toes_R.x'
    TOES_R_Y = 'Toes_R.y'
    TOES_R_Z = 'Toes_R.z'

    @classmethod
    def get_all_default_columns(cls) -> list[str]:
        return [column.value for column in cls]

    def __len__(self) -> int:
        return len(DefaultColumns.__members__)


class PositionColumns(Enum):
    JOINT_ROOT_POS_X = 'Joint_Root_pos.x'
    JOINT_ROOT_POS_Y = 'joint_root_pos.y'
    JOINT_ROOT_POS_Z = 'joint_root_pos.z'
    HIPS_POS_X = 'hips_pos.x'
    HIPS_POS_Y = 'hips_pos.y'
    HIPS_POS_Z = 'hips_pos.z'
    SPINE_POS_X = 'spine_pos.x'
    SPINE_POS_Y = 'spine_pos.y'
    SPINE_POS_Z = 'spine_pos.z'
    CHEST_POS_X = 'chest_pos.x'
    CHEST_POS_Y = 'chest_pos.y'
    CHEST_POS_Z = 'chest_pos.z'
    NECK_POS_X = 'neck_pos.x'
    NECK_POS_Y = 'neck_pos.y'
    NECK_POS_Z = 'neck_pos.z'
    HEAD_POS_X = 'head_pos.x'
    HEAD_POS_Y = 'head_pos.y'
    HEAD_POS_Z = 'head_pos.z'
    SHOULDER_L_POS_X = 'shoulder_L_pos.x'
    SHOULDER_L_POS_Y = 'shoulder_L_pos.y'
    SHOULDER_L_POS_Z = 'shoulder_L_pos.z'
    UPPERARM_L_POS_X = 'upperArm_L_pos.x'
    UPPERARM_L_POS_Y = 'upperArm_L_pos.y'
    UPPERARM_L_POS_Z = 'upperArm_L_pos.z'
    LOWERARM_L_POS_X = 'lowerArm_L_pos.x'
    LOWERARM_L_POS_Y = 'lowerArm_L_pos.y'
    LOWERARM_L_POS_Z = 'lowerArm_L_pos.z'
    HAND_L_POS_X = 'hand_L_pos.x'
    HAND_L_POS_Y = 'hand_L_pos.y'
    HAND_L_POS_Z = 'hand_L_pos.z'
    SHOULDER_R_POS_X = 'shoulder_R_pos.x'
    SHOULDER_R_POS_Y = 'shoulder_R_pos.y'
    SHOULDER_R_POS_Z = 'shoulder_R_pos.z'
    UPPERARM_R_POS_X = 'upperArm_R_pos.x'
    UPPERARM_R_POS_Y = 'upperArm_R_pos.y'
    UPPERARM_R_POS_Z = 'upperArm_R_pos.z'
    LOWERARM_R_POS_X = 'lowerArm_R_pos.x'
    LOWERARM_R_POS_Y = 'lowerArm_R_pos.y'
    LOWERARM_R_POS_Z = 'lowerArm_R_pos.z'
    HAND_R_POS_X = 'hand_R_pos.x'
    HAND_R_POS_Y = 'hand_R_pos.y'
    HAND_R_POS_Z = 'hand_R_pos.z'
    UPPERLEG_L_POS_X = 'upperLeg_L_pos.x'
    UPPERLEG_L_POS_Y = 'upperLeg_L_pos.y'
    UPPERLEG_L_POS_Z = 'upperLeg_L_pos.z'
    LOWERLEG_L_POS_X = 'lowerLeg_L_pos.x'
    LOWERLEG_L_POS_Y = 'lowerLeg_L_pos.y'
    LOWERLEG_L_POS_Z = 'lowerLeg_L_pos.z'
    FOOT_L_POS_X = 'foot_L_pos.x'
    FOOT_L_POS_Y = 'foot_L_pos.y'
    FOOT_L_POS_Z = 'foot_L_pos.z'
    TOES_L_POS_X = 'toes_L_pos.x'
    TOES_L_POS_Y = 'toes_L_pos.y'
    TOES_L_POS_Z = 'toes_L_pos.z'
    UPPERLEG_R_POS_X = 'upperLeg_R_pos.x'
    UPPERLEG_R_POS_Y = 'upperLeg_R_pos.y'
    UPPERLEG_R_POS_Z = 'upperLeg_R_pos.z'
    LOWERLEG_R_POS_X = 'lowerLeg_R_pos.x'
    LOWERLEG_R_POS_Y = 'lowerLeg_R_pos.y'
    LOWERLEG_R_POS_Z = 'lowerLeg_R_pos.z'
    FOOT_R_POS_X = 'foot_R_pos.x'
    FOOT_R_POS_Y = 'foot_R_pos.y'
    FOOT_R_POS_Z = 'foot_R_pos.z'
    TOES_R_POS_X = 'toes_R_pos.x'
    TOES_R_POS_Y = 'toes_R_pos.y'
    TOES_R_POS_Z = 'toes_R_pos.z'

    @classmethod
    def get_all_positions(cls) -> list[str]:
        return [position.value for position in cls]

    def __len__(self) -> int:
        return len(PositionColumns.__members__)


from enum import Enum


class RotationColumns(Enum):
    JOINT_ROOT_ROT_X = 'joint_root_rot.x'
    JOINT_ROOT_ROT_Y = 'joint_root_rot.y'
    JOINT_ROOT_ROT_Z = 'joint_root_rot.z'
    HIPS_ROT_X = 'hips_rot.x'
    HIPS_ROT_Y = 'hips_rot.y'
    HIPS_ROT_Z = 'hips_rot.z'
    SPINE_ROT_X = 'spine_rot.x'
    SPINE_ROT_Y = 'spine_rot.y'
    SPINE_ROT_Z = 'spine_rot.z'
    CHEST_ROT_X = 'chest_rot.x'
    CHEST_ROT_Y = 'chest_rot.y'
    CHEST_ROT_Z = 'chest_rot.z'
    NECK_ROT_X = 'neck_rot.x'
    NECK_ROT_Y = 'neck_rot.y'
    NECK_ROT_Z = 'neck_rot.z'
    HEAD_ROT_X = 'head_rot.x'
    HEAD_ROT_Y = 'head_rot.y'
    HEAD_ROT_Z = 'head_rot.z'
    SHOULDER_L_ROT_X = 'shoulder_L_rot.x'
    SHOULDER_L_ROT_Y = 'shoulder_L_rot.y'
    SHOULDER_L_ROT_Z = 'shoulder_L_rot.z'
    UPPERARM_L_ROT_X = 'upperArm_L_rot.x'
    UPPERARM_L_ROT_Y = 'upperArm_L_rot.y'
    UPPERARM_L_ROT_Z = 'upperArm_L_rot.z'
    LOWERARM_L_ROT_X = 'lowerArm_L_rot.x'
    LOWERARM_L_ROT_Y = 'lowerArm_L_rot.y'
    LOWERARM_L_ROT_Z = 'lowerArm_L_rot.z'
    HAND_L_ROT_X = 'hand_L_rot.x'
    HAND_L_ROT_Y = 'hand_L_rot.y'
    HAND_L_ROT_Z = 'hand_L_rot.z'
    SHOULDER_R_ROT_X = 'shoulder_R_rot.x'
    SHOULDER_R_ROT_Y = 'shoulder_R_rot.y'
    SHOULDER_R_ROT_Z = 'shoulder_R_rot.z'
    UPPERARM_R_ROT_X = 'upperArm_R_rot.x'
    UPPERARM_R_ROT_Y = 'upperArm_R_rot.y'
    UPPERARM_R_ROT_Z = 'upperArm_R_rot.z'
    LOWERARM_R_ROT_X = 'lowerArm_R_rot.x'
    LOWERARM_R_ROT_Y = 'lowerArm_R_rot.y'
    LOWERARM_R_ROT_Z = 'lowerArm_R_rot.z'
    HAND_R_ROT_X = 'hand_R_rot.x'
    HAND_R_ROT_Y = 'hand_R_rot.y'
    HAND_R_ROT_Z = 'hand_R_rot.z'
    UPPERLEG_L_ROT_X = 'upperLeg_L_rot.x'
    UPPERLEG_L_ROT_Y = 'upperLeg_L_rot.y'
    UPPERLEG_L_ROT_Z = 'upperLeg_L_rot.z'
    LOWERLEG_L_ROT_X = 'lowerLeg_L_rot.x'
    LOWERLEG_L_ROT_Y = 'lowerLeg_L_rot.y'
    LOWERLEG_L_ROT_Z = 'lowerLeg_L_rot.z'
    FOOT_L_ROT_X = 'foot_L_rot.x'
    FOOT_L_ROT_Y = 'foot_L_rot.y'
    FOOT_L_ROT_Z = 'foot_L_rot.z'
    TOES_L_ROT_X = 'toes_L_rot.x'
    TOES_L_ROT_Y = 'toes_L_rot.y'
    TOES_L_ROT_Z = 'toes_L_rot.z'
    UPPERLEG_R_ROT_X = 'upperLeg_R_rot.x'
    UPPERLEG_R_ROT_Y = 'upperLeg_R_rot.y'
    UPPERLEG_R_ROT_Z = 'upperLeg_R_rot.z'
    LOWERLEG_R_ROT_X = 'lowerLeg_R_rot.x'
    LOWERLEG_R_ROT_Y = 'lowerLeg_R_rot.y'
    LOWERLEG_R_ROT_Z = 'lowerLeg_R_rot.z'
    FOOT_R_ROT_X = 'foot_R_rot.x'
    FOOT_R_ROT_Y = 'foot_R_rot.y'
    FOOT_R_ROT_Z = 'foot_R_rot.z'
    TOES_R_ROT_X = 'toes_R_rot.x'
    TOES_R_ROT_Y = 'toes_R_rot.y'
    TOES_R_ROT_Z = 'toes_R_rot.z'

    @classmethod
    def get_all_rotations(cls) -> list[str]:
        return [rotation.value for rotation in cls]

    def __len__(self) -> int:
        return len(RotationColumns.__members__)


class VelocityColumns(Enum):
    JOINT_ROOT_VX = 'joint_Root.vx'
    JOINT_ROOT_VY = 'joint_Root.vy'
    JOINT_ROOT_VZ = 'joint_Root.vz'
    HIPS_VX = 'Hips.vx'
    HIPS_VY = 'Hips.vy'
    HIPS_VZ = 'Hips.vz'
    SPINE_VX = 'Spine.vx'
    SPINE_VY = 'Spine.vy'
    SPINE_VZ = 'Spine.vz'
    CHEST_VX = 'Chest.vx'
    CHEST_VY = 'Chest.vy'
    CHEST_VZ = 'Chest.vz'
    NECK_VX = 'Neck.vx'
    NECK_VY = 'Neck.vy'
    NECK_VZ = 'Neck.vz'
    HEAD_VX = 'Head.vx'
    HEAD_VY = 'Head.vy'
    HEAD_VZ = 'Head.vz'
    SHOULDER_L_VX = 'Shoulder_L.vx'
    SHOULDER_L_VY = 'Shoulder_L.vy'
    SHOULDER_L_VZ = 'Shoulder_L.vz'
    UPPERARM_L_VX = 'UpperArm_L.vx'
    UPPERARM_L_VY = 'UpperArm_L.vy'
    UPPERARM_L_VZ = 'UpperArm_L.vz'
    LOWERARM_L_VX = 'LowerArm_L.vx'
    LOWERARM_L_VY = 'LowerArm_L.vy'
    LOWERARM_L_VZ = 'LowerArm_L.vz'
    HAND_L_VX = 'Hand_L.vx'
    HAND_L_VY = 'Hand_L.vy'
    HAND_L_VZ = 'Hand_L.vz'
    SHOULDER_R_VX = 'Shoulder_R.vx'
    SHOULDER_R_VY = 'Shoulder_R.vy'
    SHOULDER_R_VZ = 'Shoulder_R.vz'
    UPPERARM_R_VX = 'UpperArm_R.vx'
    UPPERARM_R_VY = 'UpperArm_R.vy'
    UPPERARM_R_VZ = 'UpperArm_R.vz'
    LOWERARM_R_VX = 'LowerArm_R.vx'
    LOWERARM_R_VY = 'LowerArm_R.vy'
    LOWERARM_R_VZ = 'LowerArm_R.vz'
    HAND_R_VX = 'Hand_R.vx'
    HAND_R_VY = 'Hand_R.vy'
    HAND_R_VZ = 'Hand_R.vz'
    UPPERLEG_L_VX = 'UpperLeg_L.vx'
    UPPERLEG_L_VY = 'UpperLeg_L.vy'
    UPPERLEG_L_VZ = 'UpperLeg_L.vz'
    LOWERLEG_L_VX = 'LowerLeg_L.vx'
    LOWERLEG_L_VY = 'LowerLeg_L.vy'
    LOWERLEG_L_VZ = 'LowerLeg_L.vz'
    FOOT_L_VX = 'Foot_L.vx'
    FOOT_L_VY = 'Foot_L.vy'
    FOOT_L_VZ = 'Foot_L.vz'
    TOES_L_VX = 'Toes_L.vx'
    TOES_L_VY = 'Toes_L.vy'
    TOES_L_VZ = 'Toes_L.vz'
    UPPERLEG_R_VX = 'UpperLeg_R.vx'
    UPPERLEG_R_VY = 'UpperLeg_R.vy'
    UPPERLEG_R_VZ = 'UpperLeg_R.vz'
    LOWERLEG_R_VX = 'LowerLeg_R.vx'
    LOWERLEG_R_VY = 'LowerLeg_R.vy'
    LOWERLEG_R_VZ = 'LowerLeg_R.vz'
    FOOT_R_VX = 'Foot_R.vx'
    FOOT_R_VY = 'Foot_R.vy'
    FOOT_R_VZ = 'Foot_R.vz'
    TOES_R_VX = 'Toes_R.vx'
    TOES_R_VY = 'Toes_R.vy'
    TOES_R_VZ = 'Toes_R.vz'

    @classmethod
    def get_all_velocities(cls) -> list[str]:
        return [velocity.value for velocity in cls]

    def __len__(self) -> int:
        return len(VelocityColumns.__members__)


class AccelerationColumns(Enum):
    JOINT_ROOT_AX = 'joint_Root.ax'
    JOINT_ROOT_AY = 'joint_Root.ay'
    JOINT_ROOT_AZ = 'joint_Root.az'
    HIPS_AX = 'Hips.ax'
    HIPS_AY = 'Hips.ay'
    HIPS_AZ = 'Hips.az'
    SPINE_AX = 'Spine.ax'
    SPINE_AY = 'Spine.ay'
    SPINE_AZ = 'Spine.az'
    CHEST_AX = 'Chest.ax'
    CHEST_AY = 'Chest.ay'
    CHEST_AZ = 'Chest.az'
    NECK_AX = 'Neck.ax'
    NECK_AY = 'Neck.ay'
    NECK_AZ = 'Neck.az'
    HEAD_AX = 'Head.ax'
    HEAD_AY = 'Head.ay'
    HEAD_AZ = 'Head.az'
    SHOULDER_L_AX = 'Shoulder_L.ax'
    SHOULDER_L_AY = 'Shoulder_L.ay'
    SHOULDER_L_AZ = 'Shoulder_L.az'
    UPPERARM_L_AX = 'UpperArm_L.ax'
    UPPERARM_L_AY = 'UpperArm_L.ay'
    UPPERARM_L_AZ = 'UpperArm_L.az'
    LOWERARM_L_AX = 'LowerArm_L.ax'
    LOWERARM_L_AY = 'LowerArm_L.ay'
    LOWERARM_L_AZ = 'LowerArm_L.az'
    HAND_L_AX = 'Hand_L.ax'
    HAND_L_AY = 'Hand_L.ay'
    HAND_L_AZ = 'Hand_L.az'
    SHOULDER_R_AX = 'Shoulder_R.ax'
    SHOULDER_R_AY = 'Shoulder_R.ay'
    SHOULDER_R_AZ = 'Shoulder_R.az'
    UPPERARM_R_AX = 'UpperArm_R.ax'
    UPPERARM_R_AY = 'UpperArm_R.ay'
    UPPERARM_R_AZ = 'UpperArm_R.az'
    LOWERARM_R_AX = 'LowerArm_R.ax'
    LOWERARM_R_AY = 'LowerArm_R.ay'
    LOWERARM_R_AZ = 'LowerArm_R.az'
    HAND_R_AX = 'Hand_R.ax'
    HAND_R_AY = 'Hand_R.ay'
    HAND_R_AZ = 'Hand_R.az'
    UPPERLEG_L_AX = 'UpperLeg_L.ax'
    UPPERLEG_L_AY = 'UpperLeg_L.ay'
    UPPERLEG_L_AZ = 'UpperLeg_L.az'
    LOWERLEG_L_AX = 'LowerLeg_L.ax'
    LOWERLEG_L_AY = 'LowerLeg_L.ay'
    LOWERLEG_L_AZ = 'LowerLeg_L.az'
    FOOT_L_AX = 'Foot_L.ax'
    FOOT_L_AY = 'Foot_L.ay'
    FOOT_L_AZ = 'Foot_L.az'
    TOES_L_AX = 'Toes_L.ax'
    TOES_L_AY = 'Toes_L.ay'
    TOES_L_AZ = 'Toes_L.az'
    UPPERLEG_R_AX = 'UpperLeg_R.ax'
    UPPERLEG_R_AY = 'UpperLeg_R.ay'
    UPPERLEG_R_AZ = 'UpperLeg_R.az'
    LOWERLEG_R_AX = 'LowerLeg_R.ax'
    LOWERLEG_R_AY = 'LowerLeg_R.ay'
    LOWERLEG_R_AZ = 'LowerLeg_R.az'
    FOOT_R_AX = 'Foot_R.ax'
    FOOT_R_AY = 'Foot_R.ay'
    FOOT_R_AZ = 'Foot_R.az'
    TOES_R_AX = 'Toes_R.ax'
    TOES_R_AY = 'Toes_R.ay'
    TOES_R_AZ = 'Toes_R.az'

    @classmethod
    def get_all_accelerations(cls) -> list[str]:
        return [acceleration.value for acceleration in cls]

    def __len__(self) -> int:
        return len(AccelerationColumns.__members__)


class AngularVelocityColumns(Enum):
    JOINT_ROOT_WX = 'joint_Root.wx'
    JOINT_ROOT_WY = 'joint_Root.wy'
    JOINT_ROOT_WZ = 'joint_Root.wz'
    HIPS_WX = 'Hips.wx'
    HIPS_WY = 'Hips.wy'
    HIPS_WZ = 'Hips.wz'
    SPINE_WX = 'Spine.wx'
    SPINE_WY = 'Spine.wy'
    SPINE_WZ = 'Spine.wz'
    CHEST_WX = 'Chest.wx'
    CHEST_WY = 'Chest.wy'
    CHEST_WZ = 'Chest.wz'
    NECK_WX = 'Neck.wx'
    NECK_WY = 'Neck.wy'
    NECK_WZ = 'Neck.wz'
    HEAD_WX = 'Head.wx'
    HEAD_WY = 'Head.wy'
    HEAD_WZ = 'Head.wz'
    SHOULDER_L_WX = 'Shoulder_L.wx'
    SHOULDER_L_WY = 'Shoulder_L.wy'
    SHOULDER_L_WZ = 'Shoulder_L.wz'
    UPPERARM_L_WX = 'UpperArm_L.wx'
    UPPERARM_L_WY = 'UpperArm_L.wy'
    UPPERARM_L_WZ = 'UpperArm_L.wz'
    LOWERARM_L_WX = 'LowerArm_L.wx'
    LOWERARM_L_WY = 'LowerArm_L.wy'
    LOWERARM_L_WZ = 'LowerArm_L.wz'
    HAND_L_WX = 'Hand_L.wx'
    HAND_L_WY = 'Hand_L.wy'
    HAND_L_WZ = 'Hand_L.wz'
    SHOULDER_R_WX = 'Shoulder_R.wx'
    SHOULDER_R_WY = 'Shoulder_R.wy'
    SHOULDER_R_WZ = 'Shoulder_R.wz'
    UPPERARM_R_WX = 'UpperArm_R.wx'
    UPPERARM_R_WY = 'UpperArm_R.wy'
    UPPERARM_R_WZ = 'UpperArm_R.wz'
    LOWERARM_R_WX = 'LowerArm_R.wx'
    LOWERARM_R_WY = 'LowerArm_R.wy'
    LOWERARM_R_WZ = 'LowerArm_R.wz'
    HAND_R_WX = 'Hand_R.wx'
    HAND_R_WY = 'Hand_R.wy'
    HAND_R_WZ = 'Hand_R.wz'
    UPPERLEG_L_WX = 'UpperLeg_L.wx'
    UPPERLEG_L_WY = 'UpperLeg_L.wy'
    UPPERLEG_L_WZ = 'UpperLeg_L.wz'
    LOWERLEG_L_WX = 'LowerLeg_L.wx'
    LOWERLEG_L_WY = 'LowerLeg_L.wy'
    LOWERLEG_L_WZ = 'LowerLeg_L.wz'
    FOOT_L_WX = 'Foot_L.wx'
    FOOT_L_WY = 'Foot_L.wy'
    FOOT_L_WZ = 'Foot_L.wz'
    TOES_L_WX = 'Toes_L.wx'
    TOES_L_WY = 'Toes_L.wy'
    TOES_L_WZ = 'Toes_L.wz'
    UPPERLEG_R_WX = 'UpperLeg_R.wx'
    UPPERLEG_R_WY = 'UpperLeg_R.wy'
    UPPERLEG_R_WZ = 'UpperLeg_R.wz'
    LOWERLEG_R_WX = 'LowerLeg_R.wx'
    LOWERLEG_R_WY = 'LowerLeg_R.wy'
    LOWERLEG_R_WZ = 'LowerLeg_R.wz'
    FOOT_R_WX = 'Foot_R.wx'
    FOOT_R_WY = 'Foot_R.wy'
    FOOT_R_WZ = 'Foot_R.wz'
    TOES_R_WX = 'Toes_R.wx'
    TOES_R_WY = 'Toes_R.wy'
    TOES_R_WZ = 'Toes_R.wz'

    @classmethod
    def get_all_angular_velocities(cls) -> list[str]:
        return [angular_velocity.value for angular_velocity in cls]

    def __len__(self) -> int:
        return len(AngularVelocityColumns.__members__)


class AngularAccelerationColumns(Enum):
    JOINT_ROOT_AWX = 'joint_Root.awx'
    JOINT_ROOT_AWY = 'joint_Root.awy'
    JOINT_ROOT_AWZ = 'joint_Root.awz'
    HIPS_AWX = 'Hips.awx'
    HIPS_AWY = 'Hips.awy'
    HIPS_AWZ = 'Hips.awz'
    SPINE_AWX = 'Spine.awx'
    SPINE_AWY = 'Spine.awy'
    SPINE_AWZ = 'Spine.awz'
    CHEST_AWX = 'Chest.awx'
    CHEST_AWY = 'Chest.awy'
    CHEST_AWZ = 'Chest.awz'
    NECK_AWX = 'Neck.awx'
    NECK_AWY = 'Neck.awy'
    NECK_AWZ = 'Neck.awz'
    HEAD_AWX = 'Head.awx'
    HEAD_AWY = 'Head.awy'
    HEAD_AWZ = 'Head.awz'
    SHOULDER_L_AWX = 'Shoulder_L.awx'
    SHOULDER_L_AWY = 'Shoulder_L.awy'
    SHOULDER_L_AWZ = 'Shoulder_L.awz'
    UPPERARM_L_AWX = 'UpperArm_L.awx'
    UPPERARM_L_AWY = 'UpperArm_L.awy'
    UPPERARM_L_AWZ = 'UpperArm_L.awz'
    LOWERARM_L_AWX = 'LowerArm_L.awx'
    LOWERARM_L_AWY = 'LowerArm_L.awy'
    LOWERARM_L_AWZ = 'LowerArm_L.awz'
    HAND_L_AWX = 'Hand_L.awx'
    HAND_L_AWY = 'Hand_L.awy'
    HAND_L_AWZ = 'Hand_L.awz'
    SHOULDER_R_AWX = 'Shoulder_R.awx'
    SHOULDER_R_AWY = 'Shoulder_R.awy'
    SHOULDER_R_AWZ = 'Shoulder_R.awz'
    UPPERARM_R_AWX = 'UpperArm_R.awx'
    UPPERARM_R_AWY = 'UpperArm_R.awy'
    UPPERARM_R_AWZ = 'UpperArm_R.awz'
    LOWERARM_R_AWX = 'LowerArm_R.awx'
    LOWERARM_R_AWY = 'LowerArm_R.awy'
    LOWERARM_R_AWZ = 'LowerArm_R.awz'
    HAND_R_AWX = 'Hand_R.awx'
    HAND_R_AWY = 'Hand_R.awy'
    HAND_R_AWZ = 'Hand_R.awz'
    UPPERLEG_L_AWX = 'UpperLeg_L.awx'
    UPPERLEG_L_AWY = 'UpperLeg_L.awy'
    UPPERLEG_L_AWZ = 'UpperLeg_L.awz'
    LOWERLEG_L_AWX = 'LowerLeg_L.awx'
    LOWERLEG_L_AWY = 'LowerLeg_L.awy'
    LOWERLEG_L_AWZ = 'LowerLeg_L.awz'
    FOOT_L_AWX = 'Foot_L.awx'
    FOOT_L_AWY = 'Foot_L.awy'
    FOOT_L_AWZ = 'Foot_L.awz'
    TOES_L_AWX = 'Toes_L.awx'
    TOES_L_AWY = 'Toes_L.awy'
    TOES_L_AWZ = 'Toes_L.awz'
    UPPERLEG_R_AWX = 'UpperLeg_R.awx'
    UPPERLEG_R_AWY = 'UpperLeg_R.awy'
    UPPERLEG_R_AWZ = 'UpperLeg_R.awz'
    LOWERLEG_R_AWX = 'LowerLeg_R.awx'
    LOWERLEG_R_AWY = 'LowerLeg_R.awy'
    LOWERLEG_R_AWZ = 'LowerLeg_R.awz'
    FOOT_R_AWX = 'Foot_R.awx'
    FOOT_R_AWY = 'Foot_R.awy'
    FOOT_R_AWZ = 'Foot_R.awz'
    TOES_R_AWX = 'Toes_R.awx'
    TOES_R_AWY = 'Toes_R.awy'
    TOES_R_AWZ = 'Toes_R.awz'

    @classmethod
    def get_all_angular_accelerations(cls) -> list[str]:
        return [angular_acceleration.value for angular_acceleration in cls]

    def __len__(self) -> int:
        return len(AngularAccelerationColumns.__members__)
