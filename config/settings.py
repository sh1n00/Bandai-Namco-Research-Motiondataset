import torch

POSITIONCOLUMNS = [
    'joint_Root.x', 'joint_Root.y', 'joint_Root.z', 'Hips.x', 'Hips.y', 'Hips.z', 'Spine.x', 'Spine.y',
    'Spine.z', 'Chest.x', 'Chest.y', 'Chest.z', 'Neck.x', 'Neck.y', 'Neck.z', 'Head.x', 'Head.y',
    'Head.z', 'Shoulder_L.x', 'Shoulder_L.y', 'Shoulder_L.z', 'UpperArm_L.x', 'UpperArm_L.y',
    'UpperArm_L.z', 'LowerArm_L.x', 'LowerArm_L.y', 'LowerArm_L.z', 'Hand_L.x', 'Hand_L.y', 'Hand_L.z',
    'Shoulder_R.x', 'Shoulder_R.y', 'Shoulder_R.z', 'UpperArm_R.x', 'UpperArm_R.y', 'UpperArm_R.z',
    'LowerArm_R.x', 'LowerArm_R.y', 'LowerArm_R.z', 'Hand_R.x', 'Hand_R.y', 'Hand_R.z', 'UpperLeg_L.x',
    'UpperLeg_L.y', 'UpperLeg_L.z', 'LowerLeg_L.x', 'LowerLeg_L.y', 'LowerLeg_L.z', 'Foot_L.x',
    'Foot_L.y', 'Foot_L.z', 'Toes_L.x', 'Toes_L.y', 'Toes_L.z', 'UpperLeg_R.x', 'UpperLeg_R.y',
    'UpperLeg_R.z', 'LowerLeg_R.x', 'LowerLeg_R.y', 'LowerLeg_R.z', 'Foot_R.x', 'Foot_R.y', 'Foot_R.z',
    'Toes_R.x', 'Toes_R.y', 'Toes_R.z'

]

VELOCITYCOLUMNS = [
    'joint_Root.vx', 'joint_Root.vy', 'joint_Root.vz', 'Hips.vx', 'Hips.vy', 'Hips.vz', 'Spine.vx', 'Spine.vy',
    'Spine.vz', 'Chest.vx', 'Chest.vy', 'Chest.vz', 'Neck.vx', 'Neck.vy', 'Neck.vz', 'Head.vx', 'Head.vy', 'Head.vz',
    'Shoulder_L.vx', 'Shoulder_L.vy', 'Shoulder_L.vz', 'UpperArm_L.vx', 'UpperArm_L.vy', 'UpperArm_L.vz',
    'LowerArm_L.vx', 'LowerArm_L.vy', 'LowerArm_L.vz', 'Hand_L.vx', 'Hand_L.vy', 'Hand_L.vz', 'Shoulder_R.vx',
    'Shoulder_R.vy', 'Shoulder_R.vz', 'UpperArm_R.vx', 'UpperArm_R.vy', 'UpperArm_R.vz', 'LowerArm_R.vx',
    'LowerArm_R.vy', 'LowerArm_R.vz', 'Hand_R.vx', 'Hand_R.vy', 'Hand_R.vz', 'UpperLeg_L.vx', 'UpperLeg_L.vy',
    'UpperLeg_L.vz', 'LowerLeg_L.vx', 'LowerLeg_L.vy', 'LowerLeg_L.vz', 'Foot_L.vx', 'Foot_L.vy', 'Foot_L.vz',
    'Toes_L.vx', 'Toes_L.vy', 'Toes_L.vz', 'UpperLeg_R.vx', 'UpperLeg_R.vy', 'UpperLeg_R.vz', 'LowerLeg_R.vx',
    'LowerLeg_R.vy', 'LowerLeg_R.vz', 'Foot_R.vx', 'Foot_R.vy', 'Foot_R.vz', 'Toes_R.vx', 'Toes_R.vy', 'Toes_R.vz'
]

ACCELERATIONCOLUMNS = [
    'joint_Root.ax', 'joint_Root.ay', 'joint_Root.az', 'Hips.ax', 'Hips.ay', 'Hips.az', 'Spine.ax', 'Spine.ay',
    'Spine.az', 'Chest.ax', 'Chest.ay', 'Chest.az', 'Neck.ax', 'Neck.ay', 'Neck.az', 'Head.ax', 'Head.ay', 'Head.az',
    'Shoulder_L.ax', 'Shoulder_L.ay', 'Shoulder_L.az', 'UpperArm_L.ax', 'UpperArm_L.ay', 'UpperArm_L.az',
    'LowerArm_L.ax', 'LowerArm_L.ay', 'LowerArm_L.az', 'Hand_L.ax', 'Hand_L.ay', 'Hand_L.az', 'Shoulder_R.ax',
    'Shoulder_R.ay', 'Shoulder_R.az', 'UpperArm_R.ax', 'UpperArm_R.ay', 'UpperArm_R.az', 'LowerArm_R.ax',
    'LowerArm_R.ay', 'LowerArm_R.az', 'Hand_R.ax', 'Hand_R.ay', 'Hand_R.az', 'UpperLeg_L.ax', 'UpperLeg_L.ay',
    'UpperLeg_L.az', 'LowerLeg_L.ax', 'LowerLeg_L.ay', 'LowerLeg_L.az', 'Foot_L.ax', 'Foot_L.ay', 'Foot_L.az',
    'Toes_L.ax', 'Toes_L.ay', 'Toes_L.az', 'UpperLeg_R.ax', 'UpperLeg_R.ay', 'UpperLeg_R.az', 'LowerLeg_R.ax',
    'LowerLeg_R.ay', 'LowerLeg_R.az', 'Foot_R.ax', 'Foot_R.ay', 'Foot_R.az', 'Toes_R.ax', 'Toes_R.ay', 'Toes_R.az'
]

STYLES = [
    "active", "normal", "happy", "sad", "angry", "proud", "not-confident", "masculinity", "feminine", "childish",
    "old", "tired", "musical", "giant", "chimpira"
]

seed = 42

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
