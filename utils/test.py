from bvh import Bvh
with open('../dataset/Bandai-Namco-Research-Motiondataset-1/data/dataset-1_run_giant_001.bvh') as f:
    mocap = Bvh(f.read())

rotations = ['Zrotation', 'Yrotation', 'Xrotation']

for joint_name in mocap.get_joints_names():
    print(joint_name)