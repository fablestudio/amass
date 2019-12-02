import sys, os
import torch
import numpy as np
import math
from smplx.lbs import batch_rodrigues
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

RAD2DEG = 57.295779513

"""
jointmap
0 = M_pelvis
1 = L_upLeg
2 = R_upLeg
3 = M_spine_1
4 = L_loLeg
5 = R_loLeg
6 = M_spine_2
7 = L_foot
8 = R_foot
9 = M_spine_3
10 = L_footEnd
11 = R_footEnd
12 = M_neck
13 = L_clavicle
14 = R_clavicle
15 = M_head
16 = L_upArm
17 = R_upArm
18 = L_loArm
19 = R_loArm
20 = L_hand
21 = R_hand
25 = L_fingerMiddle_0
40 = R_fingerMiddle_0
"""


def normalize(vector):
    norm = np.linalg.norm(vector, axis=-1,keepdims=True)
    if norm < 0.0001:
        raise Exception('Failed to normalize vector')
    return vector/norm


def angle_between(vector1, vector2):
    return math.acos(np.dot(vector1, vector2)) * RAD2DEG


def calculate_joint_vectors(joints, joint_children):
    # Calculate vector from parent to child (ignore hips - start at 1)
    joint_vectors = []
    for i in range(1, len(joint_children)):
        children = joint_children[i]
        if not children:
            continue
        joint_vectors.append(normalize(joints[children[0]] - joints[i]))
    return joint_vectors


def calculate_joint_differences(ref_joint_vectors, posed_joint_vectors):
    joint_diffs = []
    if len(ref_joint_vectors) != len(posed_joint_vectors):
        raise Exception('Reference and Posed joint vector lists must be of equal length')

    for i in range(len(ref_joint_vectors)):
        ref = ref_joint_vectors[i]
        posed = posed_joint_vectors[i]
        joint_diffs.append(angle_between(ref, posed))
    return joint_diffs

    #TODO Calculate matrix from vector, then convert to quaternion


comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
bm = BodyModel(
    bm_path='../body_models/smplh/male/model.npz', 
    num_betas=10, 
    num_dmpls=8, 
    path_dmpl='../body_models/dmpls/male/model.npz').to(comp_device)

joint_parents = bm.kintree_table[0]


# We only care about first 22 joints (ignore hands)
joint_children = [[] for i in range(22)] 
for i in range(len(joint_parents)):
    parent = int(joint_parents[i])

    if (parent == -1):
        continue
    
    if parent < len(joint_children):
        joint_children[parent].append(i)


# Reference pose
ref_body = bm()
ref_joints = c2c(ref_body.Jtr[0])
ref_joint_vectors = calculate_joint_vectors(ref_joints, joint_children)


# Animated pose
bdata = np.load(os.path.dirname(os.path.abspath(__file__)) + '/../body_data/HumanEva/HumanEva/S1/Jog_1_poses.npz')
fId = 0 # frame id of the mocap sequence
#body = bm()

pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device)
pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device)
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)
body = bm(
    pose_body=pose_body, 
    pose_hand=pose_hand, 
    betas=betas)

full_pose = torch.cat([bm.root_orient, pose_body, pose_hand], dim=1)
batch_size = max(betas.shape[0], full_pose.shape[0])
rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=bm.dtype).view([batch_size, -1, 3, 3])

posed_joints = c2c(body.Jtr[0])
posed_mats = c2c(body.rot_mats)
print(rot_mats)
posed_joint_vectors = calculate_joint_vectors(posed_joints, joint_children)


# Calculate difference between animated and reference
joint_diffs = calculate_joint_differences(ref_joint_vectors, posed_joint_vectors)
print(joint_diffs)


import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
from notebook_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

angle = math.pi * 0.3
cosA = math.cos(angle)
sinA = math.sin(angle)
s = np.sqrt(2)/2
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, cosA, sinA, 1.5],
    [0.0, -sinA, cosA, 1.5],
    [0.0, 0.0, 0.0, 1.0],
])
mv.scene.set_pose(mv.camera_node, camera_pose)

body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
joints_mesh = points_to_spheres([posed_joints[40]], vc = colors['red'], radius=0.01)
mv.set_static_meshes([body_mesh] + joints_mesh)
body_image = mv.render(render_wireframe=True)
#show_image(body_image)