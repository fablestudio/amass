import sys, os
import torch
import numpy as np
import math
from smplx.lbs import batch_rodrigues
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

joint_names = [
    'M_pelvis',
    'L_upLeg',
    'R_upLeg',
    'M_spine_1',
    'L_loLeg',
    'R_loLeg',
    'M_spine_2',
    'L_foot',
    'R_foot',
    'M_spine_3',
    'L_footEnd',
    'R_footEnd',
    'M_neck',
    'L_clavicle',
    'R_clavicle',
    'M_head',
    'L_upArm',
    'R_upArm',
    'L_loArm',
    'R_loArm',
    'L_hand',
    'R_hand'
]


comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
bm = BodyModel(
    bm_path='../body_models/smplh/male/model.npz', 
    num_betas=10, 
    num_dmpls=8, 
    path_dmpl='../body_models/dmpls/male/model.npz').to(comp_device)

# Animated pose
bdata = np.load(os.path.dirname(os.path.abspath(__file__)) + '/../body_data/HumanEva/HumanEva/S1/Jog_1_poses.npz')
fId = 0 # frame id of the mocap sequence

pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device)
pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device)
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)
body = bm(
    pose_body=pose_body, 
    pose_hand=pose_hand, 
    betas=betas)

joint_positions = c2c(body.Jtr[0])
full_pose = torch.cat([bm.root_orient, pose_body, pose_hand], dim=1)
batch_size = max(betas.shape[0], full_pose.shape[0])
joint_matrices = batch_rodrigues(full_pose.view(-1, 3), dtype=bm.dtype).view([batch_size, -1, 3, 3])
print(joint_matrices)


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
joint_meshes = points_to_spheres([joint_positions[40]], vc = colors['red'], radius=0.01)
mv.set_static_meshes([body_mesh] + joint_meshes)
body_image = mv.render(render_wireframe=True)
show_image(body_image)