#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import trimesh
import numpy as np
from smplx import FLAME
from plyfile import PlyData, PlyElement

import torch
from torch import nn
import torch.nn.functional as F

from simple_knn._C import distCUDA2

from gaussiansplatting.utils.sh_utils import RGB2SH
from gaussiansplatting.utils.system_utils import mkdir_p
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.general_utils import strip_symmetric, build_scaling_rotation_only
from gaussiansplatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation


class GaussianFlameModel(GaussianModel):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            R, L = build_scaling_rotation_only(scaling_modifier * scaling, rotation)
            flame_T, flame_R, flame_S = self.get_trans_matrix()
            R = flame_R @ R
            L = R @ L
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, model_folder, gender='generic', device='cuda'):
        super().__init__(sh_degree)
        self.device = torch.device(device)
        self.num_betas = 300
        self.num_expression = 100
        self.model = FLAME(
            model_folder,
            gender=gender,
            ext='pkl',
            num_betas=self.num_betas,
            num_expression_coeffs=self.num_expression,
            create_global_orient=True,
        ).to(self.device)

        self.flame_scale = 0
        self.densify_scale = 1
        self.center = 0
        self.scale = 0
        self.T = torch.empty(0)
        self.R = torch.empty(0)
        self.S = torch.empty(0)
        self._shape = torch.empty(0)
        self._expression = torch.empty(0)
        self._jaw_pose = torch.zeros([1, 3], device=self.device)
        self._leye_pose = torch.zeros([1, 3], device=self.device)
        self._reye_pose = torch.zeros([1, 3], device=self.device)
        self._neck_pose = torch.zeros([1, 3], device=self.device)

    @property
    def get_shape(self):
        return self._shape

    @property
    def get_expression(self):
        return self._expression

    @property
    def get_faces(self):
        return self._faces

    @property
    def get_jaw_pose(self):
        return self._jaw_pose

    @property
    def get_leye_pose(self):
        return self._leye_pose

    @property
    def get_reye_pose(self):
        return self._reye_pose

    @property
    def get_neck_pose(self):
        return self._neck_pose

    @property
    def get_tris_scaling(self):
        flame_model = self.model
        # flame: shape and expression
        betas = self.get_shape
        expression = self.get_expression
        faces = self.get_faces
        jaw_pose = self.get_jaw_pose
        leye_pose = self.get_leye_pose
        reye_pose = self.get_reye_pose
        # global_orient = self.get_global_orient
        neck_pose = self.get_neck_pose

        # flame: triangles
        flame_output = flame_model(
            betas=betas,
            neck_pose=neck_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True
        )
        vertices = flame_output.vertices.squeeze()
        # rescale and recenter

        vertices = (vertices - self.center) * self.scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-self.flame_scale)

        tris = vertices[faces]
        T = self.centroid(tris)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        _a = torch.norm(a - T, dim=-1)
        _b = torch.norm(b - T, dim=-1)
        _c = torch.norm(c - T, dim=-1)
        return torch.stack([_a, _b, _c], dim=1)

    def get_trans_matrix(self):
        flame_model = self.model
        # flame: shape and expression
        betas = self.get_shape
        expression = self.get_expression
        faces = self.get_faces
        jaw_pose = self.get_jaw_pose
        leye_pose = self.get_leye_pose
        reye_pose = self.get_reye_pose
        # global_orient = self.get_global_orient
        neck_pose = self.get_neck_pose

        # flame: triangles
        flame_output = flame_model(
            betas=betas,
            neck_pose=neck_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True
        )
        vertices = flame_output.vertices.squeeze()
        # rescale and recenter

        vertices = (vertices - self.center) * self.scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-self.flame_scale)

        tris = vertices[faces]
        # tris = torch.tensor(tris, dtype=torch.float32, device=self.device)

        T = self.centroid(tris)
        R = self.tbn(tris)
        S = self.area(tris)
        return T, R, S

    @property
    def get_scaling(self):
        T, R, S = self.get_trans_matrix()
        scaling = (S + 1e-10).sqrt().unsqueeze(-1) * self.scaling_activation(self._scaling)
        return scaling

    @property
    def get_xyz(self):
        T, R, S = self.get_trans_matrix()
        xyz = (S + 1e-10).sqrt().unsqueeze(-1) * torch.bmm(R, self._xyz.unsqueeze(-1)).squeeze(-1) + T
        return xyz

    def extract_tris(self, betas, expression, global_orient=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                     center=None, scale=None):
        flame_model = self.model

        flame_output = flame_model(
            betas=betas,
            global_orient=global_orient,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True
        )
        vertices = flame_output.vertices.squeeze().detach().cpu().numpy()
        faces = flame_model.faces
        if center is not None and scale is not None:
            vertices = (vertices - center) * scale
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        triangles = vertices[faces]
        triangles = torch.tensor(triangles, dtype=torch.float32, device=self.device)
        return triangles

    def area(self, tris):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        n = torch.cross(b - a, c - a)
        area = 0.5 * torch.norm(n, dim=-1)
        return area

    def centroid(self, tris, dim=1):
        c = tris.sum(dim) / 3
        return c

    def tbn(self, tris):
        # triangles: Tensor[num, 3, 3]
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        n = F.normalize(torch.cross(b - a, c - a), dim=-1)
        d = b - a

        X = F.normalize(torch.cross(d, n), dim=-1)
        Y = F.normalize(torch.cross(d, X), dim=-1)
        Z = F.normalize(d, dim=-1)

        return torch.stack([X, Y, Z], dim=1)

    def create_from_flame(self, spatial_lr_scale: float, flame_scale: float, N=100000):
        self.spatial_lr_scale = spatial_lr_scale

        # flame
        flame_model = self.model
        # flame: shape and expression
        betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)
        expression = torch.zeros([1, self.num_expression], dtype=torch.float32, device=self.device)
        # flame: triangles
        flame_output = flame_model(betas=betas, expression=expression, return_verts=True)
        vertices = flame_output.vertices.squeeze()
        faces = torch.tensor(flame_model.faces.astype(np.int32), dtype=torch.int32, device=self.device)
        # rescale and recenter
        vmin = vertices.min(0)[0]
        vmax = vertices.max(0)[0]
        ori_center = (vmin + vmax) / 2
        ori_scale = 0.6 / (vmax - vmin).max()
        vertices = (vertices - ori_center) * ori_scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-flame_scale)

        self.flame_scale = flame_scale
        self.center = ori_center.detach()
        self.scale = ori_scale.detach()
        self._shape = nn.Parameter(betas.contiguous().requires_grad_(True))
        self._expression = expression.detach()
        self._faces = faces

        # 3DGS
        mesh = trimesh.Trimesh(vertices.detach().cpu(), faces.detach().cpu())
        samples, face_index = trimesh.sample.sample_surface(mesh, N)
        self.num_gs = N
        self._faces = faces[face_index]
        T, R, S = self.get_trans_matrix()
        samples = torch.from_numpy(np.asarray(samples)).float().cuda()
        fused_point_cloud = (samples - T) / (S + 1e-10).sqrt().unsqueeze(-1)
        fused_points = torch.bmm(R.inverse(), fused_point_cloud.unsqueeze(-1)).squeeze(-1)
        fused_color = torch.ones_like(fused_points, dtype=torch.float32, device=self.device) * 0.5
        features = torch.zeros((self.num_gs, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_points.shape[0])
        self._xyz = nn.Parameter(fused_points.requires_grad_(True))

        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3)
        scales = scales / (S + 1e-10).sqrt().unsqueeze(-1)
        scales = torch.log(scales)
        rots = torch.zeros((self.num_gs, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.num_gs, 1), dtype=torch.float, device="cuda"))

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.num_gs), device="cuda")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._faces.shape[1]):
            l.append('face_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        face = self._faces.cpu().numpy()
        shape = self._shape.detach().cpu().numpy()

        # FLAME shape
        shape_attributes = []
        for i in range(self._shape.shape[1]):
            shape_attributes.append("shape_{}".format(i))
        dtype_shape = [(attribute, 'f4') for attribute in shape_attributes]
        elements_shape = np.empty(shape.shape[0], dtype=dtype_shape)
        elements_shape[:] = list(map(tuple, shape))
        el2 = PlyElement.describe(elements_shape, 'shape')

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, face), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el, el2]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        face_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("face")]
        face_names = sorted(face_names, key=lambda x: int(x.split('_')[-1]))
        faces = np.zeros((xyz.shape[0], len(face_names)))
        for idx, attr_name in enumerate(face_names):
            faces[:, idx] = np.asarray(plydata.elements[0][attr_name])
        faces = faces.astype(np.int32)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._faces = torch.tensor(faces, dtype=torch.long, device="cuda")

        try:
            shape_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("shape")]
            shape_names = sorted(shape_names, key=lambda x: int(x.split('_')[-1]))
            shape = np.zeros((1, len(shape_names)))
            for idx, attr_name in enumerate(shape_names):
                shape[:, idx] = np.asarray(plydata.elements[1][attr_name])
            self._shape = nn.Parameter(torch.tensor(shape, dtype=torch.float, device="cuda").requires_grad_(True))
        except IndexError:
            pass

        self.active_sh_degree = self.max_sh_degree

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_gs, 1), device="cuda")
        self.denom = torch.zeros((self.num_gs, 1), device="cuda")

        scale = 1.0
        scale_small = 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * scale_small,
             "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * scale, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr * scale / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * scale_small, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * scale_small, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * scale_small, "name": "rotation"},
            {'params': [self._shape], 'lr': training_args.shape_lr * scale_small, "name": "flame_shape"}
        ]
        self.params_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * scale_small,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * scale_small,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'flame_shape':
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'flame_shape':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_faces, new_max_radii2D):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._faces = torch.cat([self._faces, new_faces], dim=0)

        self.num_gs = self._xyz.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_gs, 1), device="cuda")
        self.denom = torch.zeros((self.num_gs, 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)
        # self.max_radii2D = torch.zeros((self.num_gs), device="cuda")

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.num_gs = self._xyz.shape[0]

        self._faces = self._faces[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.num_gs
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # topk_grad = torch.topk(grads, k=torch.tensor(self.num_gs * 0.1, dtype=torch.int), dim=0)[0]
        # selected_pts_mask = torch.where(grads >= topk_grad[-1], True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling,
                                                                           dim=1).values > 0.01 * scene_extent)
        print(f"{selected_pts_mask.sum()} points are splitted")

        stds = self.scaling_activation(self._scaling[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds / 100)  # 由于面片放缩的关系，太大会导致离群点
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_activation(self._scaling[selected_pts_mask].repeat(N, 1)) / (0.8 * N)
        new_scaling = self.scaling_inverse_activation(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_faces = self._faces[selected_pts_mask].repeat(N, 1)
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N) / (0.8 * N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_faces, new_max_radii2D)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # topk_grad = torch.topk(grads, k=torch.tensor(grads.shape[0] * 0.1, dtype=torch.int), dim=0)[0]
        # selected_pts_mask = torch.where(grads >= topk_grad[-1], True, False)
        selected_pts_mask = torch.where(grads[0] >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= 0.01 * scene_extent)
        print(f"{selected_pts_mask.sum()} points are cloned")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_faces = self._faces[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_faces, new_max_radii2D)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.03 * extent  # 0.05
            prune_mask = torch.logical_or(big_points_vs, big_points_ws)
            print(f'{prune_mask.sum()} points are pruned')
            print(f'max_radii2D: {big_points_vs.sum()} | scaling: {big_points_ws.sum()}')
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_only(self, min_opacity=0.005, extent=0.01):
        unseen_points = (self.get_opacity < min_opacity).squeeze()
        big_points_ws = self.get_scaling.max(dim=1).values > 0.03 * extent
        prune_mask = torch.logical_or(unseen_points, big_points_ws)
        self.prune_points(prune_mask)
        print(f'{prune_mask.sum()} points are pruned')
        print(f'opacity: {unseen_points.sum()} | scaling: {big_points_ws.sum()}')

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):

        grads = viewspace_point_tensor[update_filter]
        grads[grads.isnan()] = 0.0
        grads = torch.norm(grads, dim=-1, keepdim=True)
        grads = F.normalize(grads, dim=0)
        self.xyz_gradient_accum[update_filter] += grads

        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1,
        #                                                      keepdim=True)

        self.denom[update_filter] += 1
