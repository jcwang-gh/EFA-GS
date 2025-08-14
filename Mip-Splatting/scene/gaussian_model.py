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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from math import sqrt
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from scipy.spatial import KDTree
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.lff_xyz_grad_accum = torch.empty(0)
        self.lff_xyz_grad_accum_abs = torch.empty(0)
        self.prev_lff_xyz_grad = torch.empty(0)
        self.prev_lff_xyz_grad_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.lff_denom = torch.empty(0)
        self.prev_selected_pts_mask = torch.empty(0)
        self.split_multiplier = 2.0

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.lff_xyz_grad_accum,
            self.lff_xyz_grad_accum_abs,
            self.prev_lff_xyz_grad,
            self.prev_lff_xyz_grad_abs,
            self.denom,
            self.lff_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        lff_xyz_grad_accum,
        lff_xyz_grad_accum_abs,
        prev_lff_xyz_grad,
        prev_lff_xyz_grad_abs,
        denom,
        lff_denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.lff_xyz_grad_accum = lff_xyz_grad_accum
        self.lff_xyz_grad_accum_abs = lff_xyz_grad_accum_abs
        self.prev_lff_xyz_grad = prev_lff_xyz_grad
        self.prev_lff_xyz_grad_abs = prev_lff_xyz_grad_abs
        self.denom = denom
        self.lff_denom = lff_denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def set_attributes(self, attribute, mask, changes):
        # add changes to original points
        assert attribute in ["xyz", "scaling", "opacity","rotation"], f"the attribute {attribute} doesn't exist or is forbidden to be set manually."
        assert changes.shape[0] == torch.sum(mask).item(), f"the numbers of changes({changes.shape[0]}) and mask({torch.sum(mask).item()}) don't match."
        device = self._scaling.device
        real_attribute = "_" + attribute
        if isinstance(changes, np.ndarray):
            getattr(self, real_attribute)[mask] += torch.from_numpy(changes).type(torch.float32).to(device)
        elif isinstance(changes, torch.Tensor):
            getattr(self, real_attribute)[mask] += changes.type(torch.float32).to(device)
        else:
            raise NotImplementedError
        torch.cuda.empty_cache()

    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    @property
    def get_indices(self):
        return self.prev_selected_pts_mask
    
    @staticmethod
    def convert_bool_to_int(bool_indices):
        return torch.nonzero(bool_indices).squeeze()

    @staticmethod
    def convert_int_to_bool(int_indices, dim=None):
        bool_indices = torch.zeros(dim, dtype=torch.bool, device=int_indices.device) if dim is not None else torch.zeros(int_indices.max().item()+1, dtype=torch.bool, device=int_indices.device)
        bool_indices[int_indices] = True
        return bool_indices

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        self.prune_nan_points()
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    @torch.no_grad()
    def compute_3D_interval(self, cameras):
        # print("Computing 3D interval")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()

        interval = distance / focal_length
        return interval[..., None]
    
    @staticmethod
    def normalize_interval(interval, opt='log', norm='minmax'):
        # trying to amplifying the depth differences, otherwise the model may not converge.
        if opt == 'exp':
            interval = torch.exp(interval)
        elif opt == 'log':
            interval = torch.log(interval)
        if norm == 'minmax':
            min_interval, max_interval = torch.min(interval), torch.max(interval)
            return (interval - min_interval) / (max_interval - min_interval)
        elif norm == 'one':
            return interval / interval.sum()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.lff_xyz_grad_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.lff_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        # At first all Gaussians are selected.
        self.prev_selected_pts_mask = self.convert_bool_to_int(torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device))
        self.lff_xyz_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.lff_xyz_grad_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.prev_lff_xyz_grad = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.prev_lff_xyz_grad_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
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

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_fused_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities = inverse_sigmoid(current_opacity_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        #TODO Maybe we don't need to reset the value, it's better to use moving average instead of reset the value
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_zeros = torch.zeros((new_xyz.shape[0], 1), device="cuda")

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.densification_postfix_lff(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_zeros, islff=False)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # self.prune_points(prune_filter)
        self.prune_points_lff(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_zeros = torch.zeros((new_xyz.shape[0], 1), device="cuda")

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.densification_postfix_lff(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_zeros, islff=False)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # self.prune_points(prune_mask)
        self.prune_points_lff(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1

    def densification_postfix_lff(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_zeros, new_selected_mask=None, islff=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        #TODO Maybe we don't need to reset the value, it's better to use moving average instead of reset the value
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if islff:
            prev_lff_xyz_grad = self.lff_xyz_grad_accum / self.lff_denom
            prev_lff_xyz_grad_abs = self.lff_xyz_grad_accum_abs / self.lff_denom
            prev_lff_xyz_grad[prev_lff_xyz_grad.isnan()] = 0
            prev_lff_xyz_grad_abs[prev_lff_xyz_grad_abs.isnan()] = 0
            self.prev_lff_xyz_grad = torch.concat([prev_lff_xyz_grad, new_zeros])
            self.prev_lff_xyz_grad_abs = torch.concat([prev_lff_xyz_grad_abs, new_zeros])
            self.lff_xyz_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.lff_xyz_grad_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.lff_xyz_grad_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.lff_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            prev_selected_pts_mask = torch.concat([new_selected_mask, torch.ones(new_xyz.shape[0], dtype=torch.bool, device=new_selected_mask.device)])
        else:
            prev_selected_pts_mask = self.convert_int_to_bool(self.prev_selected_pts_mask, dim=self.prev_lff_xyz_grad.shape[0])
            self.prev_lff_xyz_grad = torch.concat([self.prev_lff_xyz_grad, new_zeros])
            self.prev_lff_xyz_grad_abs = torch.concat([self.prev_lff_xyz_grad_abs, new_zeros])
            self.lff_xyz_grad_accum = torch.concat([self.lff_xyz_grad_accum, new_zeros])
            self.lff_xyz_grad_accum_abs = torch.concat([self.lff_xyz_grad_accum_abs, new_zeros])
            self.lff_xyz_grad_accum_abs_max = torch.concat([self.lff_xyz_grad_accum_abs_max, new_zeros])
            self.lff_denom = torch.concat([self.lff_denom, new_zeros])
            prev_selected_pts_mask = torch.concat([prev_selected_pts_mask, torch.ones(new_xyz.shape[0], dtype=torch.bool, device=prev_selected_pts_mask.device)])
        
        self.prev_selected_pts_mask = self.convert_bool_to_int(prev_selected_pts_mask)

    def prune_points_lff(self, mask):
        valid_points_mask = ~mask
        prev_selected_pts_mask = self.convert_int_to_bool(self.prev_selected_pts_mask, dim=self.get_xyz.shape[0])
        valid_prev_selected_pts_mask = prev_selected_pts_mask[valid_points_mask]
        self.prev_selected_pts_mask = self.convert_bool_to_int(valid_prev_selected_pts_mask)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.lff_denom = self.lff_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.lff_xyz_grad_accum = self.lff_xyz_grad_accum[valid_points_mask]
        self.lff_xyz_grad_accum_abs = self.lff_xyz_grad_accum_abs[valid_points_mask]
        self.lff_xyz_grad_accum_abs_max = self.lff_xyz_grad_accum_abs_max[valid_points_mask]
        self.prev_lff_xyz_grad = self.prev_lff_xyz_grad[valid_points_mask]
        self.prev_lff_xyz_grad_abs = self.prev_lff_xyz_grad_abs[valid_points_mask]

    def densify_lff(self, grads, grad_threshold, grads_abs, grad_abs_threshold, cameras, N=1, scaling_multiplier_max=1.0, scaling_multiplier_min=1.0, training_percent_powered=0.0, splitting_ub=1.0, splitting_lb=0.5, tolerance = 1e-5, diffscale=True):
        assert scaling_multiplier_max >= 1.0, f"{scaling_multiplier_max=} is less than 1.0."
        assert scaling_multiplier_min >= 1.0, f"{scaling_multiplier_min=} is less than 1.0."
        assert training_percent_powered <= 1.0 and training_percent_powered >= 0, f"{training_percent_powered} is not in [0,1]."
        
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

        prev_selected_pts_mask = self.convert_int_to_bool(self.prev_selected_pts_mask, self.get_xyz.shape[0])

        intersect_selected_pts_mask = torch.logical_and(prev_selected_pts_mask, selected_pts_mask)

        enlarged_mask = torch.logical_and(torch.logical_not(prev_selected_pts_mask), selected_pts_mask)
        splitted_mask = torch.zeros_like(enlarged_mask, dtype=torch.bool)
        # processing intersection and difference of selected_pts_mask and prev_selected_pts_mask
        # introducing tolerance-based comparison
        whether_decent = torch.where(torch.norm(grads[intersect_selected_pts_mask], dim=-1) <= torch.norm(self.prev_lff_xyz_grad[intersect_selected_pts_mask] - tolerance, dim=-1), True, False)
        whether_decent_abs = torch.where(torch.norm(grads_abs[intersect_selected_pts_mask], dim=-1) <= torch.norm(self.prev_lff_xyz_grad_abs[intersect_selected_pts_mask] - tolerance, dim=-1), True, False)
        whether_decent = torch.logical_or(whether_decent, whether_decent_abs)

        enlarged_mask[intersect_selected_pts_mask] = torch.logical_not(whether_decent)
        splitted_mask[intersect_selected_pts_mask] = whether_decent

        # if gradients do not decent, then enlarge the scales of Gaussians to let them accept more information from neighborhoods.
        enlarged_scaling_changes = torch.zeros_like(self._scaling[enlarged_mask], dtype=torch.float32, device=self._scaling.device)
 
        interval = self.compute_3D_interval(cameras)
        
        with torch.no_grad():
            # depth-based strategy is used.
            # deeper gaussians, lower sampling rates, lower scaling multipliers, higher opacity multipliers
            interval_coef = self.normalize_interval(interval,'log','minmax').to(self._scaling.device)
            scaling_multiplier_coef = interval_coef * scaling_multiplier_min + (1 - interval_coef) * scaling_multiplier_max
            log_scaling_multiplier = torch.log(scaling_multiplier_coef) * training_percent_powered
            # if function scaling_inverse_activation is not logarithm anymore, then use the following code.
            # scaling_changes = self.scaling_inverse_activation(scaling_multiplier * self.get_scaling) - self._scaling
            
            #diffscale is an indicator denoting whether scale-based strategy is used or not.
            if diffscale:
                coef_of_enlarge = torch.ones_like(enlarged_scaling_changes)
                enlarged_sorted_indices = torch.sort(self._scaling[enlarged_mask], dim=1, descending=False)[1]
                coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:,1].unsqueeze(1), - 1.0 / 3)
                coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:,2].unsqueeze(1), - 2.0 / 3)
                enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask] * coef_of_enlarge
            else:
                enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask]

        self.set_attributes("scaling", enlarged_mask, enlarged_scaling_changes)
        enlarged_num = enlarged_scaling_changes.shape[0]

        # if gradients decent, then shrink the gaussians.
        splitted_scaling_changes = torch.zeros_like(self._scaling[splitted_mask], dtype=torch.float32, device=self._scaling.device)
        
        with torch.no_grad():
            if diffscale:
                coef_of_split = torch.ones_like(splitted_scaling_changes)
                splitted_sorted_indices = torch.sort(self._scaling[splitted_mask], dim=1, descending=True)[1]
                coef_of_split.scatter_(1, splitted_sorted_indices[:,1].unsqueeze(1), 0.5)
                coef_of_split.scatter_(1, splitted_sorted_indices[:,2].unsqueeze(1), 0)
                splitted_scaling_changes -= 0.5 * (log_scaling_multiplier[splitted_mask] + torch.log(torch.tensor(self.split_multiplier, device = self._scaling.device))) * coef_of_split
            else:
                splitted_scaling_changes -= 0.5 * (log_scaling_multiplier[splitted_mask] + torch.log(torch.tensor(self.split_multiplier, device = self._scaling.device)))

        self.set_attributes("scaling", splitted_mask, splitted_scaling_changes)
        
        # splitting gaussians with some randomness
        # deeper gaussians, lower sampling rates, higher splitting probabilities to compensate sampling rates.
        splitting_prob_threshold = interval_coef * (splitting_ub - splitting_lb) + splitting_lb
        whether_split = torch.where(torch.rand_like(splitting_prob_threshold[intersect_selected_pts_mask]) <= splitting_prob_threshold[intersect_selected_pts_mask], True, False)
        splitted_mask[intersect_selected_pts_mask] = torch.logical_and(splitted_mask[intersect_selected_pts_mask], whether_split.squeeze())

        stds = self.get_scaling[splitted_mask].repeat(N,1)
        stds[stds.isnan()] = 1e-3
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[splitted_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[splitted_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[splitted_mask].repeat(N,1))
        new_rotation = self._rotation[splitted_mask].repeat(N,1)
        new_features_dc = self._features_dc[splitted_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[splitted_mask].repeat(N,1,1)
        new_opacity = self._opacity[splitted_mask].repeat(N,1)
        new_zeros = torch.zeros((new_xyz.shape[0], 1), device="cuda")

        self.densification_postfix_lff(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_zeros, selected_pts_mask, islff=True)
        
        splitted_num = new_xyz.shape[0]

        return enlarged_num, splitted_num

    def densify_and_prune_lff(self, max_grad, min_opacity, extent, max_screen_size, cameras, N=1, scaling_multiplier_max=1.0, scaling_multiplier_min=1.0,training_percent_powered=0.0, splitting_ub=1.0, splitting_lb=0.5, islff=False, tolerance=1e-5, diffscale=True):
        assert splitting_ub >= splitting_lb, f"splitting upper bound should be not smaller than splitting lower bound, whereas {splitting_ub}<{splitting_lb}."
        assert scaling_multiplier_max >= scaling_multiplier_min, f"scaling upper bound should be not smaller than scaling lower bound, whereas {scaling_multiplier_max}<{scaling_multiplier_min}."

        if islff:
            grads = self.lff_xyz_grad_accum / self.lff_denom
            grads_abs = self.lff_xyz_grad_accum_abs / self.lff_denom
        else:
            grads = self.xyz_gradient_accum / self.denom
            grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0

        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        before = self.get_xyz.shape[0]
        if islff:
            enlarged_num, splitted_num = self.densify_lff(grads, max_grad, grads_abs, Q, cameras, N, scaling_multiplier_max, scaling_multiplier_min, training_percent_powered, splitting_ub, splitting_lb, tolerance, diffscale)
        else:
            self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
            clone = self._xyz.shape[0]
            self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        before_prune = self.get_xyz.shape[0]
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        nan_mask = self.get_nan_mask()
        if nan_mask.sum() > 0:
            print(f"During densify_and_prune_lff stage, {nan_mask.sum()} points are NaN.")
        prune_mask = torch.logical_or(prune_mask, nan_mask)

        self.prune_points_lff(prune_mask)
        after_prune = self.get_xyz.shape[0]
        
        # if islff:
        #     print(f"LFF: Before = {before}, Enlarged = {enlarged_num}, Splitted = {splitted_num}, Pruned = {before_prune-after_prune}, Now = {after_prune}, Change = {after_prune-before}.")
        # else:
        #     print(f"Normal: Before = {before}, Cloned = {clone-before}, Splitted = {before_prune-clone}, Pruned = {before_prune-after_prune}, Now = {after_prune}, Change = {after_prune-before}.")

    def add_densification_stats_lff(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.lff_xyz_grad_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.lff_xyz_grad_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.lff_xyz_grad_accum_abs_max[update_filter] = torch.max(self.lff_xyz_grad_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
        self.lff_denom[update_filter] += 1

    def get_nan_mask(self):
        nan_mask = torch.logical_or(torch.any(self.get_xyz.isnan(), dim=1), torch.any(self.get_scaling.isnan(), dim=1))
        nan_mask = torch.logical_or(nan_mask, torch.any(self.get_rotation.isnan(), dim=1))
        nan_mask = torch.logical_or(nan_mask, torch.any(self.get_opacity.isnan(), dim=1))
        nan_mask = torch.logical_or(nan_mask, torch.any(self.get_features.isnan().reshape(self.get_features.shape[0], -1), dim=1))
        return nan_mask

    def prune_nan_points(self):
        nan_mask = self.get_nan_mask()
        if nan_mask.sum() > 0:
            print(f"{nan_mask.sum()} points are NaN.")
            self.prune_points_lff(nan_mask)
            self.filter_3D = self.filter_3D[~nan_mask]
    
    def reset_prev_grad(self):
        self.prev_lff_xyz_grad = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.prev_lff_xyz_grad_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  
