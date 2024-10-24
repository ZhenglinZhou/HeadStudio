import io
import math
import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass, field
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

import threestudio
# from threestudio.utils.poser import Skeleton
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussiansplatting.scene.cameras import Camera, MiniCam
from gaussiansplatting.scene.gaussian_flame_model import GaussianFlameModel


@threestudio.register("head-3dgs-lks-rig-system")
class Head3DGSLKsRig(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        flame_path: str = "/path/to/flame/model"
        flame_gender: str = 'generic'
        pts_num: int = 100000

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False

        area_relax: bool = False
        shape_update_end_step: int = 12000
        training_w_animation: bool = True

        # area scaling factor
        # area_scaling_factor: float = 1

    cfg: Config

    def configure(self) -> None:
        self.radius = self.cfg.radius
        # self.gaussian = GaussianModel(sh_degree=0)
        self.gaussian = GaussianFlameModel(sh_degree=0, gender=self.cfg.flame_gender, model_folder=self.cfg.flame_path)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32,
                                              device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0],
                                                                                                    dtype=torch.float32,
                                                                                                    device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

        self.cameras_extent = 4.0

        self.cfg.loss.lambda_position = 0.01 * self.cfg.loss.lambda_position
        self.cfg.loss.lambda_scaling = 0.01 * self.cfg.loss.lambda_scaling
        if self.cfg.area_relax:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.smoothl1_position = torch.nn.SmoothL1Loss(beta=1.0, reduction=reduction)
        self.l1_scaling = torch.nn.L1Loss(reduction=reduction)

    def save_gif_to_file(self, images, output_file):
        with io.BytesIO() as writer:
            images[0].save(
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
            )
            writer.seek(0)
            with open(output_file, 'wb') as file:
                file.write(writer.read())

    def get_c2w(self, dist, elev, azim):
        elev = elev * math.pi / 180
        azim = azim * math.pi / 180
        batch_size = dist.shape[0]
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                dist * torch.cos(elev) * torch.cos(azim),
                dist * torch.cos(elev) * torch.sin(azim),
                dist * torch.sin(elev),
            ],
            dim=-1,
        )
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions, device=self.device)
        up: Float[Tensor, "B 3"] = torch.as_tensor(
            [0, 0, 1], dtype=torch.float32, device=self.device)[None, :].repeat(batch_size, 1)
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1], device=self.device)], dim=1
        )
        c2w[:, 3, 3] = 1.0
        return c2w

    def set_pose(self, expression, jaw_pose, leye_pose, reye_pose, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        # self.gaussian._leye_pose = leye_pose.detach()
        # self.gaussian._reye_pose = reye_pose.detach()
        if neck_pose is not None:
            self.gaussian._neck_pose = neck_pose.detach()

    def forward(self, batch: Dict[str, Any], renderbackground=None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        self.viewspace_point_list = []

        if self.cfg.training_w_animation:
            self.set_pose(batch['expression'], batch['jaw_pose'], batch['leye_pose'], batch['reye_pose'])

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w'][id], FoVy=batch['fovy'][id], height=batch['height'],
                                   width=batch['width'])

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg[
                "radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        # depth_min = torch.amin(depths, dim=[1, 2, 3], keepdim=True)
        # depth_max = torch.amax(depths, dim=[1, 2, 3], keepdim=True)
        # depths = (depths - depth_min) / (depth_max - depth_min + 1e-10)
        # depths = depths.repeat(1, 1, 1, 3)

        self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)

        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch)

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        control_images = batch["flame_conds"]
        # control_images = out["depth"]

        guidance_eval = False

        guidance_out = self.guidance(
            images.permute(0, 3, 1, 2), control_images.permute(0, 3, 1, 2), prompt_utils,
            **batch, rgb_as_latents=False,
        )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])

        # scaling = self.gaussian.get_scaling.max(dim=1).values
        scaling = self.gaussian.get_scaling
        tris_scaling = self.gaussian.get_tris_scaling.max(dim=1).values
        big_points_ws = scaling > (0.5 * tris_scaling).unsqueeze(-1)
        loss_scaling = self.l1_scaling(scaling[big_points_ws], torch.zeros_like(scaling[big_points_ws]))
        if self.cfg.area_relax:
            T, R, S = self.gaussian.get_trans_matrix()
            loss_scaling = (loss_scaling / (
                    S.unsqueeze(-1).repeat(1, 3)[big_points_ws] + 1e-10)).mean()
        self.log("train/loss_scaling", loss_scaling)
        loss += loss_scaling * self.C(self.cfg.loss.lambda_scaling)

        if self.true_global_step >= self.cfg.prune_only_start_step:
            position_threshold = 0.5 * tris_scaling
            T, R, S = self.gaussian.get_trans_matrix()
            xyz = self.gaussian.get_xyz - T
            position = torch.norm(xyz, dim=1)
            mask = position > position_threshold
            loss_position = self.smoothl1_position(position[mask], torch.zeros_like(position[mask]))
            if self.cfg.area_relax:
                loss_position = (loss_position / (S[mask] + 1e-10)).mean()
            self.log("train/loss_position", loss_position)
            loss += loss_position * self.C(self.cfg.loss.lambda_position)

        loss_shape = torch.norm(self.gaussian._shape)
        self.log("train/loss_shape", loss_shape)
        loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):

        # return

        with torch.no_grad():

            if self.true_global_step < self.cfg.densify_prune_end_step:  # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:  # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None  # 3000
                    self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, size_threshold)

                    # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    self.gaussian.prune_only(extent=self.cameras_extent)

            if self.true_global_step > self.cfg.shape_update_end_step:
                for param_group in self.gaussian.optimizer.param_groups:
                    if param_group['name'] == 'flame_shape':
                        param_group['lr'] = 1e-10

    def on_after_backward(self):
        self.dataset.skel.betas = self.gaussian.get_shape.detach()
        # pass

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        self.gaussian.create_from_flame(self.cameras_extent, -10, N=self.cfg.pts_num)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )
