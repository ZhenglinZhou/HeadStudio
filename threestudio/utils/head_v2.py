import cv2
import numbers
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import trimesh

from smplx import FLAME
from smplx.utils import Tensor

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
)

from threestudio.utils.mediapipe_utils import draw_landmarks_468
from threestudio.utils.mediapipe_utils_v2 import draw_landmarks_105


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def draw_openpose(all_lmks, H, W, eps=0.01):
    bs = all_lmks.shape[0]
    canvas = np.zeros(shape=(bs, H, W, 3), dtype=np.uint8)
    for i, lmks in enumerate(all_lmks):
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            # x = int(x * W)
            # y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas[i], (int(x), int(y)), 3, (255, 255, 255), thickness=-1)
    return canvas


def plot_points(vis, points, radius=1, color=(255, 255, 0), shift=4, indexes=0, is_index=False, index_size=0.2):
    if isinstance(points, list):
        num_point = len(points)
    elif isinstance(points, np.ndarray):
        num_point = points.shape[0]
    else:
        raise NotImplementedError
    if isinstance(radius, numbers.Number):
        radius = np.zeros((num_point)) + radius

    if isinstance(indexes, numbers.Number):
        indexes = [indexes + i for i in range(num_point)]
    elif isinstance(indexes, list):
        pass
    else:
        raise NotImplementedError

    factor = (1 << shift)
    for (index, p, s) in zip(indexes, points, radius):
        cv2.circle(vis, (int(p[0] * factor + 0.5), int(p[1] * factor + 0.5)),
                   int(s * factor), color, 1, cv2.LINE_AA, shift=shift)
        if is_index:
            vis = cv2.putText(vis, str(index), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, index_size,
                              (255, 255, 255), 1)

    return vis


class MeshRendererWithDepth(nn.Module):
    def __init__(
            self,
            rasterizer,
            shader=None,
    ):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world):
        fragments = self.rasterizer(meshes_world)
        output = fragments.zbuf
        if self.shader is not None:
            images = self.shader(fragments, meshes_world)
            output = (output, images)

        return output


class FlamePointswRandomExp:
    def __init__(
            self,
            model_folder,
            gender='generic',
            ext='pkl',
            device='cuda',
            batch_size=1,
            image_size=512,
            flame_scale=-10,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.num_betas = 300
        self.num_expression = 100
        self.flame_scale = flame_scale
        self.image_size = image_size

        self.model = FLAME(
            model_folder,
            gender=gender,
            ext=ext,
            num_betas=self.num_betas,
            num_expression_coeffs=self.num_expression,
            use_face_contour=True,
        ).to(self.device)

        self.center = 0
        self.scale = 0

        self.init_mesh()

        # initizalize raster
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])

        self.betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)

        # facial landmarks
        # 68-DECA
        # flame_lmk_embedding_path = "/home/zhenglin/Documents/DECA/data/landmark_embedding.npy"
        # lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")
        # lmk_embeddings = lmk_embeddings[()]
        # self.full_lmk_faces_idx = torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).to(torch.long).to(self.device)
        # self.full_lmk_bary_coords = torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(torch.float32).to(self.device)

        # 468-HeadSculpt
        flame_mediapipe_lmk_embedding_path = "./ckpts/ControlNet-Mediapipe/flame2facemesh.npy"
        self.lmk_faces_idx_mediapipe_468 = np.load(flame_mediapipe_lmk_embedding_path).astype(int)
        # Add: Left Eye 4597 and Right Eye 4051
        left_eye_index = 4597  # 4597
        right_eye_index = 4051  # 4051
        self.lmk_faces_idx_mediapipe_468 = np.append(self.lmk_faces_idx_mediapipe_468,
                                                     [left_eye_index, right_eye_index])

        # 105-EMOCA
        flame_mediapipe_lmk_embedding_path = "./ckpts/FLAME-2000/mediapipe_landmark_embedding.npz"
        lmk_embeddings_mediapipe = np.load(flame_mediapipe_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        self.lmk_faces_idx_mediapipe_105 = torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64),
                                                        dtype=torch.long).to(self.device)
        self.lmk_bary_coords_mediapipe_105 = torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'],
                                                          dtype=torch.float32).to(self.device)

    def init_mesh(self):
        betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)
        expression = torch.zeros([1, self.num_expression], dtype=torch.float32, device=self.device)

        output = self.model(betas=betas, expression=expression, return_verts=True)
        vertices = output.vertices.squeeze()

        # rescale and recenter
        vmin = vertices.min(0)[0]
        vmax = vertices.max(0)[0]
        ori_center = (vmin + vmax) / 2
        ori_scale = 0.6 / (vmax - vmin).max()
        vertices = (vertices - ori_center) * ori_scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-self.flame_scale)

        self.center = ori_center
        self.scale = ori_scale

    def get_camera(self, dist=0.6, elev=0, azim=0, at=((0, 0, 0),), up=((0, 1, 0),), fov=40):
        R, T = look_at_view_transform(dist, elev, azim, degrees=True, at=at, up=up)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        return cameras

    def depth_postprocess(self, fragments):
        depthmap = fragments[..., 0]
        # depthmap: (bs, 512, 512)
        depth_min = torch.amin(depthmap, dim=[1, 2], keepdim=True)
        depth_max = torch.amax(depthmap, dim=[1, 2], keepdim=True)
        depth_images = (depthmap - depth_min) / (depth_max - depth_min + 1e-10)
        depth_images = depth_images.clip(0, 1).to(torch.float32)
        depth_images = depth_images[..., None].expand(-1, -1, -1, 3)
        return depth_images

    def camera_conversion(self, v):
        # v: [bs, 3]
        # threestudio -> FLAME
        # x, y, z -> -y, z, -x
        v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]
        temp_v = torch.stack([-v_y, v_z, -v_x])
        return temp_v.permute(1, 0)

    def render(self, cameras, mesh):
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        render = MeshRendererWithDepth(rasterizer=rasterizer)

        fragments = render(mesh)
        depths = self.depth_postprocess(fragments)
        return depths

    def render_mesh(self, cameras, mesh, colors):
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=self.lights,

        )
        render = MeshRendererWithDepth(rasterizer=rasterizer, shader=shader)
        fragments, images = render(mesh)

        # bug
        # render = MeshRenderer(rasterizer, shader=shader)
        # images = render(mesh, materials={"diffuse_color": colors.unsqueeze(0)})
        # from pytorch3d.renderer.materials import Materials
        # materials = Materials(diffuse_color=colors)
        # images = render(mesh, materials=materials)
        images = self.image_postprocee(images)
        return images

    def image_postprocee(self, images):
        return images[..., :3]

    def get_cond_lmk_openpose(self, joints, cameras):
        # lmk3d = vertices2landmarks(
        #     vertices.repeat(self.batch_size, 1, 1),
        #     faces,
        #     self.full_lmk_faces_idx.repeat(self.batch_size, 1),
        #     self.full_lmk_bary_coords.repeat(self.batch_size, 1, 1)
        # )
        lmk3d = joints[-68:]
        lmk3d = (lmk3d - self.center) * self.scale
        lmk3d *= 1.1 ** (-self.flame_scale)
        # Apply projection
        proj_lmk = cameras.transform_points(lmk3d.repeat(self.batch_size, 1, 1))[:, :, :2]

        # Map to image coordinate
        img_lmk = 0.5 * (-proj_lmk + 1) * self.image_size

        # Draw Pose
        imgs = draw_openpose(img_lmk.detach().cpu(), self.image_size, self.image_size)

        return imgs

    def get_cond_lmk_mediapipe(self, vertices, faces, cameras):
        lmk3d_105 = vertices2landmarks(
            vertices.repeat(self.batch_size, 1, 1),
            faces,
            self.lmk_faces_idx_mediapipe_105.repeat(self.batch_size, 1),
            self.lmk_bary_coords_mediapipe_105.repeat(self.batch_size, 1, 1)
        )

        lmk3d_468 = vertices[self.lmk_faces_idx_mediapipe_468]

        def proj_lmk3d(lmk3d):
            # Apply projection
            if len(lmk3d.shape) == 2:
                lmk3d = lmk3d.repeat(self.batch_size, 1, 1)
            proj_lmk = cameras.transform_points(lmk3d)[:, :, :2]

            # Map to image coordinate
            img_lmk = 0.5 * (-proj_lmk + 1) * self.image_size
            return img_lmk

        img_lmk_105 = proj_lmk3d(lmk3d_105)
        img_lmk_468 = proj_lmk3d(lmk3d_468)

        # Draw 105 Mediapipe
        canvas = np.ones((self.batch_size, 512, 512, 3))
        imgs = draw_landmarks_105(canvas, img_lmk_105.detach().cpu().numpy())

        # Draw face oval in 468 Mediapipe
        for idx, img in enumerate(imgs):
            imgs[idx] = draw_landmarks_468(img, img_lmk_468[idx].long().detach().cpu().numpy())

        # imgs = plot_points(canvas, (img_lmk_105[0]).detach().cpu().numpy(), is_index=True)

        # Draw 468 Mediapipe
        # imgs = draw_landmarks(canvas, img_lmk_468[0].long().detach().cpu().numpy())
        # imgs = np.array(imgs, dtype=np.int32)
        # imgs = draw_openpose(img_lmk.detach().cpu(), self.image_size, self.image_size)
        return imgs / 255.0

    def get_cond_depth(self, vertices, faces, cameras, mesh_vis=False, mesh_rgb=False):
        if True:
            mesh = trimesh.Trimesh(vertices.detach().cpu(), faces.detach().cpu())
            area_faces = np.log(mesh.area_faces)
            cmap = plt.cm.PiYG  # hsv
            norm = plt.Normalize(area_faces.min(), area_faces.max())
            colors = cmap(norm(area_faces))

            colors = torch.from_numpy(colors[:, :3]).to(self.device)  # [M, 3]
            colors = colors[None, :, None, None, :]
            textures = TexturesAtlas(atlas=colors)

        # verts_rgb = torch.ones(vertices.shape, device=self.device)
        # textures = TexturesVertex([verts_rgb for _ in range(self.batch_size)])
        mesh = Meshes(
            verts=[vertices for _ in range(self.batch_size)],
            faces=[faces for _ in range(self.batch_size)],
            textures=textures
        )
        if mesh_vis:
            results = self.render_mesh(cameras, mesh, colors)
        else:
            results = self.render(cameras, mesh)
        return results

    def get_cond(self, dist=0.6, elev=0, azim=0, at=((0, 0, 0),), up=((0, 1, 0),), fov=40,
                 betas=None, expression=None, jaw_pose=None, leye_pose=None, reye_pose=None, neck_pose=None,
                 lmk=False, mediapipe=True, mesh_vis=False):
        if betas is None:
            betas = self.betas
        if expression is None:
            expression = torch.zeros([1, self.num_expression], device=self.device)
        output = self.model(
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            neck_pose=neck_pose,
            return_verts=True
        )
        vertices = output.vertices.squeeze()
        vertices = (vertices - self.center) * self.scale
        vertices *= 1.1 ** (-self.flame_scale)

        joints = output.joints.detach().squeeze()

        faces = torch.tensor(self.model.faces.astype(np.int32), dtype=torch.int32, device=self.device)

        # threestudio -> FLAME
        # R, T = look_at_view_transform(
        #     dist, elev, (azim - 90),
        #     degrees=True,
        #     at=self.camera_conversion(at),
        #     up=self.camera_conversion(up))
        # cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        cameras = self.get_camera(
            dist, elev, (azim - 90),
            self.camera_conversion(at),
            self.camera_conversion(up),
            fov
        )

        if lmk:
            if mediapipe:
                cond_lmks = self.get_cond_lmk_mediapipe(vertices, faces, cameras)
            else:
                cond_lmks = self.get_cond_lmk_openpose(joints, cameras)
            # result = {
            #     'depths': cond_depths,
            #     'lmks': torch.from_numpy(cond_lmks)
            # }
            result = torch.from_numpy(cond_lmks)
        else:
            cond_depths = self.get_cond_depth(vertices, faces, cameras, mesh_vis)
            result = cond_depths

        return result
