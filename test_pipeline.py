# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: test_wilor_pipeline.py

"""
you need to install trimesh and pyrender if you want to render mesh
pip install trimesh
pip install pyrender
"""

import os
import pdb
import time

import trimesh
import pyrender
import numpy as np
import torch


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class Renderer:

    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(),
                                   vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        if is_right:
            mesh_base_color = mesh_base_color[::-1]
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


def test_wilor_image_pipeline():
    import cv2
    import torch
    import numpy as np
    import os
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    img_path = "/home/wangrx/Pictures/屏幕截图-Video_20241217152150624.avi-3.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _ in range(20):  #打印每次预测的时间
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces) #渲染器

    render_image = image.copy() #复制图像
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0 #转换图像格式
    pred_keypoints_2d_all = []
    for i, out in enumerate(outputs):
        verts = out["wilor_preds"]['pred_vertices'][0] #预测的3D顶点，有778个？反正很多
        is_right = out['is_right'] #是否是右手
        cam_t = out["wilor_preds"]['pred_cam_t_full'][0] #相机位置。相机位置是估算出来的，用于渲染
        scaled_focal_length = out["wilor_preds"]['scaled_focal_length'] #缩放焦距，没懂
        pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"] #预测的2D关键点，有21个，可以用于后续的延长线
        pred_keypoints_2d_all.append(pred_keypoints_2d)
        misc_args = dict(  #定义渲染参数，包括网格颜色、背景颜色和焦距
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
        tmesh.export(os.path.join(save_dir, f'{os.path.basename(img_path)}_{int(time.time())}_hand{i:02d}.obj')) #保存网格到文件
        cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                        is_right=is_right,
                                        **misc_args) #渲染3D网格，生成带有透明通道的图像。

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:] #将渲染结果叠加到原始图像上。

    render_image = (255 * render_image).astype(np.uint8) #将渲染图像转换为8位无符号整数格式。
    for pred_keypoints_2d in pred_keypoints_2d_all:  #遍历所有2D关键点，并在渲染图像上绘制红色圆点
        for j in range(pred_keypoints_2d[0].shape[0]):
            color = (0, 0, 255)
            radius = 3
            x, y = pred_keypoints_2d[0][j]
            cv2.circle(render_image, (int(x), int(y)), radius, color, -1)
    output_image_path = os.path.join(save_dir, f'{os.path.basename(img_path)}_{int(time.time())}.png')
    cv2.imwrite(output_image_path, render_image)  # 保存渲染结果
    print(output_image_path)

def test_wilor_image_pipeline_extension():
    import cv2
    import torch
    import numpy as np
    import os
    import time
    import open3d as o3d
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    img_path = "/home/wangrx/Projects/kingfisher/kingfisher_slim/res/L1_20241219_160716_rect_left.png"   # 图片路径
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _ in range(20):  # 打印每次预测的时间
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)  # 渲染器

    render_image = image.copy()  # 复制图像
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0  # 转换图像格式
    pred_keypoints_2d_all = []
    for i, out in enumerate(outputs):
        verts = out["wilor_preds"]['pred_vertices'][0]  # 预测的3D顶点
        is_right = out['is_right']  # 是否是右手
        cam_t = out["wilor_preds"]['pred_cam_t_full'][0]  # 相机位置
        scaled_focal_length = out["wilor_preds"]['scaled_focal_length']  # 缩放焦距
        pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]  # 预测的2D关键点
        pred_keypoints_2d_all.append(pred_keypoints_2d)
        misc_args = dict(  # 定义渲染参数
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )

        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
        tmesh.export(os.path.join(save_dir, f'{os.path.basename(img_path)}_{int(time.time())}_hand{i:02d}.obj'))  # 保存网格到文件
        cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                        is_right=is_right,
                                        **misc_args)  # 渲染3D网格

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]  # 将渲染结果叠加到原始图像上

    render_image = (255 * render_image).astype(np.uint8)  # 将渲染图像转换为8位无符号整数格式

    # 读取深度图
    depth_map = np.load('/home/wangrx/Projects/kingfisher/kingfisher_slim/res/L1_20241219_160716_depth.npy')   # 深度图路径

    # 相机内参
    cam1_k = np.array([
        [1269.379133098410, 0.000000000000, 621.965297248433],
        [0.000000000000, 1269.550731910414, 516.472341387013],
        [0.000000000000, 0.000000000000, 1.000000000000]
    ])

    # 计算3D点
    points_3d = []
    points_2d = []
    index_finger_indices = [5, 6, 7, 8]  # 假设食指的关键点索引为 5 到 8
    for pred_keypoints_2d in pred_keypoints_2d_all:
        for idx in index_finger_indices:
            x, y = pred_keypoints_2d[0][idx]
            z = depth_map[int(y), int(x)]
            point_2d = np.array([x, y, 1])
            point_xyz = np.array([x, y, z])
            points_2d.append(point_xyz)
            point_3d = np.linalg.inv(cam1_k) @ (point_2d * z) 
            points_3d.append(point_3d)

            # 只绘制这四个关键点的圆点
            color = (0, 0, 255)
            radius = 3
            cv2.circle(render_image, (int(x), int(y)), radius, color, -1)

    # 打印出这四个点的坐标
    for i, point in enumerate(points_2d):
        print(f"Point {i + 1}: {point}")

    # 使用食指最顶端的两个点做延长线
    p1 = points_2d[-2] 
    p2 = points_2d[-1]
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    print(f"p1: {p1}, p2: {p2}, direction: {direction}")

    # 找到与深度图几何相交的点
    intersection_point = None
    for t in np.linspace(0, 500, 1000):
        extended_point = p2 + direction * t
        x_ext, y_ext = int(extended_point[0]), int(extended_point[1])
        if 0 <= x_ext < depth_map.shape[1] and 0 <= y_ext < depth_map.shape[0]:
            z_ext = depth_map[y_ext, x_ext]
            #print(f"Extended Point: {extended_point}, Depth: {z_ext}")
            if t>100 and np.abs(z_ext - extended_point[2]) < 0.01:  # 判断是否相交
                intersection_point = extended_point
                break

    if intersection_point is not None:
        print(f"Intersection Point: {intersection_point}")
        # 在图像上标出相交点
        cv2.circle(render_image, (int(intersection_point[0]), int(intersection_point[1])), radius, (0, 255, 0), -1)

    # 保存点云
    points_3d = np.array(points_3d)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    #o3d.io.write_point_cloud('/home/wangrx/Projects/WiLoR/pointcloud/L0.ply', point_cloud)

    # 显示手指4点的点云
    o3d.visualization.draw_geometries([point_cloud]) #

    output_image_path = os.path.join(save_dir, f'{os.path.basename(img_path)}_{int(time.time())}.png')
    cv2.imwrite(output_image_path, render_image)  # 保存渲染结果
    print(output_image_path)



def test_wilor_video_pipeline():
    import cv2
    import torch
    import numpy as np
    import os
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    video_path = "/opt/MVS/bin/Temp/Data/MV-CU013-A0GC+DA3245534/Video_20241217152150624.avi"
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    output_path = os.path.join(save_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
        render_image = image.copy()
        render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

        for i, out in enumerate(outputs):
            verts = out["wilor_preds"]['pred_vertices'][0]
            is_right = out['is_right']
            cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
            scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
            cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                            is_right=is_right,
                                            **misc_args)

            # Overlay image
            render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        render_image = (255 * render_image).astype(np.uint8)

        # Write the frame to the output video
        vout.write(render_image)

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Release everything
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved to {output_path}")


if __name__ == '__main__':
    # test_wilor_image_pipeline()
    # test_wilor_video_pipeline()
    test_wilor_image_pipeline_extension()