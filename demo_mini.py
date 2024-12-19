import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import json

# 自定义的 JSON 编码器，用于处理 numpy 数组
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# 设置 Hugging Face Hub 的 URL 为国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16
print(f"Running on device: {device}")
pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
img_path = "/home/wangrx/Pictures/vlcsnap-2024-12-16-19h39m57s655.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = pipe.predict(image)

# 保存输出结果到文件
output_file_path = "./hand_pose_output.json"
with open(output_file_path, 'w') as f:
    json.dump(outputs, f, indent=4, cls=NumpyEncoder)

print(f"输出结果已保存到: {output_file_path}")

# 获取3D顶点
vertices = outputs[0]['wilor_preds']['pred_vertices'][0]

# 绘制3D网格
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 保存3D网格图像
mesh_img_path = "/home/wangrx/Pictures/hand_mesh.png"
plt.savefig(mesh_img_path)
plt.show()

print(f"3D网格图像已保存到: {mesh_img_path}")

# 保存原始图像
original_img_path = "/home/wangrx/Pictures/original_image.png"
cv2.imwrite(original_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print(f"原始图像已保存到: {original_img_path}")

