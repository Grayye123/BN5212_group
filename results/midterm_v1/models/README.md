# 模型权重

真实训练得到两个 ImageNet ResNet18 微调检查点，最后一层为一个输出，输出经 sigmoid 得到预测概率。最佳模型仅按共享验证集 BCE 选择：seen 第 3 轮、unseen 第 4 轮。

| 本机文件 | 字节数 | 公开仓库状态 |
| --- | ---: | --- |
| `seen.pt` | 44,780,491 | 未上传 |
| `unseen.pt` | 44,781,899 | 未上传 |

两文件已复制到维护者本机的本目录，原文件仍在 `private/midterm_v1_quick/`，均被 Git 忽略。校验值见 [manifest.json](manifest.json)。本目录在 GitHub 上只有说明和校验值，克隆不会得到权重。

未上传的原因不是文件大小。PhysioNet 的 [MIMIC 衍生数据与模型共享指导](https://physionet.org/content/mimiciv/3.1/) 要求将衍生模型视为含敏感信息，并通过 PhysioNet 在与源数据相同协议下共享。本项目据此不将 MIMIC-CXR 微调权重公开分发。此项不妨碍使用汇总图表制作 PPT。

获得合规访问的权重后，可用如下方式加载。需使用与训练相同的 DICOM 预处理、224×224 灰度三通道及 ImageNet 归一化；不能直接把任意原图送入模型。完整处理见 [pilot_train.py](../../../code/pilot_train.py)。

```python
import torch
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
state = torch.load("seen.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

本版仅作课程探索性实验，不作为临床诊断工具。
