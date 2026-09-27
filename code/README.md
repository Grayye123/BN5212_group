# 代码

2026-09-27：维护者已明确授权期中第一版实现与训练。目标是一位目标医生、一个种子的配对实验，不等待期末级别的完整人工验证。原始 ZIP 不改写；只生成本机训练缩略副本。当前已完成标签、划分、合成检查及期中小规模真实配对训练。

## 第一版入口

制作 PPT 请直接使用 [公开结果文件夹](../results/midterm_v1/README.md)，不必安装训练环境。以下命令用于具备授权数据与私有标签输入的重跑；克隆仓库本身不包含这些输入。所需 Python 包见 [pilot_requirements.txt](pilot_requirements.txt)，原运行环境见 [environment.json](../results/midterm_v1/environment.json)。请在自己的 Python 环境安装依赖；GPU 版 PyTorch 按自己的 CUDA 环境配置。

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONIOENCODING = 'utf-8'
python code/pilot_prepare.py --review '/path/to/authorized/BN5212_标签裁决.xlsx' --dest private/midterm_v1_quick --max-train-per-group 256 --max-valid 128
python code/test_pilot.py --smoke
python code/pilot_train.py --work private/midterm_v1_quick --cache private/midterm_v1/pixels224 --weights-cache private/midterm_v1/torchhub --epochs 5
python code/pilot_report.py --work private/midterm_v1_quick --dest private/midterm_v1_quick/PRESENTATION_NOTES.md
```

上述小规模命令将逐病例清单、模型和预测写入 Git 忽略的 `private/midterm_v1_quick/`；图像与 ImageNet 权重缓存复用 `private/midterm_v1/`。现有 `out/audit/` 只读。`--prepare-only` 仅准备图像缓存；缓存后训练可复用，不需要完整解压。`--random-init` 是明确可选的另一实验设定，默认不使用；本次采用已下载并核验官方哈希的 ImageNet ResNet18 权重。

## 输入与规则

- 标签输入：CheXbert 全量表、272 条人工判定及已有报告上下文。保留各来源和规则调整原因，不覆盖原文件。
- 用户规则：likely 不确定，most likely 阳性；volume loss 排除非常轻微表述。代码将 `very mild/minimal/trace/slight/very small` 作为非常轻微的暂定词表。这是首版操作定义，不是医学同义断言。
- 只对包含规则词的目标句调整；多句冲突、否定或鉴别等难以可靠解析的情况保守排除。未随机生成标签，未声称逐条完成专家裁决。机械未提及部分暂保留，未补人工复核作为明确限制。
- 每检查选一张 PA 优先、其次 AP 的图像，同体位按图像键排序；病人 60/20/20 分层划分，种子 20260927。
- 按可用独立病人数最多选择目标医生，不按模型表现选择。训练两组按标签与体位精确匹配，共用排除目标医生病例的验证集和同一目标医生测试集。
- 图像：DICOM 解码、rescale、MONOCHROME1 反转、每图 0.5–99.5 百分位截断，保持比例缩放并补边至 224×224。无水平翻转或其他增强。ImageNet 归一化。
- 全参数 ResNet18，AdamW，学习率 1e-4，weight decay 1e-4，batch 32，AMP，5 轮；只按共享验证集 BCE 选检查点。
- 指标：主要为 AUROC，另报 AUPRC、Brier。差值 seen−unseen，按病人配对 bootstrap 1,000 次。单种子、单医生结果仅作探索性展示。

## 已执行检查

- 初始完整规模清单核查（未按此规模训练）：3,351 个唯一标签、272 个唯一人工判定、病人划分无交叉；两组训练各 1,409 条，阳性和体位数量相同；seen 保留目标医生 249 条，unseen 与验证集排除该医生病例。实际训练采用下述每组 256 条的期中预算。
- 3 组标签规则边界测试通过；合成 DICOM 解码和尺寸检查通过；随机图像的 GPU 反向传播、模型保存恢复、预测和配对 bootstrap 检查通过。
- 合成检查输出在 `private/midterm_v1/smoke/`，不属于真实实验结果。
- 真实运行已完成：718 张图像解码成功；两组各 5 轮完成；109 行预测与测试清单对应，三个指标经独立计算核验。读取失败会停止，不会静默替换病例。

## 输出

`summary.json` 是标签与样本汇总；`run_config.json` 固定训练设置和划分摘要；真实实验成功后才会生成 `results.json`、`results.png`、两份模型权重和逐病例 `predictions.csv`。共享前仅提取汇总与无病例信息的图表。


## 期中小规模运行

首次网络读取耗时较长，在任何训练/测试表现产生前收敛预算：两训练组各 256 次、各 59 次阳性、PA 110 / AP 146，seen 有目标医生 46 次。验证 128 次；测试仍为原 109 次，文件逐字节一致。实际结果写入 `private/midterm_v1_quick/`。共需 718 张不同图像，已生成的原缓存直接复用。图像缓存顺序按 ZIP 物理位置排序，单路读取，避免 SSHFS 并发性能波动。


## 本次真实结果

运行目录 `private/midterm_v1_quick/`：AUROC 0.672894 / 0.676923，差值 -0.004029，病人配对 95% 区间 [-0.080486, 0.060237]。最佳验证 BCE 分别来自第 3 / 4 轮。结果图、原始指标、预测及权重已生成，区间基于 1,000 次配对病人抽样。完整汇总见 [PROJECT.md](../docs/PROJECT.md)，汇报稿见 [MIDTERM_V1.md](../slides/MIDTERM_V1.md)。
