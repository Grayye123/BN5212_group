# Idea 1 · 实验资料

更新：2026-09-08。以下包含设计草案、本地表格核查和合成显存测试；真实胸片训练尚未开始。

## 研究问题

对主治医生 A，比较训练中**包含 A 关联病例**与**排除 A 关联病例**的两个图像模型；两者都在同一批 A 的测试病例上评估。模型只输入图像，医生匿名编号用于分组。

先按病人隔离训练、验证和测试集。两组训练数量匹配，尽量平衡体位；共用不含 A 病例的验证集与相同训练设置。匹配后，包含 A 的组必须确实保留 A 的训练病例。

医生关联病例不是随机分配。病情、体位和报告抽取误差都可能影响差异；本实验不能直接判断医生水平或证明医生的因果作用。

## 数据核查

范围是课程子集，计数来自本地表格与 ZIP 索引，尚未验证真实像素内容。

| 项目 | 已核查结果 |
| --- | --- |
| 规模 | 5,534 张影像 / 3,351 次检查 / 1,000 位病人 |
| 主治匿名编号 | 3,351 次检查全部关联，共 35 位 |
| 住院医匿名编号 | 1,211 次检查有记录，共 92 位 |
| 自定义标签 | 3,351 次检查全部关联；来自报告规则抽取 |
| 肺不张标签 | 阳性 832 / 明确阴性 13 / 不确定 213 / 未提及 2,293 |
| 已提取体位字段 | 1,893 / 5,534 张非空，不能按完整体位信息使用 |

医生表记录检查关联的匿名医生，不能等同于逐图独立打标者。[MIMIC-CXR 2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/)

官方 CheXpert 标签本地尚未找到；当前自定义标签尚未充分验证。官方标签也来自报告，不能称为独立影像诊断。[MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

## 流程与接口

```text
索引 + 标签 + 医生表 → 关联核查 → 获准后核查图像
→ 每检查按固定规则选一张正面片 → 按病人划分
→ 为每位纳入医生配对训练 → 同一套测试病例评估
```

| 函数 | 约定 |
| --- | --- |
| `build_manifest(index, labels, providers)` | 检查影像键唯一、标签和医生表每检查唯一；按检查多对一关联，核对病人编号；保留关联失败记录 |
| `define_target(data, policy)` | 保留原始四态及来源；缺失整行另行排除。草案：阳性 1，明确阴性/未提及 0，不确定排除 |
| `prepare_images(data, policy)` | 获准后读取完整 DICOM，核对体位、灰度转换和选片规则，记录排除原因 |
| `split_by_patient(data, policy)` | 同一病人及重复影像不跨集合；保存划分清单、种子和摘要值 |
| `make_reader_pair(train, valid, test, doctor)` | 匹配训练数量及体位，共用不含 A 的验证集与同一 A 测试集；核查独立正负例病人数 |
| `train_model(train, valid, config, seed)` | 草案用预训练 ResNet18 单输出；只用验证集选模型和阈值，不在测试集调参 |
| `compare_predictions(seen, unseen, test)` | 比较 AUROC、AUPRC、Brier；按病人配对重复抽样给出区间 |

<details>
<summary>主流程伪代码（尚未实现）</summary>

```python
data, join_audit = build_manifest(index, labels, providers)
data, exclusions = define_target(data, label_policy)
images, image_audit = prepare_images(data, image_policy)  # 等待数据处理获准
train, valid, test = split_by_patient(images, split_policy)
for doctor in eligible_doctors:  # 纳入条件预先确定
    seen, unseen, common_valid, test_A = make_reader_pair(train, valid, test, doctor)
    for seed in training_seeds:
        # 两个模型依次训练、预测并释放，使用相同设置。
        pred_seen = fit_predict_release(seen, common_valid, test_A, config, seed)
        pred_unseen = fit_predict_release(unseen, common_valid, test_A, config, seed)
        save_summary(compare_predictions(pred_seen, pred_unseen, test_A))
```

此处为接口草图，包含说明性函数，不能直接运行。`fit_predict_release` 表示调用 `train_model`、预测并释放模型。

</details>

0 表示未明确报出肺不张，不代表医学上确定无病。图像路径、真实 ID、逐病例划分和预测保留在获准的数据位置；仓库只接收可共享的汇总。

每位医生报告独立病人数、检查数、阳性比例、训练数量、体位构成、两组指标及差值。若定义差值为 `seen - unseen`，AUROC/AUPRC 正值、Brier 负值表示 seen 更好。病人抽样须连同其全部检查一起抽；跨种子变化单列，不能用来虚增病人数。单类别样本的 AUROC 不填 0，样本不足时报告不可稳定估计。

## 尚未冻结的设置

- 官方或自定义标签的最终选择，以及四态处理规则。
- 多张正面片的固定选片办法、体位缺失和图像质检规则。
- 病人划分比例、种子与清单；医生最低独立阳性/阴性病人数。
- 训练配对与体位平衡方法，包含 A 组的最低 A 病例数。
- 分辨率、预训练来源、训练参数和随机种子。
- 主要指标、跨医生汇总、区间计算与多重比较处理。

这些设置应在正式评估前记录理由和版本，不按测试表现挑医生、参数或随机种子。

## 存储与显存

| 保存方式 | 估算 |
| --- | --- |
| 课程 ZIP | 约 43.34 GB；压缩成员合计 43,334,031,377 字节，另有文件头和目录 |
| 全部 DICOM 解压 | 75,439,424,259 字节，约 75.44 GB |
| ZIP + 完整解压副本 | 约 118.78 GB，未计临时文件和训练副本 |
| ZIP 逐成员转换，只保留小尺寸副本 | 建议预留 55–65 GB，未实施 |
| ZIP、完整 DICOM 和训练副本均保留 | 建议预留 135–150 GB，未实施 |

GB = 10^9 字节；GiB = 2^30 字节。5,534 张 512×512、单通道 8-bit 未压缩像素约 1.45 GB；每检查一张最多 3,351 张，约 0.88 GB。最终正面片数和压缩文件大小仍待核查。

显存测试使用 RTX 3060 Laptop（约 6 GB）、随机初始化 ResNet18、随机三通道输入、AdamW、AMP float16、全部参数更新，每设置三个更新步。没有下载权重或使用真实胸片。

| 输入 / batch | 张量峰值 | 框架保留峰值 | 正式进程预算建议 |
| --- | --- | --- | --- |
| 224×224 / 16 | 0.46 GiB | 0.54 GiB | 约 2 GiB |
| 512×512 / 16 | 1.25 GiB | 1.58 GiB | 约 3–4 GiB |

环境：PyTorch 2.12.0+cu126，torchvision 0.27.0+cu126。峰值不包含整张显卡的其他占用；真实处理与增强可能增加开销。这是资源测试，不是模型性能结果。

## 本地依据

核查依据保留在 `out/cxr_subset_ids.csv`、`out/cxr_metadata.csv`、`out/dataset_zip_index.json`、`raw/cxr-provider-list.csv.gz`、`out/report_labels.csv` 和 `tmp/pdfs/resource_profile.json`。

以上为维护者本地路径，不是公开下载入口；本仓库未包含这些文件。旧的全量报告文本计划也不作为当前图像实验的完成证据。

合作者讨论稿已收录为 [PDF](BN5212_Idea1_Team_Brief.pdf) 和 [Markdown](BN5212_Idea1_Team_Brief.md)，保留 2026-09-08 的讨论版本；后续实验设置继续在本页更新。
