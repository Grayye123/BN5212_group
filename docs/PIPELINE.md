# Idea 1：实验设计与函数草图

状态：设计草案，未实现真实影像训练。日期：2026-09-08。

## 要比较的两种情况

对某位主治医生 A，模型一训练时包含 A 关联的其他病人病例，模型二排除 A 的训练病例。两个模型在**同一批 A 的测试病人病例**上比较。两组训练规模匹配，使用相同的不含 A 病例的验证集、模型和训练设置。

病例由哪位医生关联并非随机分配。体位、病情和报告写法都可能有差异；这里测量适应能力，不能直接分离出医生的因果作用。

```text
影像索引 + 检查级标签 + 医生表
    → 关联与缺失核查
    → 获准后核查真实图像、固定每检查一张正面片规则
    → 按病人划分训练 / 验证 / 测试
    → 预先确定可评估的医生
    → 含 A / 不含 A 的配对训练（顺序运行）
    → 同一套 A 测试病例上的预测
    → 按病人配对的指标差值与置信区间
```

## 数据接口

以下仅说明字段；包含真实 ID 的表不提交 GitHub。

| 内容 | 所需字段与约束 |
| --- | --- |
| 影像索引 | `subject_id`, `study_id`, `dicom_id`；影像编号唯一 |
| 报告标签 | 检查编号、标签原值、是否存在标签整行、抽取方法和版本；每检查唯一行 |
| 医生表 | 检查编号、主治匿名编号、住院医匿名编号；检查关联，不是逐图独立标注 |
| 图像处理 | 图像路径、真实体位、转换配置与质检状态；不以不完整头信息强行决定体位 |
| 划分 | 病人编号、集合、种子和清单摘要值；同一病人只能属于一个集合 |

## 伪代码

```python
def build_manifest(image_index, labels, providers):
    # 先验证各表键是否唯一，再按 study_id 进行多对一关联。
    # 检查 subject_id 是否一致；保存关联失败记录和原始标签来源。
    return manifest, join_audit

def define_target(manifest, label_policy):
    # 标签整行缺失：排除并单独记录。
    # 草案：阳性 -> 1；明确阴性 / 未提及 -> 0；不确定先排除。
    # 保留四态原值。目标是报告是否明确报出肺不张，不是独立诊断。
    return eligible, exclusion_audit

def prepare_images(manifest, image_policy):
    # 仅在数据处理获准后运行。读取完整 DICOM 并检查灰度转换。
    # 按预先确定的规则每检查选一张正面片，保留排除原因。
    return image_dataset, image_audit

def split_by_patient(data, split_policy):
    # 先按病人分组，再分 train / valid / test。
    # 检查病人、重复影像均不跨集合；保存清单和摘要值。
    return train, valid, test

def make_reader_pair(train, valid, test, doctor, inclusion_policy):
    test_A = test[doctor_is_A]
    valid_common = valid[doctor_is_not_A]
    unseen = train[doctor_is_not_A]
    seen = train  # 匹配后仍必须含有 A 的训练病例。
    seen, unseen = match_training_size_and_view(seen, unseen)
    # 核查独立病人数及正负例病人数；不足则报告不可估计。
    return seen, unseen, valid_common, test_A

def train_model(train, valid, config, seed):
    # 拟采用 ImageNet 预训练 ResNet18，单输出；仅输入图像。
    # 未获准前不下载权重。训练和验证配置须先冻结。
    # 不在测试集选择模型、阈值或超参数。
    return model

def compare_on_same_patients(pred_seen, pred_unseen, test_A):
    # AUROC / AUPRC / Brier；保留两组原始指标和差值。
    # 配对抽样时按病人抽，连同该病人的全部检查一起重采样。
    # 单类别样本无法算 AUROC 时标记无效，不填 0 或虚构数值。
    return metrics, patient_bootstrap_intervals

def run_idea1(config):
    # 串联以上步骤；医生与种子清单在正式结果前固定。
    # 每位医生每个种子：训练 seen → 预测并释放 → 训练 unseen → 比较。
    # 保存配置、划分摘要、排除统计、日志、权重及概率预测。
    # GitHub 仅接收经检查的汇总结果，行级结果在授权位置保留。
    return aggregate_results
```

这段代码用于约定职责，包含说明性符号，不能直接运行。正式函数实现应另建任务，并用合成数据先检查数据关联、缺失处理和病人隔离。

## 结果应报告什么

每位纳入医生的独立病人数、检查数、阳性比例，两个模型的 AUROC、AUPRC、Brier 及配对差值；记录体位构成和训练数量。预先声明差值方向，例如 `seen - unseen`：AUROC / AUPRC 正值表示 seen 更好，Brier 负值表示 seen 更好。跨种子的变化单独报告，不把不同种子当作独立病人扩大样本量。

先确定主要指标、医生纳入阈值和多医生比较的汇总方式，再开展正式评估。测试例数不足时诚实报告不能稳定估计。
