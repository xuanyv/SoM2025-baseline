# ? SoM2025 Challenge Baseline

本仓库提供 **SoM2025: Adapting WiFo for Wireless Physical Layer Tasks** 的官方 Baseline 实现。参赛者可在此基础上做 **模型微调、参数优化、轻量化** 等改进。

---

## 1) 赛题与数据概览

本挑战基于 **预训练 WiFo-Base 模型**，包含三个物理层任务：

● 赛题一：稀疏导频信道估计（Channel Estimation）
- 目标：利用导频位置的观测对完整信道矩阵进行恢复/内插。
- 数据来源：Quadriga
- 场景：UMa NLoS
- 训练样本：900
- 信道维度：4 × 32 × 64
- 任务相关配置：导频每 4 个子载波放置
- Baseline 指标：NMSE = 0.318

● 赛题二：LoS / NLoS 判别（Link Type Classification）
- 目标：根据信道数据判断是否为 LoS。
- 数据来源：Quadriga
- 场景：UMi LoS + UMi NLoS
- 训练样本：300
- 信道维度：24 × 8 × 128
- Baseline 指标：F1 = 0.828

● 赛题三：视觉辅助无线定位（Vision-aided Localization）
- 目标：结合视觉与信道数据实现车辆位置精准定位。
- 数据来源：SynthSoM
- 场景：Cross Road
- 训练样本：500
- 信道维度：1 × 128 × 32
- Baseline 指标：MAE = 9.83

综合 Baseline 总分（限制在 0~1）：
((1 - 0.318/1.000) + 0.828 + (1 - 9.83/20)) × (1 + 0.11)/3.6 ≈ 0.622

---

## 2) 数据集构成与格式

下载官方数据集后放入项目根目录的 `./dataset/` 下（链接见下节）。目录结构建议如下：

dataset/
├─ Task1/           # 信道估计数据（.npy / .pt），shape: [N, 4, 32, 64]
│  ├─ X_train.mat ...
│  └─ X_val.mat   ...
├─ Task2/           # LoS/NLoS 分类数据，shape: [N, 24, 8, 128]；标签 0/1
│  ├─ X_train.mat ...
│  └─ X_val.mat   ...
└─ Task3/           # 定位数据，输入 shape: [N, 1, 128, 32]；目标为位置向量
│  ├─ X_train.mat ...
│  └─ X_val.mat   ...

如使用自定义格式，请在各任务脚本中适配相应的 `Dataset` 读取逻辑。

---

## 3) 快速开始（环境与依赖）

1) 克隆仓库
git clone https://github.com/SoM2025/Baseline.git
cd Baseline

2) 创建与激活环境（任选一法）
# Conda
conda create -n som2025 python=3.9 -y
conda activate som2025
# 或 venv
python -m venv .venv && source .venv/bin/activate    # Windows 使用 .venv\Scripts\activate

3) 安装依赖
pip install -r requirements.txt
 
---

## 4) 下载官方数据集

请从以下地址下载并解压到项目根目录的 ./dataset/ 下：
【官方数据集下载】https://YOUR-DATA-HOST/som2025-dataset
（如为私有下载，请替换为实际链接；保持 Task1/Task2/Task3 子目录结构不变）

---

## 5) 训练与评测

分别运行下列脚本训练三项任务（会自动在 ./logs 下保存模型与结果）：

# 赛题一：信道估计
python main_fine_tune_T1.py

# 赛题二：LoS/NLoS 判别
python main_fine_tune_T2.py

# 赛题三：视觉辅助定位
python main_fine_tune_T3.py
 
---

## 6) 提交规范（submission）

● 提交文件一：推理结果 JSON（与 submission_demo.json 格式一致）
文件名：submission.json
格式：
{
  "Task1": Array_t1,  // 信道估计：预测矩阵，建议 shape [N, 4, 32, 64]
  "Task2": Array_t2,  // LoS/NLoS：预测标签数组（0/1），shape [N] 或 [N, 1]
  "Task3": Array_t3   // 定位：位置预测矩阵，默认 shape [N, 2] 表示 (x, y)
}

注意：
- JSON 中的数组应为可序列化的嵌套列表（list of lists）。
- 不要包含 NaN/Inf；请使用 float/整数。

● 提交文件二：平均训练参数量（必填）
- 统计方法：训练时可记录 "模型中 requires_grad 的参数总量" 作为可训练参数量（若使用 LoRA/Adapter 等，仅统计可训练部分）；不同阶段若参数规模有变化，可按训练期间的时长/epoch 加权平均。
 

- 参考统计脚本（PyTorch）：
# tools/count_params.py
import torch
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
---

## 7) 许可证与致谢

- 预训练基座：WiFo（开源地址请参考项目主页/论文引用）
- 本 Baseline 仅用于 SoM2025 挑战赛学术评测目的。
- 若在论文或报告中使用本项目，请在引用中致谢 SoM2025 组委会与 WiFo 项目。

祝各位在 SoM2025 挑战赛中取得优异成绩！
