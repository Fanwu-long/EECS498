# EECS 498/598 课程资源清单

> 原官网已失效：`web.eecs.umich.edu/~justincj/` 整个目录 404（Justin Johnson 已离开密歇根）。
> 以下为核实可用的替代资源，最后核验时间 2026-09-21。

---

## 一、作业（已完成，共 182 个 TODO）

**来源仓库**：https://github.com/nizne9/EECS498-WI22

- `main` 分支 = **未完成的骨架**（本目录内容，就是它）
- `solutions` 分支 = **答案，严禁查看**

本地路径：`C:\Users\Lenovo\WorkBuddy\2026-09-20-21-47-00\EECS498-WI22-main\`

| 作业 | 内容 | 待填空 |
|---|---|---|
| A1 | PyTorch 101、kNN 分类器 | 29 |
| A2 | SVM/Softmax 线性分类器、两层网络（手写梯度 + 数值校验）、MNIST 挑战题 | 20 |
| A3 | 全连接网络、CNN（手写优化器/BN/Dropout/卷积/池化） | 39 |
| A4 | 单阶段检测器 FCOS、两阶段 Faster R-CNN（**从此改用 autograd**） | 27 |
| A5 | RNN/LSTM、Image Captioning、**Transformer**（25 个空）、ViT | 40 |
| A6 | VAE、GAN、网络可视化、风格迁移 | 27 |

每份作业 = 一个 `.ipynb`（题目讲解 + 预期输出）+ 一个 `.py`（你要填的代码）。

**没有 Autograder**（仅对密歇根在校生开放）。自检方式：对照 notebook 里写的预期结果，例如
A1 的 kNN 约 28%、A3 的 CNN 约 71%。

---

## 二、课程视频

| 版本 | 链接 |
|---|---|
| YouTube 官方播放列表（WI2019 录像，需梯子） | https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r |
| B 站搬运 · FA2019（国内直连） | https://www.bilibili.com/video/BV1eD4y1U7uu/ |
| B 站搬运 · WI2022 | https://www.bilibili.com/video/BV1zg411a7Wi/ |

注意：录像是 **WI2019** 的，作业是 **WI2022** 的，两者不完全对应 ——
A4 的 FCOS、A5 的 Transformer 是后来新增的，录像里没有。视频只当补充。

---

## 三、课件（slides）

EECS 498 原版 slides 随官网一起没了。可用 **CS231n 的 slides** 替代（同源，Justin Johnson 主讲，内容高度重合）：

```
https://cs231n.stanford.edu/slides/2025/lecture_N.pdf      # N = 1, 2, 3, ...
```

例如：
- https://cs231n.stanford.edu/slides/2025/lecture_2.pdf  图像分类
- https://cs231n.stanford.edu/slides/2025/lecture_5.pdf  神经网络
- https://cs231n.stanford.edu/slides/2025/lecture_7.pdf  卷积网络
- https://cs231n.stanford.edu/slides/2025/lecture_13.pdf 注意力

课程安排页（含每讲标题）：https://cs231n.stanford.edu/2025/schedule.html

---

## 四、笔记

| 笔记 | 链接 | 说明 |
|---|---|---|
| ShowMeAI 中文笔记 | https://blog.showmeai.tech/eecs498/ | **中文，推荐**。视频的图文版总结，还有速查表 |
| Michael-Jetson 笔记 | https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch | 20–30 万字，非常详细 |
| YANG-SOBER 笔记 | https://github.com/YANG-SOBER/Deep-Learning-for-Computer-Vision | README 含 Lecture 1–22 列表 |
| CS自学指南 · 中文介绍 | https://csdiy.wiki/en/深度学习/EECS498-007/ | 课程评价与资源汇总 |

---

## 五、讲义 ↔ 作业对应（卡住时查哪一讲）

| 作业 | 对应 Lecture |
|---|---|
| A1 | L2 图像分类 |
| A2 | L3 线性分类器、L4 优化、L5 神经网络、L6 反向传播 |
| A3 | L7 卷积网络、L8 CNN 架构、L10–11 训练神经网络 |
| A4 | L15 目标检测、L16 检测与分割 |
| A5 | L12 RNN、L13 注意力与自注意力、L18 Vision Transformer |
| A6 | L14 可视化与理解、L19–20 生成模型 |

---

## 六、环境

作业按 **Google Colab** 设计（notebook 里有挂载 Google Drive 的 cell）。

本机状态：系统 Python（`D:\python\python.exe`）有 numpy + matplotlib，**无 torch/torchvision**，尚未配置。
