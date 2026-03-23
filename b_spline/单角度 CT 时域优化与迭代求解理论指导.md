# 单角度 CT 时域优化与迭代求解理论指导

本文档旨在为单角度 CT 逆问题的算法代码实现提供理论支撑与整体架构指导。核心思路是：**在时域（探测器域）精确构建基于样条函数的物理正向模型，并采用交替迭代的优化算法框架进行求解。**

该框架将“物理约束”与“先验约束（如未来的深度学习模块）”进行了解耦，确保算法在工程实现上的灵活性与可扩展性。

## 第一阶段：数学模型的时域转化

我们首先需要摒弃直接的解析求逆或频域变换，将物理投影过程精确转化为时域的线性代数方程组。

### 1.1 连续域的物理表达

假设待重建的图像 $f$ 位于由生成元 $\phi$（如 $m$ 阶基数 B 样条 $B_m$ 组合而成的 Box-spline）张成的平移不变空间中： 

$$f(x) = \sum_{k \in E^{+}} c_{k} \phi(x - k)$$

 其中，$\{c_k\}$ 是我们需要求解的未知系数序列（长度记为 $N$）。

在单角度 $\alpha$ 下，对该连续图像进行 Radon 变换，得到连续的投影函数： 

$$\mathcal{R}_{\alpha}f(s) = \sum_{k \in E^{+}} c_{k} \mathcal{R}_{\alpha}\phi(s - \alpha \cdot k)$$

### 1.2 探测器域的离散采样

在真实的 CT 系统中，探测器在扫描域 $s$ 上具有离散的传感器排列。假设有 $M$ 个探测器采样点 $S = \{s_1, s_2, \dots, s_M\}$，我们采集到的真实带噪物理数据记为向量 $\mathbf{g} \in \mathbb{R}^{M}$。

对于第 $j$ 个采样点 $s_j$，理论投影值可以表示为： 

$$[\mathcal{R}_{\alpha}f](s_j) = \sum_{m=1}^{N} c_{k_m} \mathcal{R}_{\alpha}\phi(s_j - \alpha \cdot k_m)$$

### 1.3 构建时域系统矩阵

我们将上述离散关系写成极其标准的矩阵-向量乘法形式： 

$$\mathbf{A}\mathbf{c} = \mathbf{g}_{true} \approx \mathbf{g}$$

- $\mathbf{c} \in \mathbb{R}^{N}$：待求的系数向量。

- $\mathbf{g} \in \mathbb{R}^{M}$：实际观测到的单角度投影数据。

- $\mathbf{A} \in \mathbb{R}^{M \times N}$：**正向投影矩阵（系统矩阵）**。其矩阵元素严格由 B 样条基函数在对应方向和采样点上的解析解计算得出： 

  $$A_{j,m} = \mathcal{R}_{\alpha}\phi(s_j - \alpha \cdot k_m)$$

> **工程实现提示**：矩阵 $\mathbf{A}$ 的构建是整个算法的物理基础。由于基函数 $\phi$ 是已知的（例如 $B_2(x)B_1(y)$），$\mathcal{R}_{\alpha}\phi$ 存在显式的解析数学表达式。**矩阵** $\mathbf{A}$ **可以在算法主循环开始前，进行一次性的离线计算并保存在内存中。**

## 第二阶段：时域优化问题的构建

由于单角度问题极度的欠定性（零空间大）以及 $\mathbf{A}$ 可能存在的极高条件数（病态性），直接求解 $\mathbf{A}^{-1}\mathbf{g}$ 或最小二乘伪逆会极大地放大噪声。因此，必须引入优化方程：

$$\arg\min_{\mathbf{c}} \mathcal{J}(\mathbf{c}) = \arg\min_{\mathbf{c}} \left( \frac{1}{2} \| \mathbf{A}\mathbf{c} - \mathbf{g} \|_2^2 + \lambda R(\mathbf{c}) \right)$$

- **数据保真项** $\frac{1}{2} \| \mathbf{A}\mathbf{c} - \mathbf{g} \|_2^2$：确保求出的图像重新投影后，符合真实的探测器读数。
- **正则化项 (先验项)** $R(\mathbf{c})$：用于抑制伪影和噪声。
- **权重参数** $\lambda$：平衡物理观测与先验假设。

## 第三阶段：迭代求解算法的拆解

为了在未来能够无缝接入任意形式的先验网络，我们不使用将 $R(\mathbf{c})$ 显式求导的方法，而是采用**近端梯度下降法（Proximal Gradient Descent, PGD）或半二次分裂法（HQS）**。

这种算法框架的核心是将上述优化方程拆解为交替执行的**两个独立子步骤**，迭代 $T$ 次：

### 步骤 A：数据一致性更新（Data Fidelity Step）

仅仅针对数据保真项进行一步（或多步）梯度下降。这一步的任务是**“尊重物理测量”**。

目标是最小化 $f(\mathbf{c}) = \frac{1}{2} \| \mathbf{A}\mathbf{c} - \mathbf{g} \|_2^2$，其关于 $\mathbf{c}$ 的梯度为 $\nabla f(\mathbf{c}) = \mathbf{A}^\top(\mathbf{A}\mathbf{c} - \mathbf{g})$。 采用步长（学习率） $\eta$ 更新系数： 

$$\mathbf{z}^{(t)} = \mathbf{c}^{(t-1)} - \eta \mathbf{A}^\top (\mathbf{A}\mathbf{c}^{(t-1)} - \mathbf{g})$$

- *算子解释*：$\mathbf{A}$ 相当于正向投影（Forward Projection），$\mathbf{A}^\top$ 相当于反投影（Back Projection）。物理约束在这里被完美体现。

### 步骤 B：先验正则化更新（Proximal / Prior Step）

针对正则化项进行求解，任务是**“消除病态伪影”**。

数学上，这一步等价于计算近端算子（Proximal Operator）： 

$$\mathbf{c}^{(t)} = \text{prox}_{\lambda \eta R} (\mathbf{z}^{(t)}) = \arg\min_{\mathbf{c}} \left( \frac{1}{2} \| \mathbf{c} - \mathbf{z}^{(t)} \|_2^2 + \lambda \eta R(\mathbf{c}) \right)$$

**理论指导精髓：** 在实际算法实现中，**你完全不需要知道** $R(\mathbf{c})$ **的具体数学公式**。你可以直接将算子 $\text{prox}_{\lambda \eta R}(\cdot)$ 视为一个**黑盒映射函数** $\mathcal{D}(\cdot)$。 未来，无论你选择 CNN、U-Net、残差网络还是扩散模型，它们的作用就是在这个阶段读取含伪影的中间状态 $\mathbf{z}^{(t)}$，并输出去噪后的干净系数 $\mathbf{c}^{(t)}$： 

$$\mathbf{c}^{(t)} = \mathcal{D}(\mathbf{z}^{(t)})$$

## 第四阶段：算法实现总体流程图

以下是供代码实现参考的宏观执行流程：

**[1. 初始化阶段 (离线计算)]**

- 确定问题维度 $N$（图像网格大小）和探测器采样点 $M$。
- 根据指定的单角度 $\alpha$ 和基函数 $\phi$ 的解析表达式 $\mathcal{R}_{\alpha}\phi$，计算生成系统矩阵 $\mathbf{A} \in \mathbb{R}^{M \times N}$。
- 计算伴随矩阵（反投影矩阵） $\mathbf{A}^\top$。
- *(可选优化)* 计算步长 $\eta \le 1 / \|\mathbf{A}^\top\mathbf{A}\|_2$ 以保证梯度下降的绝对收敛。

**[2. 算法主循环阶段 (在线迭代)]**

- **输入**: 观测数据 $\mathbf{g}$，初始化系数 $\mathbf{c}^{(0)}$（例如设为全 0 向量，或直接设为 $\mathbf{A}^\top \mathbf{g}$）。
- **For** $t = 1, 2, \dots, T$ **do**:
  1. **物理误差计算**: $\mathbf{e} = \mathbf{A}\mathbf{c}^{(t-1)} - \mathbf{g}$
  2. **反投影求梯度**: $\nabla = \mathbf{A}^\top \mathbf{e}$
  3. **梯度下降更新 (步骤A)**: $\mathbf{z}^{(t)} = \mathbf{c}^{(t-1)} - \eta \nabla$
  4. **黑盒先验映射 (步骤B)**: $\mathbf{c}^{(t)} = \mathcal{D}(\mathbf{z}^{(t)})$   *(注：目前无网络时，此步可跳过或替换为传统的 Soft-Thresholding 等简单滤波)*
- **End For**

**[3. 结果输出阶段]**

- 输出最终的优化系数序列 $\mathbf{c}^{(T)}$。
- 利用连续公式 $f(x) = \sum c_k^{(T)} \phi(x-k)$ 在空间网格上进行重构，渲染并保存最终的可视化图像。

## 总结

这一理论框架为你将来的算法编写打下了坚实的基础。系统矩阵 $\mathbf{A}$ 负责**严格锁定单角度 CT 的空间物理约束**，而独立的先验映射步骤 $\mathcal{D}(\cdot)$ 则为你**预留了深度学习的接入端口**。在编写代码时，你可以先将 $\mathcal{D}(\cdot)$ 设置为一个简单的恒等映射（即不进行任何额外操作），专心验证 $\mathbf{A}$ 和梯度下降部分的代码正确性。