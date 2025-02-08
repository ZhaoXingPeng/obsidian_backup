# PINN (Physics-Informed Neural Networks)
## 基础概念
### PINN是什么？
PINN（Physics-Informed Neural Networks，物理信息神经网络）是一种将物理规律作为约束条件嵌入到神经网络训练过程中的深度学习模型
#### 主要特点：
1. **物理约束**
   - 在神经网络的损失函数中加入物理方程作为约束条件
   - 确保模型预测符合基本物理规律
2. **混合学习方式**
   - 结合数据驱动（传统深度学习）
   - 结合物理规律（方程约束）
### 典型架构示例
```python
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    def physics_loss(self, x, t):
        u = self.forward(x, t)
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        physics_loss = u_t - 0.01 * u_x
        return physics_loss.mean()
```

举例说明：对是否携带武器的行人的微多普勒谱图进行增强（GAN生成+物理约束）
Focante E. Physics-informed Data Augmentation for Human Radar Signatures[J]. 2023.
> *Physics-informed machine learning aims to incorporate physical prior knowledge*
> *and governing equations of the target domain into the machine learning pipeline to*
> *improve the performance of NNs in fields with limited available data but with well-*
> *defined physical models. In this thesis, physics-informed DA in the radar domain is*
> *addressed to improve the task of classifying armed and unarmed walking individuals*
> *through micro-Doppler spectrograms.*
![[Pasted image 20250208170523.png]]
![[Pasted image 20250208172711.png]]
![[Pasted image 20250208172729.png]]
[27] “Physics-informed machine learning,” Nature Reviews Physics 2021 3:6, vol. 3,
pp. 422–440, 5 2021. [Online]. Available: https://www.nature.com/articles/
s42254-021-00314-5
# 机器学习中嵌入物理知识的三种偏置方法

## 1. 观察偏置 (Observational Biases)

### 主要特点：
- 通过输入训练数据来引入物理知识
- 结合模拟数据和实际数据
- 确保数据的物理可行性
### 实现方式：
```python
# 示例：数据增强保持物理约束
def physics_aware_augmentation(data):
    augmented_data = []
    for sample in data:
        # 保持物理约束的数据增强
        augmented = apply_physics_constraints(sample)
        augmented_data.append(augmented)
    return augmented_data
```
## 2. 归纳偏置 (Inductive Biases)

### 主要特点：
- 通过模型架构设计引入物理知识
- 强制执行特定的物理特征提取
- 适合处理对称性等几何属性
### 示例实现：
```python
class PhysicsAwareNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 设计特殊的网络结构以保持物理特性
        self.symmetric_layer = SymmetricLayer()
        self.conservation_layer = ConservationLayer()
    
    def forward(self, x):
        # 强制执行物理约束的前向传播
        x = self.symmetric_layer(x)
        x = self.conservation_layer(x)
        return x
```
## 3. 学习偏置 (Learning Biases)

### 主要特点：
- 通过损失函数设计引入物理知识
- 添加基于物理定律的惩罚项
- 引导模型向物理合规的解收敛
### 示例实现：
```python
def physics_aware_loss(predictions, targets, model):
    # 常规损失
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # 物理约束损失
    physics_violation = calculate_physics_violation(predictions)
    
    # 组合损失
    total_loss = mse_loss + lambda_physics * physics_violation
    return total_loss
```
#### 三种方法的对比
| 方法   | 优势                       | 局限性                       | 适用场景       |
| ---- | ------------------------ | ------------------------- | ---------- |
| 观测偏置 | - 直接作用于数据层面<br>- 易于理解和实现 | - 可能计算开销大                 | 有可靠仿真模型的领域 |
| 归纳偏置 | - 保证强制执行<br>- 较好的鲁棒性     | - 难以实现复杂物理规律<br>- 架构设计难度大 | 简单几何性质和对称性 |
| 学习偏置 | - 实现灵活<br>- 易于集成         | - 约束相对松散<br>- 可能需要精细调优    | 复杂物理系统建模   |

----
# Physics-Aware（物理感知）

## 基本定义
Physics-Aware是一个更广泛的概念，指的是在机器学习系统中以任何形式考虑物理知识的方法。它是一个总括性术语，包含了所有让机器学习模型"意识到"物理规律的方法。
## 与PINN的关系
1. **范围区别**：
   - Physics-Aware是更宽泛的概念
   - PINN是Physics-Aware方法的一个具体实现
2. **实现方式**：
   - Physics-Aware可以有多种实现形式
   - PINN专注于通过神经网络损失函数嵌入物理方程
## Physics-Aware的实现方式

```python
# 示例1：物理感知的数据预处理
class PhysicsAwarePreprocessing:
    def __init__(self):
        self.physical_constraints = {}
    
    def apply_constraints(self, data):
        # 应用物理约束进行数据清洗
        return physically_valid_data

# 示例2：物理感知的模型架构
class PhysicsAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 物理感知层
        self.physical_layers = nn.ModuleList([
            ConservationLayer(),
            SymmetryLayer(),
            PhysicalConstraintLayer()
        ])
    
    def forward(self, x):
        # 在推理过程中考虑物理规律
        for layer in self.physical_layers:
            x = layer(x)
        return x
```
## Physics-Aware vs PINN 对比表

| 特性 | Physics-Aware | PINN |
|------|--------------|------|
| 范围 | 更广泛，包含所有物理感知方法 | 专注于通过神经网络求解物理方程 |
| 实现方式 | 多样化，可在任何阶段引入 | 主要通过损失函数嵌入物理方程 |
| 灵活性 | 非常灵活，可根据需求选择方式 | 相对固定的实现框架 |
| 复杂度 | 可以很简单也可以很复杂 | 通常较为复杂 |
| 精确性 | 取决于具体实现 | 通常较高 |
![[Pasted image 20250208214432.png]]
# 三种建模方法的对比分析

## 1. 物理基础模型 (Physics-Based Models)

### 特点：
- 无需数据 (No Data)
- 高度依赖物理知识 (High Knowledge)
- 包含的要素：
  - 现象学 (Phenomenology)
  - 传感器特性 (Sensor Properties)
  - 目标模型 (Target Model)
  - 杂波模型 (Clutter Model)

```python
# 物理基础模型示例
class PhysicsBasedModel:
    def __init__(self):
        self.phenomenology = PhenomenologyModel()
        self.sensor = SensorModel()
        self.target = TargetModel()
        self.clutter = ClutterModel()
    
    def simulate(self, conditions):
        # 基于物理定律的模拟
        return physical_simulation_result
```

## 2. 数据驱动深度学习 (Data-Driven Deep Learning)

### 特点：
- 需要大量数据 (Lots of Data)
- 不依赖物理知识 (No Physics)
- 关注：
  - 环境动态变化 (Dynamic Changes)
  - 目标特性 (Target Properties)
  - 传感器伪影 (Sensor Artifacts)

```python
# 数据驱动模型示例
class DataDrivenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
```

## 3. 物理感知机器学习 (Physics-Aware ML)

### 特点：
- 位于两者之间的折中方案
- 需要少量数据 (Little Data)
- 包含可处理的物理知识 (Tractable Physics)

```python
# 物理感知模型示例
class PhysicsAwareML(nn.Module):
    def __init__(self):
        super().__init__()
        self.ml_layers = nn.Sequential(...)
        self.physics_constraints = PhysicsConstraints()
    
    def forward(self, x):
        # 结合物理约束和机器学习
        ml_output = self.ml_layers(x)
        return self.physics_constraints(ml_output)
```
## 三种方法的比较

| 特性    | 物理基础模型 | 物理感知ML | 数据驱动深度学习 |
| ----- | ------ | ------ | -------- |
| 数据需求  | 无需数据   | 少量数据   | 大量数据     |
| 物理知识  | 高度依赖   | 部分依赖   | 不依赖      |
| 计算复杂度 | 较高     | 中等     | 较低       |
| 可解释性  | 强      | 中等     | 弱        |
| 适应性   | 较差     | 良好     | 很好       |

以

> Rahman M M, Gurbuz S Z, Amin M G. Physics-aware generative adversarial networks for radar-based human activity recognition[J]. IEEE Transactions on Aerospace and Electronic Systems, 2022, 59(3): 2994-3008.

为例
# physics-aware GANs (PhGAN)的物理感知方法实现

## 1. 架构创新：多分支GAN (MBGAN)

### 主要特点：
- 在GAN架构中集成了领域知识
- 使用多分支结构处理微多普勒特征

```python
class MBGAN(nn.Module):
    def __init__(self):
        super().__init__()
        # 主分支处理完整微多普勒信号
        self.main_branch = MainBranchNet()
        # 辅助分支处理包络线特征
        self.upper_envelope_branch = EnvelopeBranchNet()
        self.lower_envelope_branch = EnvelopeBranchNet()
    
    def forward(self, x):
        main_features = self.main_branch(x)
        upper_env = self.upper_envelope_branch(x)
        lower_env = self.lower_envelope_branch(x)
        return self.combine_features(main_features, upper_env, lower_env)
```

## 2. 物理感知损失函数

### 包含的物理度量：
1. 动态时间规整距离
2. 离散Frechet距离
3. Pearson相关系数

```python
class PhysicsAwareLoss:
    def __init__(self):
        self.dtw_weight = 0.3
        self.frechet_weight = 0.3
        self.correlation_weight = 0.4
    
    def calculate_loss(self, real, synthetic):
        # 物理约束损失
        dtw_loss = dynamic_time_warping(real, synthetic)
        frechet_loss = frechet_distance(real, synthetic)
        correlation_loss = pearson_correlation(real, synthetic)
        
        # 组合损失
        physics_loss = (self.dtw_weight * dtw_loss + 
                       self.frechet_weight * frechet_loss +
                       self.correlation_weight * correlation_loss)
        return physics_loss
```

## 3. 物理特性验证

### 评估指标：
1. **运动学一致性**：
   - 步态不对称性分析
   - 步幅持续时间

2. **信号质量度量**：
   - 均方误差(MSE)
   - 结构相似性指数(SSIM)

```python
class PhysicsValidator:
    def validate_kinematics(self, synthetic_data):
        # 验证运动学特性
        gait_asymmetry = calculate_gait_asymmetry(synthetic_data)
        stride_duration = analyze_stride_duration(synthetic_data)
        
        # 验证信号质量
        mse = calculate_mse(real_data, synthetic_data)
        ssim = calculate_ssim(real_data, synthetic_data)
        
        return {
            'gait_asymmetry': gait_asymmetry,
            'stride_duration': stride_duration,
            'mse': mse,
            'ssim': ssim
        }
```
## 5. 应用效果

1. **数据质量提升**：
   - 生成的样本更符合物理规律
   - 运动学特性更准确

2. **分类性能提升**：
   - 提高了人类活动识别准确率
   - 更好的特征泛化能力

----
