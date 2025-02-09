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
![[Pasted image 20250208221657.png]]
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
# physics-aware transformer (PAT)
以

> Huang Q, Hu M, Brady D J. Array camera image fusion using physics-aware transformers[J]. arxiv preprint arxiv:2207.02250, 2022.

为例
# 多相机阵列数据融合任务

## 主要目标
- 融合来自不同相机的图像数据
- 生成一个从选定视角(α视角)的融合结果
# PAT的物理感知实现方法
## 1. 物理感知的注意力引擎

### 核心实现：
```python
class PhysicsAwareAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_conv = nn.Conv2d(D, D, 1)
        self.key_conv = nn.Conv2d(D, D, 1)
        self.value_conv = nn.Conv2d(D, D, 1)
        
    def forward(self, alpha_feature, beta_features):
        # 生成查询特征
        query = self.query_conv(alpha_feature)  # [αH×αW×D]
        
        # 基于物理接收域的特征对齐
        for beta_feature in beta_features:
            key = self.key_conv(beta_feature)   # [βH×βW×D]
            value = self.value_conv(beta_feature)
            
            # C3操作：收集、关联、组合
            collected = self.collect(query, key)
            correlated = self.correlate(collected)
            combined = self.combine(correlated, value)
            
        return combined
```
## 2. 物理约束的集成

### 主要物理约束：
1. 相机内参和外参
2. 对极几何
3. 景深
4. 视差

```python
class PhysicalConstraints:
    def __init__(self, camera_params):
        self.intrinsics = camera_params['intrinsics']
        self.extrinsics = camera_params['extrinsics']
        
    def get_receptive_field(self, alpha_point):
        # 基于对极几何计算接收域
        epipolar_line = self.compute_epipolar_line(alpha_point)
        max_disparity = self.estimate_max_disparity()
        return self.truncate_field(epipolar_line, max_disparity)
```

## 3. 物理感知的数据合成

### 实现方式：
```python
class PhysicsAwareDataSynthesis:
    def __init__(self, blender_config):
        self.camera_array = []
        self.scene_params = {}
        
    def setup_cameras(self):
        # 设置具有不同物理特性的相机
        for cam_config in self.camera_configs:
            camera = Camera(
                focal_length=cam_config.focal_length,
                sensor_size=cam_config.sensor_size,
                pixel_pitch=cam_config.pixel_pitch,
                resolution=cam_config.resolution
            )
            self.camera_array.append(camera)
    
    def generate_scene(self):
        # 生成符合物理规律的场景
        scene = Scene(
            meshes=self.create_meshes(),
            materials=self.setup_materials(),
            lighting=self.setup_lighting()
        )
        return scene.render(self.camera_array)
```

---
# physics-driven

## 1. 基本定义

Physics-Driven指的是由物理定律和原理驱动的方法或模型，这些方法直接使用物理方程或规律来指导系统的行为。

```python
class PhysicsDrivenModel:
    def __init__(self):
        self.physical_equations = {}
        self.constraints = {}
        
    def solve(self, initial_conditions):
        # 直接使用物理方程求解
        solution = self.apply_physical_laws(initial_conditions)
        return self.enforce_constraints(solution)
```

## 2. 主要特点

1. **直接使用物理定律**
2. **确定性**：结果可预测
3. **计算特性**：通常计算密集，求解精确
## 3. 与其他方法的对比

| 特性     | Physics-Driven | Data-Driven | Physics-Aware | PINN |
| ------ | -------------- | ----------- | ------------- | ---- |
| 物理知识使用 | 完全依赖           | 不使用         | 作为约束          | 深度集成 |
| 数据需求   | 无需数据           | 大量数据        | 适量数据          | 少量数据 |
| 计算复杂度  | 高              | 中等          | 中等            | 较高   |
| 可解释性   | 极强             | 弱           | 中等            | 强    |
| 泛化能力   | 好              | 依赖数据        | 较好            | 很好   |


以

> Li L, Wang L, Zhou X, et al. A novel physics-driven fast parallel three-dimension radar imaging method[C]//2016 URSI Asia-Pacific Radio Science Conference (URSI AP-RASC). IEEE, 2016: 543-546.

为例
# Physics-Driven在雷达成像中的应用

## 1. 核心物理模型

论文使用了三个关键的物理模型/假设：

```python
class PhysicsDrivenRadarImaging:
    def __init__(self):
        # 三个核心物理模型
        self.reflectivity_model = GeneralReflectivityModel()
        self.far_field_approximation = FarFieldApproximation()
        self.neighbor_cell_approximation = NeighborCellApproximation()
```

1. **通用反射率模型**（General Reflectivity Model）
2. **远场近似**（Far-field-approximation）
3. **邻域单元近似**（Neighbor-cell Approximation）

## 2. 物理驱动的实现步骤

```python
class RadarImagingMethod:
    def process(self, imaging_region):
        # 1. 区域分解
        sub_regions = self.decompose_region(imaging_region)
        
        # 2. 应用远场格林函数
        green_function = self.apply_far_field_greens_function()
        
        # 3. 双重变换
        system_response = self.dual_transform(green_function)
        
        # 4. 并行重建
        reconstructions = self.parallel_reconstruction(sub_regions)
        
        # 5. 结果融合
        final_image = self.fuse_results(reconstructions)
        
        return final_image
```

这篇论文通过将物理定律（反射率模型、远场近似、邻域单元近似）直接嵌入到算法设计中，实现了一个高效的三维雷达成像方法。这是一个典型的Physics-Driven方法，因为它直接基于物理原理构建解决方案，而不是依赖于数据学习。

以

> Yari M, Ibikunle O, Varshney D, et al. Airborne snow radar data simulation with deep learning and physics-driven methods[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021, 14: 12035-12047.

为例
# Physics-Driven雪地雷达数据模拟

## 1. 物理驱动核心思想

```python
class PhysicsRadarSimulator:
    def __init__(self):
        # 基于两个主要物理过程
        self.layering_process = LayeringProcess()  # 分层过程
        self.backscatter_process = BackscatterProcess()  # 后向散射
        
        # 参数化设置
        self.radar_params = {
            'flight_line': 1200,  # km
            'range_lines': 200000,
            'scattering_power': (0, 255)  # 8-bit范围
        }
```
### 主要特点：
1. 基于已知的物理过程
2. 使用实际数据统计参数化
3. 模拟底层随机过程

## 2. 物理过程建模

```python
class PhysicsProcesses:
    def simulate_layer(self):
        # 模拟雪层的物理特性
        return {
            'dielectric_contrasts': self.calculate_contrasts(),
            'layer_smoothness': self.simulate_smoothness(),
            'vertical_variations': self.calculate_variations()
        }
    
    def simulate_backscatter(self, layer_properties):
        # 模拟后向散射
        return self.calculate_backscatter_power(layer_properties)
```

### 关键物理特性：
1. **雪层特性**：
2. **散射特性**：
## 3. 数据驱动的参数化

```python
class ParameterEstimation:
    def __init__(self, sample_dataset):
        self.sample_data = sample_dataset
        
    def estimate_parameters(self):
        # 从实际数据估计统计参数
        statistics = {
            'accumulation_conditions': self.analyze_accumulation(),
            'layer_distribution': self.analyze_layers(),
            'backscatter_statistics': self.analyze_backscatter()
        }
        return statistics
```
### 参数来源：
- 格陵兰飞行数据
- 1200公里飞行线
- 200,000个测距线
## 4. 图像生成过程

```python
class EchogramGenerator:
    def generate_echogram(self):
        # 1. 创建数据矩阵
        matrix = np.zeros((height, width))
        
        # 2. 模拟层
        for rangeline in range(width):
            layers = self.physics_model.simulate_layers()
            backscatter = self.physics_model.simulate_backscatter(layers)
            matrix[:, rangeline] = backscatter
            
        # 3. 应用8位量化
        return self.quantize_to_8bit(matrix)
```

## 5. 与cGAN方法的对比
### Physics-Driven优势：
1. **结构相似性好**：
   - 更好地保持物理结构
   - 层次关系明确

2. **物理一致性**：
   - 符合雷达散射物理
   - 保持层的单值性
### cGAN优势：
1. **纹理相似性好**：
   - 更好的细节表现
   - 更真实的视觉效果

这篇论文的Physics-Driven方法主要通过模拟雪地雷达的物理散射过程，并结合实际数据的统计特性来生成模拟数据。这种方法与纯数据驱动的cGAN方法形成互补，各有优势。最终，两种方法生成的模拟数据都能用于改善深度学习模型在跟踪雪的内部层方面的性能。
