# car_rl_train

## 1. 项目介绍

本项目用于在 Isaac Sim / Isaac Lab 中训练、评估和导出差速小车局部导航策略。训练环境包含静态障碍物、动态障碍物、课程学习、动力学随机化、真实车辆标定约束和确定性评估。部署接口固定为 86 维归一化观测输入和 2 维左右轮控制输出，生产模型使用 ONNX 格式。

项目只有一条正式流水线，根目录下的 `scripts/01_*.py` 至 `scripts/07_*.py`、`scripts/09_*.py` 和 `scripts/10_*.py` 必须按编号执行；`scripts/08_play.py` 只用于人工回放检查，可按需要运行。脚本默认读取项目内固定位置，不依赖原始 ROS 工作区：

```text
assets/
├── urdf/                         # Xacro 源文件和展开后的 car.urdf
├── usd/car.usd                   # Isaac Sim 使用的车辆资产
└── asset_manifest.json           # 源文件、配置、URDF、USD 摘要
calibration/                      # 实车标定 YAML 和原始 CSV
configs/                          # 资产、训练、评估、回放和导出配置
observations/                     # 实车观测 CSV、元数据、NPZ 和 manifest
scripts/                          # 01～10 正式执行入口
source/car_rl_train/              # Python 包、环境、模型和导出实现
logs/                             # 训练记录和 campaign 指针
evaluations/                      # 正式评估结果和最新评估指针
exports/car_rl_model/             # 固定名称的最终部署包
```

资产以 `assets/urdf/car.urdf.xacro` 为唯一入口，所引用的描述文件全部保存在 `assets/urdf/car/`。步骤 01 先用 ROS 2 Xacro 生成 `assets/urdf/car.urdf`，再转换、扁平化并校验 `assets/usd/car.usd`。发布时同时更新 `assets/asset_manifest.json`；任何阶段失败都保留上一版有效资产。

训练默认执行种子 42、43、44。每个 checkpoint 内保存策略结构、训练配置、标定信息和资产源指纹；恢复训练时发现标定或资产变化会立即拒绝继续。步骤 07 默认运行 core、dynamic、stress 三套评估，生产导出至少要求 core 1000 回合、dynamic 500 回合、4000 条真实观测，并满足 `configs/evaluate.yaml` 中的成功率、安全和精度门限。

步骤 10 只发布生产合格结果。所有文件先在临时目录完成 ONNX 数值一致性、推理延迟、模型包和 ZIP 完整性校验，再原子替换 `exports/car_rl_model/`。最终关键文件固定为：

```text
exports/car_rl_model/
├── bundle.yaml
├── environment-lock.json
├── onnx_validation.json
├── car_rl_model.zip
├── car_rl_model.zip.sha256
└── controller/
    ├── car_rl_model.onnx
    ├── metadata.yaml
    └── evaluation.json
```

## 2. 环境、前置文件与执行步骤

### 2.1 系统与硬件

- Ubuntu 22.04 x86_64
- 支持 Isaac Sim 5.1 的 NVIDIA GPU 和驱动
- 建议显存不低于 12 GB；显存较小时降低 `configs/train.yaml` 的 `num_envs`
- 训练磁盘空间建议不低于 50 GB
- ROS 2 Humble，至少安装 `xacro`
- Conda 或 Miniconda，Python 3.11

先安装系统工具和 Xacro：

```bash
sudo apt update
sudo apt install -y git build-essential cmake zip ros-humble-xacro
source /opt/ros/humble/setup.bash
```

### 2.2 Python、Isaac Sim 与 Isaac Lab

本项目验证通过的版本如下。Isaac Lab 源码版本为 v2.3.2，其中 Python distribution 版本显示为 0.54.2。

| 组件 | 版本 |
|---|---:|
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | v2.3.2 / 0.54.2 |
| Isaac Lab RL | 0.4.7 |
| Isaac Lab Tasks | 0.11.12 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| Gymnasium | 1.2.1 |
| RSL-RL | 3.1.2 |
| NumPy | 1.26.0 |
| h5py | 3.16.0 |
| TensorDict | 0.13.0 |
| PyYAML | 6.0.2 |
| psutil | 5.9.8 |
| ONNX | 1.18.0 |
| ONNX Runtime | 1.28.0 |

创建独立环境并安装 Isaac Sim：

```bash
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0.0" \
  --extra-index-url https://pypi.nvidia.com
```

安装 Isaac Lab v2.3.2：

```bash
WORKSPACE_ROOT="$(pwd)"
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git "$WORKSPACE_ROOT/IsaacLab"
cd "$WORKSPACE_ROOT/IsaacLab"
./isaaclab.sh --install rsl_rl
```

安装本项目及导出依赖：

```bash
cd "$WORKSPACE_ROOT"
git clone https://github.com/xiaohao1208/car_rl_train.git
cd car_rl_train
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
ISAACLAB_ROOT="$(cd "$PROJECT_ROOT/../IsaacLab" && pwd)"
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install \
  -c constraints-isaacsim-5.1.txt \
  -e "source/car_rl_train[export]"
```

每次运行前激活环境并进入项目目录：

```bash
conda activate isaaclab
source /opt/ros/humble/setup.bash
cd car_rl_train
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
```

如果本机使用自定义环境激活命令，只需保证 `python` 能同时导入 `isaacsim`、`isaaclab`、`torch` 和 `car_rl_train`。

### 2.3 必需的前置文件

车辆描述文件必须直接放入以下位置。重新从 ROS 工程复制时，应保持相同的目录层级和文件名：

```text
assets/urdf/car.urdf.xacro
assets/urdf/car/base.urdf.xacro
assets/urdf/car/inertia.urdf.xacro
assets/urdf/car/actuator/caster.urdf.xacro
assets/urdf/car/actuator/wheel.urdf.xacro
assets/urdf/car/sensor/imu.urdf.xacro
assets/urdf/car/sensor/laser.urdf.xacro
assets/urdf/car/sensor/ultrasonic.urdf.xacro
```

实车标定必须直接放入固定目录：

```text
calibration/calibration.yaml
calibration/calibration_raw.csv
```

实车标定和观测数据不随公开仓库分发，首次运行前必须由当前车辆重新采集并放入上述固定位置。

`calibration.yaml` 必须为 schema v1，包含测量时间、质量、轮胎摩擦、左右电机增益、执行延迟、电机时间常数以及线速度和角速度加减速参数。其 `source_log_sha256` 必须与 `calibration_raw.csv` 一致。CSV 必须包含完整 A0～A3 阶段以及每个规定速度档位至少 4 个有效 trial。步骤 02 会从 CSV 重新计算全部统计量，不只读取 YAML 汇总值。

生产导出还需要部署端记录的真实观测：

```text
observations/real_observations.csv
observations/real_observations.metadata.yaml
```

CSV 必须严格包含 `observation_0` 至 `observation_85` 共 86 列，至少 4000 行有限数值，归一化范围为 `[-1, 1]`，且至少有 100 条不同观测。元数据必须声明 contract v1、采集完成状态、行数和原始 CSV 的 SHA-256。

### 2.4 完整执行顺序

1. **必选：**生成并校验车辆资产。

   ```bash
   python scripts/01_prepare_assets.py --headless --device cuda:0
   ```

   成功后得到 `assets/urdf/car.urdf`、`assets/usd/car.usd` 和 `assets/asset_manifest.json`。不得手工修改生成的 URDF、USD 或 manifest；描述文件变化后重新执行本步骤。

2. **必选：**校验实车标定。

   ```bash
   python scripts/02_validate_calibration.py
   ```

3. **必选：**检查全部环境、配置、CUDA、资产和标定。

   ```bash
   python scripts/03_check_environment.py
   ```

   仅在无 GPU 的配置检查机器上可使用 `--no-require-cuda`；正式训练和动力学校验必须使用 CUDA。

4. **必选：**校验全部课程阶段的地图生成、路径可达性和安全净空。

   ```bash
   python scripts/04_validate_maps.py \
     --seeds-per-stage 10000 \
     --workers 8 \
     --output logs/validation/maps.json
   ```

5. **必选：**在 Isaac Lab 中校验轮关节、驱动方向和车辆动力学。

   ```bash
   python scripts/05_check_dynamics.py \
     --device cuda:0 \
     --num-envs 16 \
     --steps 900 \
     --expect-goal \
     --headless \
     --output logs/validation/dynamics.json
   ```

6. **必选：**按三个固定种子执行正式训练和自动候选预检。

   ```bash
   python scripts/06_train.py --campaign-name car_rl_model
   ```

   训练记录位于 `logs/rsl_rl/car_local_navigation/`，campaign 状态位于 `logs/rsl_rl/car_local_navigation/campaigns/`。全部种子完成后会更新 `campaigns/latest_campaign.json`。恢复单个运行时必须显式指定种子和 checkpoint：

   ```bash
   python scripts/06_train.py \
     --single-seed 42 \
     --checkpoint logs/rsl_rl/car_local_navigation/<run>/model_<iteration>.pt \
     --run-name car_rl_model_resume
   ```

7. **必选：**运行完整正式评估并选择最佳 checkpoint。

   ```bash
   python scripts/07_evaluate.py --suite all
   ```

   默认读取 `campaigns/latest_campaign.json`，结果写入 `evaluations/`，并原子更新 `evaluations/latest_evaluation.json`。任何生产门限未通过都不会产生可导出的正式选择。

8. **可选：**回放评估选出的策略，人工检查实际运动行为。本步骤不生成后续导出依赖的文件。

   ```bash
   python scripts/08_play.py --device cuda:0
   ```

9. **必选：**把真实 86 维观测转换为带摘要的 NPZ。

   ```bash
   python scripts/09_prepare_observations.py
   ```

   成功后得到 `observations/real_observations.npz` 和 `observations/real_observations.manifest.json`。为防止误混用旧数据，本步骤拒绝覆盖已有输出；重新生成前应先人工确认并移走这两个旧文件。

10. **必选：**执行生产门禁并发布固定名称模型。

    ```bash
    python scripts/10_export_model.py
    ```

    默认读取 `evaluations/latest_evaluation.json`、实车观测 NPZ/manifest 和 checkpoint 内嵌标定。成功后只发布到 `exports/car_rl_model/`。再次执行会在新结果全部校验成功后安全替换旧目录；导出失败时旧模型保持不变。
