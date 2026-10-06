# 项目结构说明

本文件定义仓库中各目录的职责。后续添加代码时应遵守模块边界，避免把数据处理、训练、设备运行和实验脚本混在一起。

## 目录职责

| 路径 | 职责 | 允许提交的内容 |
| --- | --- | --- |
| `docs/` | 需求、架构、算法、部署、评测和计划 | Markdown、架构图源文件 |
| `src/` | 可复用的项目实现 | Python/C++ 源码和模块说明 |
| `data/` | 数据格式和本地目录说明 | README、少量匿名样例、索引模板 |
| `models/` | 模型注册、版本和导出规范 | README、模型卡、校验清单 |
| `configs/` | 实验与运行参数 | YAML/TOML/JSON 配置，不保存密钥 |
| `experiments/` | 可复现实验和结果汇总 | 实验说明、指标表、图表 |
| `hardware/` | 硬件选型、连接和装配 | BOM、接线说明、结构设计文件 |
| `tests/` | 单元、集成、回放和设备验收 | 测试代码、测试数据说明 |
| `tools/` | 一次性入口和辅助工具 | 标注、导出、基准测试脚本 |

## 未来代码模块

`src/` 建议按以下边界扩展：

```text
src/
├─ capture/          # 双摄像头采集、时间戳、曝光和缓冲控制
├─ preprocessing/    # 缩放、裁剪、颜色转换、坐标映射
├─ teacher/          # MediaPipe 教师推理，仅用于开发和标注
├─ hand/             # 手部检测、关键点、静态与动态手势
├─ eye/              # 眼部 ROI、眼睑/虹膜特征、注视与眨眼
├─ distillation/     # 教师学生损失、伪标签过滤和训练流程
├─ export/           # ONNX/TFLite/设备格式导出与量化
├─ runtime/          # CPU/NPU 推理后端、调度和性能统计
├─ interaction/      # 眼动与手势融合、防抖和事件状态机
├─ telemetry/        # 延迟、帧率、内存、温度和功耗日志
└─ app/              # 演示程序和命令行入口
```

## 依赖方向

推荐保持单向依赖：

```text
capture → preprocessing → hand / eye → interaction → app
                               ↓
                         runtime / telemetry

teacher → data → distillation → export → runtime
```

`teacher` 不进入最终端侧运行链路；它只在开发机上生成标签、建立精度上限和辅助蒸馏。端侧只能加载导出的本地学生模型。

## 数据与模型命名

- 数据集版本：`dataset_name-vMAJOR.MINOR`
- 训练运行：`YYYYMMDD-task-model-tag`
- 模型版本：`task_arch_data_precision-vMAJOR.MINOR.PATCH`
- 实验目录：`experiments/YYYYMMDD-short-description/`
- 设备日志：`device_session_timestamp.jsonl`

每个发布模型至少应记录训练数据版本、输入尺寸、输出定义、量化方式、模型文件哈希、运行后端和验证指标。

## 不应提交到 Git 的内容

- 原始或可识别身份的眼部、面部和手部视频。
- 大型中间数据、缓存、权重、编译产物和设备日志。
- 私钥、访问令牌、无线网络信息和个人路径配置。
- 未确认许可证允许再分发的第三方模型或数据。

