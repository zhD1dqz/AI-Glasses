# 测试目录

## 已有手部标注测试

在项目根目录使用独立环境运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_hand_annotation.py" -v
```

`test_hand_annotation.py` 包含13项检查：时间戳、关键点格式、越界标记、无检测处理、文件扫描、结果校验与重复运行、外部精确帧数验收，以及本地模型对图片和视频的真实 CPU 推理。测试创建的临时样本会在结束后清理，不修改原始数据。

正样本使用随项目提供的 `fixtures/public_hand_frame.jpg`，来源为 CC0 公开视频，许可和帧信息见 [素材说明](fixtures/README.md)。不再依赖临时下载目录；测试本身从不联网。未安装推理依赖或缺少教师模型时，4项真实推理测试会跳过，不应将跳过当作完整通过。

2026-10-19 的历史记录见 [验证记录](../docs/hand-annotation-validation.md)；公开样例的完整性检查与边界说明见 `data-local/annotations/public_pilot/woman_counting_on_fingers-686650a174d5/quality_check/`。

## 后续测试层级

后续测试分为四层：

- 单元测试：坐标变换、滤波、状态机和配置解析。
- 模型回归：固定数据上的精度与导出一致性。
- 管线回放：使用录制视频验证实时队列和事件输出。
- 设备验收：离线启动、长时间运行、功耗、温升和故障降级。

测试数据必须匿名、小型并具有明确许可证。

