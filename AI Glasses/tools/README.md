# 工具目录

`annotate_hands.py` 已提供本地视频和图片的 MediaPipe 手部批量标注，输出逐帧伪标签、训练图像、预览和统计。最简入口为根目录的 `开始手部标注.cmd`，使用方法见 [手部标注指南](../docs/hand-annotation-quickstart.md)。

`audit_hand_labels.py` 可检查使用 `--save-every 1` 生成的视频任务：核对原始文件、标签、全部图像、时间戳，以及独立解码帧数；生成初筛候选和待复核清单。它只做结构检查，不把伪标签变成人工真值。可选依赖见 `requirements-annotation-audit.txt`，使用方式为 `python tools/audit_hand_labels.py 标注任务目录`，首次运行生成 `quality_check/`，不会覆盖已有检查目录。

后续工具包括数据采集、标签可视化、数据检查、模型导出、量化、基准测试和日志汇总。

工具应提供明确输入输出、支持断点续跑，并把版本和配置写入产物元数据。会修改数据的工具必须保留原始数据，只向新目录写结果。

