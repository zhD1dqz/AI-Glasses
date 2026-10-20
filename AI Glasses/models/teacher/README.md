# 手部教师模型

本机使用用户从官网下载的 `hand_landmarker.task`，已复制到本目录；下载目录中的原文件保持不变。模型包包含手部检测和手部关键点两个 TFLite 模型，已通过包完整性检查和 CPU 加载检查。

来源说明、文件大小和 SHA-256 见 `hand_landmarker.source.json`。参考下载地址不等于对用户下载来源的独立认证。模型文件受根目录 `.gitignore` 管理，不提交 Git；在其他电脑可按工具的 `--download-model` 命令准备。

该模型用于生成21个手部关键点伪标签。它不是眼部追踪模型，不直接输出张掌/握拳等手势类别，也不是本项目训练后的轻量学生模型。本阶段先验证教师标注流程，眼部数据和模型另行准备。

本工具固定使用 CPU 推理，运行时不会下载模型或上传输入数据。后续部署前还需依据官方模型说明核对许可和设备兼容性。

参考：[官方模型说明](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models)。
