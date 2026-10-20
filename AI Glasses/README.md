# AI Glasses：轻量化手势识别与眼部追踪

毕业设计项目。目标是用 MediaPipe 在电脑端生成伪标签，再训练轻量学生模型，最终在眼镜端离线识别手势、追踪佩戴者眼部。**目前完成的是手部数据标注与检查流程，不是最终识别系统。**

## 现在能做什么

- 用本地 MediaPipe Hand Landmarker 处理视频或图片，输出每帧的手部21点、时间戳、左右手信息和质量检查标记。
- 保存不带关键点的原始图像，以及供人工检查的标注预览和统计报告。训练使用“原图 + 标签”，不是带连线的预览视频。
- 检查全部帧、图像、时间戳及文件哈希，生成初筛候选和待复核清单。
- 提供项目方案、目录规划及13项自动测试。

**还没完成：**轻量学生模型训练、手势类别标注与分类、眼部伪标签与追踪、模型量化、眼镜硬件部署。现有 `.task` 是官方教师模型，不是我们训练的学生模型；自动伪标签也不等于人工真值。

## 组员先看这里

拿到完整项目资料后，不用安装环境也可以先查看已生成的 [标注报告](data-local/annotations/public_pilot/woman_counting_on_fingers-686650a174d5/report.md)、[抽查图](data-local/annotations/public_pilot/woman_counting_on_fingers-686650a174d5/quality_check/spot_check.jpg) 和 [数据检查说明](data-local/annotations/public_pilot/woman_counting_on_fingers-686650a174d5/quality_check/README.md)。注意：这些示例在本地 `data-local/` 中，默认不会被 Git 上传，单独获取代码时可能没有这些文件。

当前公开样例共377帧，原图与标签全部保留；其中277帧为初筛候选、100帧待复核。候选尚未全部人工确认，且没有手势类别标签，不能直接当作完整手势分类训练集。它是第三人称流程验证样例，不代表眼镜第一视角效果。

## 文件夹怎么用

```text
tools/                  已实现：手部标注、标签检查脚本
models/teacher/         本地 MediaPipe 教师模型与来源记录
data-local/input/       把自己要标注的视频放这里
data-local/examples/    公开示例视频及许可、来源信息
data-local/annotations/ 已生成的原图、伪标签、预览和检查结果
tests/                  自动测试及小型离线测试素材
docs/                   使用指南、架构、蒸馏与评测等方案文档
src/                    后续学生模型与运行时代码规划，尚未实现
configs/                后续配置规范
data/                   数据组织规范，不是实际样本目录
experiments/            后续实验记录规范
hardware/               后续硬件资料规范
.venv/                  本机 Python 环境，不上传、不在组员间复制
```

根目录另有原始课题说明 `Topic3-Hand Recgonition.docx`、已填写的开题报告 `1. Proposal Template (BIT)_20230922.docx`（保留了模板文件名），以及 [开题报告 PDF 预览](docs/proposal-preview.pdf)。更完整的规划见 [项目结构](PROJECT_STRUCTURE.md)、[系统架构](docs/architecture.md) 和 [数据与蒸馏方案](docs/data-and-distillation.md)；这些方案不代表对应功能已实现。

## 组员如何运行

获取项目文件后，在项目根目录打开 PowerShell。已验证环境为 Windows x64 + Python 3.13.2；每位组员应创建自己的环境，不要复制别人的 `.venv`：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-annotation.txt
```

当前本地文件夹已包含教师模型；如果只获取 Git 中的代码且模型缺失，先运行 `.\.venv\Scripts\python.exe tools\annotate_hands.py --download-model` 显式下载。把视频放入 `data-local/input/`，双击 `开始手部标注.cmd` 即可；结果生成在 `data-local/annotations/`。默认检测一只手、每5帧保存一张原图，标签逐帧保存。更密集保存、双手检测及公开样例复现见 [手部标注指南](docs/hand-annotation-quickstart.md)。依赖安装完成且本地模型齐全后，标注过程不需联网。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_hand_annotation.py" -v
```

如需对新任务执行完整数据检查，还需安装 `requirements-annotation-audit.txt`，详见 [工具说明](tools/README.md)。已有样例不必重新运行。

## 同步说明与下一步

直接同步项目文件即可，不需要额外生成分享压缩包。代码、文档、依赖清单和小型测试素材可以上传；不要上传本机环境 `.venv/`、缓存、失败任务或 `archives/` 清理备份。

当前 `.gitignore` 默认排除 `data-local/` 和模型二进制，避免误传以后采集的私人视频。需要让组员一起获得现有公开示例、标注结果及教师模型时，应明确将这部分公开资料纳入同步范围或另外分发，不要直接取消整个数据目录的隐私保护。

下一步先确定手势类别并补充多人员、第一视角数据，复核伪标签后训练一个小型手部学生模型，在电脑端验证。眼部方向需另行采集面向佩戴者眼睛的数据并验证标注方案，不能直接用现有手部模型替代。

