# 手部标注最简使用方式

## 你需要做的事

首次在组员电脑上使用，先安装 Python（本项目验证版本为 Windows x64 Python 3.13.2），在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-annotation.txt
```

当前本地项目已包含 `models/teacher/hand_landmarker.task`；如果仅获取了 Git 中的代码且缺少模型，再执行 `.\.venv\Scripts\python.exe tools\annotate_hands.py --download-model`。依赖安装与显式下载需要网络，后续标注不需要。

录一段 20～30 秒的手部视频，复制到项目的 `data-local/input/` 文件夹，然后双击根目录的 `开始手部标注.cmd`。

视频中依次展示张掌、握拳、食指指向、捏合，并包含几秒无手画面。让手指清晰可见，先使用正常光照和普通 MP4 视频即可。录制时不需要准确计时，也不需要手动逐帧标注。

## 工具自动完成的事

- 从本地视频逐帧运行 MediaPipe Hand Landmarker 教师模型。
- 保存全部帧的21个关键点、时间戳、左右手分类与检查标记。
- 每5帧保存一张原图，给后续图像到关键点的学生模型训练使用。
- 生成带关键点连线的预览视频、机器可读统计和中文报告。
- 同样的输入和配置已经成功处理时，跳过已完成文件；失败尝试保留并在新目录重试。

原视频保持不变。标注过程使用本地模型，不要求联网，不会自动打开摄像头。第一次下载依赖和模型需要网络。

原开发机的独立环境已安装，用户提供的 `hand_landmarker.task` 已放到 `models/teacher/`。组员须创建自己的环境；不需要移动下载目录中的原模型。该模型只处理手部，眼部追踪不在本次工具范围内。

## 查看结果

结果在 `data-local/annotations/` 下，每个输入文件有独立目录：

```text
视频名-运行标识/
├─ labels.jsonl      # 每行一帧的教师伪标签
├─ images/           # 按间隔保存的原始图像，不含绘制的关键点
├─ preview.mp4       # 标注预览；不支持 MP4 写入时回退到 AVI
├─ manifest.json     # 输入、模型哈希、版本、参数和坐标约定
├─ summary.json      # 处理统计；成功结束后才生成
└─ report.md         # 中文说明和统计
```

图像输入会生成 `preview.jpg`。视频若无法解码，尝试从手机导出 H.264 编码的 MP4。失败时 `error.json` 会记录原因，不会把不完整结果标为成功。

## 标签含义

`landmarks_image` 是21×3数组；x、y按原图宽高归一化，z是以手腕为原点的相对深度，不是米制距离。`handedness_score` 是左右手分类分数，不能用于证明关键点精度。

未检测到手的帧标记 `no_detection`，它既可能是确实无手，也可能是教师漏检。自动预测不是人工真值，不能把检测覆盖率当成准确率。`gesture_label` 暂时为空；张掌、握拳和点击等类别需要另行标注或实现规则。

样本图未经裁剪或镜像，坐标与保存原图一致。`subject_id` 默认为 `unassigned`，正式训练前需补匿名受试者编号，并按受试者隔离训练集和测试集。

## 开发者命令

在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-annotation.txt
.\.venv\Scripts\python.exe tools\annotate_hands.py --download-model
.\.venv\Scripts\python.exe tools\annotate_hands.py
```

指定输入或更密集地保存原图：

```powershell
.\.venv\Scripts\python.exe tools\annotate_hands.py --input data-local\input --save-every 1 --subject-id pilot_001
```

输入支持常见图片，以及 MP4、MOV、AVI、MKV、WEBM、M4V 视频。目录输入会递归查找媒体文件，标注默认只检测一只手，可用 `--num-hands 2` 修改。

少数可变帧率视频的容器帧数是估算值。工具默认遇到解码帧数不足会报错；只有使用独立解码器核验完整帧数后，才能对单个视频加 `--expected-frames 实际帧数`。该参数会写入运行记录，并严格要求实际解码帧数与指定值相等，不应用它掩盖损坏输入。

## 复现已提供的公开样例

公开素材放在 `data-local/examples/finger-counting/`，与自己的待标注视频分开。来源与 CC0 许可信息见同目录的 `.source.json`。该视频实际完整帧数已独立核验为377，复现命令为：

```powershell
.\.venv\Scripts\python.exe tools\annotate_hands.py --input data-local\examples\finger-counting\woman_counting_on_fingers.webm --output data-local\annotations\public_pilot --save-every 1 --num-hands 2 --subject-id public_counting_clip_001 --expected-frames 377
```

相同模型、依赖与配置下会校验并跳过已完成任务，不会覆盖原标签。当前本地目录已保存377条标签、377张原图及初筛结果，拿到这些资料后不必为查看结果重新标注。`data-local/` 默认被 Git 忽略，仅获取代码的组员需另行获取公开样例。运行记录新增项目相对路径，源视频和模型按相同目录结构同步后，换电脑仍能定位；旧绝对路径仅保留为历史记录。

## 环境与测试

验证过的环境为 Windows x64、Python 3.13.2、MediaPipe 1.1.1、OpenCV contrib 4.14.0.94、NumPy 2.5.4，直接依赖版本已固定。

运行自动测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_hand_annotation.py" -v
```

实现参考：[MediaPipe Hand Landmarker Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)。本机使用用户提供的官网模型下载，来源说明与 SHA-256 保存在模型旁的 `.source.json` 中；其他电脑也可使用工具显式下载官方模型。
