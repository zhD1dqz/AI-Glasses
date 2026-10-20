"""Offline teacher labels for videos/images; raw inputs are never changed."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TOOL_VERSION = "1.0.1"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = ROOT / "models" / "teacher" / "hand_landmarker.task"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
    (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
    (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative_path(path: Path) -> str | None:
    """Keep portable locators alongside historical absolute paths."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def download_model(path: Path) -> None:
    """Explicit setup action. Annotation itself never downloads a model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print(f"模型已存在，保留原文件：{path}")
        return
    print("正在下载 Google 官方手部教师模型……", flush=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AI-Glasses-Annotation/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if not payload.startswith(b"PK") or not 100_000 < len(payload) <= 32 * 1024 * 1024:
        raise ValueError("下载内容不是预期的 .task 模型包，未写入模型文件。")
    # Exclusive creation: do not overwrite a model produced by another setup.
    with path.open("xb") as stream:
        stream.write(payload)
    write_json(path.with_suffix(".source.json"), {
        "source_url": MODEL_URL,
        "sha256": file_sha256(path),
        "bytes": len(payload),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "MediaPipe teacher; not a student model",
    })
    print(f"模型已保存：{path}")


def collect_inputs(path: Path, output: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    supported = IMAGE_SUFFIXES | VIDEO_SUFFIXES
    if path.is_file():
        if path.suffix.lower() not in supported:
            raise ValueError(f"不支持的文件类型：{path.suffix}")
        return [path]
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
        and not p.resolve().is_relative_to(output)
    )


def frame_timestamp(raw_ms: float, index: int, fps: float, previous: int) -> tuple[int, str]:
    """Prefer source timestamps; retain strict monotonicity required by Tasks."""
    if math.isfinite(raw_ms) and raw_ms >= 0 and (index == 0 or raw_ms > previous):
        candidate = round(raw_ms)
        origin = "source"
    else:
        candidate = round(index * 1000.0 / fps)
        origin = "frame_index_and_fps"
    if candidate <= previous:
        candidate = previous + 1
        origin += "_adjusted"
    return candidate, origin


def check_frame_count(actual: int, reported: int, expected: int | None) -> None:
    """Keep strict defaults; allow an externally verified exact count for VFR."""
    if expected is not None:
        if actual != expected:
            raise ValueError(f"视频解码得到 {actual} 帧，与外部指定的 {expected} 帧不符；未标为完成。")
    elif reported and actual < reported:
        raise ValueError(
            f"视频仅解码 {actual}/{reported} 帧；结果保留但未标为完成。"
            "部分视频的元数据帧数为估算值，独立核验后可用 --expected-frames 指定真实帧数。")


def serialize_hands(result: object) -> tuple[list[dict], list[str]]:
    hands = []
    flags = []
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        coordinates = [[float(p.x), float(p.y), float(p.z)] for p in landmarks]
        if len(coordinates) != 21 or not all(math.isfinite(v) for p in coordinates for v in p):
            flags.append("invalid_landmarks")
            continue
        if any(not 0 <= p[0] <= 1 or not 0 <= p[1] <= 1 for p in coordinates):
            flags.append("landmarks_outside_image")
        categories = result.handedness[hand_index] if hand_index < len(result.handedness) else []
        category = categories[0] if categories else None
        hands.append({
            "handedness": category.category_name if category else None,
            # This score classifies left/right; it is NOT landmark accuracy.
            "handedness_score": float(category.score) if category else None,
            "landmarks_image": coordinates,
        })
    if not hands:
        flags.append("no_detection")
    return hands, sorted(set(flags))


def draw_preview(cv: object, frame: object, hands: list[dict], index: int, timestamp: int) -> object:
    preview = frame.copy()
    height, width = preview.shape[:2]
    for hand in hands:
        points = [
            (round(max(-1, min(2, p[0])) * (width - 1)),
             round(max(-1, min(2, p[1])) * (height - 1)))
            for p in hand["landmarks_image"]
        ]
        for start, end in CONNECTIONS:
            cv.line(preview, points[start], points[end], (50, 220, 70), 2, cv.LINE_AA)
        for point in points:
            cv.circle(preview, point, 3, (40, 80, 255), -1, cv.LINE_AA)
    text = f"frame={index} time={timestamp}ms hands={len(hands)}"
    cv.putText(preview, text, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(preview, text, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return preview


def save_jpeg(cv: object, path: Path, frame: object) -> None:
    # imencode + tofile supports Chinese Windows paths across OpenCV versions.
    success, encoded = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise OSError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def reserve_job(output: Path, stem: str, signature: str) -> tuple[Path, dict | None]:
    """File-level resume. Keep incomplete attempts intact for diagnosis."""
    base = output / f"{stem}-{signature[:12]}"
    for attempt in range(1, 10000):
        job = base if attempt == 1 else base.with_name(f"{base.name}-attempt{attempt}")
        if not job.exists():
            job.mkdir(parents=True, exist_ok=False)
            return job, None
        summary_path = job / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                labels = job / "labels.jsonl"
                preview = job / summary.get("preview_file", "missing")
                if (summary.get("status") == "complete" and summary.get("signature") == signature
                        and labels.is_file() and preview.is_file()
                        and file_sha256(labels) == summary.get("labels_sha256")
                        and all((job / p).is_file() for p in summary.get("sample_images", []))):
                    return job, summary
            except (OSError, ValueError, TypeError):
                pass
    raise RuntimeError("输出尝试次数过多，请选择新的输出目录。")


def build_report(summary: dict) -> str:
    count = summary["processed_frames"]
    ratio = summary["frames_with_hands"] / count if count else 0
    return (
        "# 手部教师标注报告\n\n"
        f"- 输入文件：{summary['source_name']}\n"
        f"- 处理帧数：{count}\n"
        f"- 检测到手的帧数：{summary['frames_with_hands']}（{ratio:.1%}）\n"
        f"- 未检测到手的帧数：{summary['frames_without_detection']}\n"
        f"- 保存的图像样本：{summary['saved_images']}\n"
        f"- 需要检查的帧数：{summary['frames_needing_review']}\n"
        f"- 本机处理耗时：{summary['elapsed_seconds']:.2f} 秒\n"
        f"- 标注预览：{summary['preview_file']}\n\n"
        "## 结果的使用范围\n\n"
        "这是 MediaPipe 自动生成的伪标签，不是人工真值。检测到手的比例不等于准确率；"
        "未检测到手也可能是漏检，不能直接当作可靠的无手负样本。\n\n"
        "landmarks_image 为原始图像坐标下的 21×3 数组。x、y 按图像宽高归一化；"
        "z 为以手腕为原点的相对深度，不是米制距离。handedness_score 仅表示左右手分类分数。\n\n"
        "每一帧均保存标签，按 save_every 间隔保存原图。没有保存图像的帧，image_path 为 null，"
        "仍可通过源视频和 frame_index 找回。图像未经镜像或 ROI 裁剪。"
        "本工具不生成张掌、握拳或点击等手势类别标签。\n\n"
        "本机离线处理速度不代表眼镜端实时延迟或功耗；预览视频采用固定帧率，"
        "可变帧率输入的精确时间以 labels.jsonl 为准。\n"
    )


def annotate_one(source: Path, output: Path, model: Path, model_hash: str,
                 options: argparse.Namespace, cv: object, mp: object, np: object) -> tuple[dict, bool]:
    is_image = source.suffix.lower() in IMAGE_SUFFIXES
    expected_frames = getattr(options, "expected_frames", None)
    versions = {name: importlib.metadata.version(name)
                for name in ("mediapipe", "opencv-contrib-python", "numpy")}
    config = {
        "tool_version": TOOL_VERSION, "source_sha256": file_sha256(source),
        "model_sha256": model_hash, "packages": versions,
        "num_hands": options.num_hands, "save_every": options.save_every,
        "subject_id": options.subject_id,
        "expected_frames": expected_frames,
        "min_hand_detection_confidence": 0.5,
        "min_hand_presence_confidence": 0.5, "min_tracking_confidence": 0.5,
        "mode": "IMAGE" if is_image else "VIDEO",
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    job, previous = reserve_job(output, source.stem, signature)
    if previous is not None:
        print(f"跳过已完成文件：{source.name}\n  结果：{job}", flush=True)
        return previous, True
    (job / "images").mkdir()
    write_json(job / "manifest.json", {
        **config, "source_name": source.name, "source_path": str(source),
        "signature": signature, "model_path": str(model),
        "source_relative_path": project_relative_path(source),
        "model_relative_path": project_relative_path(model),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": "full image; x/y normalized; z wrist-relative, not meters",
    })
    print(f"正在标注：{source.name}\n  输出目录：{job}", flush=True)
    capture = None
    writer = None
    frames = hand_frames = review_frames = 0
    samples = []
    flags_count = Counter()
    origins = Counter()
    started = time.perf_counter()
    preview_name = "preview.jpg" if is_image else "preview.mp4"
    fps = 0.0
    fps_fallback = False
    reported_frame_count = 0
    image = None
    try:
        if is_image:
            image = cv.imdecode(np.fromfile(str(source), dtype=np.uint8), cv.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码图像。")
        else:
            capture = cv.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError("无法打开视频；请尝试手机导出为 H.264 编码的 MP4。")
            fps = capture.get(cv.CAP_PROP_FPS)
            if not math.isfinite(fps) or not 0 < fps <= 1000:
                fps, fps_fallback = 25.0, True
            reported = capture.get(cv.CAP_PROP_FRAME_COUNT)
            reported_frame_count = round(reported) if math.isfinite(reported) and reported > 0 else 0
        vision = mp.tasks.vision
        task_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE if is_image else vision.RunningMode.VIDEO,
            num_hands=options.num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        previous_timestamp = -1
        with vision.HandLandmarker.create_from_options(task_options) as landmarker, \
                (job / "labels.jsonl").open("x", encoding="utf-8") as labels:
            while True:
                if is_image:
                    if frames:
                        break
                    frame = image
                    raw_timestamp = 0.0
                else:
                    success, frame = capture.read()
                    if not success:
                        break
                    raw_timestamp = capture.get(cv.CAP_PROP_POS_MSEC)
                timestamp, origin = (0, "image") if is_image else frame_timestamp(
                    raw_timestamp, frames, fps, previous_timestamp)
                previous_timestamp = timestamp
                origins[origin] += 1
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(media_image) if is_image else landmarker.detect_for_video(media_image, timestamp)
                hands, flags = serialize_hands(result)
                image_path = None
                if frames % options.save_every == 0:
                    image_path = f"images/frame_{frames:08d}.jpg"
                    save_jpeg(cv, job / image_path, frame)
                    samples.append(image_path)
                record = {
                    "schema_version": "1.0", "source_sha256": config["source_sha256"],
                    "subject_id": options.subject_id, "frame_index": frames,
                    "timestamp_ms": timestamp, "timestamp_origin": origin,
                    "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    "image_path": image_path, "hands": hands,
                    "label_source": "mediapipe_teacher", "gesture_label": None,
                    "quality_flags": flags, "needs_review": bool(flags),
                }
                labels.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                annotated = draw_preview(cv, frame, hands, frames, timestamp)
                if is_image:
                    save_jpeg(cv, job / preview_name, annotated)
                else:
                    # Only pad the preview; original images and coordinates stay unchanged.
                    height, width = annotated.shape[:2]
                    if width % 2 or height % 2:
                        annotated = cv.copyMakeBorder(
                            annotated, 0, height % 2, 0, width % 2,
                            cv.BORDER_CONSTANT, value=(0, 0, 0))
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv.VideoWriter(str(job / preview_name), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            writer.release()
                            preview_name = "preview.avi"
                            writer = cv.VideoWriter(str(job / preview_name), cv.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                        if not writer.isOpened():
                            raise OSError("无法创建预览视频，请检查视频编码器与输出目录。")
                    writer.write(annotated)
                hand_frames += int(bool(hands))
                review_frames += int(bool(flags))
                flags_count.update(flags)
                frames += 1
                if frames % 100 == 0:
                    print(f"  已处理 {frames} 帧，检测到手 {hand_frames} 帧", flush=True)
        if frames == 0:
            raise ValueError("未读到任何图像帧，请检查输入视频是否损坏或编码不受支持。")
        if not is_image:
            check_frame_count(frames, reported_frame_count, expected_frames)
    except Exception as error:
        write_json(job / "error.json", {"status": "failed", "error": str(error), "processed_frames": frames})
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete", "signature": signature, "source_name": source.name,
        "processed_frames": frames, "frames_with_hands": hand_frames,
        "frames_without_detection": frames - hand_frames,
        "frames_needing_review": review_frames, "quality_flag_counts": dict(flags_count),
        "saved_images": len(samples), "sample_images": samples,
        "labels_sha256": file_sha256(job / "labels.jsonl"),
        "elapsed_seconds": round(elapsed, 4), "source_fps": fps or None,
        "fps_fallback_used": fps_fallback, "timestamp_origins": dict(origins),
        "reported_frame_count": reported_frame_count, "preview_file": preview_name,
        "expected_frames": expected_frames,
        "frame_count_check": "external_exact_count" if expected_frames is not None else "container_metadata",
        "save_every": options.save_every,
    }
    (job / "report.md").write_text(build_report(summary), encoding="utf-8")
    # Completion is published last, after previews have been closed and flushed.
    write_json(job / "summary.json", summary)
    print(f"完成：{frames} 帧，检测到手 {hand_frames} 帧，保存图像 {len(samples)} 张。", flush=True)
    return summary, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地手部批量标注：不打开摄像头，不修改输入文件。")
    parser.add_argument("--input", type=Path, default=ROOT / "data-local" / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "data-local" / "annotations")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-every", type=int, default=5, help="每隔多少帧保存一张原图；标签仍逐帧保存")
    parser.add_argument("--num-hands", type=int, default=1, choices=(1, 2))
    parser.add_argument("--subject-id", default="unassigned", help="匿名受试者编号；用于后续按人划分数据")
    parser.add_argument("--expected-frames", type=int, help="独立核验后的真实视频帧数，仅限单个视频输入")
    parser.add_argument("--download-model", action="store_true", help="仅下载官方教师模型，然后退出")
    options = parser.parse_args(argv)
    if options.save_every < 1:
        parser.error("--save-every 必须大于零")
    if options.expected_frames is not None:
        if options.expected_frames < 1:
            parser.error("--expected-frames 必须大于零")
        if not options.input.is_file() or options.input.suffix.lower() not in VIDEO_SUFFIXES:
            parser.error("--expected-frames 只能和单个视频文件输入一起使用")
    try:
        model = options.model.resolve()
        if options.download_model:
            download_model(model)
            return 0
        source = options.input.resolve()
        output = options.output.resolve()
        if source.is_dir() and source == output:
            raise ValueError("输入与输出目录不能相同。")
        if source == (ROOT / "data-local" / "input").resolve():
            source.mkdir(parents=True, exist_ok=True)
        inputs = collect_inputs(source, output)
        if not inputs:
            print(f"请把录制好的手部视频放进：\n{source}\n然后重新双击‘开始手部标注.cmd’。")
            return 0
        if not model.is_file():
            raise FileNotFoundError("缺少教师模型。先运行：python tools/annotate_hands.py --download-model")
        import cv2
        import mediapipe as mp
        import numpy as np
        output.mkdir(parents=True, exist_ok=True)
        model_hash = file_sha256(model)
        failures = 0
        for path in inputs:
            try:
                annotate_one(path, output, model, model_hash, options, cv2, mp, np)
            except Exception as error:
                failures += 1
                print(f"失败：{path.name}：{error}", file=sys.stderr, flush=True)
        print(f"\n处理结束：{len(inputs) - failures} 个文件成功或已完成，{failures} 个失败。\n结果目录：{output}")
        return 1 if failures else 0
    except (ImportError, importlib.metadata.PackageNotFoundError):
        print("项目依赖不完整。请在项目 .venv 中安装 requirements-annotation.txt。", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
