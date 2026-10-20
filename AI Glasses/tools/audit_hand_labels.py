"""Audit an all-frames annotation job without changing raw labels or images."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import av
import cv2
import numpy as np

from annotate_hands import ROOT, draw_preview, file_sha256, save_jpeg, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_manifest_asset(manifest: dict, name: str) -> Path:
    """Prefer the shared project's copy; retain support for older manifests."""
    relative = manifest.get(f"{name}_relative_path")
    if relative:
        candidate = (ROOT / relative).resolve()
        require(candidate.is_relative_to(ROOT), "Asset path escapes project")
        if candidate.is_file():
            return candidate
    return Path(manifest[f"{name}_path"])


def audit(job: Path) -> dict:
    summary = json.loads((job / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (job / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    require(summary["status"] == "complete", "The job is not complete")
    require(len(rows) == summary["processed_frames"], "Label/frame count mismatch")
    require(file_sha256(job / "labels.jsonl") == summary["labels_sha256"], "Labels checksum mismatch")
    source = resolve_manifest_asset(manifest, "source")
    require(file_sha256(source) == manifest["source_sha256"], "Source checksum mismatch")
    require(file_sha256(resolve_manifest_asset(manifest, "model")) == manifest["model_sha256"], "Model checksum mismatch")
    with av.open(str(source)) as video:
        source_times = [round(float(frame.time) * 1000) for frame in video.decode(video=0)]
    require(len(source_times) == len(rows), "Independent decode frame count mismatch")
    previous = -1
    for index, row in enumerate(rows):
        require(row["frame_index"] == index, "Frame indices are not contiguous")
        require(row["source_sha256"] == manifest["source_sha256"], "Row source mismatch")
        require(row["timestamp_ms"] > previous, "Non-increasing timestamp")
        require(abs(row["timestamp_ms"] - source_times[index]) <= 1, "Timestamp differs from independent decode")
        previous = row["timestamp_ms"]
        require(bool(row["image_path"]), "Audit expects --save-every 1")
        image = cv2.imdecode(np.fromfile(str(job / row["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None, f"Missing or unreadable image at frame {index}")
        require(image.shape[:2] == (row["height"], row["width"]), "Image size mismatch")
        require(row["needs_review"] == bool(row["quality_flags"]), "Review flag mismatch")
        if not row["hands"]:
            require("no_detection" in row["quality_flags"], "Missing no-detection flag")
        for hand in row["hands"]:
            points = hand["landmarks_image"]
            require(len(points) == 21 and all(len(point) == 3 for point in points), "Invalid keypoint shape")
            require(all(math.isfinite(v) for point in points for v in point), "Non-finite keypoint")
            outside = any(not 0 <= p[0] <= 1 or not 0 <= p[1] <= 1 for p in points)
            require(not outside or "landmarks_outside_image" in row["quality_flags"], "Unflagged outside keypoint")
    with av.open(str(job / summary["preview_file"])) as video:
        preview_frames = sum(1 for _ in video.decode(video=0))
    require(preview_frames == len(rows), "Preview frame count mismatch")
    positives = sum(bool(row["hands"]) for row in rows)
    require(positives == summary["frames_with_hands"], "Detection summary mismatch")
    flags = dict(Counter(flag for row in rows for flag in row["quality_flags"]))
    require(flags == summary["quality_flag_counts"], "Flag summary mismatch")
    candidates = [row for row in rows if row["hands"] and not row["needs_review"]]
    review = [row for row in rows if row["needs_review"]]
    require(len(candidates) + len(review) == len(rows), "Manifest partition mismatch")
    require(len(review) == summary["frames_needing_review"], "Review summary mismatch")
    output = job / "quality_check"
    output.mkdir(exist_ok=False)
    for filename, subset in (("training_candidates.jsonl", candidates), ("review_queue.jsonl", review)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            for row in subset:
                # Manifest image paths resolve relative to this manifest's parent.
                entry = {**row, "image_path": f"../{row['image_path']}",
                         "human_verified": False, "dataset_split": "unassigned"}
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    indices = sorted(set(round(i * (len(rows) - 1) / 8) for i in range(9)))
    tiles = []
    for index in indices:
        row = rows[index]
        image = cv2.imdecode(np.fromfile(str(job / row["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        tile = cv2.resize(image, (480, 270))
        tile = draw_preview(cv2, tile, row["hands"], index, row["timestamp_ms"])
        status = ",".join(row["quality_flags"]) or "candidate (not ground truth)"
        cv2.putText(tile, status, (8, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (30, 30, 255) if row["needs_review"] else (30, 220, 50), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) < 9:
        tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    sheet = np.vstack([np.hstack(tiles[start:start + 3]) for start in (0, 3, 6)])
    save_jpeg(cv2, output / "spot_check.jpg", sheet)
    result = {
        "status": "structural_checks_passed", "decoded_frames": len(rows),
        "independent_decoder": f"PyAV {av.__version__}", "independent_decode_frames": len(source_times),
        "container_reported_frames": summary["reported_frame_count"],
        "saved_and_decoded_images": len(rows), "preview_frames": preview_frames,
        "frames_with_teacher_hands": positives,
        "teacher_hand_instances": sum(len(row["hands"]) for row in rows),
        "candidate_frames": len(candidates), "review_frames": len(review), "quality_flags": flags,
        "gesture_class_labeled_frames": sum(row["gesture_label"] is not None for row in rows),
        "spot_check_frames": indices, "source_sha256": manifest["source_sha256"],
        "model_sha256": manifest["model_sha256"], "labels_sha256": summary["labels_sha256"],
        "limits": ["Teacher predictions are not human ground truth", "Structural checks do not measure accuracy",
                   "Not a complete gesture classification dataset", "No train/test split from this single clip"],
    }
    write_json(output / "audit.json", result)
    (output / "README.md").write_text(
        "# 公开手势视频伪标签结果\n\n"
        f"已完成 {len(rows)} 帧标注，并保存 {len(rows)} 张无叠加标记的图像。"
        f"其中 {positives} 帧检出手，共 {result['teacher_hand_instances']} 个手部关键点实例。\n\n"
        f"初筛候选：{len(candidates)} 帧；待复核：{len(review)} 帧。"
        f"待复核原因：{json.dumps(flags, ensure_ascii=False)}。\n\n"
        "## 直接查看\n\n"
        "- `../labels.jsonl`：全部帧的原始教师预测，没有删除失败帧。\n"
        "- `../images/`：每一帧的干净原图。\n"
        "- `training_candidates.jsonl`：非空、无已知质量标记的关键点训练候选。\n"
        "- `review_queue.jsonl`：需要复核的帧，不能直接当成可靠无手负样本。\n"
        "- `spot_check.jpg`：9帧标签抽查拼图，不作为训练输入。\n"
        "- `audit.json`：原文件/标签哈希、独立解码、全部样本图、预览帧数和时间戳检查。\n\n"
        "两份筛选清单中的 image_path 相对清单所在目录解析；每只手是21×3数组。"
        "x/y按原图宽高归一化，z是手腕相对深度，不是米制距离。"
        "handedness_score只表示左右手分类分数，不是定位准确率。\n\n"
        "## 使用边界\n\n"
        "候选只代表通过结构和边界检查，不保证教师预测准确；human_verified=false。"
        "全部 gesture_label 为空，目前是手部关键点伪标签，不是数字或握拳等类别标签。"
        "一段第三视角视频不能作为完整的眼镜端训练集，也不能把相邻帧随机分到训练和测试。"
        "未训练任何学生模型。\n\n"
        "## 来源与帧数\n\n"
        "素材：Mvolz，Woman counting on fingers.webm，来源页标明CC0 1.0。"
        "[来源及许可](https://commons.wikimedia.org/wiki/File:Woman_counting_on_fingers.webm)。"
        "完整来源记录见输入文件旁的 `.source.json`。\n\n"
        f"容器报告 {summary['reported_frame_count']} 帧，但视频未记录精确总帧数。"
        f"独立解码得到 {len(source_times)} 帧，与 OpenCV、全部标签和样本图一致；"
        "逐帧时间戳亦匹配，原文件大小与来源页SHA-1吻合。"
        "预览使用固定帧率，因此时长可能略有变化，应以原视频和标签时间戳为准。\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all-frame MediaPipe hand labels and make candidate/review manifests")
    parser.add_argument("job", type=Path)
    options = parser.parse_args()
    try:
        print(json.dumps(audit(options.job.resolve()), ensure_ascii=True, indent=2))
    except Exception as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
