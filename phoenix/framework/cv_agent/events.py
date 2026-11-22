"""Lightweight Computer Vision Event & Detection schemas (zero heavy dependencies).

Uses `slots=True` dataclasses to keep per-event RAM overhead minimal on phone CPUs
and constrained edge hardware running 24/7 vision pipelines.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_DC_SLOTS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


class CVSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(**_DC_SLOTS)
class Detection:
    """Single bounding-box / classification result emitted by a CV model (YOLO, ONNX, OpenCV, etc.)."""

    label: str
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    track_id: Optional[str] = None
    zone: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_compact_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "label": self.label,
            "conf": round(float(self.confidence), 3),
        }
        if self.track_id is not None:
            out["track_id"] = str(self.track_id)
        if self.zone:
            out["zone"] = self.zone
        if self.bbox is not None:
            out["bbox"] = [round(float(v), 1) for v in self.bbox]
        if self.attributes:
            out["attrs"] = dict(self.attributes)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        raw_bbox = data.get("bbox")
        bbox_tuple: Optional[Tuple[float, float, float, float]] = None
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox_tuple = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
        conf = data.get("confidence", data.get("conf", data.get("score", 1.0)))
        tid = data.get("track_id", data.get("id"))
        attrs = data.get("attributes") or data.get("attrs") or {}
        return cls(
            label=str(data.get("label") or data.get("class") or data.get("name") or "unknown"),
            confidence=float(conf),
            bbox=bbox_tuple,
            track_id=str(tid) if tid is not None else None,
            zone=str(data["zone"]) if data.get("zone") else None,
            attributes=dict(attrs) if isinstance(attrs, dict) else {},
        )


@dataclass(**_DC_SLOTS)
class CVEvent:
    """Normalized Computer Vision event consumed by CVAgent."""

    event_type: str
    camera_id: str = "cam_0"
    zone: str = "default"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    frame_id: Optional[int] = None
    detections: List[Detection] = field(default_factory=list)
    severity_hint: CVSeverity = CVSeverity.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(float(d.confidence) for d in self.detections)

    @property
    def track_ids(self) -> List[str]:
        return [d.track_id for d in self.detections if d.track_id is not None]

    def to_compact_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "ts": round(float(self.timestamp), 2),
            "frame_id": self.frame_id,
            "severity_hint": self.severity_hint.value,
            "detections": [d.to_compact_dict() for d in self.detections[:12]],
            "meta": dict(self.metadata),
        }

    def to_micro_line(self) -> str:
        """Single-line compact summary (< 40 tokens) tailored for <= 1B MoE context windows."""
        det_counts: Dict[str, int] = {}
        for d in self.detections:
            det_counts[d.label] = det_counts.get(d.label, 0) + 1
        det_str = ", ".join(f"{k}x{v}" for k, v in det_counts.items()) if det_counts else "none"
        tids = ",".join(self.track_ids[:4])
        tid_part = f" tracks=[{tids}]" if tids else ""
        meta_part = ""
        if self.metadata:
            kv = [f"{k}={v}" for k, v in list(self.metadata.items())[:5]]
            meta_part = " meta={" + ", ".join(kv) + "}"
        return (
            f"type={self.event_type} cam={self.camera_id} zone={self.zone} "
            f"sev={self.severity_hint.value} conf={self.max_confidence:.2f} "
            f"objs=[{det_str}]{tid_part}{meta_part}"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVEvent":
        raw_dets = data.get("detections") or data.get("objects") or []
        dets = [
            d if isinstance(d, Detection) else Detection.from_dict(d)
            for d in raw_dets
            if isinstance(d, (Detection, dict))
        ]
        raw_sev = data.get("severity_hint", data.get("severity", "low"))
        if isinstance(raw_sev, CVSeverity):
            sev = raw_sev
        else:
            try:
                sev = CVSeverity(str(raw_sev).lower())
            except ValueError:
                sev = CVSeverity.LOW
        ts = data.get("timestamp", data.get("ts"))
        return cls(
            event_type=str(data.get("event_type") or data.get("type") or "detection"),
            camera_id=str(data.get("camera_id") or data.get("camera") or "cam_0"),
            zone=str(data.get("zone") or "default"),
            timestamp=float(ts) if ts is not None else time.time(),
            event_id=str(data.get("event_id") or f"ev_{uuid.uuid4().hex[:10]}"),
            frame_id=int(data["frame_id"]) if data.get("frame_id") is not None else None,
            detections=dets,
            severity_hint=sev,
            metadata=dict(data.get("metadata") or data.get("meta") or {}),
        )

    @classmethod
    def from_boxes(
        cls,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Optional[Sequence[float]] = None,
        *,
        track_ids: Optional[Sequence[Any]] = None,
        event_type: str = "object_detected",
        camera_id: str = "cam_0",
        zone: str = "default",
        frame_id: Optional[int] = None,
        min_confidence: float = 0.25,
        severity_hint: CVSeverity = CVSeverity.LOW,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CVEvent":
        """Create a CVEvent directly from raw detector arrays (OpenCV DNN, ONNX, MediaPipe, TFLite)."""
        dets: List[Detection] = []
        n = min(len(boxes), len(labels))
        for i in range(n):
            conf = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            if conf < min_confidence:
                continue
            b = boxes[i]
            bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3])) if len(b) >= 4 else None
            tid = str(track_ids[i]) if track_ids is not None and i < len(track_ids) and track_ids[i] is not None else None
            dets.append(Detection(label=str(labels[i]), confidence=conf, bbox=bbox, track_id=tid, zone=zone))
        return cls(
            event_type=event_type,
            camera_id=camera_id,
            zone=zone,
            frame_id=frame_id,
            detections=dets,
            severity_hint=severity_hint,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_yolo(
        cls,
        yolo_result: Any,
        *,
        event_type: str = "frame_detections",
        camera_id: str = "cam_0",
        zone: str = "default",
        frame_id: Optional[int] = None,
        min_confidence: float = 0.25,
        severity_hint: CVSeverity = CVSeverity.LOW,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CVEvent":
        """Adapt an Ultralytics YOLO Result object (duck-typed; zero `ultralytics` import needed)."""
        names = getattr(yolo_result, "names", None) or {}
        boxes_obj = getattr(yolo_result, "boxes", None)
        dets: List[Detection] = []
        if boxes_obj is not None:
            xyxy_seq = getattr(boxes_obj, "xyxy", [])
            conf_seq = getattr(boxes_obj, "conf", [])
            cls_seq = getattr(boxes_obj, "cls", [])
            id_seq = getattr(boxes_obj, "id", None)
            if hasattr(xyxy_seq, "tolist"):
                xyxy_seq = xyxy_seq.tolist()
            if hasattr(conf_seq, "tolist"):
                conf_seq = conf_seq.tolist()
            if hasattr(cls_seq, "tolist"):
                cls_seq = cls_seq.tolist()
            if id_seq is not None and hasattr(id_seq, "tolist"):
                id_seq = id_seq.tolist()
            for idx in range(len(xyxy_seq)):
                conf = float(conf_seq[idx]) if idx < len(conf_seq) else 1.0
                if conf < min_confidence:
                    continue
                cls_idx = int(cls_seq[idx]) if idx < len(cls_seq) else 0
                label = str(names.get(cls_idx, cls_idx)) if isinstance(names, dict) else (
                    str(names[cls_idx]) if isinstance(names, (list, tuple)) and cls_idx < len(names) else str(cls_idx)
                )
                coords = xyxy_seq[idx]
                bbox = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])) if len(coords) >= 4 else None
                tid = str(int(id_seq[idx])) if id_seq is not None and idx < len(id_seq) and id_seq[idx] is not None else None
                dets.append(Detection(label=label, confidence=conf, bbox=bbox, track_id=tid, zone=zone))
        return cls(
            event_type=event_type,
            camera_id=camera_id,
            zone=zone,
            frame_id=frame_id,
            detections=dets,
            severity_hint=severity_hint,
            metadata=dict(metadata or {}),
        )
