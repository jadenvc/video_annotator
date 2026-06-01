#!/usr/bin/env python3
"""
CTAG Annotator — Scene label annotation for underwater video.

Stripped-down version of EdgeTAM Feature Tracker.  All torch / SAM2 / hydra /
timm dependencies have been removed.  This file contains only:
  - Scene label (environment + substrate) annotation bars
  - Read-only feature list (features loaded from a saved JSON, if any)
  - Video playback and frame seeking
  - Bottom timeline scrubber
  - JSON export and matplotlib analysis plots

Usage:
    python run_annotator.py video.mp4
    python run_annotator.py                      # opens file-picker dialog
    python run_annotator.py /path/to/frames/     # --frames-dir mode
    python run_annotator.py /path/ --frames-dir
"""

import sys
import os
import json
import argparse
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Qt compatibility: prefer PyQt6, fall back to PyQt5
# --------------------------------------------------------------------------
try:
    from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog,
    )
    _QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog,
    )
    _QT6 = False

# --------------------------------------------------------------------------
# Modern UI constants
# --------------------------------------------------------------------------

ENV_OPTIONS = ["Mangrove", "Reef", "Open Ocean"]
SUBSTRATE_OPTIONS = ["Sand", "Rock", "Coral", "Seagrass", "Rubble", "Mud", "Other"]

FEATURE_TYPES = ["Fish", "Shark", "Ray", "Turtle", "Diver", "Other"]

BEHAVIOR_OPTIONS = ["Swimming", "Resting", "Foraging", "Interacting"]

# Modern-ish palette (RGB) for tracked features
FEATURE_COLORS = [
    (16, 185, 129),   # emerald-500
    (245, 158, 11),   # amber-500
    (14, 165, 233),   # sky-500
    (217, 70, 239),   # fuchsia-500
    (132, 204, 22),   # lime-500
    (6, 182, 212),    # cyan-500
    (244, 63, 94),    # rose-500
    (34, 197, 94),    # green-500
]

APP_QSS = r"""
QMainWindow { background: #F6F7FB; }
* { font-family: "Inter","SF Pro Text","SF Pro Display","Segoe UI","Helvetica","Arial"; font-size: 12px; color: #0F172A; }

QFrame#Card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
}

QLabel#Title {
  font-size: 16px;
  font-weight: 600;
  color: #0F172A;
}

QLabel#Subtle {
  color: #64748B;
}

QLabel#BarTitle {
  color: #334155;
  font-weight: 600;
}

QLineEdit, QComboBox, QDoubleSpinBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 10px;
  padding: 7px 10px;
}

QComboBox::drop-down {
  border: none;
  width: 26px;
}

QCheckBox { spacing: 10px; }
QRadioButton { spacing: 8px; }

QPushButton {
  background: #111827;
  color: #FFFFFF;
  border: none;
  border-radius: 12px;
  padding: 9px 12px;
  font-weight: 600;
}
QPushButton:hover { background: #0B1220; }
QPushButton:pressed { background: #030712; }
QPushButton:disabled { background: #9CA3AF; color: #F9FAFB; }

QPushButton#Secondary {
  background: #EEF2FF;
  color: #1E293B;
  border: 1px solid #C7D2FE;
}
QPushButton#Secondary:hover { background: #E0E7FF; }

QToolButton[segmented="true"] {
  background: #F1F5F9;
  border: 1px solid #E2E8F0;
  border-radius: 999px;
  padding: 7px 12px;
  font-weight: 600;
  color: #0F172A;
}
QToolButton[segmented="true"]:hover {
  background: #E2E8F0;
}
QToolButton[segmented="true"]:checked {
  background: #4F46E5;
  border: 1px solid #4F46E5;
  color: #FFFFFF;
}

QListWidget {
  background: transparent;
  border: none;
}
QListWidget::item {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 8px 10px;
  margin: 4px 0px;
}
QListWidget::item:selected {
  background: #EEF2FF;
  border: 1px solid #C7D2FE;
}

QGroupBox { border: none; }
"""

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class TrackedFeature:
    name: str
    feature_type: str
    init_frame: int
    end_frame: int
    init_type: str              # "point" | "bbox"
    init_coords: list           # [x,y] or [x1,y1,x2,y2]
    confidence_threshold: float
    color_idx: int = 0
    behavior: str = "Swimming"


class FeatureStore:
    """Holds every tracked feature together with its per-frame bboxes + scene labels."""

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 video_w: int, video_h: int):
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self.video_w = video_w
        self.video_h = video_h

        self.features: List[TrackedFeature] = []
        self.feature_masks: List[Dict[int, np.ndarray]] = []
        self.feature_bboxes: List[Dict[int, Tuple[int, int, int, int]]] = []
        self._next_color = 0

        # per-frame scene labels (persist until changed)
        self.env_per_frame: List[Optional[str]] = [None] * total_frames
        self.substrate_per_frame: List[Optional[str]] = [None] * total_frames

    def add_feature(self, feat: TrackedFeature,
                    masks: Dict[int, np.ndarray],
                    bboxes: Dict[int, Tuple[int, int, int, int]]):
        feat.color_idx = self._next_color
        self._next_color = (self._next_color + 1) % len(FEATURE_COLORS)
        self.features.append(feat)
        self.feature_masks.append(masks)
        self.feature_bboxes.append(bboxes)

    def remove_feature(self, idx: int):
        if 0 <= idx < len(self.features):
            self.features.pop(idx)
            self.feature_masks.pop(idx)
            self.feature_bboxes.pop(idx)

    def features_at(self, frame_idx: int):
        """(index, feature, mask-or-None, bbox-or-None) for every active feature."""
        out = []
        for i, feat in enumerate(self.features):
            if feat.init_frame <= frame_idx <= feat.end_frame:
                mask = self.feature_masks[i].get(frame_idx)
                bbox = self.feature_bboxes[i].get(frame_idx)
                out.append((i, feat, mask, bbox))
        return out

    def set_env(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.env_per_frame[frame_idx] = label

    def set_substrate(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.substrate_per_frame[frame_idx] = label

    @staticmethod
    def _compress_timeline(labels: List[Optional[str]]):
        """Run-length encode a per-frame label list into segments."""
        if not labels:
            return []
        segs = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                segs.append({"start": start, "end": i - 1, "value": cur})
                cur = labels[i]
                start = i
        segs.append({"start": start, "end": len(labels) - 1, "value": cur})
        return segs

    def save_json(self, path: str):
        data = {
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "video_w": self.video_w,
            "video_h": self.video_h,
            "scene": {
                "environment_segments": self._compress_timeline(self.env_per_frame),
                "substrate_segments": self._compress_timeline(self.substrate_per_frame),
            },
            "features": [],
        }
        for i, feat in enumerate(self.features):
            data["features"].append({
                "name": feat.name,
                "feature_type": feat.feature_type,
                "init_frame": feat.init_frame,
                "end_frame": feat.end_frame,
                "init_type": feat.init_type,
                "init_coords": feat.init_coords,
                "confidence_threshold": feat.confidence_threshold,
                "color_idx": feat.color_idx,
                "behavior": feat.behavior,
                "bboxes": {str(k): list(v) for k, v in self.feature_bboxes[i].items()},
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "FeatureStore":
        """Load a previously saved annotation JSON and return a populated FeatureStore."""
        with open(path) as f:
            data = json.load(f)

        store = cls(
            video_path=data.get("video_path", ""),
            fps=float(data.get("fps", 30.0)),
            total_frames=int(data.get("total_frames", 0)),
            video_w=int(data.get("video_w", 0)),
            video_h=int(data.get("video_h", 0)),
        )

        scene = data.get("scene", {})
        for seg in scene.get("environment_segments", []):
            for fi in range(int(seg["start"]), int(seg["end"]) + 1):
                store.set_env(fi, seg.get("value"))
        for seg in scene.get("substrate_segments", []):
            for fi in range(int(seg["start"]), int(seg["end"]) + 1):
                store.set_substrate(fi, seg.get("value"))

        for fd in data.get("features", []):
            feat = TrackedFeature(
                name=fd.get("name", "feature"),
                feature_type=fd.get("feature_type", "Other"),
                init_frame=int(fd.get("init_frame", 0)),
                end_frame=int(fd.get("end_frame", 0)),
                init_type=fd.get("init_type", "point"),
                init_coords=fd.get("init_coords", [0, 0]),
                confidence_threshold=float(fd.get("confidence_threshold", 0.5)),
                color_idx=int(fd.get("color_idx", 0)),
                behavior=fd.get("behavior", "Swimming"),
            )
            bboxes_raw = fd.get("bboxes", {})
            bboxes: Dict[int, Tuple[int, int, int, int]] = {
                int(k): tuple(v) for k, v in bboxes_raw.items()  # type: ignore[misc]
            }
            store.features.append(feat)
            store.feature_masks.append({})
            store.feature_bboxes.append(bboxes)
            store._next_color = (feat.color_idx + 1) % len(FEATURE_COLORS)

        return store


# --------------------------------------------------------------------------
# VideoLabel — displays frames; click/drag signals kept for possible future use
# --------------------------------------------------------------------------

class VideoLabel(QLabel):
    pointClicked = pyqtSignal(object)           # QPoint
    bboxDrawn = pyqtSignal(object, object)      # QPoint, QPoint

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._drawing = False
        self._start = None
        self._end = None
        self._is_drag = False

    def set_frame_pixmap(self, pix):
        self.setPixmap(pix)

    @staticmethod
    def _ev_pos(ev):
        if _QT6:
            return ev.position().toPoint()
        return ev.pos()

    def paintEvent(self, ev):
        super().paintEvent(ev)


# --------------------------------------------------------------------------
# Bottom Timeline Widget (habitat + features + timestamp scrubber)
# --------------------------------------------------------------------------

class AnnotationTimeline(QWidget):
    """
    Bottom timeline widget:
      - Habitat/Environment segments over time
      - Feature presence segments over time (color-coded)
      - Scrubber row with big knob + timestamp labels
    Click/drag anywhere to seek.
    """
    frameSelected = pyqtSignal(int)

    def __init__(self, store: FeatureStore, fps: float, total_frames: int, parent=None):
        super().__init__(parent)
        self.store = store
        self.fps = float(fps) if fps else 30.0
        self.total_frames = max(1, int(total_frames))
        self.current_frame = 0
        self._dragging = False

        # Size policy compatibility
        try:
            exp = QSizePolicy.Policy.Expanding
            fixed = QSizePolicy.Policy.Fixed
        except Exception:
            exp = QSizePolicy.Expanding
            fixed = QSizePolicy.Fixed

        self.setSizePolicy(exp, fixed)
        self.setMinimumHeight(130)
        self.setMouseTracking(True)

        # Colors for environment segments (soft)
        self._env_colors = {
            "Mangrove": QColor(16, 185, 129),
            "Reef": QColor(14, 165, 233),
            "Open Ocean": QColor(245, 158, 11),
            "Open-Ocean": QColor(245, 158, 11),
            None: QColor(203, 213, 225),
        }

    def set_current_frame(self, idx: int):
        self.current_frame = max(0, min(int(idx), self.total_frames - 1))
        self.update()

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        s = int(seconds + 0.5)
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        if h > 0:
            return f"{h}:{m:02d}:{ss:02d}"
        return f"{m}:{ss:02d}"

    def _bar_geometry(self):
        label_w = 150
        pad_l = 14
        pad_r = 14
        x0 = pad_l + label_w
        x1 = self.width() - pad_r
        w = max(1, x1 - x0)
        return label_w, x0, x1, w

    def _frame_to_x(self, frame_idx: int, x0: int, w: int) -> int:
        if self.total_frames <= 1:
            return x0
        t = frame_idx / (self.total_frames - 1)
        return int(x0 + t * w)

    def _x_to_frame(self, x: int, x0: int, w: int) -> int:
        x = max(x0, min(x0 + w, x))
        if w <= 0 or self.total_frames <= 1:
            return 0
        t = (x - x0) / w
        return int(round(t * (self.total_frames - 1)))

    def mousePressEvent(self, ev):
        btn = Qt.MouseButton.LeftButton if _QT6 else Qt.LeftButton
        if ev.button() != btn:
            return
        _, x0, _, w = self._bar_geometry()
        pos = ev.position().toPoint() if _QT6 else ev.pos()
        if pos.x() < x0 - 10:
            return
        self._dragging = True
        f = self._x_to_frame(pos.x(), x0, w)
        self.set_current_frame(f)
        self.frameSelected.emit(f)

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        _, x0, _, w = self._bar_geometry()
        pos = ev.position().toPoint() if _QT6 else ev.pos()
        f = self._x_to_frame(pos.x(), x0, w)
        self.set_current_frame(f)
        self.frameSelected.emit(f)

    def mouseReleaseEvent(self, ev):
        self._dragging = False

    def paintEvent(self, ev):
        p = QPainter(self)
        if _QT6:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        else:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.TextAntialiasing, True)

        label_w, x0, x1, w = self._bar_geometry()
        top = 10
        row_h = 22
        gap = 14

        env_y = top
        animals_y = env_y + row_h + gap
        scrub_y = animals_y + row_h + gap

        # Background
        p.fillRect(self.rect(), QColor(246, 247, 251))

        def draw_left_label(text: str, y: int):
            p.setPen(QColor(15, 23, 42))
            f = p.font()
            f.setPointSize(14)
            f.setBold(True)
            p.setFont(f)
            p.drawText(14, y + row_h - 5, text)

        def draw_bar_outline(y: int):
            p.setPen(QPen(QColor(148, 163, 184), 1))
            p.setBrush(QColor(255, 255, 255))
            p.drawRoundedRect(x0, y, w, row_h, 6, 6)

        # --- Environment row
        draw_left_label("Environment", env_y)
        draw_bar_outline(env_y)

        env_labels = self.store.env_per_frame if self.store is not None else [None] * self.total_frames
        segs = FeatureStore._compress_timeline(env_labels)

        for seg in segs:
            v = seg.get("value", None)
            c = self._env_colors.get(v, QColor(203, 213, 225))
            s = int(seg["start"])
            e = int(seg["end"])

            # Use total_frames for segment boundaries so (e+1) maps cleanly
            x_s = int(x0 + (s / max(1, self.total_frames)) * w)
            x_e = int(x0 + ((e + 1) / max(1, self.total_frames)) * w)
            if x_e <= x_s:
                continue

            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 70))
            p.drawRoundedRect(x_s, env_y + 1, max(1, x_e - x_s), row_h - 2, 6, 6)

            label = v if v is not None else "Unknown"
            if (x_e - x_s) > 90:
                p.setPen(QColor(15, 23, 42))
                f = p.font()
                f.setBold(False)
                f.setPointSize(11)
                p.setFont(f)
                p.drawText(x_s + 8, env_y + row_h - 6, label)

        # --- Other animals row
        draw_left_label("Other animals:", animals_y)
        draw_bar_outline(animals_y)

        intervals = []
        if self.store is not None:
            for feat in self.store.features:
                intervals.append((feat.init_frame, feat.end_frame, feat))
        intervals.sort(key=lambda t: (t[0], t[1]))

        # Greedy lane assignment (compact)
        max_lanes = 3
        lane_ends = [-1] * max_lanes
        assigned = []
        for s, e, feat in intervals:
            lane = None
            for li in range(max_lanes):
                if s > lane_ends[li]:
                    lane = li
                    lane_ends[li] = e
                    break
            if lane is None:
                lane = 0
            assigned.append((s, e, feat, lane))

        lane_h = max(6, (row_h - 4) // max_lanes)

        for s, e, feat, lane in assigned:
            x_s = int(x0 + (s / max(1, self.total_frames)) * w)
            x_e = int(x0 + ((e + 1) / max(1, self.total_frames)) * w)
            if x_e <= x_s:
                continue

            c_rgb = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)]
            c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])

            y = animals_y + 2 + lane * lane_h
            h = lane_h - 2

            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 170))
            p.drawRoundedRect(x_s, y, max(2, x_e - x_s), h, 4, 4)

            # Name tag above row (like screenshot)
            if (x_e - x_s) > 65:
                p.setPen(QColor(15, 23, 42))
                f = p.font()
                f.setPointSize(10)
                f.setBold(False)
                p.setFont(f)
                # clamp to not go out of bounds
                tx = min(max(x_s + 2, x0), x0 + w - 60)
                p.drawText(tx, animals_y - 3, feat.name)

        # --- Scrubber row (timestamps)
        draw_bar_outline(scrub_y)

        # mid guide line
        p.setPen(QPen(QColor(148, 163, 184), 2))
        mid_y = scrub_y + row_h // 2
        p.drawLine(x0 + 8, mid_y, x0 + w - 8, mid_y)

        # Cursor line + knob
        cx = self._frame_to_x(self.current_frame, x0, w)
        p.setPen(QPen(QColor(15, 23, 42), 2))
        p.drawLine(cx, env_y, cx, scrub_y + row_h)

        p.setBrush(QColor(15, 23, 42))
        p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
        p.drawEllipse(cx - 14, mid_y - 14, 28, 28)

        # Time labels
        p.setPen(QColor(100, 116, 139))
        f = p.font()
        f.setPointSize(10)
        f.setBold(False)
        p.setFont(f)

        start_t = self._format_time(0)
        end_t = self._format_time((self.total_frames - 1) / max(1e-6, self.fps))
        cur_t = self._format_time(self.current_frame / max(1e-6, self.fps))

        p.drawText(x0, scrub_y + row_h + 16, start_t)
        end_w = p.fontMetrics().horizontalAdvance(end_t)
        p.drawText(x0 + w - end_w, scrub_y + row_h + 16, end_t)

        # current time above knob
        p.setPen(QColor(15, 23, 42))
        cur_w = p.fontMetrics().horizontalAdvance(cur_t)
        tx = max(x0, min(cx - cur_w // 2, x0 + w - cur_w))
        p.drawText(tx, scrub_y - 6, cur_t)

        # little "Frame counts" label like screenshot vibe
        p.setPen(QColor(100, 116, 139))
        p.drawText(x0, scrub_y + row_h + 32, "Frame counts")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, video_path: str, fps: float, total_frames: int,
                 video_w: int, video_h: int,
                 is_frame_dir: bool = False,
                 temp_dir: Optional[str] = None,
                 store_video_path: Optional[str] = None,
                 preloaded_store: Optional[FeatureStore] = None):
        super().__init__()
        self.setWindowTitle("CTAG Annotator")

        self.video_path = video_path
        self.is_frame_dir = is_frame_dir
        self.temp_dir = temp_dir

        self.fps = float(fps) or 30.0
        self.total_frames = int(total_frames)
        self.video_w = int(video_w)
        self.video_h = int(video_h)

        # For frame directories, load frame list
        if is_frame_dir:
            self.frame_files = sorted([
                os.path.join(video_path, f) for f in os.listdir(video_path)
                if f.lower().endswith(('.jpg', '.jpeg'))
            ], key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            self.cap = None
            # Trust the values passed in; re-derive total_frames from files if needed
            if self.total_frames == 0:
                self.total_frames = len(self.frame_files)
        else:
            self.frame_files = None
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

        self.current_frame_idx = 0
        self.playing = False

        # state — either use preloaded store or create fresh one
        store_path = store_video_path or video_path
        if preloaded_store is not None:
            self.store = preloaded_store
            # override path to match current video
            self.store.video_path = store_path
        else:
            self.store = FeatureStore(store_path, self.fps,
                                      self.total_frames, self.video_w, self.video_h)

        # environment/substrate selection (persist-until-changed)
        self._current_env = ENV_OPTIONS[0]
        self._current_substrate = SUBSTRATE_OPTIONS[0]

        # playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(max(1, int(1000 / self.fps)))

        self._build_ui()
        self.seek_to(0)

    # ------------------------------------------------------------------ UI
    def _make_segment_bar(self, title: str, options: List[str], on_select):
        row = QWidget()
        hl = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        lab = QLabel(title)
        lab.setObjectName("BarTitle")
        hl.addWidget(lab)

        try:
            from PyQt6.QtWidgets import QButtonGroup
        except ImportError:
            from PyQt5.QtWidgets import QButtonGroup

        group = QButtonGroup(self)
        group.setExclusive(True)
        btns = {}

        for opt in options[:10]:
            b = QToolButton()
            b.setText(opt)
            b.setCheckable(True)
            b.setProperty("segmented", True)
            b.clicked.connect(lambda checked, t=opt: on_select(t))
            group.addButton(b)
            btns[opt] = b
            hl.addWidget(b)

        hl.addStretch(1)
        row.setLayout(hl)
        return row, btns

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # -------------------------------- video display
        self.video_label = VideoLabel()
        align_flag = Qt.AlignmentFlag.AlignCenter if _QT6 else Qt.AlignCenter
        self.video_label.setAlignment(align_flag)
        self.video_label.setMinimumSize(QSize(740, 420))

        # Environment + Substrate bars (below video)
        self.env_bar, self._env_btns = self._make_segment_bar(
            "Environment:", ENV_OPTIONS, self._on_env_selected
        )
        self.substrate_bar, self._sub_btns = self._make_segment_bar(
            "Substrate:", SUBSTRATE_OPTIONS, self._on_substrate_selected
        )
        # initial highlight
        self._env_btns[self._current_env].setChecked(True)
        self._sub_btns[self._current_substrate].setChecked(True)

        video_col = QVBoxLayout()
        video_col.setSpacing(10)
        video_col.addWidget(self.video_label, stretch=1)
        video_col.addWidget(self.env_bar)
        video_col.addWidget(self.substrate_bar)

        video_wrap = QWidget()
        video_wrap.setLayout(video_col)

        # -------------------------------- right panel (feature list — read-only)
        # feature list (populated from JSON if available; cannot add without tracking)
        self.feat_list = QListWidget()

        fa = QHBoxLayout()
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("Secondary")
        del_btn.clicked.connect(self._delete_feature)
        ren_btn = QPushButton("Rename")
        ren_btn.setObjectName("Secondary")
        ren_btn.clicked.connect(self._rename_feature)
        fa.addWidget(del_btn)
        fa.addWidget(ren_btn)

        self.name_edit = QLineEdit("feature")

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_row.addWidget(QLabel("Name:"))
        name_row.addWidget(self.name_edit, stretch=1)

        # export
        exp_btn = QPushButton("Export Video")
        exp_btn.clicked.connect(self._export_video)
        json_btn = QPushButton("Save JSON")
        json_btn.setObjectName("Secondary")
        json_btn.clicked.connect(self._save_json)
        load_json_btn = QPushButton("Load JSON")
        load_json_btn.setObjectName("Secondary")
        load_json_btn.clicked.connect(self._load_json)

        # Pack into a "card"
        card_layout = QVBoxLayout()
        card_layout.addWidget(QLabel("Annotated Features (read-only):"))
        card_layout.addWidget(self.feat_list, stretch=1)
        card_layout.addLayout(name_row)
        card_layout.addLayout(fa)
        card_layout.addWidget(load_json_btn)
        card_layout.addWidget(exp_btn)
        card_layout.addWidget(json_btn)

        card = QFrame()
        card.setObjectName("Card")
        card_v = QVBoxLayout()
        card_v.setContentsMargins(14, 14, 14, 14)
        card_v.setSpacing(10)
        title = QLabel("Controls")
        title.setObjectName("Title")
        subtitle = QLabel("Label environment & substrate. Load a JSON to review features.")
        subtitle.setObjectName("Subtle")
        card_v.addWidget(title)
        card_v.addWidget(subtitle)
        card_v.addLayout(card_layout)
        card.setLayout(card_v)

        rw = QWidget()
        rw_l = QVBoxLayout()
        rw_l.setContentsMargins(0, 0, 0, 0)
        rw_l.addWidget(card)
        rw.setLayout(rw_l)
        rw.setMaximumWidth(320)

        # ---- playback controls ----
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setObjectName("Secondary")

        self.back_btn = QPushButton("◀")
        self.back_btn.setObjectName("Secondary")
        self.back_btn.clicked.connect(lambda: self.seek_to(self.current_frame_idx - 1))

        self.fwd_btn = QPushButton("▶")
        self.fwd_btn.setObjectName("Secondary")
        self.fwd_btn.clicked.connect(lambda: self.seek_to(self.current_frame_idx + 1))

        self.frame_lbl = QLabel()
        self.frame_lbl.setObjectName("Subtle")

        ctrls = QHBoxLayout()
        ctrls.setSpacing(8)
        ctrls.addWidget(self.back_btn)
        ctrls.addWidget(self.play_btn)
        ctrls.addWidget(self.fwd_btn)
        ctrls.addWidget(self.frame_lbl)
        ctrls.addStretch(1)

        # ---- bottom timeline scrubber ----
        self.timeline = AnnotationTimeline(self.store, self.fps, self.total_frames)
        self.timeline.frameSelected.connect(self.seek_to)

        # ---- assemble ----
        top = QHBoxLayout()
        top.setSpacing(14)
        top.addWidget(video_wrap, stretch=3)
        top.addWidget(rw, stretch=1)

        lay = QVBoxLayout()
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        lay.addLayout(top)
        lay.addWidget(self.timeline)
        lay.addLayout(ctrls)
        central.setLayout(lay)

        self.statusBar().showMessage("Ready.")
        self._refresh_features()

    # ------------------------------------------------------------ env/sub
    def _on_env_selected(self, label: str):
        self._current_env = label
        self.store.set_env(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Environment: {label}")

    def _on_substrate_selected(self, label: str):
        self._current_substrate = label
        self.store.set_substrate(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Substrate: {label}")

    def _ensure_scene_labels(self, idx: int):
        # Persist-until-changed behavior: if current frame has no label, inherit current selection.
        if self.store.env_per_frame[idx] is None:
            self.store.set_env(idx, self._current_env)
        else:
            self._current_env = self.store.env_per_frame[idx]  # keep UI consistent

        if self.store.substrate_per_frame[idx] is None:
            self.store.set_substrate(idx, self._current_substrate)
        else:
            self._current_substrate = self.store.substrate_per_frame[idx]

        # Sync highlight state
        if self._current_env in self._env_btns:
            self._env_btns[self._current_env].setChecked(True)
        if self._current_substrate in self._sub_btns:
            self._sub_btns[self._current_substrate].setChecked(True)

    # ------------------------------------------------------- feature list
    def _refresh_features(self):
        self.feat_list.clear()
        for i, f in enumerate(self.store.features):
            c = FEATURE_COLORS[f.color_idx % len(FEATURE_COLORS)]
            it = QListWidgetItem(f"{f.feature_type} • {f.name}  [{f.behavior}]  (f{f.init_frame}–f{f.end_frame})")
            user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
            it.setData(user_role, i)
            it.setForeground(QColor(*c))
            self.feat_list.addItem(it)

    def _delete_feature(self):
        it = self.feat_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        idx = it.data(user_role)
        if idx is not None:
            self.store.remove_feature(int(idx))
            self._refresh_features()
            self._display_frame(self.current_frame_idx)
            if hasattr(self, "timeline"):
                self.timeline.update()

    def _rename_feature(self):
        it = self.feat_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        idx = it.data(user_role)
        name = (self.name_edit.text() or "").strip()
        if idx is not None and name and 0 <= idx < len(self.store.features):
            self.store.features[idx].name = name
            self._refresh_features()
            self._display_frame(self.current_frame_idx)
            if hasattr(self, "timeline"):
                self.timeline.update()

    def _load_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load annotation JSON", "", "JSON (*.json)")
        if not p:
            return
        try:
            loaded = FeatureStore.load_json(p)
            # Merge scene labels and features into current store
            for fi in range(min(len(loaded.env_per_frame), self.store.total_frames)):
                if loaded.env_per_frame[fi] is not None:
                    self.store.env_per_frame[fi] = loaded.env_per_frame[fi]
                if loaded.substrate_per_frame[fi] is not None:
                    self.store.substrate_per_frame[fi] = loaded.substrate_per_frame[fi]
            for feat, masks, bboxes in zip(loaded.features, loaded.feature_masks, loaded.feature_bboxes):
                self.store.features.append(feat)
                self.store.feature_masks.append(masks)
                self.store.feature_bboxes.append(bboxes)
            self._refresh_features()
            if hasattr(self, "timeline"):
                self.timeline.update()
            self._display_frame(self.current_frame_idx)
            self.statusBar().showMessage(f"Loaded {p}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    # ----------------------------------------------------------- coord map
    def _map(self, lx: int, ly: int) -> Optional[Tuple[int, int]]:
        """Label-widget coords → video-frame coords."""
        pix = self.video_label.pixmap()
        if pix is None:
            return None
        lw, lh = self.video_label.width(), self.video_label.height()
        pw, ph = pix.width(), pix.height()
        ox, oy = (lw - pw) // 2, (lh - ph) // 2
        x, y = lx - ox, ly - oy
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None
        fx = max(0, min(int(x * self.video_w / pw), self.video_w - 1))
        fy = max(0, min(int(y * self.video_h / ph), self.video_h - 1))
        return (fx, fy)

    # ------------------------------------------------------------- playback
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        self.timer.start() if self.playing else self.timer.stop()

    def _tick(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
            return
        self.seek_to(self.current_frame_idx + 1)

    def seek_to(self, idx: int):
        idx = max(0, min(int(idx), self.total_frames - 1))
        self.current_frame_idx = idx
        self._ensure_scene_labels(idx)
        self._display_frame(idx)

        if hasattr(self, "timeline"):
            self.timeline.set_current_frame(idx)

        self.frame_lbl.setText(
            f"Frame {idx}/{self.total_frames - 1} • {self._current_env} • {self._current_substrate}"
        )

    # -------------------------------------------------------------- display
    def _read_frame(self, idx):
        if self.is_frame_dir:
            if 0 <= idx < len(self.frame_files):
                return cv2.imread(self.frame_files[idx])
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def _display_frame(self, idx):
        frame = self._read_frame(idx)
        if frame is None:
            return
        preview = frame.copy()

        # overlay committed features (bboxes only — no SAM2 masks)
        for _, feat, mask, bbox in self.store.features_at(idx):
            bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
            self._draw_mask(preview, mask, bgr, f"{feat.feature_type}: {feat.name}", bbox)

        # scene text (anti-aliased)
        env = self.store.env_per_frame[idx] or self._current_env
        sub = self.store.substrate_per_frame[idx] or self._current_substrate
        text = f"{env} • {sub}"
        cv2.putText(preview, text, (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, text, (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w,
                      QImage.Format.Format_RGB888 if _QT6 else QImage.Format_RGB888)
        aspect_mode = Qt.AspectRatioMode.KeepAspectRatio if _QT6 else Qt.KeepAspectRatio
        transform_mode = Qt.TransformationMode.SmoothTransformation if _QT6 else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), aspect_mode, transform_mode)
        self.video_label.set_frame_pixmap(pix)

    @staticmethod
    def _draw_mask(img, mask, bgr_color, label, bbox):
        if mask is not None and mask.any():
            bgr = np.array(bgr_color, dtype=np.float64)
            img[mask] = (bgr * 0.35 + img[mask].astype(np.float64) * 0.65).astype(np.uint8)
            mu8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, bgr_color, 2, lineType=cv2.LINE_AA)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2, lineType=cv2.LINE_AA)
            if label:
                cv2.putText(img, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv2.LINE_AA)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()

    # -------------------------------------------------------------- export
    def _save_plots(self, base_path: str) -> list:
        """Generate and save three analysis PNG plots alongside the exported video."""
        try:
            import matplotlib.pyplot as plt
            plt.switch_backend("agg")   # safe even if pyplot was already imported
            from collections import Counter, defaultdict
        except ImportError:
            QMessageBox.warning(self, "Plots skipped",
                                "matplotlib is not installed.\n"
                                "Run: pip install matplotlib")
            return []
        except Exception as e:
            QMessageBox.warning(self, "Plots skipped", f"Could not initialise matplotlib:\n{e}")
            return []

        try:
            return self._render_plots(base_path, plt, Counter, defaultdict)
        except Exception as e:
            QMessageBox.warning(self, "Plot error", f"Error generating plots:\n{e}")
            return []

    def _render_plots(self, base_path, plt, Counter, defaultdict) -> list:

        store  = self.store
        fps    = max(float(self.fps), 1e-6)
        total  = self.total_frames
        saved  = []

        # ── palette matching the app UI ───────────────────────────────────
        HAB_COLORS = {
            "Mangrove":   "#10B981",
            "Reef":       "#0EA5E9",
            "Open Ocean": "#F59E0B",
        }
        DEFAULT_CLR = "#94A3B8"

        def _style(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=15, fontweight="bold", pad=14, color="#0F172A")
            ax.set_xlabel(xlabel, fontsize=11, labelpad=8, color="#334155")
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8, color="#334155")
            ax.tick_params(axis="both", colors="#475569", labelsize=10)
            ax.set_facecolor("#F8FAFC")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)

        env_counts = Counter(lab for lab in store.env_per_frame if lab)

        # ── 1. Time spent in each habitat ─────────────────────────────────
        if env_counts:
            envs = list(env_counts.keys())
            secs = [env_counts[e] / fps for e in envs]
            cols = [HAB_COLORS.get(e, DEFAULT_CLR) for e in envs]
            peak = max(secs)

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#FFFFFF")
            bars = ax.bar(envs, secs, color=cols, width=0.45,
                          edgecolor="#FFFFFF", linewidth=1.8, zorder=3)
            for bar, s in zip(bars, secs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        s + peak * 0.025,
                        f"{s:.1f}s",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="600", color="#1E293B")
            _style(ax, "Time Spent in Each Habitat", "Habitat", "Duration (seconds)")
            plt.tight_layout(pad=1.8)
            out = f"{base_path}_1_habitat_time.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
            plt.close(fig)
            saved.append(out)

        # ── 2. Animal sightings per minute by habitat ─────────────────────
        hab_sightings = Counter()
        for feat in store.features:
            habs = [store.env_per_frame[f]
                    for f in range(feat.init_frame, min(feat.end_frame + 1, total))
                    if store.env_per_frame[f]]
            if habs:
                hab_sightings[Counter(habs).most_common(1)[0][0]] += 1

        all_envs = sorted(set(list(env_counts.keys()) + list(hab_sightings.keys())))
        if all_envs:
            rates = []
            for e in all_envs:
                mins = env_counts.get(e, 0) / fps / 60.0
                rates.append(hab_sightings.get(e, 0) / mins if mins > 0 else 0.0)
            cols = [HAB_COLORS.get(e, DEFAULT_CLR) for e in all_envs]
            peak = max(rates) if rates else 0.0

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#FFFFFF")
            bars = ax.bar(all_envs, rates, color=cols, width=0.45,
                          edgecolor="#FFFFFF", linewidth=1.8, zorder=3)
            for bar, r in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        r + max(peak * 0.025, 0.005),
                        f"{r:.2f}",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="600", color="#1E293B")
            _style(ax, "Animal Sightings per Minute by Habitat",
                   "Habitat", "Sightings / minute")
            plt.tight_layout(pad=1.8)
            out = f"{base_path}_2_sightings_per_min.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
            plt.close(fig)
            saved.append(out)

        # ── 3. Behavior distribution by habitat ───────────────────────────
        beh_hab = defaultdict(Counter)
        for feat in store.features:
            habs = [store.env_per_frame[f]
                    for f in range(feat.init_frame, min(feat.end_frame + 1, total))
                    if store.env_per_frame[f]]
            if habs:
                dom = Counter(habs).most_common(1)[0][0]
                beh_hab[feat.behavior][dom] += 1

        behaviors = sorted(beh_hab.keys())
        plot_habs = sorted(set(h for c in beh_hab.values() for h in c))

        if behaviors and plot_habs:
            n_habs  = len(plot_habs)
            group_w = 0.65
            bar_w   = group_w / max(n_habs, 1)

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#FFFFFF")
            x = np.arange(len(behaviors))
            for j, hab in enumerate(plot_habs):
                counts = [beh_hab[b].get(hab, 0) for b in behaviors]
                offset = (j - n_habs / 2 + 0.5) * bar_w
                ax.bar(x + offset, counts, width=bar_w * 0.88,
                       label=hab, color=HAB_COLORS.get(hab, DEFAULT_CLR),
                       edgecolor="#FFFFFF", linewidth=1.5, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([b.capitalize() for b in behaviors], fontsize=11)
            leg = ax.legend(title="Habitat", fontsize=9, title_fontsize=10,
                            framealpha=0.95, edgecolor="#E2E8F0")
            leg.get_frame().set_linewidth(0.8)
            _style(ax, "Behavior Distribution by Habitat",
                   "Behavior", "Number of Animals")
            plt.tight_layout(pad=1.8)
            out = f"{base_path}_3_behavior_habitat.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
            plt.close(fig)
            saved.append(out)

        # ── 4. Sightings over time with habitat shading ───────────────────────
        import matplotlib.ticker as mticker
        from matplotlib.transforms import blended_transform_factory

        total_secs = total / fps

        # Rolling window: target ~60 s, clamped between 15 s and ⅛ of video
        window_s      = max(15.0, min(60.0, total_secs / 8))
        window_frames = max(1, min(int(window_s * fps), total))

        # Count new feature appearances per frame
        new_sightings = np.zeros(total, dtype=float)
        for feat in store.features:
            if 0 <= feat.init_frame < total:
                new_sightings[feat.init_frame] += 1

        # Rolling sum → rate per minute
        kernel       = np.ones(window_frames)
        rolling_sum  = np.convolve(new_sightings, kernel, mode="same")
        window_mins  = window_frames / fps / 60.0
        rolling_rate = rolling_sum / max(window_mins, 1e-9)

        times = np.arange(total) / fps   # x axis in seconds

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("#FFFFFF")

        # ── habitat background shading ──
        segs = FeatureStore._compress_timeline(store.env_per_frame)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            c   = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.axvspan(t_s, t_e, color=c, alpha=0.13, zorder=0, lw=0)
            if seg["start"] > 0:
                ax.axvline(t_s, color=c, linewidth=1.0,
                           linestyle="--", alpha=0.55, zorder=1)

        # ── sightings line + fill ──
        ax.plot(times, rolling_rate, color="#4F46E5", linewidth=2.2, zorder=3)
        ax.fill_between(times, rolling_rate, alpha=0.14, color="#4F46E5", zorder=2)

        ax.set_ylim(bottom=0)
        ax.set_xlim(0, times[-1] if len(times) > 1 else 1)

        # ── habitat labels pinned to top of axes ──
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            seg_secs = t_e - t_s
            if seg_secs < total_secs * 0.04:
                continue
            mid = (t_s + t_e) / 2
            c   = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.text(mid, 0.97, v, ha="center", va="top",
                    transform=trans, fontsize=9,
                    fontweight="600", color=c)

        # ── time axis formatting ──
        def _fmt_time(x, _):
            s = int(max(0, x))
            return f"{s // 60}:{s % 60:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(10, integer=False))

        _style(ax,
               f"Animal Sightings Over Time  (rolling {window_s:.0f} s window)",
               "Time (m:ss)", "Sightings / minute")

        plt.tight_layout(pad=1.8)
        out = f"{base_path}_4_sightings_over_time.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        saved.append(out)

        return saved

    def _save_json(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save annotations",
            self.video_path + ".annotated.json", "JSON (*.json)")
        if p:
            self.store.save_json(p)
            self.statusBar().showMessage(f"Saved {p}")

    def _export_video(self):
        if self.playing:
            self.toggle_play()
        p, _ = QFileDialog.getSaveFileName(
            self, "Export annotated video",
            self.video_path + ".annotated.mp4", "MP4 (*.mp4)")
        if not p:
            return
        try:
            self._render(p)
            base = os.path.splitext(p)[0]
            saved_plots = self._save_plots(base)
            msg = f"Exported:\n{p}"
            if saved_plots:
                msg += "\n\nPlots saved:\n" + "\n".join(
                    os.path.basename(pp) for pp in saved_plots
                )
            QMessageBox.information(self, "Done", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _render(self, out_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(out_path, fourcc, self.fps, (self.video_w, self.video_h))

        def draw_scene(frame, fidx):
            env = self.store.env_per_frame[fidx] or self._current_env
            sub = self.store.substrate_per_frame[fidx] or self._current_substrate
            text = f"{env} • {sub}"
            cv2.putText(frame, text, (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if self.is_frame_dir:
            for fidx in range(self.total_frames):
                frame = cv2.imread(self.frame_files[fidx])
                if frame is None:
                    continue
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, f"{feat.feature_type}: {feat.name}", bbox)
                draw_scene(frame, fidx)
                w.write(frame)
        else:
            cap = cv2.VideoCapture(self.video_path)
            fidx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, f"{feat.feature_type}: {feat.name}", bbox)
                draw_scene(frame, fidx)
                w.write(frame)
                fidx += 1
            cap.release()

        w.release()

    # -------------------------------------------------------------- close
    def closeEvent(self, ev):
        if self.cap is not None:
            self.cap.release()
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        ev.accept()


# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------

def _extract_frames(video_path: str, out_dir: str) -> int:
    """Extract every frame from video_path as a JPEG into out_dir using cv2.

    Files are named 00000.jpg, 00001.jpg … so MainWindow's frame-dir loader
    can sort them by integer stem. Returns the number of frames written.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    modal = Qt.WindowModality.ApplicationModal if _QT6 else Qt.ApplicationModal

    dlg = QProgressDialog("Extracting frames...", None, 0, max(total, 1))
    dlg.setWindowTitle("Loading video")
    dlg.setMinimumDuration(0)
    dlg.setWindowModality(modal)
    dlg.setValue(0)
    dlg.show()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(
            os.path.join(out_dir, f"{idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        idx += 1
        dlg.setValue(idx)
        QApplication.processEvents()

    cap.release()
    dlg.close()
    return idx


def _enable_hidpi():
    """Best-effort crispness across Qt5/Qt6."""
    try:
        if _QT6:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        else:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="CTAG Annotator — Scene label annotation (no torch/SAM2 required)"
    )
    parser.add_argument("video", nargs="?",
                        help="Path to video file OR directory of JPEG frames")
    parser.add_argument("--frames-dir", action="store_true",
                        help="Input is a directory of JPEG frames (not a video file)")
    args = parser.parse_args()

    _enable_hidpi()
    app = QApplication(sys.argv)

    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setStyleSheet(APP_QSS)

    f = QFont()
    f.setPointSize(12)
    app.setFont(f)

    video_path = args.video
    is_frame_dir = args.frames_dir
    temp_frames_dir = None
    original_video_path = None

    if is_frame_dir:
        # --- frames directory mode (explicit --frames-dir flag) ---
        if not video_path:
            video_path = QFileDialog.getExistingDirectory(None, "Select frames directory")
        if not video_path:
            sys.exit(0)
        if not os.path.isdir(video_path):
            sys.exit(f"Frames directory not found: {video_path}")
        jpegs = [fn for fn in os.listdir(video_path) if fn.lower().endswith(('.jpg', '.jpeg'))]
        if not jpegs:
            sys.exit(f"No JPEG files found in: {video_path}")
        print(f"Found {len(jpegs)} JPEG frames")

        # Derive video properties from first frame
        first_file = sorted(jpegs, key=lambda fn: int(os.path.splitext(fn)[0]))[0]
        first = cv2.imread(os.path.join(video_path, first_file))
        if first is None:
            sys.exit(f"Cannot read first frame: {first_file}")
        video_h, video_w = first.shape[:2]
        total_frames = len(jpegs)
        fps = 30.0

    else:
        # --- video file mode: show dialog, then extract frames internally ---
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(
                None, "Open video", "",
                "Video (*.mp4 *.MP4 *.mov *.avi *.mkv)"
            )
        if not video_path:
            sys.exit(0)
        if not os.path.isfile(video_path):
            sys.exit(f"Video not found: {video_path}")

        # Read video metadata before extracting
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        original_video_path = video_path
        temp_frames_dir = tempfile.mkdtemp(prefix="ctag_frames_")
        print(f"Extracting frames to: {temp_frames_dir}")
        try:
            n_frames = _extract_frames(video_path, temp_frames_dir)
        except Exception as e:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            sys.exit(f"Frame extraction failed: {e}")
        if n_frames == 0:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            sys.exit(f"No frames could be read from: {video_path}")
        print(f"Extracted {n_frames} frames")
        total_frames = n_frames
        video_path = temp_frames_dir
        is_frame_dir = True

    print(f"Opening annotator: {video_path}  ({total_frames} frames @ {fps:.2f} fps)")

    win = MainWindow(
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
        video_w=video_w,
        video_h=video_h,
        is_frame_dir=is_frame_dir,
        temp_dir=temp_frames_dir,
        store_video_path=original_video_path,
    )
    win.resize(1320, 860)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
