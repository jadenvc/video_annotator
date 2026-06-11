#!/usr/bin/env python3
"""
CTAG Annotator — Scene label annotation for underwater video.

Stripped-down version of EdgeTAM Feature Tracker.  All torch / SAM2 / hydra /
timm dependencies have been removed.  This file contains only:
  - Scene label (behavior + habitat) annotation bars
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
import re
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
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QShortcut, QKeySequence
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog, QSpinBox, QCompleter,
    )
    _QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QShortcut, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog, QSpinBox, QCompleter,
    )
    _QT6 = False

# --------------------------------------------------------------------------
# Modern UI constants
# --------------------------------------------------------------------------

BEHAVIOR_OPTIONS = [
    "1 - Burst swimming",
    "2 - Low-speed swimming",
    "3 - Stationary (alone)",
    "4 - Foraging",
    "5 - Following a shark",
    "6 - Shark following tagged",
    "7 - Parallel swimming",
    "8 - Brief interaction",
    "9 - Stationary with 1+ sharks",
]

HABITAT_OPTIONS = ["Mangrove", "Rocky reef", "Sandy bottom", "Gravel", "Mud"]

SPECIES_CATEGORIES = {
    "Shark": [
        # Species
        "Lemon shark (Negaprion brevirostris)",
        "Pacific nurse shark (Ginglymostoma unami)",
        # Genus
        "Ginglymostoma sp.",
        "Negaprion sp.",
        # Family
        "Carcharhinidae",
        "Ginglymostomatidae",
    ],
    "Ray": [
        # Species
        "Longtail stingray (Hypanus longus)",
        "Pacific chupare stingray (Styracura pacifica)",
        "Longtail butterfly ray (Gymnura crebipunctata)",
        "Munk's devil ray (Mobula munkiana)",
        "Pacific eagle ray (Aetobatus laticeps)",
        "Speckled guitarfish (Pseudobatos glaucostygma)",
        "Prahl's guitarfish (Pseudobatos prahli)",
        "Pacific cownose ray (Rhinoptera steindachneri)",
        "Leopard round ray (Urobatis pardalis)",
        "Chilean round ray (Urotrygon chilensis)",
        # Genus
        "Aetobatus sp.",
        "Gymnura sp.",
        "Hypanus sp.",
        "Mobula sp.",
        "Pseudobatos sp.",
        "Rhinoptera sp.",
        "Styracura sp.",
        "Urobatis sp.",
        "Urotrygon sp.",
        # Family
        "Dasyatidae",
        "Gymnuridae",
        "Mobulidae",
        "Myliobatidae",
        "Rhinobatidae",
        "Rhinopteridae",
        "Urotrygonidae",
    ],
    "Teleost fish": [
        # Species
        "Convict surgeonfish (Acanthurus triostegus)",
        "Yellowfin surgeonfish (Acanthurus xanthopterus)",
        "Razor surgeonfish (Prionurus laticlavius)",
        "Dovi's cardinalfish (Apogon dovii)",
        "Chinese trumpetfish (Aulostomus chinensis)",
        "Finescale triggerfish (Balistes polylepis)",
        "Stone triggerfish (Pseudobalistes naufragium)",
        "Orangeside triggerfish (Sufflamen verres)",
        "Pacific agujon needlefish (Tylosurus fodiator)",
        "Large-banded blenny (Ophioblennius steindachneri)",
        "Sabertooth blenny (Plagiotremus azaleus)",
        "Leopard flounder (Bothus leopardinus)",
        "African pompano (Alectis ciliaris)",
        "Green jack (Caranx caballus)",
        "Pacific crevalle jack (Caranx caninus)",
        "Bigeye trevally (Caranx sexfasciatus)",
        "Rainbow runner (Elagatis bipinnulata)",
        "Golden trevally (Gnathanodon speciosus)",
        "Almaco jack (Seriola rivoliana)",
        "Paita pompano (Trachinotus paitensis)",
        "Blackblotch pompano (Trachinotus kennedyi)",
        "Gafftopsail pompano (Trachinotus rhodopus)",
        "Blackfin snook (Centropomus medius)",
        "Roughcheek barnacle blenny (Acanthemblemaria exilispinus)",
        "Hancock's blenny (Acanthemblemaria hancocki)",
        "Delta pikeblenny (Chaenopsis deltarrhis)",
        "Threebanded butterflyfish (Chaetodon humeralis)",
        "Barberfish (Johnrandallia nigrirostris)",
        "Giant hawkfish (Cirrhitus rivulatus)",
        "Coral hawkfish (Cirrhitichthys oxycephalus)",
        "Dolphinfish / mahi-mahi (Coryphaena hippurus)",
        "Spotfin burrfish (Chilomycterus reticulatus)",
        "Balloonfish (Diodon holocanthus)",
        "Spotted porcupinefish (Diodon hystrix)",
        "Pacific ladyfish (Elops affinis)",
        "Pacific spadefish (Chaetodipterus zonatus)",
        "Bluespotted cornetfish (Fistularia commersonii)",
        "Pacific flagfin mojarra (Eucinostomus currani)",
        "Slender mojarra (Gerres simillimus)",
        "Redlight goby (Coryphopterus urospilus)",
        "Spotted cleaner goby (Elacatinus puncticulatus)",
        "Burrito grunt (Anisotremus interruptus)",
        "Panamic porkfish (Anisotremus taeniatus)",
        "Yellowspotted grunt (Haemulon flaviguttatum)",
        "Spottail grunt (Haemulon maculicauda)",
        "Scudder's grunt (Haemulon scudderii)",
        "Greybar grunt (Haemulon sexfasciatum)",
        "Chere-chere grunt (Haemulon steindachneri)",
        "Goldeneye grunt (Microlepidotus brevipinnis)",
        "Panamic soldierfish (Myripristis leiognathus)",
        "Tinsel squirrelfish (Sargocentron suborbitale)",
        "Cortez sea chub (Kyphosus elegans)",
        "Blue-bronze sea chub (Kyphosus ocyurus)",
        "Mexican hogfish (Bodianus diplotaenia)",
        "Wounded wrasse (Halichoeres chierchiae)",
        "Chameleon wrasse (Halichoeres dispilus)",
        "Spinster wrasse (Halichoeres nicholsi)",
        "Bicolor wrasse (Halichoeres notospilus)",
        "Peacock razorfish (Iniistius pavo)",
        "Rockmover wrasse (Novaculichthys taeniourus)",
        "Sunset wrasse (Thalassoma grammaticum)",
        "Cortez rainbow wrasse (Thalassoma lucasanum)",
        "Panamic fanged blenny (Malacoctenus sudensis)",
        "Barred pargo (Hoplopagrus guentherii)",
        "Mullet snapper (Lutjanus aratus)",
        "Yellow snapper (Lutjanus argentiventris)",
        "Colorado snapper (Lutjanus colorado)",
        "Spotted rose snapper (Lutjanus guttatus)",
        "Golden snapper (Lutjanus inermis)",
        "Pacific dog snapper (Lutjanus novemfasciatus)",
        "Shortjaw tilefish (Malacanthus brevirostris)",
        "Unicorn filefish (Aluterus monoceros)",
        "Scrawled filefish (Aluterus scriptus)",
        "White mullet (Mugil curema)",
        "Mexican goatfish (Mulloidichthys dentatus)",
        "Bigscale goatfish (Pseudupeneus grandisquamis)",
        "Snowflake moray (Echidna nebulosa)",
        "Freckled moray (Echidna nocturna)",
        "Zebra moray (Gymnomuraena zebra)",
        "Chestnut moray (Gymnothorax castaneus)",
        "Dovi's moray (Gymnothorax dovii)",
        "Yellowmargin moray (Gymnothorax flavimarginatus)",
        "Panamic green moray (Gymnothorax panamensis)",
        "Undulated moray (Gymnothorax undulatus)",
        "Argus moray (Muraena argus)",
        "Hourglass moray (Muraena clepsydra)",
        "Jeweled moray (Muraena lentiginosa)",
        "Tiger reef-eel (Scuticaria tigrina)",
        "Roosterfish (Nematistius pectoralis)",
        "Tiger snake eel (Myrichthys tigrinus)",
        "Pacific snake eel (Ophichthus triseralis)",
        "Notched-fin snake eel (Quassiremus nothochir)",
        "Whitespotted boxfish (Ostracion meleagris)",
        "King angelfish (Holacanthus passer)",
        "Cortez angelfish (Pomacanthus zonipectus)",
        "Night sergeant (Abudefduf concolor)",
        "Panamic sergeant major (Abudefduf troschelii)",
        "Scissortail chromis (Azurina atrilobata)",
        "Bumphead damselfish (Microspathodon bairdii)",
        "Giant damselfish (Microspathodon dorsalis)",
        "Acapulco damselfish (Stegastes acapulcoensis)",
        "Beaubrummel (Stegastes flavilatus)",
        "Blue-barred parrotfish (Scarus ghobban)",
        "Greenblotch parrotfish (Scarus perrico)",
        "Ember parrotfish (Scarus rubroviolaceus)",
        "Black skipjack (Euthynnus lineatus)",
        "Pacific sierra (Scomberomorus sierra)",
        "Stone scorpionfish (Scorpaena mystes)",
        "Pacific mutton hamlet (Alphestes immaculatus)",
        "Pacific graysby (Cephalopholis colonus)",
        "Creolefish (Paranthias panamensis)",
        "Leather bass (Dermatolepis dermatolepis)",
        "Spotted grouper (Epinephelus analogus)",
        "Starry grouper (Epinephelus labriformis)",
        "Pacific goliath grouper (Epinephelus quinquefasciatus)",
        "Broomtail grouper (Mycteroperca xenarcha)",
        "Mottled soapfish (Rypticus bicolor)",
        "Flag serrano (Serranus psittacinus)",
        "Pacific porgy (Calamus brachysomus)",
        "Mexican barracuda (Sphyraena ensis)",
        "Bluestripe pipefish (Doryrhamphus excisus)",
        "Calico lizardfish (Synodus lacertinus)",
        "Whitespotted puffer (Arothron hispidus)",
        "Guineafowl puffer (Arothron meleagris)",
        "Spotted sharpnose puffer (Canthigaster punctatissima)",
        "Bullseye puffer (Sphoeroides annulatus)",
        "Lobed puffer (Sphoeroides lobatus)",
        "Lucilla's triplefin (Axoclinus lucillae)",
        # Genus
        "Abudefduf sp.",
        "Acanthemblemaria sp.",
        "Acanthurus sp.",
        "Alectis sp.",
        "Alphestes sp.",
        "Aluterus sp.",
        "Anisotremus sp.",
        "Apogon sp.",
        "Arothron sp.",
        "Aulostomus sp.",
        "Axoclinus sp.",
        "Azurina sp.",
        "Balistes sp.",
        "Bodianus sp.",
        "Bothus sp.",
        "Calamus sp.",
        "Canthigaster sp.",
        "Caranx sp.",
        "Centropomus sp.",
        "Cephalopholis sp.",
        "Chaenopsis sp.",
        "Chaetodipterus sp.",
        "Chaetodon sp.",
        "Chilomycterus sp.",
        "Cirrhitichthys sp.",
        "Cirrhitus sp.",
        "Coryphaena sp.",
        "Coryphopterus sp.",
        "Dermatolepis sp.",
        "Diodon sp.",
        "Doryrhamphus sp.",
        "Echidna sp.",
        "Elacatinus sp.",
        "Elagatis sp.",
        "Elops sp.",
        "Epinephelus sp.",
        "Eucinostomus sp.",
        "Euthynnus sp.",
        "Fistularia sp.",
        "Gerres sp.",
        "Gnathanodon sp.",
        "Gymnomuraena sp.",
        "Gymnothorax sp.",
        "Haemulon sp.",
        "Halichoeres sp.",
        "Holacanthus sp.",
        "Hoplopagrus sp.",
        "Iniistius sp.",
        "Johnrandallia sp.",
        "Kyphosus sp.",
        "Lutjanus sp.",
        "Malacanthus sp.",
        "Malacoctenus sp.",
        "Microlepidotus sp.",
        "Microspathodon sp.",
        "Mugil sp.",
        "Mulloidichthys sp.",
        "Muraena sp.",
        "Myrichthys sp.",
        "Mycteroperca sp.",
        "Myripristis sp.",
        "Nematistius sp.",
        "Novaculichthys sp.",
        "Ophioblennius sp.",
        "Ophichthus sp.",
        "Ostracion sp.",
        "Paranthias sp.",
        "Plagiotremus sp.",
        "Pomacanthus sp.",
        "Prionurus sp.",
        "Pseudobalistes sp.",
        "Pseudupeneus sp.",
        "Quassiremus sp.",
        "Rypticus sp.",
        "Sargocentron sp.",
        "Scarus sp.",
        "Scomberomorus sp.",
        "Scorpaena sp.",
        "Scuticaria sp.",
        "Seriola sp.",
        "Serranus sp.",
        "Sphoeroides sp.",
        "Sphyraena sp.",
        "Stegastes sp.",
        "Sufflamen sp.",
        "Synodus sp.",
        "Thalassoma sp.",
        "Trachinotus sp.",
        "Tylosurus sp.",
        # Family
        "Acanthuridae",
        "Apogonidae",
        "Aulostomidae",
        "Balistidae",
        "Belonidae",
        "Blenniidae",
        "Bothidae",
        "Carangidae",
        "Centropomidae",
        "Chaenopsidae",
        "Chaetodontidae",
        "Cirrhitidae",
        "Coryphaenidae",
        "Diodontidae",
        "Elopidae",
        "Ephippidae",
        "Fistulariidae",
        "Gerreidae",
        "Gobiidae",
        "Haemulidae",
        "Holocentridae",
        "Kyphosidae",
        "Labridae",
        "Labrisomidae",
        "Lutjanidae",
        "Malacanthidae",
        "Monacanthidae",
        "Mugilidae",
        "Mullidae",
        "Muraenidae",
        "Nematistiidae",
        "Ophichthidae",
        "Ostraciidae",
        "Pomacanthidae",
        "Pomacentridae",
        "Scaridae",
        "Scombridae",
        "Scorpaenidae",
        "Serranidae",
        "Sparidae",
        "Sphyraenidae",
        "Syngnathidae",
        "Synodontidae",
        "Tetraodontidae",
        "Tripterygiidae",
    ],
    "Sea turtle": [
        # Species
        "Hawksbill sea turtle (Eretmochelys imbricata)",
        "Olive ridley sea turtle (Lepidochelys olivacea)",
        "Green sea turtle (Chelonia mydas)",
        # Genus
        "Chelonia sp.",
        "Eretmochelys sp.",
        "Lepidochelys sp.",
        # Family
        "Cheloniidae",
    ],
    "Other": [],
}
ALL_SPECIES = [s for species in SPECIES_CATEGORIES.values() for s in species]

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
    init_frame: int
    end_frame: int
    init_type: str              # "point" | "bbox"
    init_coords: list           # [x,y] or [x1,y1,x2,y2]
    color_idx: int = 0
    count: int = 1
    species_category: str = ""  # Shark / Ray / Teleost fish / Sea turtle / Other
    species: str = ""           # specific species name, optional


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
        self.behavior_per_frame: List[Optional[str]] = [None] * total_frames
        self.habitat_per_frame: List[Optional[str]] = [None] * total_frames

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

    def set_behavior(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.behavior_per_frame[frame_idx] = label

    def set_habitat(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.habitat_per_frame[frame_idx] = label

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
                "behavior_segments": self._compress_timeline(self.behavior_per_frame),
                "habitat_segments": self._compress_timeline(self.habitat_per_frame),
            },
            "features": [],
        }
        for i, feat in enumerate(self.features):
            data["features"].append({
                "name": feat.name,
                "init_frame": feat.init_frame,
                "end_frame": feat.end_frame,
                "init_type": feat.init_type,
                "init_coords": feat.init_coords,
                "color_idx": feat.color_idx,
                "count": feat.count,
                "species_category": feat.species_category,
                "species": feat.species,
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
        # Support both new field names and old field names for backward compatibility
        for seg in scene.get("behavior_segments", scene.get("environment_segments", [])):
            for fi in range(int(seg["start"]), int(seg["end"]) + 1):
                store.set_behavior(fi, seg.get("value"))
        for seg in scene.get("habitat_segments", scene.get("substrate_segments", [])):
            for fi in range(int(seg["start"]), int(seg["end"]) + 1):
                store.set_habitat(fi, seg.get("value"))

        for fd in data.get("features", []):
            feat = TrackedFeature(
                name=fd.get("name", "feature"),
                init_frame=int(fd.get("init_frame", 0)),
                end_frame=int(fd.get("end_frame", 0)),
                init_type=fd.get("init_type", "point"),
                init_coords=fd.get("init_coords", [0, 0]),
                color_idx=int(fd.get("color_idx", 0)),
                count=int(fd.get("count", 1)),
                species_category=fd.get("species_category", fd.get("feature_type", "")),
                species=fd.get("species", ""),
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
        self._bbox_mode = False

    def set_bbox_mode(self, on: bool):
        self._bbox_mode = on
        cursor = Qt.CursorShape.CrossCursor if (_QT6 and on) else (Qt.CrossCursor if on else Qt.ArrowCursor) if not _QT6 else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def set_frame_pixmap(self, pix):
        self.setPixmap(pix)

    @staticmethod
    def _ev_pos(ev):
        if _QT6:
            return ev.position().toPoint()
        return ev.pos()

    def mousePressEvent(self, ev):
        btn = Qt.MouseButton.LeftButton if _QT6 else Qt.LeftButton
        if self._bbox_mode and ev.button() == btn:
            self._drawing = True
            self._start = self._ev_pos(ev)
            self._end = self._start
            self._is_drag = False
            self.update()

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._end = self._ev_pos(ev)
            if (abs(self._end.x() - self._start.x()) > 5 or
                    abs(self._end.y() - self._start.y()) > 5):
                self._is_drag = True
            self.update()

    def mouseReleaseEvent(self, ev):
        btn = Qt.MouseButton.LeftButton if _QT6 else Qt.LeftButton
        if ev.button() == btn and self._drawing:
            self._drawing = False
            self._end = self._ev_pos(ev)
            if self._is_drag:
                self.bboxDrawn.emit(self._start, self._end)
            self._start = self._end = None
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._drawing and self._start and self._end and self._is_drag:
            p = QPainter(self)
            pen_style = Qt.PenStyle.SolidLine if _QT6 else Qt.SolidLine
            p.setPen(QPen(QColor(79, 70, 229), 2, pen_style))
            x1, y1 = self._start.x(), self._start.y()
            x2, y2 = self._end.x(), self._end.y()
            p.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))


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
        self.setMinimumHeight(160)
        self.setMouseTracking(True)

        # Colors for behavior segments — cycle through FEATURE_COLORS palette
        self._behavior_colors = {
            opt: QColor(*FEATURE_COLORS[i % len(FEATURE_COLORS)])
            for i, opt in enumerate(BEHAVIOR_OPTIONS)
        }
        self._behavior_colors[None] = QColor(203, 213, 225)

        # Colors for habitat segments
        self._habitat_colors = {
            "Mangrove": QColor(16, 185, 129),
            "Rocky reef": QColor(14, 165, 233),
            "Sandy bottom": QColor(245, 158, 11),
            "Gravel": QColor(156, 163, 175),
            "Mud": QColor(120, 100, 80),
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
        gap = 10

        behavior_y = top
        habitat_y = behavior_y + row_h + gap
        animals_y = habitat_y + row_h + gap
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

        def draw_segment_row(row_y, labels, color_dict):
            draw_bar_outline(row_y)
            the_segs = FeatureStore._compress_timeline(
                labels if labels is not None else [None] * self.total_frames
            )
            for seg in the_segs:
                v = seg.get("value", None)
                c = color_dict.get(v, QColor(203, 213, 225))
                s = int(seg["start"])
                e = int(seg["end"])
                x_s = int(x0 + (s / max(1, self.total_frames)) * w)
                x_e = int(x0 + ((e + 1) / max(1, self.total_frames)) * w)
                if x_e <= x_s:
                    continue
                p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
                p.setBrush(QColor(c.red(), c.green(), c.blue(), 70))
                p.drawRoundedRect(x_s, row_y + 1, max(1, x_e - x_s), row_h - 2, 6, 6)
                lbl = v if v is not None else "Unknown"
                if (x_e - x_s) > 90:
                    p.setPen(QColor(15, 23, 42))
                    ff = p.font()
                    ff.setBold(False)
                    ff.setPointSize(11)
                    p.setFont(ff)
                    p.drawText(x_s + 8, row_y + row_h - 6, lbl)

        # --- Behavior row
        draw_left_label("Behavior", behavior_y)
        behavior_labels = self.store.behavior_per_frame if self.store is not None else None
        draw_segment_row(behavior_y, behavior_labels, self._behavior_colors)

        # --- Habitat row
        draw_left_label("Habitat", habitat_y)
        habitat_labels = self.store.habitat_per_frame if self.store is not None else None
        draw_segment_row(habitat_y, habitat_labels, self._habitat_colors)

        # --- Other animals row
        draw_left_label("Features:", animals_y)
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

            # Name tag above row
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
        p.drawLine(cx, behavior_y, cx, scrub_y + row_h)

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
        self._labeling = True  # False = navigate-only, no label writes

        # state — either use preloaded store or create fresh one
        store_path = store_video_path or video_path
        if preloaded_store is not None:
            self.store = preloaded_store
            # override path to match current video
            self.store.video_path = store_path
        else:
            self.store = FeatureStore(store_path, self.fps,
                                      self.total_frames, self.video_w, self.video_h)

        # behavior/habitat selection (persist-until-changed)
        self._current_behavior = BEHAVIOR_OPTIONS[0]
        self._current_habitat = HABITAT_OPTIONS[0]
        self._type_counters: Dict[str, int] = {}

        # playback speed
        self._play_speed = 1.0

        # playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(max(1, int(1000 / self.fps)))

        self._build_ui()
        self.seek_to(0)

    # ------------------------------------------------------------------ UI
    def _make_segment_bar(self, title: str, options: List[str], on_select, rows: int = 1):
        container = QWidget()
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        try:
            from PyQt6.QtWidgets import QButtonGroup
        except ImportError:
            from PyQt5.QtWidgets import QButtonGroup

        group = QButtonGroup(self)
        group.setExclusive(True)
        btns = {}

        # Split options across rows evenly
        import math
        per_row = math.ceil(len(options) / rows)
        chunks = [options[i:i+per_row] for i in range(0, len(options), per_row)]

        for r, chunk in enumerate(chunks):
            hl = QHBoxLayout()
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            if r == 0:
                lab = QLabel(title)
                lab.setObjectName("BarTitle")
                hl.addWidget(lab)
            else:
                spacer = QLabel("")
                spacer.setFixedWidth(60)
                hl.addWidget(spacer)
            for opt in chunk:
                b = QToolButton()
                b.setText(opt)
                b.setCheckable(True)
                b.setProperty("segmented", True)
                b.clicked.connect(lambda checked, t=opt: on_select(t))
                group.addButton(b)
                btns[opt] = b
                hl.addWidget(b)
            hl.addStretch(1)
            row_w = QWidget()
            row_w.setLayout(hl)
            outer.addWidget(row_w)

        container.setLayout(outer)
        return container, btns

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # -------------------------------- video display
        self.video_label = VideoLabel()
        align_flag = Qt.AlignmentFlag.AlignCenter if _QT6 else Qt.AlignCenter
        self.video_label.setAlignment(align_flag)
        self.video_label.setMinimumSize(QSize(740, 420))

        # Behavior + Habitat bars (below video)
        self.behavior_bar, self._behavior_btns = self._make_segment_bar(
            "Behavior:", BEHAVIOR_OPTIONS, self._on_behavior_selected, rows=2
        )
        self.habitat_bar, self._habitat_btns = self._make_segment_bar(
            "Habitat:", HABITAT_OPTIONS, self._on_habitat_selected
        )
        # initial highlight
        self._behavior_btns[self._current_behavior].setChecked(True)
        self._habitat_btns[self._current_habitat].setChecked(True)

        video_col = QVBoxLayout()
        video_col.setSpacing(10)
        video_col.addWidget(self.video_label, stretch=1)
        video_col.addWidget(self.behavior_bar)
        video_col.addWidget(self.habitat_bar)

        video_wrap = QWidget()
        video_wrap.setLayout(video_col)

        # -------------------------------- right panel (feature list — read-only)
        # feature list (populated from JSON if available; cannot add without tracking)
        self.feat_list = QListWidget()
        self.feat_list.currentItemChanged.connect(self._on_feature_selected)

        fa = QHBoxLayout()
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("Secondary")
        del_btn.clicked.connect(self._delete_feature)
        ren_btn = QPushButton("Update")
        ren_btn.setObjectName("Secondary")
        ren_btn.clicked.connect(self._update_feature)
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
        plots_btn = QPushButton("Export Plots")
        plots_btn.setObjectName("Secondary")
        plots_btn.clicked.connect(self._export_plots_only)
        excel_btn = QPushButton("Export Excel")
        excel_btn.setObjectName("Secondary")
        excel_btn.clicked.connect(self._export_excel)
        json_btn = QPushButton("Save JSON")
        json_btn.setObjectName("Secondary")
        json_btn.clicked.connect(self._save_json)
        load_json_btn = QPushButton("Load JSON")
        load_json_btn.setObjectName("Secondary")
        load_json_btn.clicked.connect(self._load_json)

        # Category combo
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Shark", "Ray", "Teleost fish", "Sea turtle", "Other"])

        category_row = QHBoxLayout()
        category_row.addWidget(QLabel("Category:"))
        category_row.addWidget(self.category_combo, stretch=1)

        # Species combo (editable with autocomplete)
        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.addItem("")
        self.species_combo.addItems(SPECIES_CATEGORIES.get("Shark", []))
        completer = QCompleter(SPECIES_CATEGORIES.get("Shark", ALL_SPECIES))
        try:
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        except AttributeError:
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.species_combo.setCompleter(completer)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)

        species_row = QHBoxLayout()
        species_row.addWidget(QLabel("Species:"))
        species_row.addWidget(self.species_combo, stretch=1)

        # Count spinbox
        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(9999)
        self.count_spin.setValue(1)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Count:"))
        count_row.addWidget(self.count_spin, stretch=1)

        # Draw bbox toggle button
        self.bbox_btn = QPushButton("Draw Bbox")
        self.bbox_btn.setCheckable(True)
        self.bbox_btn.toggled.connect(self._bbox_mode_toggled)

        # Wire bbox signal
        self.video_label.bboxDrawn.connect(self._on_bbox)

        # Pack into a "card"
        card_layout = QVBoxLayout()
        card_layout.addLayout(category_row)
        card_layout.addLayout(species_row)
        card_layout.addLayout(count_row)
        card_layout.addLayout(name_row)
        card_layout.addWidget(self.bbox_btn)
        card_layout.addWidget(QLabel("Annotated Features:"))
        card_layout.addWidget(self.feat_list, stretch=1)
        card_layout.addLayout(fa)
        card_layout.addWidget(load_json_btn)
        card_layout.addWidget(exp_btn)
        card_layout.addWidget(plots_btn)
        card_layout.addWidget(excel_btn)
        card_layout.addWidget(json_btn)

        card = QFrame()
        card.setObjectName("Card")
        card_v = QVBoxLayout()
        card_v.setContentsMargins(14, 14, 14, 14)
        card_v.setSpacing(10)
        title = QLabel("Controls")
        title.setObjectName("Title")
        subtitle = QLabel("Label behavior & habitat. Draw bboxes to annotate features on individual frames.")
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

        # Speed combo
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)

        self.label_mode_btn = QPushButton("Labeling: ON")
        self.label_mode_btn.setCheckable(True)
        self.label_mode_btn.setChecked(True)
        self.label_mode_btn.setObjectName("Secondary")
        self.label_mode_btn.setStyleSheet(
            "QPushButton:checked { background:#16A34A; color:#FFFFFF; border:none; }"
            "QPushButton:!checked { background:#6B7280; color:#FFFFFF; border:none; }"
        )
        self.label_mode_btn.clicked.connect(self._toggle_label_mode)

        ctrls = QHBoxLayout()
        ctrls.setSpacing(8)
        ctrls.addWidget(self.back_btn)
        ctrls.addWidget(self.play_btn)
        ctrls.addWidget(self.fwd_btn)
        ctrls.addWidget(self.frame_lbl)
        ctrls.addStretch(1)
        ctrls.addWidget(self.label_mode_btn)
        ctrls.addWidget(QLabel("Speed:"))
        ctrls.addWidget(self.speed_combo)

        # ---- bottom timeline scrubber ----
        self.timeline = AnnotationTimeline(self.store, self.fps, self.total_frames)
        self.timeline.frameSelected.connect(self._on_slider_seek)

        # ---- keyboard shortcuts for frame stepping ----
        QShortcut(
            QKeySequence(Qt.Key.Key_Left if _QT6 else Qt.Key_Left), self
        ).activated.connect(lambda: self.seek_to(self.current_frame_idx - 1))
        QShortcut(
            QKeySequence(Qt.Key.Key_Right if _QT6 else Qt.Key_Right), self
        ).activated.connect(lambda: self.seek_to(self.current_frame_idx + 1))

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
    def _bbox_mode_toggled(self, on: bool):
        self.video_label.set_bbox_mode(on)
        self.statusBar().showMessage("Draw a bbox on the video to annotate a feature." if on else "")

    def _on_bbox(self, p1, p2):
        m1 = self._map(p1.x(), p1.y())
        m2 = self._map(p2.x(), p2.y())
        if m1 is None or m2 is None:
            return
        x1, y1 = min(m1[0], m2[0]), min(m1[1], m2[1])
        x2, y2 = max(m1[0], m2[0]), max(m1[1], m2[1])
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        species_category = self.category_combo.currentText() or "Other"
        species = self.species_combo.currentText().strip()
        count = self.count_spin.value()
        name = (self.name_edit.text() or "").strip()
        if not name:
            n = self._type_counters.get(species_category, 1)
            name = f"{species_category.lower().replace(' ', '_')}_{n}"
            self._type_counters[species_category] = n + 1
            self.name_edit.setText(name)

        fidx = self.current_frame_idx
        feat = TrackedFeature(
            name=name,
            init_frame=fidx,
            end_frame=fidx,
            init_type="bbox",
            init_coords=[x1, y1, x2, y2],
            count=count,
            species_category=species_category,
            species=species,
        )
        self.store.add_feature(feat, {}, {fidx: (x1, y1, x2, y2)})
        self._refresh_features()
        self._display_frame(fidx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        if species:
            self.statusBar().showMessage(
                f"Added {species_category} ({species}) ×{count} '{name}' on frame {fidx}."
            )
        else:
            self.statusBar().showMessage(
                f"Added {species_category} ×{count} '{name}' on frame {fidx}."
            )
        # clear name field so next annotation gets a fresh auto-name
        self.name_edit.clear()
        # auto-disarm so the mode state is always explicit (arm → draw → off)
        self.bbox_btn.setChecked(False)

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

    def _on_behavior_selected(self, label: str):
        self._current_behavior = label
        self.store.set_behavior(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Behavior: {label}")

    def _on_habitat_selected(self, label: str):
        self._current_habitat = label
        self.store.set_habitat(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Habitat: {label}")

    def _on_category_changed(self, category: str):
        self.species_combo.clear()
        self.species_combo.addItem("")  # blank = no species selected
        species_list = SPECIES_CATEGORIES.get(category, [])
        if not species_list:
            self.species_combo.addItems(ALL_SPECIES)
        else:
            self.species_combo.addItems(species_list)
        # reset completer
        completer = QCompleter(SPECIES_CATEGORIES.get(category, ALL_SPECIES))
        try:
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        except AttributeError:
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.species_combo.setCompleter(completer)

    def _ensure_scene_labels(self, idx: int):
        stored_beh = self.store.behavior_per_frame[idx]
        stored_hab = self.store.habitat_per_frame[idx]

        if not self.playing:
            # Seek/drag: update _current_* from stored values so the UI shows
            # what was annotated here, and future playback continues from it.
            if stored_beh is not None:
                self._current_behavior = stored_beh
            if stored_hab is not None:
                self._current_habitat = stored_hab
        # During playback: _current_* stays fixed; _tick writes it to each frame.

        if self._current_behavior in self._behavior_btns:
            self._behavior_btns[self._current_behavior].setChecked(True)
        if self._current_habitat in self._habitat_btns:
            self._habitat_btns[self._current_habitat].setChecked(True)

    # ------------------------------------------------------- feature list
    def _refresh_features(self):
        self.feat_list.clear()
        for i, feat in enumerate(self.store.features):
            c = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)]
            label = feat.species_category
            if feat.species:
                label = f"{feat.species_category}: {feat.species}"
            display = f"{label}  ×{feat.count}  [f{feat.init_frame}]  {feat.name}"
            it = QListWidgetItem(display)
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

    def _on_feature_selected(self, current, previous):
        """Populate edit fields when a feature is selected in the list."""
        if current is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        idx = current.data(user_role)
        if idx is None or not (0 <= idx < len(self.store.features)):
            return
        feat = self.store.features[idx]
        self.name_edit.setText(feat.name)
        self.count_spin.setValue(feat.count)
        # Set category combo
        cat_idx = self.category_combo.findText(feat.species_category)
        if cat_idx >= 0:
            self.category_combo.setCurrentIndex(cat_idx)
        # Set species combo (after category filter is applied)
        sp_idx = self.species_combo.findText(feat.species)
        if sp_idx >= 0:
            self.species_combo.setCurrentIndex(sp_idx)
        else:
            self.species_combo.setEditText(feat.species)

    def _update_feature(self):
        """Save name/species/count changes back to the selected feature."""
        it = self.feat_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        idx = it.data(user_role)
        if idx is None or not (0 <= int(idx) < len(self.store.features)):
            return
        feat = self.store.features[int(idx)]
        name = (self.name_edit.text() or "").strip()
        if name:
            feat.name = name
        feat.species_category = self.category_combo.currentText()
        feat.species = self.species_combo.currentText().strip()
        feat.count = self.count_spin.value()
        self._refresh_features()
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Updated '{feat.name}'.")

    def _load_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load annotation JSON", "", "JSON (*.json)")
        if not p:
            return
        try:
            loaded = FeatureStore.load_json(p)
            # Merge scene labels and features into current store
            for fi in range(min(len(loaded.behavior_per_frame), self.store.total_frames)):
                if loaded.behavior_per_frame[fi] is not None:
                    self.store.behavior_per_frame[fi] = loaded.behavior_per_frame[fi]
                if loaded.habitat_per_frame[fi] is not None:
                    self.store.habitat_per_frame[fi] = loaded.habitat_per_frame[fi]
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
        if self.playing:
            interval = max(1, int(33 / self._play_speed))
            self.timer.start(interval)
        else:
            self.timer.stop()

    def _toggle_label_mode(self, checked: bool):
        self._labeling = checked
        self.label_mode_btn.setText("Labeling: ON" if checked else "Labeling: OFF")

    def _on_speed_changed(self, text: str):
        self._play_speed = float(text.replace('x', ''))
        if self.timer.isActive():
            interval = max(1, int(33 / self._play_speed))
            self.timer.setInterval(interval)

    def _on_slider_seek(self, idx: int):
        if self._labeling and idx > self.current_frame_idx:
            for f in range(self.current_frame_idx, idx + 1):
                self.store.set_behavior(f, self._current_behavior)
                self.store.set_habitat(f, self._current_habitat)
            if hasattr(self, "timeline"):
                self.timeline.update()
        self.seek_to(idx)

    def _tick(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
            return
        if self._labeling:
            self.store.set_behavior(self.current_frame_idx, self._current_behavior)
            self.store.set_habitat(self.current_frame_idx, self._current_habitat)
        self.seek_to(self.current_frame_idx + 1)

    def seek_to(self, idx: int):
        idx = max(0, min(int(idx), self.total_frames - 1))
        self.current_frame_idx = idx
        self._ensure_scene_labels(idx)
        self._display_frame(idx)

        if hasattr(self, "timeline"):
            self.timeline.set_current_frame(idx)

        self.frame_lbl.setText(
            f"Frame {idx}/{self.total_frames - 1} • {self._current_behavior} • {self._current_habitat}"
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
            feat_label = feat.species_category
            if feat.species:
                feat_label = f"{feat.species_category}: {feat.species}"
            if feat.count > 1:
                feat_label = f"{feat_label} ×{feat.count}"
            self._draw_mask(preview, mask, bgr, feat_label, bbox)

        # scene text (anti-aliased)
        beh = self.store.behavior_per_frame[idx] or self._current_behavior
        hab = self.store.habitat_per_frame[idx] or self._current_habitat
        text = f"{beh} • {hab}"
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

    @staticmethod
    def _propagate(timeline):
        """Forward-fill a per-frame label list (None gaps inherit the last set value)."""
        out     = list(timeline)
        current = None
        for i, v in enumerate(out):
            if v is not None:
                current = v
            elif current is not None:
                out[i] = current
        return out

    def _render_plots(self, base_path, plt, Counter, defaultdict) -> list:
        from matplotlib.patches import Patch

        store = self.store
        total = self.total_frames
        saved = []

        # Forward-propagate so every frame inherits its nearest prior label.
        # This means a habitat annotated at frame 20 applies to frames 20-N
        # until the next annotation, matching the "persist until changed" UX.
        hab_timeline = self._propagate(store.habitat_per_frame)
        beh_timeline = self._propagate(store.behavior_per_frame)

        WHITE = "#FFFFFF"
        BG    = "#F8FAFC"

        HAB_COLORS = {
            "Mangrove":     "#10B981",
            "Rocky reef":   "#0EA5E9",
            "Sandy bottom": "#F59E0B",
            "Gravel":       "#9CA3AF",
            "Mud":          "#78644A",
        }
        BEH_COLORS = [
            "#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
            "#EC4899", "#8B5CF6", "#0EA5E9", "#14B8A6",
        ]
        CAT_COLORS = {
            "Shark":        "#3B82F6",
            "Ray":          "#06B6D4",
            "Teleost fish": "#10B981",
            "Sea turtle":   "#F59E0B",
            "Other":        "#94A3B8",
        }
        DEFAULT_CLR = "#94A3B8"

        def _donut(ax, short_labels, vals, colors, title, title_color="#0F172A",
                   center_text=""):
            _, _, autotexts = ax.pie(
                vals, labels=None, colors=colors,
                autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
                pctdistance=0.78,
                startangle=90, counterclock=False,
                wedgeprops={"linewidth": 3, "edgecolor": WHITE, "width": 0.52},
            )
            for at in autotexts:
                at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("#1E293B")
            ax.text(0, 0, center_text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#0F172A", linespacing=1.5)
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20,
                         color=title_color)

        def _hbar(ax, species, counts, colors, cat_for_sp, title,
                  title_color="#0F172A"):
            peak = max(counts)
            y    = np.arange(len(species))
            bars = ax.barh(y, counts, color=colors, height=0.58,
                           edgecolor=WHITE, linewidth=1.5, zorder=3)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_width() + peak * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        str(c), va="center", ha="left",
                        fontsize=10, fontweight="700", color="#1E293B")
            ax.set_yticks(y)
            ax.set_yticklabels(species, fontsize=10, color="#334155")
            ax.set_xlabel("Animals Observed", fontsize=11, labelpad=8, color="#334155")
            ax.set_title(title, fontsize=16, fontweight="bold",
                         pad=14, color=title_color)
            ax.set_xlim(0, peak * 1.18)
            ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", colors="#475569", labelsize=10)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#CBD5E1")
            seen_cats = sorted({cat_for_sp.get(sp, "Other") for sp in species})
            if len(seen_cats) > 1:
                handles = [Patch(facecolor=CAT_COLORS.get(c, DEFAULT_CLR),
                                 edgecolor=WHITE, label=c) for c in seen_cats]
                leg = ax.legend(handles=handles, title="Category",
                                fontsize=9, title_fontsize=10,
                                framealpha=0.95, edgecolor="#E2E8F0", loc="lower right")
                leg.get_frame().set_linewidth(0.8)

        def _fish_badge(ax, species, counts_map, cat_for_sp):
            fish = [s for s in species if cat_for_sp.get(s) == "Teleost fish"]
            if not fish:
                return
            n_sp  = len(fish)
            total_fish = sum(counts_map[s] for s in fish)
            label = f"Teleost fish: {n_sp} {'species' if n_sp != 1 else 'species'}  ·  {total_fish} individuals"
            ax.text(0.01, 0.99, label, transform=ax.transAxes,
                    fontsize=9.5, va="top", ha="left",
                    color=CAT_COLORS["Teleost fish"], fontweight="600",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5",
                              edgecolor="#6EE7B7", linewidth=1.2, alpha=0.92))

        def _feat_habitat(feat):
            # Use propagated timeline so a bbox at frame 5 picks up habitat
            # annotated at frame 1 even if frame 5 itself has no raw label.
            habs = [hab_timeline[f]
                    for f in range(feat.init_frame, min(feat.end_frame + 1, total))
                    if hab_timeline[f]]
            return Counter(habs).most_common(1)[0][0] if habs else None

        # ── build global species / cat maps and habitat list ─────────
        species_counts = Counter()
        cat_for_species = {}
        for feat in store.features:
            sp = feat.species or feat.species_category or "Unknown"
            species_counts[sp] += feat.count
            cat_for_species[sp] = feat.species_category or "Other"

        all_habitats = sorted(set(h for h in hab_timeline if h))

        # ── 1. Species observed — overall + per-habitat side by side ───
        if species_counts:
            # All species sorted by total count (ascending = top of chart = most)
            all_sp  = sorted(species_counts, key=lambda s: species_counts[s])
            n_sp    = len(all_sp)
            y_pos   = np.arange(n_sp)

            # Per-habitat species data (same _feat_habitat logic)
            sp_hab_counts = {}   # hab -> Counter
            sp_hab_cats   = {}   # hab -> {sp: cat}
            for hab in all_habitats:
                c, cat = Counter(), {}
                for feat in store.features:
                    if _feat_habitat(feat) == hab:
                        sp = feat.species or feat.species_category or "Unknown"
                        c[sp]   += feat.count
                        cat[sp]  = feat.species_category or "Other"
                sp_hab_counts[hab] = c
                sp_hab_cats[hab]   = cat

            n_cols  = 1 + len(all_habitats)
            height  = max(5.0, n_sp * 0.42 + 2.2)
            # First col needs room for species labels; rest just bars + count labels
            col_w   = [3.8] + [2.6] * len(all_habitats)
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(sum(col_w), height),
                sharey=True,
                gridspec_kw={"width_ratios": col_w},
            )
            if n_cols == 1:
                axes = [axes]
            fig.patch.set_facecolor(WHITE)
            fig.subplots_adjust(wspace=0.06)

            def _hbar_col(ax, counts_map, cat_map, title, title_color="#0F172A",
                          show_ylabels=True):
                counts = [counts_map.get(s, 0) for s in all_sp]
                colors = [CAT_COLORS.get(cat_map.get(s, "Other"), DEFAULT_CLR)
                          for s in all_sp]
                peak   = max((c for c in counts if c > 0), default=1)
                bars   = ax.barh(y_pos, counts, color=colors, height=0.62,
                                 edgecolor=WHITE, linewidth=1.2, zorder=3)
                for bar, c in zip(bars, counts):
                    if c > 0:
                        ax.text(bar.get_width() + peak * 0.04,
                                bar.get_y() + bar.get_height() / 2,
                                str(c), va="center", ha="left",
                                fontsize=8.5, fontweight="700", color="#1E293B")
                ax.set_yticks(y_pos)
                if show_ylabels:
                    ax.set_yticklabels(all_sp, fontsize=9, color="#334155")
                else:
                    ax.set_yticklabels([])
                ax.set_title(title, fontsize=11, fontweight="bold",
                             pad=10, color=title_color)
                ax.set_xlim(0, peak * 1.22)
                ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.7, zorder=0)
                ax.set_axisbelow(True)
                ax.set_facecolor(BG)
                ax.tick_params(axis="both", colors="#475569", labelsize=8.5)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#CBD5E1")
                ax.set_xlabel("Count", fontsize=9, labelpad=5, color="#334155")
                # Fish badge
                fish = [s for s in all_sp
                        if cat_map.get(s) == "Teleost fish"
                        and counts_map.get(s, 0) > 0]
                if fish:
                    n_f  = len(fish)
                    tot  = sum(counts_map[s] for s in fish)
                    lbl  = f"🐟 {n_f} sp · {tot} ind"
                    ax.text(0.97, 0.01, lbl, transform=ax.transAxes,
                            fontsize=8, va="bottom", ha="right",
                            color=CAT_COLORS["Teleost fish"], fontweight="600",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5",
                                      edgecolor="#6EE7B7", linewidth=1.0, alpha=0.9))

            # Overall column
            _hbar_col(axes[0], species_counts, cat_for_species,
                      "All Habitats", show_ylabels=True)

            # Per-habitat columns
            for i, hab in enumerate(all_habitats):
                _hbar_col(axes[i + 1],
                          sp_hab_counts[hab], sp_hab_cats[hab],
                          hab, title_color=HAB_COLORS.get(hab, DEFAULT_CLR),
                          show_ylabels=False)

            # Category legend on the overall column
            seen_cats = sorted({cat_for_species[s] for s in all_sp})
            if len(seen_cats) > 1:
                handles = [Patch(facecolor=CAT_COLORS.get(c, DEFAULT_CLR),
                                 edgecolor=WHITE, label=c) for c in seen_cats]
                leg = axes[0].legend(handles=handles, title="Category",
                                     fontsize=8, title_fontsize=9,
                                     framealpha=0.95, edgecolor="#E2E8F0",
                                     loc="lower right")
                leg.get_frame().set_linewidth(0.8)

            out = f"{base_path}_1_species.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── 2. Behavior composition — donut chart ─────────────────────
        beh_counts = Counter(lab for lab in beh_timeline if lab)
        if beh_counts:
            ordered = [b for b in BEHAVIOR_OPTIONS if b in beh_counts]
            ordered += [b for b in beh_counts if b not in ordered]
            vals    = [beh_counts[b] for b in ordered]
            colors  = []
            for b in ordered:
                try:
                    colors.append(BEH_COLORS[BEHAVIOR_OPTIONS.index(b) % len(BEH_COLORS)])
                except ValueError:
                    colors.append(DEFAULT_CLR)
            short = [b.split(" - ", 1)[-1] if " - " in b else b for b in ordered]
            pct   = 100 * sum(vals) / max(total, 1)
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor(WHITE)
            _donut(ax, short, vals, colors, "Behavior Composition",
                   center_text=f"{pct:.0f}%\nannotated")
            handles = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                       for c, lbl in zip(colors, short)]
            leg = ax.legend(handles=handles,
                            loc="lower center", bbox_to_anchor=(0.5, -0.14),
                            ncol=3, fontsize=9, framealpha=0.95, edgecolor="#E2E8F0",
                            columnspacing=1.0, handlelength=1.2)
            leg.get_frame().set_linewidth(0.8)
            plt.tight_layout(pad=1.5)
            out = f"{base_path}_2_behavior.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── 3. Habitat composition — donut chart ──────────────────────
        hab_counts = Counter(lab for lab in hab_timeline if lab)
        if hab_counts:
            ordered = sorted(hab_counts, key=lambda h: hab_counts[h], reverse=True)
            vals    = [hab_counts[h] for h in ordered]
            colors  = [HAB_COLORS.get(h, DEFAULT_CLR) for h in ordered]
            pct     = 100 * sum(vals) / max(total, 1)
            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor(WHITE)
            _donut(ax, ordered, vals, colors, "Habitat Composition",
                   center_text=f"{pct:.0f}%\nannotated")
            handles = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                       for c, lbl in zip(colors, ordered)]
            leg = ax.legend(handles=handles,
                            loc="lower center", bbox_to_anchor=(0.5, -0.08),
                            ncol=3, fontsize=9, framealpha=0.95, edgecolor="#E2E8F0",
                            columnspacing=1.0, handlelength=1.2)
            leg.get_frame().set_linewidth(0.8)
            plt.tight_layout(pad=1.5)
            out = f"{base_path}_3_habitat.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── per-habitat plots ──────────────────────────────────────────
        if all_habitats:
            hab_dir = base_path + "_habitat_plots"
            os.makedirs(hab_dir, exist_ok=True)

            for hab in all_habitats:
                hab_slug  = hab.lower().replace(" ", "_")
                hab_color = HAB_COLORS.get(hab, DEFAULT_CLR)

                # ── behavior donut for this habitat ──────────────────
                beh_in_hab = Counter(
                    beh_timeline[f]
                    for f in range(total)
                    if hab_timeline[f] == hab and beh_timeline[f]
                )
                if beh_in_hab:
                    ord_b = [b for b in BEHAVIOR_OPTIONS if b in beh_in_hab]
                    ord_b += [b for b in beh_in_hab if b not in ord_b]
                    vals_b = [beh_in_hab[b] for b in ord_b]
                    cols_b = []
                    for b in ord_b:
                        try:
                            cols_b.append(BEH_COLORS[BEHAVIOR_OPTIONS.index(b)
                                                      % len(BEH_COLORS)])
                        except ValueError:
                            cols_b.append(DEFAULT_CLR)
                    short_b = [b.split(" - ", 1)[-1] if " - " in b else b
                               for b in ord_b]
                    fig, ax = plt.subplots(figsize=(9, 7))
                    fig.patch.set_facecolor(WHITE)
                    _donut(ax, short_b, vals_b, cols_b,
                           f"Behavior — {hab}", title_color=hab_color,
                           center_text=f"{sum(vals_b)}\nframes")
                    handles_b = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                                 for c, lbl in zip(cols_b, short_b)]
                    leg = ax.legend(handles=handles_b,
                                    loc="lower center", bbox_to_anchor=(0.5, -0.14),
                                    ncol=3, fontsize=9, framealpha=0.95,
                                    edgecolor="#E2E8F0", columnspacing=1.0,
                                    handlelength=1.2)
                    leg.get_frame().set_linewidth(0.8)
                    plt.tight_layout(pad=1.5)
                    out = os.path.join(hab_dir, f"{hab_slug}_behavior.png")
                    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
                    plt.close(fig)
                    saved.append(out)

                # ── species bar for this habitat ─────────────────────
                sp_in_hab  = Counter()
                cat_in_hab = {}
                for feat in store.features:
                    if _feat_habitat(feat) == hab:
                        sp = feat.species or feat.species_category or "Unknown"
                        sp_in_hab[sp] += feat.count
                        cat_in_hab[sp] = feat.species_category or "Other"
                if sp_in_hab:
                    sp_list  = sorted(sp_in_hab, key=lambda s: sp_in_hab[s])
                    c_list   = [sp_in_hab[s] for s in sp_list]
                    col_list = [CAT_COLORS.get(cat_in_hab.get(s, "Other"), DEFAULT_CLR)
                                for s in sp_list]
                    n_sp   = len(sp_list)
                    height = max(4.0, n_sp * 0.48 + 2.0)
                    fig, ax = plt.subplots(figsize=(10, height))
                    fig.patch.set_facecolor(WHITE)
                    ax.set_facecolor(BG)
                    _hbar(ax, sp_list, c_list, col_list, cat_in_hab,
                          f"Species — {hab}", title_color=hab_color)
                    _fish_badge(ax, sp_list, sp_in_hab, cat_in_hab)
                    plt.tight_layout(pad=1.8)
                    out = os.path.join(hab_dir, f"{hab_slug}_species.png")
                    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
                    plt.close(fig)
                    saved.append(out)

        # ── 4. Sightings over time with habitat shading ───────────────
        import matplotlib.ticker as mticker
        from matplotlib.transforms import blended_transform_factory

        fps        = max(float(self.fps), 1e-6)
        total_secs = total / fps

        window_s      = 14.0
        window_frames = max(1, min(int(window_s * fps), total))

        new_sightings = np.zeros(total, dtype=float)
        for feat in store.features:
            if 0 <= feat.init_frame < total:
                new_sightings[feat.init_frame] += feat.count

        kernel       = np.ones(window_frames)
        rolling_sum  = np.convolve(new_sightings, kernel, mode="same")
        window_mins  = window_frames / fps / 60.0
        rolling_rate = rolling_sum / max(window_mins, 1e-9)
        times        = np.arange(total) / fps

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(BG)

        segs = FeatureStore._compress_timeline(hab_timeline)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            c   = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.axvspan(t_s, t_e, color=c, alpha=0.15, zorder=0, lw=0)
            if seg["start"] > 0:
                ax.axvline(t_s, color=c, linewidth=1.0,
                           linestyle="--", alpha=0.5, zorder=1)

        ax.plot(times, rolling_rate, color="#4F46E5", linewidth=2.2, zorder=3)
        ax.fill_between(times, rolling_rate, alpha=0.14, color="#4F46E5", zorder=2)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, times[-1] if len(times) > 1 else 1)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            if (t_e - t_s) < total_secs * 0.04:
                continue
            c = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.text((t_s + t_e) / 2, 0.97, v,
                    ha="center", va="top", transform=trans,
                    fontsize=9, fontweight="600", color=c)

        def _fmt_time(x, _):
            s = int(max(0, x))
            return f"{s // 60}:{s % 60:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(10, integer=False))

        ax.set_title(f"Animal Sightings Over Time  (rolling {window_s:.0f} s window)",
                     fontsize=15, fontweight="bold", pad=14, color="#0F172A")
        ax.set_xlabel("Time (m:ss)", fontsize=11, labelpad=8, color="#334155")
        ax.set_ylabel("Sightings / minute", fontsize=11, labelpad=8, color="#334155")
        ax.tick_params(axis="both", colors="#475569", labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        # Habitat color legend
        hab_seen = sorted({seg["value"] for seg in segs if seg.get("value")})
        if hab_seen:
            from matplotlib.patches import Patch as _Patch
            handles = [_Patch(facecolor=HAB_COLORS.get(h, DEFAULT_CLR),
                              alpha=0.55, edgecolor="none", label=h)
                       for h in hab_seen]
            leg = ax.legend(handles=handles, title="Habitat",
                            fontsize=9, title_fontsize=10,
                            framealpha=0.95, edgecolor="#E2E8F0",
                            loc="upper right")
            leg.get_frame().set_linewidth(0.8)

        plt.tight_layout(pad=1.8)
        out = f"{base_path}_4_sightings_over_time.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
        plt.close(fig)
        saved.append(out)

        # ── 5. Per-species sighting timelines ─────────────────────────
        if species_counts and total > 1:
            sp_dir = base_path + "_species_timelines"
            os.makedirs(sp_dir, exist_ok=True)

            def _timeline_ax(ax, sightings_arr, line_color, title):
                roll = np.convolve(sightings_arr, np.ones(window_frames),
                                   mode="same") / max(window_mins, 1e-9)
                for seg in segs:
                    v = seg.get("value")
                    if v is None:
                        continue
                    t_s = seg["start"] / fps
                    t_e = (seg["end"] + 1) / fps
                    c   = HAB_COLORS.get(v, DEFAULT_CLR)
                    ax.axvspan(t_s, t_e, color=c, alpha=0.15, zorder=0, lw=0)
                    if seg["start"] > 0:
                        ax.axvline(t_s, color=c, linewidth=1.0,
                                   linestyle="--", alpha=0.5, zorder=1)
                ax.plot(times, roll, color=line_color, linewidth=2.2, zorder=3)
                ax.fill_between(times, roll, alpha=0.14,
                                color=line_color, zorder=2)
                ax.set_ylim(bottom=0)
                ax.set_xlim(0, times[-1] if len(times) > 1 else 1)
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                for seg in segs:
                    v = seg.get("value")
                    if v is None:
                        continue
                    t_s = seg["start"] / fps
                    t_e = (seg["end"] + 1) / fps
                    if (t_e - t_s) < total_secs * 0.04:
                        continue
                    ax.text((t_s + t_e) / 2, 0.97, v,
                            ha="center", va="top", transform=trans,
                            fontsize=9, fontweight="600",
                            color=HAB_COLORS.get(v, DEFAULT_CLR))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
                ax.xaxis.set_major_locator(mticker.MaxNLocator(10, integer=False))
                ax.set_title(title, fontsize=13, fontweight="bold",
                             pad=12, color="#0F172A")
                ax.set_xlabel("Time (m:ss)", fontsize=10, labelpad=6,
                              color="#334155")
                ax.set_ylabel("Sightings / minute", fontsize=10,
                              labelpad=6, color="#334155")
                ax.tick_params(axis="both", colors="#475569", labelsize=9)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.spines["left"].set_color("#CBD5E1")
                ax.spines["bottom"].set_color("#CBD5E1")
                ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.7, zorder=0)
                ax.set_axisbelow(True)
                if hab_seen:
                    from matplotlib.patches import Patch as _P
                    hleg = ax.legend(
                        handles=[_P(facecolor=HAB_COLORS.get(h, DEFAULT_CLR),
                                    alpha=0.55, edgecolor="none", label=h)
                                 for h in hab_seen],
                        title="Habitat", fontsize=8, title_fontsize=9,
                        framealpha=0.95, edgecolor="#E2E8F0",
                        loc="upper right")
                    hleg.get_frame().set_linewidth(0.8)

            for sp_name in sorted(species_counts):
                sp_arr = np.zeros(total, dtype=float)
                for feat in store.features:
                    key = feat.species or feat.species_category or "Unknown"
                    if key == sp_name and 0 <= feat.init_frame < total:
                        sp_arr[feat.init_frame] += feat.count
                if not sp_arr.any():
                    continue
                cat        = cat_for_species.get(sp_name, "Other")
                line_color = CAT_COLORS.get(cat, DEFAULT_CLR)
                slug       = re.sub(r"[^\w]+", "_", sp_name).strip("_").lower()
                fig, ax    = plt.subplots(figsize=(13, 4))
                fig.patch.set_facecolor(WHITE)
                ax.set_facecolor(BG)
                _timeline_ax(ax, sp_arr, line_color,
                             f"{sp_name}  (rolling {window_s:.0f} s window)")
                plt.tight_layout(pad=1.5)
                out = os.path.join(sp_dir, f"{slug}_timeline.png")
                fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
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
                msg += "\n\n" + self._plots_summary(saved_plots)
            QMessageBox.information(self, "Done", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_plots_only(self):
        if self.playing:
            self.toggle_play()
        p, _ = QFileDialog.getSaveFileName(
            self, "Export plots — choose base filename",
            self.video_path + ".plots", "All files (*)")
        if not p:
            return
        base = os.path.splitext(p)[0] if "." in os.path.basename(p) else p
        try:
            saved_plots = self._save_plots(base)
            if not saved_plots:
                QMessageBox.information(self, "Export Plots", "No plots were generated.")
                return
            QMessageBox.information(self, "Done", self._plots_summary(saved_plots))
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_excel(self):
        if self.playing:
            self.toggle_play()
        try:
            import openpyxl
        except ImportError:
            QMessageBox.warning(self, "Missing dependency",
                                "openpyxl is required for Excel export.\n"
                                "Install with: pip install openpyxl")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export Excel",
            self.video_path + ".annotated.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        try:
            wb = self._build_workbook(openpyxl)
            wb.save(p)
            QMessageBox.information(self, "Done", f"Excel saved:\n{p}")
        except Exception as e:
            QMessageBox.critical(self, "Excel Export Error", str(e))

    def _build_workbook(self, openpyxl):
        from openpyxl.chart import BarChart, PieChart, Reference
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from collections import Counter

        wb    = openpyxl.Workbook()
        store = self.store
        fps   = max(float(self.fps), 1e-6)
        total = self.total_frames

        hab_tl = self._propagate(store.habitat_per_frame)
        beh_tl = self._propagate(store.behavior_per_frame)

        HEADER_FILL = PatternFill("solid", fgColor="0F172A")
        HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
        ALT_FILL    = PatternFill("solid", fgColor="F1F5F9")

        def _hrow(ws, cols):
            for c, t in enumerate(cols, 1):
                cell = ws.cell(row=1, column=c, value=t)
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 18

        def _autowidth(ws, mn=8, mx=40):
            for col in ws.columns:
                w = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[
                    get_column_letter(col[0].column)].width = min(mx, max(mn, w + 2))

        def _t(frames):
            s = int(frames / fps)
            return f"{s // 60}:{s % 60:02d}"

        # ── Sheet 1: Sightings flat table ─────────────────────────
        ws1 = wb.active
        ws1.title = "Sightings"
        _hrow(ws1, ["Frame", "Time", "Name", "Category", "Species",
                    "Count", "Behavior", "Habitat"])
        ws1.freeze_panes = "A2"
        for r, feat in enumerate(
                sorted(store.features, key=lambda f: f.init_frame), 2):
            fi = feat.init_frame
            for c, v in enumerate([
                fi, _t(fi), feat.name,
                feat.species_category or "",
                feat.species or "",
                feat.count,
                beh_tl[fi] or "",
                hab_tl[fi] or "",
            ], 1):
                cell = ws1.cell(row=r, column=c, value=v)
                if r % 2 == 0:
                    cell.fill = ALT_FILL
        _autowidth(ws1)

        # ── Sheet 2: Species Summary + bar chart ──────────────────
        ws2 = wb.create_sheet("Species Summary")
        _hrow(ws2, ["Species", "Category", "Count"])
        ws2.freeze_panes = "A2"
        sp_counts = Counter()
        sp_cats   = {}
        for feat in store.features:
            sp = feat.species or feat.species_category or "Unknown"
            sp_counts[sp] += feat.count
            sp_cats[sp]    = feat.species_category or "Other"
        sp_list = sorted(sp_counts, key=lambda s: sp_counts[s], reverse=True)
        for r, sp in enumerate(sp_list, 2):
            ws2.cell(row=r, column=1, value=sp)
            ws2.cell(row=r, column=2, value=sp_cats[sp])
            ws2.cell(row=r, column=3, value=sp_counts[sp])
        _autowidth(ws2)
        if sp_list:
            n   = len(sp_list)
            ch  = BarChart()
            ch.type   = "bar"
            ch.barDir = "bar"
            ch.title  = "Species Observed"
            ch.x_axis.title = "Count"
            ch.y_axis.title = "Species"
            ch.style  = 10
            ch.add_data(Reference(ws2, min_col=3, min_row=1, max_row=1+n),
                        titles_from_data=True)
            ch.set_categories(Reference(ws2, min_col=1, min_row=2, max_row=1+n))
            ch.width  = 22
            ch.height = max(10, n * 0.55)
            ws2.add_chart(ch, "E2")

        # ── Sheet 3: Behavior Summary + pie chart ─────────────────
        ws3 = wb.create_sheet("Behavior Summary")
        _hrow(ws3, ["Behavior", "Frames", "Duration (s)", "% of Annotated"])
        ws3.freeze_panes = "A2"
        beh_counts = Counter(b for b in beh_tl if b)
        beh_total  = sum(beh_counts.values()) or 1
        beh_rows   = [b for b in BEHAVIOR_OPTIONS if b in beh_counts]
        beh_rows  += [b for b in beh_counts if b not in beh_rows]
        for r, b in enumerate(beh_rows, 2):
            f = beh_counts[b]
            ws3.cell(row=r, column=1, value=b)
            ws3.cell(row=r, column=2, value=f)
            ws3.cell(row=r, column=3, value=round(f / fps, 2))
            ws3.cell(row=r, column=4, value=round(100 * f / beh_total, 1))
        _autowidth(ws3)
        if beh_rows:
            n  = len(beh_rows)
            ch = PieChart()
            ch.title  = "Behavior Distribution"
            ch.style  = 10
            ch.add_data(Reference(ws3, min_col=3, min_row=1, max_row=1+n),
                        titles_from_data=True)
            ch.set_categories(Reference(ws3, min_col=1, min_row=2, max_row=1+n))
            ch.width  = 18
            ch.height = 14
            ws3.add_chart(ch, "F2")

        # ── Sheet 4: Habitat Summary + pie chart ──────────────────
        ws4 = wb.create_sheet("Habitat Summary")
        _hrow(ws4, ["Habitat", "Frames", "Duration (s)", "% of Annotated"])
        ws4.freeze_panes = "A2"
        hab_counts = Counter(h for h in hab_tl if h)
        hab_total  = sum(hab_counts.values()) or 1
        hab_rows   = sorted(hab_counts, key=lambda h: hab_counts[h], reverse=True)
        for r, h in enumerate(hab_rows, 2):
            f = hab_counts[h]
            ws4.cell(row=r, column=1, value=h)
            ws4.cell(row=r, column=2, value=f)
            ws4.cell(row=r, column=3, value=round(f / fps, 2))
            ws4.cell(row=r, column=4, value=round(100 * f / hab_total, 1))
        _autowidth(ws4)
        if hab_rows:
            n  = len(hab_rows)
            ch = PieChart()
            ch.title  = "Habitat Distribution"
            ch.style  = 10
            ch.add_data(Reference(ws4, min_col=3, min_row=1, max_row=1+n),
                        titles_from_data=True)
            ch.set_categories(Reference(ws4, min_col=1, min_row=2, max_row=1+n))
            ch.width  = 18
            ch.height = 14
            ws4.add_chart(ch, "F2")

        # ── Sheet 5: Behavior Timeline (segments) ─────────────────
        ws5 = wb.create_sheet("Behavior Timeline")
        _hrow(ws5, ["Start Frame", "Start Time", "End Frame",
                    "End Time", "Duration (s)", "Behavior"])
        ws5.freeze_panes = "A2"
        for r, seg in enumerate(
                (s for s in FeatureStore._compress_timeline(
                    store.behavior_per_frame) if s.get("value")), 2):
            dur = (seg["end"] - seg["start"] + 1) / fps
            for c, v in enumerate([
                seg["start"], _t(seg["start"]),
                seg["end"],   _t(seg["end"]),
                round(dur, 2), seg["value"],
            ], 1):
                ws5.cell(row=r, column=c, value=v)
        _autowidth(ws5)

        # ── Sheet 6: Habitat Timeline (segments) ──────────────────
        ws6 = wb.create_sheet("Habitat Timeline")
        _hrow(ws6, ["Start Frame", "Start Time", "End Frame",
                    "End Time", "Duration (s)", "Habitat"])
        ws6.freeze_panes = "A2"
        for r, seg in enumerate(
                (s for s in FeatureStore._compress_timeline(
                    store.habitat_per_frame) if s.get("value")), 2):
            dur = (seg["end"] - seg["start"] + 1) / fps
            for c, v in enumerate([
                seg["start"], _t(seg["start"]),
                seg["end"],   _t(seg["end"]),
                round(dur, 2), seg["value"],
            ], 1):
                ws6.cell(row=r, column=c, value=v)
        _autowidth(ws6)

        return wb

    @staticmethod
    def _plots_summary(saved_plots: list) -> str:
        main  = [f for f in saved_plots
                 if "_habitat_plots" not in f and "_species_timelines" not in f]
        hab   = [f for f in saved_plots if "_habitat_plots" in f]
        sp    = [f for f in saved_plots if "_species_timelines" in f]
        lines = [f"{len(saved_plots)} plots saved."]
        if main:
            lines.append("Summary plots:\n" + "\n".join(
                os.path.basename(p) for p in main))
        if hab:
            lines.append(f"{len(hab)} habitat plots:\n{os.path.dirname(hab[0])}")
        if sp:
            lines.append(f"{len(sp)} species timelines:\n{os.path.dirname(sp[0])}")
        return "\n\n".join(lines)

    def _render(self, out_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(out_path, fourcc, self.fps, (self.video_w, self.video_h))

        def draw_scene(frame, fidx):
            beh = self.store.behavior_per_frame[fidx] or self._current_behavior
            hab = self.store.habitat_per_frame[fidx] or self._current_habitat
            text = f"{beh} • {hab}"
            cv2.putText(frame, text, (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        def feat_label(feat):
            lbl = feat.species_category
            if feat.species:
                lbl = f"{feat.species_category}: {feat.species}"
            if feat.count > 1:
                lbl = f"{lbl} x{feat.count}"
            return lbl

        if self.is_frame_dir:
            for fidx in range(self.total_frames):
                frame = cv2.imread(self.frame_files[fidx])
                if frame is None:
                    continue
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, feat_label(feat), bbox)
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
                    self._draw_mask(frame, mask, bgr, feat_label(feat), bbox)
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
