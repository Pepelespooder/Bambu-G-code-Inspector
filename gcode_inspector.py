#!/usr/bin/env python3

from __future__ import annotations

USAGE = (
    "G-code Inspector\n\n"
    "Usage:\n"
    "  python gcode_inspector.py <path-to-file.gcode> [--filament-diameter 1.75] [--plot] [--sideview] [--flow] [--interactive] [--e-per-mm] [--cooling]\n\n"
    "Notes:\n"
    "  - Drag-and-drop onto this .py or .bat is supported.\n"
    "  - Use --plot to save a per-layer metrics PNG.\n"
    "  - Use --sideview to save a sideview (layers vs. path) PNG with speeds/travel coloring.\n"
    "  - Use --flow to save volumetric flow (mm^3/s) over time with safe-limit flags.\n"
    "  - Use --e-per-mm to save Extrusion-per-distance (E/mm) plot with rolling std-dev.\n"
    "  - Use --cooling to save a cooling plot (fan vs layer time) highlighting low-fan short layers.\n"
)

import math
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Matplotlib is optional; import lazily where needed


# ---------------------------- Parsing Utilities ---------------------------- #


float_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
FLOAT_RE_C = re.compile(float_re)

# Precompiled regex caches for performance on large files
_WORD_RE_CACHE: Dict[str, re.Pattern[str]] = {}
def _word_re(letter: str) -> re.Pattern[str]:
    # Match like 'X123.4' or 'X 123.4' with boundaries
    key = letter.upper()
    p = _WORD_RE_CACHE.get(key)
    if p is None:
        p = re.compile(fr"\b{re.escape(key)}\s*({float_re})\b", re.IGNORECASE)
        _WORD_RE_CACHE[key] = p
    return p

KV_RE = re.compile(r"([A-Za-z0-9_\- ]+)\s*[:=]\s*([^;]+)")
LAYER0_RE = re.compile(r"\blayer\s*:?\s*0\b|^layer:0", re.IGNORECASE)
LAYER1_RE = re.compile(r"\blayer\s*:?\s*1\b|^layer:1", re.IGNORECASE)
BAMBU_LAYER1_RE = re.compile(r"layer\s+num/total_layer_count\s*:\s*1/", re.IGNORECASE)
BAMBU_LAYER2_RE = re.compile(r"layer\s+num/total_layer_count\s*:\s*2/", re.IGNORECASE)
TYPE_BRIDGE_RE = re.compile(r"type:bridge", re.IGNORECASE)
TYPE_SKIRT_BRIM_RE = re.compile(r"type:(skirt|brim)", re.IGNORECASE)
MATERIAL_RE = re.compile(r"(?<![A-Za-z])(pla|petg|abs|asa|tpu|nylon|pc|hips|pa|pet)(?![A-Za-z])", re.IGNORECASE)
DIAMETER_RE = re.compile(fr"(\b(1\.75|2\.85|3\.00)\b\s*mm)|diameter\s*({float_re})", re.IGNORECASE)
LAYER_INDEX_RE = re.compile(r"\blayer\s*:?\s*(\d+)\b|^layer:(\d+)", re.IGNORECASE)
BAMBU_LAYER_PROGRESS_RE = re.compile(r"layer\s+num/total_layer_count\s*:\s*(\d+)/", re.IGNORECASE)

# Sentinel to ensure we only try to auto-install matplotlib once per process
_MPL_TRIED_INSTALL = False

# Corner-stress defaults
CS_ANGLE_MIN_DEG = 45.0       # minimum corner angle to consider (stricter)
CS_SEG_MIN_MM = 1.5           # minimum segment length for both sides (filter infill jitter)
CS_RADIUS_FLOOR_MM = 1.5      # minimum effective radius for heuristic (less pessimistic)
CS_STRESS_MIN = 0.25          # ignore tiny stress values to reduce noise
CS_MARGIN = 1.60              # safety margin when comparing incoming vs. limit
# Additional gating: require a minimum time on both segments so the head actually reaches speed
CS_SEG_TIME_MIN_S = 0.02

# Layer-metrics jump detection defaults (tuned to reduce false positives)
LM_ABS_FLOOR_MM_FACTOR = 0.18     # fraction of median layer move for abs floor
LM_ABS_FLOOR_MM_MIN = 100.0       # minimum absolute floor for movement jumps
LM_PCT_MIN = 55.0                 # minimum percent change for movement jumps
LM_SIGMA_K = 3.5                  # robust z-score threshold on diffs
LM_MIN_BASELINE_MM = 30.0         # ignore jumps when prior layer movement is tiny

LT_ABS_FLOOR_S_FACTOR = 0.18      # fraction of median layer time for abs floor
LT_ABS_FLOOR_S_MIN = 15.0         # minimum absolute floor for time jumps
LT_PCT_MIN = 55.0                 # minimum percent change for time jumps
LT_SIGMA_K = 3.5                  # robust z-score threshold on diffs
LT_MIN_BASELINE_S = 5.0           # ignore jumps when prior layer time is tiny

# Printer model detection patterns
PRN_PATS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\ba1\s*mini\b", re.I), "bambu_a1"),
    (re.compile(r"\ba1\b", re.I), "bambu_a1"),
    (re.compile(r"\bp1s\b", re.I), "bambu_p1s"),
    (re.compile(r"\bp1p\b", re.I), "bambu_p1s"),
    (re.compile(r"\bx1c\b|\bx1e\b|\bx1\b", re.I), "bambu_x1c"),
    (re.compile(r"\bh2[-_]?d\b", re.I), "bambu_h2d"),
    (re.compile(r"\bh2[-_]?s\b", re.I), "bambu_h2s"),
]


def parse_word(line: str, letter: str) -> Optional[float]:
    m = _word_re(letter).search(line)
    return float(m.group(1)) if m else None


def strip_comment(line: str) -> Tuple[str, Optional[str]]:
    # Cura/PrusaSlicer use ';' comments. Some use parentheses. We'll handle ';'.
    if ";" in line:
        code, comment = line.split(";", 1)
        return code.strip(), comment.strip()
    return line.strip(), None


# ------------------------------- Data Models ------------------------------- #


@dataclass
class TempProfile:
    nozzle_setpoints: List[float] = field(default_factory=list)
    bed_setpoints: List[float] = field(default_factory=list)
    nozzle_print_setpoints: List[float] = field(default_factory=list)
    bed_print_setpoints: List[float] = field(default_factory=list)
    first_layer_nozzle: Optional[float] = None
    first_layer_bed: Optional[float] = None
    # Optional chamber values (setpoints)
    chamber_setpoints: List[float] = field(default_factory=list)
    chamber_print_setpoints: List[float] = field(default_factory=list)
    first_layer_chamber: Optional[float] = None


@dataclass
class RetractionStats:
    samples: List[float] = field(default_factory=list)  # retraction distance (mm)
    speeds: List[float] = field(default_factory=list)  # mm/min from F during retract
    z_hops: int = 0

    @property
    def avg_distance(self) -> Optional[float]:
        return sum(self.samples) / len(self.samples) if self.samples else None

    @property
    def max_distance(self) -> Optional[float]:
        return max(self.samples) if self.samples else None

    @property
    def avg_speed_mms(self) -> Optional[float]:
        if not self.speeds:
            return None
        # speeds recorded in mm/min; convert to mm/s
        return (sum(self.speeds) / len(self.speeds)) / 60.0


@dataclass
class SpeedStats:
    # Track feed rates for extrusion and travel moves
    extrusion_feeds_mm_min: List[float] = field(default_factory=list)
    travel_feeds_mm_min: List[float] = field(default_factory=list)

    def add(self, feed: Optional[float], extruding: bool):
        if feed is None:
            return
        if extruding:
            self.extrusion_feeds_mm_min.append(feed)
        else:
            self.travel_feeds_mm_min.append(feed)

    @property
    def avg_extrusion_mms(self) -> Optional[float]:
        if not self.extrusion_feeds_mm_min:
            return None
        return (sum(self.extrusion_feeds_mm_min) / len(self.extrusion_feeds_mm_min)) / 60.0

    @property
    def avg_travel_mms(self) -> Optional[float]:
        if not self.travel_feeds_mm_min:
            return None
        return (sum(self.travel_feeds_mm_min) / len(self.travel_feeds_mm_min)) / 60.0

    @property
    def max_extrusion_mms(self) -> Optional[float]:
        return max(self.extrusion_feeds_mm_min) / 60.0 if self.extrusion_feeds_mm_min else None

    @property
    def max_travel_mms(self) -> Optional[float]:
        return max(self.travel_feeds_mm_min) / 60.0 if self.travel_feeds_mm_min else None


@dataclass
class FanStats:
    # Records latest setpoint 0..255 or 0..100, we normalize to 0..255 scale
    samples: List[int] = field(default_factory=list)

    @staticmethod
    def normalize(value: float) -> int:
        # Heuristic: if <= 1.0 treat as 0..1 scale, if <= 100 treat as percent
        if value <= 1.0:
            return int(round(value * 255))
        if value <= 100.0:
            return int(round(value / 100.0 * 255))
        return int(round(value))

    @property
    def avg_percent(self) -> Optional[float]:
        if not self.samples:
            return None
        return (sum(self.samples) / len(self.samples)) / 255.0 * 100.0

    @property
    def max_percent(self) -> Optional[float]:
        return (max(self.samples) / 255.0 * 100.0) if self.samples else None


@dataclass
class ExtrusionStats:
    total_e_mm: float = 0.0
    positive_extrusions: int = 0
    resets: int = 0
    absolute_mode: Optional[bool] = None  # True for M82, False for M83


@dataclass
class HeuristicStats:
    travel_no_retract_count: int = 0
    max_travel_no_retract_mm: float = 0.0
    mode_switches: int = 0
    first_layer_found: bool = False
    first_layer_extrusion_max_mms: Optional[float] = None
    first_layer_travel_max_mms: Optional[float] = None
    first_layer_extrusion_sum_mms: float = 0.0
    first_layer_extrusion_count: int = 0
    first_layer_travel_sum_mms: float = 0.0
    first_layer_travel_count: int = 0
    first_layer_fan_max_percent: Optional[float] = None
    skirt_brim_present: bool = False
    bridge_segments: int = 0
    bridge_low_fan_segments: int = 0
    pauses_detected: int = 0
    feedrate_override: Optional[float] = None  # M220 S%
    flow_override: Optional[float] = None      # M221 S%
    bed_leveling_disabled: bool = False
    g92_resets: int = 0
    # Volumetric flow tracking: store (delta_e_mm, time_s, in_first_layer, abs_time_s)
    vol_samples: List[Tuple[float, float, bool, float]] = field(default_factory=list)
    max_volumetric_mm3_s: Optional[float] = None
    max_volumetric_first_layer_mm3_s: Optional[float] = None
    avg_volumetric_mm3_s: Optional[float] = None
    avg_volumetric_first_layer_mm3_s: Optional[float] = None
    # First-layer dynamics (accel/jerk)
    fl_accel_print_max: Optional[float] = None  # from M204 P or S
    fl_accel_travel_max: Optional[float] = None  # from M204 T or S
    fl_jerk_x_max: Optional[float] = None  # from M205 X
    fl_jerk_y_max: Optional[float] = None  # from M205 Y
    fl_jerk_z_max: Optional[float] = None  # from M205 Z
    fl_jerk_e_max: Optional[float] = None  # from M205 E
    printer_model: Optional[str] = None  # e.g., bambu_a1, bambu_p1s, bambu_x1c, bambu_h2d, bambu_h2s
    # Bed leveling state tracking
    bed_leveling_enabled_current: Optional[bool] = None
    bed_leveling_enabled_at_first_layer: Optional[bool] = None
    # Per-layer aggregates
    layer_move_mm: Dict[int, float] = field(default_factory=dict)
    layer_time_s: Dict[int, float] = field(default_factory=dict)
    # Time-weighted fan per layer (sum of fan*seg_time), to compute avg fan% per layer
    layer_fan_time_255: Dict[int, float] = field(default_factory=dict)
    # Per-segment extrusion per distance samples (E/mm)
    e_per_mm_samples: List[float] = field(default_factory=list)
    # Optional layer index alongside E/mm samples
    e_per_mm_layers: List[Optional[int]] = field(default_factory=list)
    # Current/last seen dynamics (global, not only first layer)
    accel_print_current: Optional[float] = None  # from M204 P/S
    accel_travel_current: Optional[float] = None  # from M204 T/S
    jerk_x_current: Optional[float] = None  # from M205 X
    jerk_y_current: Optional[float] = None  # from M205 Y
    junction_dev_current: Optional[float] = None  # from M205 J (Marlin JD)
    klipper_scv_current: Optional[float] = None  # from SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY
    # Corner stress samples: (time_s, stress_index, angle_deg, v_in, v_limit, r_est)
    corner_samples: List[Tuple[float, float, float, float, Optional[float], float]] = field(default_factory=list)
    # Total elapsed time across moves (s)
    total_time_s: float = 0.0
    # Timelapse filtering (Bambu/Orca SKIPPABLE timelapse blocks)
    timelapse_blocks: int = 0
    timelapse_lines_skipped: int = 0


@dataclass
class GcodeSummary:
    file: Path
    filament_diameter_mm: float
    inferred_material: str
    material_confidence: str
    estimated_length_mm: float
    estimated_volume_cm3: float
    estimated_mass_g: float
    density_used_g_cm3: float
    temps: TempProfile
    retraction: RetractionStats
    speeds: SpeedStats
    fan: FanStats
    flags: List[str]
    notes: List[str]
    printer_model: Optional[str] = None
    # Filament branding
    filament_brand: Optional[str] = None
    filament_name: Optional[str] = None
    filament_color: Optional[str] = None
    # Filament Max Volumetric Speed (mm^3/s) if provided/overridden
    filament_mvs_mm3_s: Optional[float] = None
    # Additional insights
    heuristics: Optional[HeuristicStats] = None
    extrusion: Optional[ExtrusionStats] = None
    # Slicer-provided aggregates
    slicer_length_mm: Optional[float] = None
    slicer_volume_cm3: Optional[float] = None
    slicer_mass_g: Optional[float] = None
    # Header-derived metadata
    total_layers: Optional[int] = None
    estimated_time_total_s: Optional[int] = None
    model_print_time_s: Optional[int] = None
    max_z_height_mm: Optional[float] = None
    # Layer aggregates (distance/time)
    avg_layer_move_mm: Optional[float] = None
    first_layer_move_mm: Optional[float] = None
    last_layer_move_mm: Optional[float] = None
    avg_layer_time_s: Optional[float] = None
    first_layer_time_s: Optional[float] = None
    last_layer_time_s: Optional[float] = None


# ------------------------------ Core Parser ------------------------------- #


class GcodeInspector:
    def __init__(self, filament_diameter_mm: float = 1.75, printer_override: Optional[str] = None, flow_limit_override_mm3_s: Optional[float] = None):
        self.filament_diameter_mm = filament_diameter_mm
        self.printer_override = printer_override
        self.flow_limit_override_mm3_s = flow_limit_override_mm3_s

    def inspect(self, lines: Iterable[str], filename: Optional[Path] = None) -> GcodeSummary:
        temps = TempProfile()
        retraction = RetractionStats()
        speeds = SpeedStats()
        fan = FanStats()
        extrusion = ExtrusionStats()
        flags: List[str] = []
        notes: List[str] = []

        # State
        absolute_positioning = True  # G90 default
        absolute_extrusion = True  # Many firmwares default to absolute E
        current_e = 0.0
        last_e_move_was_extrude = False
        last_feed = None
        last_z = 0.0
        seen = {
            "G28": False,  # homing
            "M104": False,
            "M109": False,
            "M140": False,
            "M190": False,
            "G29": False,  # ABL
            "M420": False,  # ABL mesh use
            "START_GCODE": False,
            "END_GCODE": False,
            "RETRACTION": False,
        }

        # Slicer hints
        comment_kv: Dict[str, str] = {}
        header_kv: Dict[str, object] = {}
        commented_material: Optional[str] = None
        commented_diameter: Optional[float] = None
        # Filament branding/name/color hints
        commented_brand_guess: Optional[str] = None
        commented_name_hint: Optional[str] = None
        commented_color_hint: Optional[str] = None

        in_header = False
        # Layer tracking
        in_first_layer = False
        # Movement tracking
        last_x: Optional[float] = None
        last_y: Optional[float] = None
        current_layer_index: Optional[int] = None
        # Retraction/travel pattern tracking
        retracted_since_last_extrude = True
        heur = HeuristicStats()
        # Fan tracking (current value)
        current_fan_0_255 = 0
        # Extrusion mode changes
        last_extrusion_mode: Optional[bool] = None

        # Track current setpoints and first layer detection to avoid using purge temps
        current_nozzle_setpoint: Optional[float] = None
        current_bed_setpoint: Optional[float] = None
        current_chamber_setpoint: Optional[float] = None
        saw_layer0 = False
        in_print_phase = False
        end_print_detected = False
        # Timelapse skip-block tracking (e.g., Bambu/Orca SKIPPABLE blocks)
        in_skippable_block = False
        current_skip_type: Optional[str] = None

        line_no = 0
        for raw in lines:
            line_no += 1
            line = raw.strip()
            if not line:
                continue

            code, comment = strip_comment(line)

            # Parse common comment metadata
            if comment:
                cstripped = comment.strip()
                lc = cstripped.lower()
                # Detect Bambu/Orca SKIPPABLE blocks and their types
                if "skippable_start" in lc:
                    in_skippable_block = True
                    current_skip_type = None
                if in_skippable_block and "skiptype" in lc:
                    # Example: '; SKIPTYPE: timelapse'
                    m = re.search(r"skiptype\s*:\s*([a-z0-9_\-]+)", lc, re.I)
                    if m:
                        current_skip_type = (m.group(1) or "").strip().lower()
                if "skippable_end" in lc:
                    # Close block; if it was a timelapse one, count it
                    if in_skippable_block and (current_skip_type == "timelapse"):
                        heur.timelapse_blocks += 1
                    in_skippable_block = False
                    current_skip_type = None
                # Bambu Studio / OrcaSlicer header/config blocks
                if lc.startswith("header_start") or lc.startswith("header_block_start") or lc.startswith("config_block_start"):
                    in_header = True
                elif lc.startswith("header_end") or lc.startswith("header_block_end") or lc.startswith("config_block_end"):
                    in_header = False

                # Key-value styles
                if (":" in comment) or ("=" in comment):
                    kv_match = KV_RE.findall(comment)
                else:
                    kv_match = []
                for k, v in kv_match:
                    key = k.strip().lower().replace(" ", "_")
                    comment_kv[key] = v.strip()
                    if in_header:
                        # Try to parse list/scalar values from header
                        val_txt = v.strip()
                        # Remove unit suffixes like 'g', 'cm3'
                        # Keep raw string too for fallback
                        parsed: object = val_txt
                        try:
                            if val_txt.startswith("[") and val_txt.endswith("]"):
                                parsed = ast.literal_eval(val_txt)
                            else:
                                # Try number
                                num_m = re.match(fr"^\s*({float_re})\s*", val_txt)
                                if num_m:
                                    parsed = float(num_m.group(1))
                        except Exception:
                            parsed = val_txt
                        header_kv[key] = parsed
                        # Try to infer printer model from header values
                        pv = f"{key} {val_txt}".lower()
                        model = self._detect_printer_model(pv)
                        if model and not heur.printer_model:
                            heur.printer_model = model
                # Detect layer 0 start to lock first-layer temps
                # Common patterns: ";LAYER:0", "; layer 0", and Bambu: "; layer num/total_layer_count: 1/XXX"
                if (LAYER0_RE.search(lc) or BAMBU_LAYER1_RE.search(lc)) and not saw_layer0:
                    saw_layer0 = True
                    in_first_layer = True
                    in_print_phase = True
                    current_layer_index = 0
                    heur.first_layer_found = True
                    # Record bed leveling state at start of print
                    heur.bed_leveling_enabled_at_first_layer = heur.bed_leveling_enabled_current
                    if temps.first_layer_nozzle is None:
                        temps.first_layer_nozzle = current_nozzle_setpoint
                    if temps.first_layer_bed is None:
                        temps.first_layer_bed = current_bed_setpoint
                    if temps.first_layer_chamber is None:
                        temps.first_layer_chamber = current_chamber_setpoint
                    # Seed print-phase temps with current setpoints
                    # Avoid seeding with purge/cooldown values
                    if current_nozzle_setpoint is not None and current_nozzle_setpoint >= 150:
                        temps.nozzle_print_setpoints.append(current_nozzle_setpoint)
                    if current_bed_setpoint is not None and current_bed_setpoint >= 40:
                        temps.bed_print_setpoints.append(current_bed_setpoint)
                    if current_chamber_setpoint is not None and current_chamber_setpoint >= 20:
                        temps.chamber_print_setpoints.append(current_chamber_setpoint)
                # Detect end of first layer when next layer begins (LAYER:1 or Bambu 2/XXX)
                if in_first_layer and (LAYER1_RE.search(lc) or BAMBU_LAYER2_RE.search(lc)):
                    in_first_layer = False

                # Update current layer index from generic markers
                mli = LAYER_INDEX_RE.search(lc)
                if mli:
                    try:
                        idx = int(mli.group(1) or mli.group(2))
                        current_layer_index = idx
                        in_print_phase = True
                        if idx == 0 and not heur.first_layer_found:
                            heur.first_layer_found = True
                        
                    except Exception:
                        pass
                else:
                    mb = BAMBU_LAYER_PROGRESS_RE.search(lc)
                    if mb:
                        try:
                            n = int(mb.group(1))
                            current_layer_index = max(0, n - 1)
                            in_print_phase = True
                            if current_layer_index == 0 and not heur.first_layer_found:
                                heur.first_layer_found = True
                        except Exception:
                            pass

                # Detect printer model tokens in free comments
                pm = self._detect_printer_model(lc)
                if pm and not heur.printer_model:
                    heur.printer_model = pm

                # Detect skirt/brim markers
                if "type:" in lc:
                    if TYPE_SKIRT_BRIM_RE.search(lc):
                        heur.skirt_brim_present = True
                    if TYPE_BRIDGE_RE.search(lc):
                        heur.bridge_segments += 1
                        # Check current fan level for bridge, normalize to percent
                        fan_pct = (current_fan_0_255 / 255.0) * 100.0
                        # Consider low if <40%
                        if fan_pct < 40.0:
                            heur.bridge_low_fan_segments += 1
                # Material name hints
                m = MATERIAL_RE.search(comment)
                if m and not commented_material:
                    commented_material = m.group(1).upper()
                # Brand/name/color hints from common keys
                for key in ("filament_brand", "brand", "vendor", "manufacturer"):
                    if key in comment_kv and not commented_brand_guess:
                        commented_brand_guess = comment_kv.get(key)
                for key in ("filament_name", "name", "filament_fullname"):
                    if key in comment_kv and not commented_name_hint:
                        commented_name_hint = comment_kv.get(key)
                for key in ("filament_color", "color", "colour"):
                    if key in comment_kv and not commented_color_hint:
                        commented_color_hint = comment_kv.get(key)
                # Free-text brand token search
                if not commented_brand_guess:
                    brand = self._detect_filament_brand((comment or "") + " " + (code or ""))
                    if brand:
                        commented_brand_guess = brand
                # Filament diameter hints
                m2 = DIAMETER_RE.search(comment)
                if m2 and not commented_diameter:
                    try:
                        # groups: 1 is the full mm match; 2 is explicit 1.75/2.85/3.00; 3 is diameter <float>
                        val_txt = m2.group(2) or m2.group(3)
                        val = float(val_txt)
                        if 1.0 < val < 3.2:
                            commented_diameter = val
                    except Exception:
                        pass

            # If inside a timelapse SKIPPABLE block, skip this line's command analysis entirely
            if in_skippable_block and (current_skip_type == "timelapse"):
                heur.timelapse_lines_skipped += 1
                continue

            # Tokenize command
            if not code:
                continue

            word0 = code.split()[0].upper()

            # Detect common blocks
            if "start" in (comment or "").lower():
                seen["START_GCODE"] = True
            if "end" in (comment or "").lower():
                seen["END_GCODE"] = True

            if word0 in {"G90", "G91"}:
                absolute_positioning = word0 == "G90"

            if word0 in {"M82", "M83"}:
                new_mode = word0 == "M82"
                if last_extrusion_mode is not None and last_extrusion_mode != new_mode:
                    heur.mode_switches += 1
                absolute_extrusion = new_mode
                last_extrusion_mode = new_mode
                extrusion.absolute_mode = absolute_extrusion
            if word0 == "G92":
                # Track E resets for diagnostics
                if re.search(r"\bE\s*", code, re.IGNORECASE):
                    heur.g92_resets += 1

            if word0 == "G92":
                e = parse_word(code, "E")
                if e is not None:
                    current_e = e
                    extrusion.resets += 1

            if word0.startswith("M10") or word0.startswith("M1"):
                # Temps, bed
                if word0 == "M104":
                    seen["M104"] = True
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.nozzle_setpoints.append(s)
                        current_nozzle_setpoint = s
                        # Detect print end (cooldown) before appending to print-phase lists
                        if s <= 30:
                            end_print_detected = True
                        # Only record plausible print-phase temps
                        if in_print_phase and not end_print_detected and s >= 150:
                            temps.nozzle_print_setpoints.append(s)
                elif word0 == "M109":
                    seen["M109"] = True
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.nozzle_setpoints.append(s)
                        current_nozzle_setpoint = s
                        if s <= 30:
                            end_print_detected = True
                        if in_print_phase and not end_print_detected and s >= 150:
                            temps.nozzle_print_setpoints.append(s)
                elif word0 == "M140":
                    seen["M140"] = True
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.bed_setpoints.append(s)
                        current_bed_setpoint = s
                        if in_print_phase and not end_print_detected and s >= 40:
                            temps.bed_print_setpoints.append(s)
                elif word0 == "M190":
                    seen["M190"] = True
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.bed_setpoints.append(s)
                        current_bed_setpoint = s
                        if in_print_phase and not end_print_detected and s >= 40:
                            temps.bed_print_setpoints.append(s)
                elif word0 == "M141":
                    # Chamber temperature setpoint
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.chamber_setpoints.append(s)
                        current_chamber_setpoint = s
                        if in_print_phase and not end_print_detected and s >= 20:
                            temps.chamber_print_setpoints.append(s)
                elif word0 == "M191":
                    # Wait for chamber temperature to reach setpoint
                    s = parse_word(code, "S")
                    if s is not None:
                        temps.chamber_setpoints.append(s)
                        current_chamber_setpoint = s
                        if in_print_phase and not end_print_detected and s >= 20:
                            temps.chamber_print_setpoints.append(s)
                elif word0 == "M204":
                    # Acceleration settings: S=default/print, P=print, T=travel (Marlin style)
                    s = parse_word(code, "S")
                    p = parse_word(code, "P")
                    t = parse_word(code, "T")
                    # Update global current values
                    val_p_glob = p or s
                    val_t_glob = t or s
                    if val_p_glob is not None:
                        heur.accel_print_current = val_p_glob
                    if val_t_glob is not None:
                        heur.accel_travel_current = val_t_glob
                    if in_first_layer:
                        val_p = p or s
                        val_t = t or s
                        if val_p is not None:
                            if heur.fl_accel_print_max is None:
                                heur.fl_accel_print_max = val_p
                            else:
                                heur.fl_accel_print_max = max(heur.fl_accel_print_max, val_p)
                        if val_t is not None:
                            if heur.fl_accel_travel_max is None:
                                heur.fl_accel_travel_max = val_t
                            else:
                                heur.fl_accel_travel_max = max(heur.fl_accel_travel_max, val_t)
                elif word0 == "M205":
                    # Jerk settings: X/Y/Z/E values in mm/s on many firmwares
                    # Also, J may represent Junction Deviation (mm) on some firmwares
                    jparam = parse_word(code, "J")
                    if jparam is not None:
                        heur.junction_dev_current = jparam
                    jx_g = parse_word(code, "X")
                    jy_g = parse_word(code, "Y")
                    if jx_g is not None:
                        heur.jerk_x_current = jx_g
                    if jy_g is not None:
                        heur.jerk_y_current = jy_g
                    if in_first_layer:
                        jx = parse_word(code, "X")
                        jy = parse_word(code, "Y")
                        jz = parse_word(code, "Z")
                        je = parse_word(code, "E")
                        if jx is not None:
                            heur.fl_jerk_x_max = jx if heur.fl_jerk_x_max is None else max(heur.fl_jerk_x_max, jx)
                        if jy is not None:
                            heur.fl_jerk_y_max = jy if heur.fl_jerk_y_max is None else max(heur.fl_jerk_y_max, jy)
                        if jz is not None:
                            heur.fl_jerk_z_max = jz if heur.fl_jerk_z_max is None else max(heur.fl_jerk_z_max, jz)
                        if je is not None:
                            heur.fl_jerk_e_max = je if heur.fl_jerk_e_max is None else max(heur.fl_jerk_e_max, je)

            if word0 in {"M106", "M107"}:
                if word0 == "M107":
                    fan.samples.append(0)
                    current_fan_0_255 = 0
                else:
                    s = parse_word(code, "S")
                    if s is not None:
                        nf = FanStats.normalize(s)
                        fan.samples.append(nf)
                        current_fan_0_255 = nf
                        if in_first_layer:
                            fl_pct = (nf / 255.0) * 100.0
                            if heur.first_layer_fan_max_percent is None:
                                heur.first_layer_fan_max_percent = fl_pct
                            else:
                                heur.first_layer_fan_max_percent = max(heur.first_layer_fan_max_percent, fl_pct)

            if word0 in {"G28", "G29"}:
                seen[word0] = True
            if word0 == "M420":
                seen[word0] = True
                s = parse_word(code, "S")
                if s is not None:
                    heur.bed_leveling_enabled_current = (s != 0)
            # Bambu turns off leveling via G29.2 S0
            if word0.startswith("G29"):
                if word0.startswith("G29.2") or "G29.2" in code:
                    s = parse_word(code, "S")
                    if s is not None:
                        heur.bed_leveling_enabled_current = (s != 0)

            # Pause commands present
            if word0 in {"M0", "M1", "M25"}:
                heur.pauses_detected += 1

            # Feedrate/flow overrides
            if word0 == "M220":
                s = parse_word(code, "S")
                if s is not None:
                    heur.feedrate_override = s
            if word0 == "M221":
                s = parse_word(code, "S")
                if s is not None:
                    heur.flow_override = s

            # Movement and extrusion
            if word0 in {"G0", "G1"}:
                # Feedrate
                f = parse_word(code, "F")
                if f is not None:
                    last_feed = f

                x = parse_word(code, "X")
                y = parse_word(code, "Y")
                z = parse_word(code, "Z")
                e = parse_word(code, "E")

                # Track Z movement distance for timing
                z_dist = 0.0
                if z is not None:
                    # For absolute positioning, Z is absolute; for relative G91 Z is delta
                    if absolute_positioning:
                        try:
                            z_dist = abs(z - last_z)
                        except Exception:
                            z_dist = 0.0
                        # Treat layer changes and z-hop detection heuristically
                        last_z = z
                    else:
                        if z > 0:
                            # relative z up movement could be z-hop
                            retraction.z_hops += 1
                        z_dist = abs(z)
                        last_z += z

                # Compute XY travel distance for this move
                xy_dist = 0.0
                dx = 0.0
                dy = 0.0
                if x is not None or y is not None:
                    # Get current/next XY based on positioning mode
                    nx = last_x if last_x is not None else 0.0
                    ny = last_y if last_y is not None else 0.0
                    if absolute_positioning:
                        if x is not None:
                            nx = x
                        if y is not None:
                            ny = y
                        if last_x is not None and last_y is not None:
                            dx = (nx - last_x)
                            dy = (ny - last_y)
                            xy_dist = math.hypot(dx, dy)
                    else:
                        dx = (x or 0.0)
                        dy = (y or 0.0)
                        nx = (last_x or 0.0) + dx
                        ny = (last_y or 0.0) + dy
                        xy_dist = math.hypot(dx, dy)
                    last_x, last_y = nx, ny

                # Retraction detection via E-negative moves or G10/G11 in some firmwares
                if e is not None:
                    if absolute_extrusion:
                        delta_e = e - current_e
                        current_e = e
                    else:
                        delta_e = e

                    if delta_e < -0.01:
                        seen["RETRACTION"] = True
                        retraction.samples.append(abs(delta_e))
                        if last_feed is not None and abs(delta_e) >= 0.25:
                            retraction.speeds.append(last_feed)
                        last_e_move_was_extrude = False
                        retracted_since_last_extrude = True
                        # Many slicers pair retraction with Z hop; detect explicit Z delta in relative mode above
                    elif delta_e > 0.0:
                        # Extruding
                        extrusion.total_e_mm += delta_e
                        extrusion.positive_extrusions += 1
                        last_e_move_was_extrude = True
                        retracted_since_last_extrude = False
                        # Track E/mm for this segment when XY distance is meaningful
                        # Ignore very short segments to avoid noisy division and purge/wipe artifacts
                        if xy_dist is not None and xy_dist > 0.2:
                            try:
                                heur.e_per_mm_samples.append(delta_e / xy_dist)
                                heur.e_per_mm_layers.append(current_layer_index)
                            except Exception:
                                pass
                        # Corner stress estimation: compare this segment to previous extruding segment
                        # Requires previous extruding vector and speeds
                        try:
                            speed_mms = (last_feed / 60.0) if last_feed else None
                        except Exception:
                            speed_mms = None
                        if speed_mms and xy_dist > 0.3:
                            # Initialize persistent previous-segment state container lazily on the instance
                            if not hasattr(self, "_prev_extr"):
                                self._prev_extr = {"vec": None, "len": 0.0, "speed": None, "t_end": 0.0}
                            prev = self._prev_extr
                            pv = prev.get("vec")
                            pl = float(prev.get("len") or 0.0)
                            pin = prev.get("speed")
                            t_prev_end = float(prev.get("t_end") or 0.0)
                            # Compute angle with previous vector
                            if pv and pin and pl > CS_SEG_MIN_MM and xy_dist > CS_SEG_MIN_MM:
                                vx0, vy0 = pv
                                v0_len = math.hypot(vx0, vy0)
                                v1_len = math.hypot(dx, dy)
                                if v0_len > 0 and v1_len > 0:
                                    dot = vx0 * dx + vy0 * dy
                                    cos_t = max(-1.0, min(1.0, dot / (v0_len * v1_len)))
                                    theta = math.acos(cos_t)
                                    # Determine turn direction via z-component of cross product to detect continuous curves
                                    cross_z = vx0 * dy - vy0 * dx
                                    turn_sign = 0
                                    if cross_z > 1e-9:
                                        turn_sign = 1
                                    elif cross_z < -1e-9:
                                        turn_sign = -1
                                    # Track runs of same-sign turning; long runs with moderate average angle are curves, not corners
                                    if not hasattr(self, "_turn_run"):
                                        self._turn_run = {"sign": None, "count": 0, "avg": 0.0}
                                    tr = self._turn_run
                                    if turn_sign == 0 or tr["sign"] is None or turn_sign != tr["sign"]:
                                        tr["sign"], tr["count"], tr["avg"] = (turn_sign, 1 if turn_sign != 0 else 0, float(theta) if turn_sign != 0 else 0.0)
                                    else:
                                        tr["count"] += 1
                                        # online mean for theta
                                        tr["avg"] += (float(theta) - tr["avg"]) / float(max(1, tr["count"]))
                                    # Skip small direction changes; require a more pronounced corner
                                    if theta > math.radians(CS_ANGLE_MIN_DEG):
                                        # Suppress continuous curves: consecutive same-sign turns with moderate per-step angle
                                        is_curve_run = (self._turn_run.get("count", 0) >= 4 and self._turn_run.get("avg", 0.0) <= math.radians(60.0))
                                        if is_curve_run:
                                            pass  # treat as curve, not a corner
                                        else:
                                            # Corner radius estimate (heuristic)
                                            # Use a slightly larger floor to reflect slicer corner smoothing/arc fitting
                                            r_est = max(CS_RADIUS_FLOOR_MM, min(pl, v1_len) * abs(math.sin(theta / 2.0)))
                                            # Limits from accel and Klipper SCV; ignore Marlin jerk (unreliable for this calc)
                                            a = heur.accel_print_current or 2000.0
                                            v_acc = math.sqrt(max(0.0, a * r_est))
                                            v_scv = heur.klipper_scv_current if heur.klipper_scv_current is not None else None
                                            v_jd = None
                                            if heur.junction_dev_current is not None and heur.junction_dev_current > 0:
                                                # Coarse cap using JD and accel; do not scale by angle to avoid over-restricting
                                                v_jd = math.sqrt(max(0.0, a * heur.junction_dev_current))
                                            # Aggregate corner limit
                                            v_candidates = [v for v in (v_acc, v_scv, v_jd) if v is not None and v > 0]
                                            v_limit = min(v_candidates) if v_candidates else None
                                            # Cap incoming speed by reachable speed given segment lengths and accel
                                            # If the segments are short, the toolhead cannot maintain the commanded speed
                                            v_reach_prev = math.sqrt(max(0.0, 2.0 * a * pl))
                                            v_reach_cur = math.sqrt(max(0.0, 2.0 * a * v1_len))
                                            v_in = min(float(pin), v_reach_prev, v_reach_cur)
                                            # Require a minimum time on both segments to avoid micro-zigzag false positives
                                            t_prev_seg = pl / float(pin) if (pin and pin > 0) else 0.0
                                            t_cur_seg = v1_len / float(speed_mms) if (speed_mms and speed_mms > 0) else 0.0
                                            if (t_prev_seg >= CS_SEG_TIME_MIN_S) and (t_cur_seg >= CS_SEG_TIME_MIN_S):
                                                # Slightly damp mid-angles for fewer false positives
                                                angle_factor = abs(math.sin(theta / 2.0)) ** 1.25
                                                margin = CS_MARGIN  # conservative margin to reduce false positives
                                                if v_limit is not None and v_limit > 0 and v_in > 0:
                                                    stress = (v_in / (v_limit * margin)) * angle_factor
                                                else:
                                                    # Fallback: acceleration-only heuristic
                                                    denom = max(1e-3, v_acc)
                                                    stress = (v_in / (denom * margin)) * angle_factor
                                                # Suppress very low stress values to reduce noise
                                                if stress >= CS_STRESS_MIN:
                                                    heur.corner_samples.append((t_prev_end, float(stress), math.degrees(theta), v_in, v_limit, r_est))
                            # Update previous segment state to current
                            # We need segment time to update end-time; estimate using XY and Z contributions
                            seg_time = 0.0
                            if last_feed is not None and last_feed > 0:
                                speed_cur = last_feed / 60.0
                                if speed_cur > 0:
                                    if xy_dist > 0.0:
                                        seg_time += xy_dist / speed_cur
                                    if z_dist > 0.0:
                                        seg_time += z_dist / speed_cur
                            # Accumulate end time based on a running counter on self
                            if not hasattr(self, "_cum_time_s"):
                                self._cum_time_s = 0.0
                            # Previous end time already stored in prev["t_end"]
                            # Now set new prev to current
                            t_end = float(self._cum_time_s) + float(seg_time)
                            self._prev_extr = {"vec": (dx, dy), "len": xy_dist, "speed": speed_mms, "t_end": t_end}
                            # Update cumulative timer
                            self._cum_time_s = t_end
                        # Volumetric flow sample if we can compute time from feed and travel length
                        # Be permissive so long prints are fully represented; still avoid zero-time spikes.
                        if last_feed and xy_dist is not None and xy_dist > 0.2 and delta_e >= 0.02:
                            speed_mms = last_feed / 60.0
                            if speed_mms > 0:
                                # Floor segment time to avoid divide-by-very-small spikes on tiny moves
                                t = max(xy_dist / speed_mms, 0.002)
                                # Store absolute end time so plotting covers travels and skipped micro segments
                                t_end_abs = float(getattr(self, "_cum_time_s", 0.0))
                                # Append line number for diagnostics (stored as 5th element; consumers ignore extra fields)
                                heur.vol_samples.append((delta_e, t, in_first_layer, t_end_abs, line_no))
                    else:
                        # Pure travel or zero E change
                        last_e_move_was_extrude = False
                        # Update cumulative time for travels as well
                        # Estimate time similarly to extrusion segments
                        if last_feed is not None and last_feed > 0:
                            speed_cur = last_feed / 60.0
                            seg_time = 0.0
                            if xy_dist > 0.0:
                                seg_time += xy_dist / speed_cur
                            if z_dist > 0.0:
                                seg_time += z_dist / speed_cur
                            if not hasattr(self, "_cum_time_s"):
                                self._cum_time_s = 0.0
                            self._cum_time_s += seg_time

                # Handle arc moves (G2/G3): update position/time/extrusion to avoid mis-attributing flow to next segment
                if word0 in {"G2", "G3"}:
                    # Default in case parsing fails
                    xy_dist_arc = 0.0
                    # Feedrate update if provided on the arc
                    f_arc = parse_word(code, "F")
                    if f_arc is not None:
                        last_feed = f_arc
                    x_arc = parse_word(code, "X")
                    y_arc = parse_word(code, "Y")
                    e_arc = parse_word(code, "E")
                    # Estimate chord length from current position to arc endpoint
                    try:
                        cur_x = last_x
                        cur_y = last_y
                        nx = cur_x if cur_x is not None else 0.0
                        ny = cur_y if cur_y is not None else 0.0
                        if absolute_positioning:
                            if x_arc is not None:
                                nx = x_arc
                            if y_arc is not None:
                                ny = y_arc
                        else:
                            nx = (cur_x or 0.0) + (x_arc or 0.0)
                            ny = (cur_y or 0.0) + (y_arc or 0.0)
                        if (cur_x is not None) and (cur_y is not None):
                            dx_a = float(nx) - float(cur_x)
                            dy_a = float(ny) - float(cur_y)
                            xy_dist_arc = math.hypot(dx_a, dy_a)
                        # Update stored position to arc end
                        last_x = nx
                        last_y = ny
                    except Exception:
                        pass
                    # Update cumulative time using chord length and current feed
                    if last_feed is not None and last_feed > 0 and xy_dist_arc > 0.0:
                        speed_mms = last_feed / 60.0
                        if speed_mms > 0:
                            seg_time = xy_dist_arc / speed_mms
                            if not hasattr(self, "_cum_time_s"):
                                self._cum_time_s = 0.0
                            self._cum_time_s += seg_time
                    # Update extrusion bookeeping so the next line's delta_e is not inflated
                    if e_arc is not None:
                        if absolute_extrusion:
                            delta_e_arc = e_arc - current_e
                            current_e = e_arc
                        else:
                            delta_e_arc = e_arc
                            current_e += e_arc
                        if delta_e_arc < -0.01:
                            seen["RETRACTION"] = True
                            retraction.samples.append(abs(delta_e_arc))
                            if last_feed is not None and abs(delta_e_arc) >= 0.25:
                                retraction.speeds.append(last_feed)
                            last_e_move_was_extrude = False
                            retracted_since_last_extrude = True
                        elif delta_e_arc > 0.0:
                            extrusion.total_e_mm += delta_e_arc
                            extrusion.positive_extrusions += 1
                            last_e_move_was_extrude = True
                            retracted_since_last_extrude = False
                    # Use arc distance for subsequent feed classification in this iteration
                    xy_dist = xy_dist_arc

                # Feed rate classification for move
                # For extrusion max speed stats, ignore micro-extrusions with tiny XY distance
                if last_feed is not None:
                    if last_e_move_was_extrude and xy_dist >= 2.0:
                        speeds.add(last_feed, extruding=True)
                    elif not last_e_move_was_extrude:
                        speeds.add(last_feed, extruding=False)

                # First-layer speed tracking
                if in_first_layer and last_feed is not None:
                    mms = last_feed / 60.0
                    if last_e_move_was_extrude and xy_dist >= 2.0:
                        if heur.first_layer_extrusion_max_mms is None:
                            heur.first_layer_extrusion_max_mms = mms
                        else:
                            heur.first_layer_extrusion_max_mms = max(heur.first_layer_extrusion_max_mms, mms)
                        heur.first_layer_extrusion_sum_mms += mms
                        heur.first_layer_extrusion_count += 1
                    else:
                        if heur.first_layer_travel_max_mms is None:
                            heur.first_layer_travel_max_mms = mms
                        else:
                            heur.first_layer_travel_max_mms = max(heur.first_layer_travel_max_mms, mms)
                        heur.first_layer_travel_sum_mms += mms
                        heur.first_layer_travel_count += 1

                # Travel-without-retraction detection
                if not last_e_move_was_extrude and xy_dist > 0.0 and not retracted_since_last_extrude:
                    if xy_dist >= 5.0:  # threshold for significant travel
                        heur.travel_no_retract_count += 1
                        heur.max_travel_no_retract_mm = max(heur.max_travel_no_retract_mm, xy_dist)

                # Per-layer aggregates (distance and time)
                if in_print_phase and current_layer_index is not None:
                    # Ignore micro jitters to reduce noise in per-layer movement
                    if xy_dist >= 0.1:
                        heur.layer_move_mm[current_layer_index] = heur.layer_move_mm.get(current_layer_index, 0.0) + xy_dist
                    if last_feed is not None:
                        speed_mms = last_feed / 60.0
                        if speed_mms > 0.0:
                            seg_t = 0.0
                            if xy_dist >= 0.1:
                                seg_t += xy_dist / speed_mms
                            if z_dist > 0.0:
                                seg_t += z_dist / speed_mms
                            # Ignore skippable timelapse segments in per-layer time metrics
                            skip_time = in_skippable_block and (current_skip_type == "timelapse")
                            if seg_t > 0.0 and not skip_time:
                                heur.layer_time_s[current_layer_index] = heur.layer_time_s.get(current_layer_index, 0.0) + seg_t
                                # Accumulate fan*time for time-weighted average per layer
                                heur.layer_fan_time_255[current_layer_index] = heur.layer_fan_time_255.get(current_layer_index, 0.0) + (current_fan_0_255 * seg_t)

        # Post-process slicer hints
        # Also check common keys for printer model
        for k in ("printer_model", "machine", "machine_name", "machine_type"):
            v = header_kv.get(k)
            if isinstance(v, str):
                pm = self._detect_printer_model(v.lower())
                if pm:
                    heur.printer_model = heur.printer_model or pm
            elif isinstance(v, list) and v:
                pm = self._detect_printer_model(str(v[0]).lower())
                if pm:
                    heur.printer_model = heur.printer_model or pm
        # Prefer header diameter if present (Bambu/Orca), else comment fallback
        header_diam = header_kv.get("filament_diameter")
        if isinstance(header_diam, list) and header_diam:
            try:
                first_d = float(header_diam[0])
                if 1.0 < first_d < 3.2:
                    self.filament_diameter_mm = first_d
            except Exception:
                pass
        elif isinstance(header_diam, (int, float)) and 1.0 < float(header_diam) < 3.2:
            self.filament_diameter_mm = float(header_diam)
        elif commented_diameter and 1.0 < commented_diameter < 3.2:
            self.filament_diameter_mm = commented_diameter

        # Filament MVS (Max Volumetric Speed) from header or user override
        filament_mvs: Optional[float] = None
        try:
            # Common keys used by slicers
            for k in ("max_volumetric_speed", "filament_max_volumetric_speed", "maximum_volumetric_speed"):
                if k in header_kv:
                    val = header_kv.get(k)
                    if isinstance(val, (int, float)):
                        filament_mvs = float(val)
                        break
                    if isinstance(val, list) and val:
                        try:
                            filament_mvs = float(val[0])
                            break
                        except Exception:
                            pass
                    if isinstance(val, str):
                        m = re.match(fr"^\s*({float_re})\s*$", val)
                        if m:
                            filament_mvs = float(m.group(1))
                            break
            # Fallback: search any header key containing both 'volumetric' and 'speed'
            if filament_mvs is None:
                for k, v in header_kv.items():
                    lk = str(k).lower()
                    if ("volumetric" in lk) and ("speed" in lk):
                        try:
                            if isinstance(v, (int, float)):
                                filament_mvs = float(v)
                                break
                            if isinstance(v, list) and v:
                                filament_mvs = float(v[0])
                                break
                            if isinstance(v, str):
                                m = re.search(fr"({float_re})", v)
                                if m:
                                    filament_mvs = float(m.group(1))
                                    break
                        except Exception:
                            continue
        except Exception:
            filament_mvs = None
        # User override takes precedence if provided
        if self.flow_limit_override_mm3_s is not None:
            filament_mvs = float(self.flow_limit_override_mm3_s)

        # Filament brand/name/color from header or comments
        def _str_first(val: object) -> Optional[str]:
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            elif isinstance(val, str) and val.strip():
                return val.strip()
            return None

        brand = _str_first(header_kv.get("filament_brand")) or _str_first(header_kv.get("filament_vendor")) or _str_first(header_kv.get("manufacturer"))
        name = _str_first(header_kv.get("filament_name")) or _str_first(header_kv.get("filament_fullname"))
        color = _str_first(header_kv.get("filament_color")) or _str_first(header_kv.get("filament_colour")) or _str_first(header_kv.get("color"))
        if not brand:
            brand = _str_first(comment_kv.get("filament_brand")) or _str_first(comment_kv.get("brand")) or _str_first(comment_kv.get("vendor")) or _str_first(comment_kv.get("manufacturer"))
        if not name:
            name = _str_first(comment_kv.get("filament_name")) or _str_first(comment_kv.get("filament_fullname")) or _str_first(comment_kv.get("name"))
        if not color:
            color = _str_first(comment_kv.get("filament_color")) or _str_first(comment_kv.get("filament_colour")) or _str_first(comment_kv.get("color"))
        # Fallback to comment free-text
        if not brand:
            brand = commented_brand_guess
        if not name:
            name = commented_name_hint
        if not color:
            color = commented_color_hint

        # Prefer header initial-layer temps if available (Bambu/Orca), else keep detected layer0 temps
        def _get_header_first(values: object) -> Optional[float]:
            if isinstance(values, list) and values:
                try:
                    return float(values[0])
                except Exception:
                    return None
            if isinstance(values, (int, float)):
                return float(values)
            return None

        for nz_key in (
            "nozzle_temperature_initial_layer",
            "initial_layer_temperature",
            "first_layer_temperature",
            "first_layer_nozzle_temperature",
        ):
            v = header_kv.get(nz_key)
            fl = _get_header_first(v)
            if fl is not None:
                temps.first_layer_nozzle = fl
                break

        for bd_key in (
            "bed_temperature_initial_layer",
            "initial_layer_bed_temperature",
            "first_layer_bed_temperature",
        ):
            v = header_kv.get(bd_key)
            fl = _get_header_first(v)
            if fl is not None:
                temps.first_layer_bed = fl
                break

        # If layer marker was not found and first-layer temps still None, fall back to earliest seen
        if temps.first_layer_nozzle is None and temps.nozzle_setpoints:
            temps.first_layer_nozzle = temps.nozzle_setpoints[0]
        if temps.first_layer_bed is None and temps.bed_setpoints:
            temps.first_layer_bed = temps.bed_setpoints[0]

        # Material inference
        inferred_material, confidence, density = self._infer_material(
            commented_material, temps, comment_kv, header_kv
        )

        # Compute volume and mass from extruded filament length
        radius_mm = self.filament_diameter_mm / 2.0
        cross_section_mm2 = math.pi * radius_mm * radius_mm
        length_mm = max(extrusion.total_e_mm, 0.0)
        volume_mm3 = cross_section_mm2 * length_mm
        volume_cm3 = volume_mm3 / 1000.0
        mass_g = volume_cm3 * density

        # Summarize corner stress exceedances for users without matplotlib
        if heur.corner_samples:
            try:
                n_ex = sum(1 for (_, s, *_rest) in heur.corner_samples if float(s) >= 1.0)
                n_warn = sum(1 for (_, s, *_rest) in heur.corner_samples if 0.8 <= float(s) < 1.0)
                if n_ex or n_warn:
                    notes.append(f"Corner stress exceedances (>=1.0): {n_ex}; near-threshold (0.8–1.0): {n_warn}")
            except Exception:
                pass

        # Compute volumetric flow stats (mm^3/s)
        if heur.vol_samples:
            mm3_s_vals: List[float] = []
            mm3_s_fl_vals: List[float] = []
            for sample in heur.vol_samples:
                try:
                    de = float(sample[0])
                    t = float(sample[1])
                    fl = bool(sample[2]) if len(sample) >= 3 else False
                except Exception:
                    continue
                if t > 0:
                    rate = (de / t) * cross_section_mm2
                    mm3_s_vals.append(rate)
                    if fl:
                        mm3_s_fl_vals.append(rate)
            # Prefer print-phase values if available to avoid purge/cali spikes
            if mm3_s_fl_vals:
                heur.max_volumetric_mm3_s = max(mm3_s_fl_vals)
                heur.avg_volumetric_mm3_s = sum(mm3_s_fl_vals) / len(mm3_s_fl_vals)
            else:
                heur.max_volumetric_mm3_s = max(mm3_s_vals) if mm3_s_vals else None
                heur.avg_volumetric_mm3_s = (sum(mm3_s_vals) / len(mm3_s_vals)) if mm3_s_vals else None
            heur.max_volumetric_first_layer_mm3_s = max(mm3_s_fl_vals) if mm3_s_fl_vals else None
            heur.avg_volumetric_first_layer_mm3_s = (sum(mm3_s_fl_vals) / len(mm3_s_fl_vals)) if mm3_s_fl_vals else None

        # Flags and notes
        flags.extend(
            self._detect_flags(
                seen,
                temps,
                retraction,
                speeds,
                fan,
                inferred_material,
                length_mm,
                heur,
                extrusion,
                filament_mvs_mm3_s=filament_mvs,
            )
        )

        # Apply printer override if provided
        if self.printer_override:
            heur.printer_model = self.printer_override

        # Slicer estimates (Bambu/Orca): totals if available
        slicer_length_mm: Optional[float] = None
        slicer_volume_cm3: Optional[float] = None
        slicer_mass_g: Optional[float] = None
        # total_filament_used may be list per-tool in mm
        tfu = header_kv.get("total_filament_used")
        if isinstance(tfu, list):
            try:
                slicer_length_mm = float(sum(float(x) for x in tfu))
            except Exception:
                pass
        elif isinstance(tfu, (int, float)):
            slicer_length_mm = float(tfu)
        # total_volume in cm3 (commonly) and total_weight in g
        tv = header_kv.get("total_volume")
        if isinstance(tv, (int, float)):
            slicer_volume_cm3 = float(tv)
        tw = header_kv.get("total_weight")
        if isinstance(tw, (int, float)):
            slicer_mass_g = float(tw)

        # Header-derived: times, layer count, height
        def _parse_time_to_seconds(txt: str) -> Optional[int]:
            if not isinstance(txt, str):
                return None
            t = txt.strip().lower()
            total = 0
            m = re.search(r"(\d+)\s*h", t)
            if m:
                total += int(m.group(1)) * 3600
            m = re.search(r"(\d+)\s*m", t)
            if m:
                total += int(m.group(1)) * 60
            m = re.search(r"(\d+)\s*s", t)
            if m:
                total += int(m.group(1))
            # Also support formats like HH:MM:SS
            if total == 0 and re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", t):
                parts = [int(p) for p in t.split(":")]
                if len(parts) == 2:
                    total = parts[0] * 60 + parts[1]
                elif len(parts) == 3:
                    total = parts[0] * 3600 + parts[1] * 60 + parts[2]
            return total or None

        total_layers: Optional[int] = None
        for key in ("total_layer_number", "total_layer_count"):
            if key in header_kv:
                try:
                    total_layers = int(float(header_kv[key]))
                    break
                except Exception:
                    pass
        max_z_height_mm: Optional[float] = None
        for key in ("max_z_height", "object_height", "model_height"):
            if key in header_kv and isinstance(header_kv[key], (int, float)):
                max_z_height_mm = float(header_kv[key])
                break
        model_print_time_s = _parse_time_to_seconds(str(header_kv.get("model_printing_time", "")))
        estimated_time_total_s = _parse_time_to_seconds(str(header_kv.get("total_estimated_time", "")))
        # Fallback: some slicers may not wrap these in a formal header block; try general comment_kv
        if model_print_time_s is None:
            try:
                model_print_time_s = _parse_time_to_seconds(str(comment_kv.get("model_printing_time", "")))
            except Exception:
                pass
        if estimated_time_total_s is None:
            try:
                estimated_time_total_s = _parse_time_to_seconds(str(comment_kv.get("total_estimated_time", "")))
            except Exception:
                pass

        # Chamber temperature from header if available
        def _extract_first_number(obj: object) -> Optional[float]:
            if isinstance(obj, (int, float)):
                return float(obj)
            if isinstance(obj, list) and obj:
                try:
                    return float(obj[0])
                except Exception:
                    return None
            if isinstance(obj, str):
                m = re.search(fr"({float_re})", obj)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
            return None
        chamber_header_keys = [
            "chamber_temperature",
            "chamber_temperatures",
            "chamber_temp",
            "chamber",
        ]
        chamber_header_value: Optional[float] = None
        for k in chamber_header_keys:
            if k in header_kv:
                chamber_header_value = _extract_first_number(header_kv[k])
                if chamber_header_value is not None:
                    break
        # If chamber info is present in header and we didn't capture any print-phase chamber setpoints, record it
        if chamber_header_value is not None and chamber_header_value > 0:
            # Avoid duplicates if already recorded
            if not temps.chamber_setpoints:
                temps.chamber_setpoints.append(chamber_header_value)
            if not temps.chamber_print_setpoints:
                temps.chamber_print_setpoints.append(chamber_header_value)
            if temps.first_layer_chamber is None:
                temps.first_layer_chamber = chamber_header_value

        # Layer aggregates: compute averages and first/last
        avg_layer_move_mm: Optional[float] = None
        first_layer_move_mm: Optional[float] = None
        last_layer_move_mm: Optional[float] = None
        avg_layer_time_s: Optional[float] = None
        first_layer_time_s: Optional[float] = None
        last_layer_time_s: Optional[float] = None
        if heur.layer_move_mm:
            keys = sorted(heur.layer_move_mm.keys())
            vals = [heur.layer_move_mm[k] for k in keys]
            if vals:
                avg_layer_move_mm = sum(vals) / len(vals)
                first_layer_move_mm = heur.layer_move_mm.get(0, heur.layer_move_mm.get(keys[0]))
                last_layer_move_mm = heur.layer_move_mm.get(keys[-1])
        if heur.layer_time_s:
            tkeys = sorted(heur.layer_time_s.keys())
            tvals = [heur.layer_time_s[k] for k in tkeys]
            if tvals:
                avg_layer_time_s = sum(tvals) / len(tvals)
                first_layer_time_s = heur.layer_time_s.get(0, heur.layer_time_s.get(tkeys[0]))
                last_layer_time_s = heur.layer_time_s.get(tkeys[-1])

        # Note about timelapse filtering
        if heur.timelapse_blocks > 0:
            notes.append(
                f"Timelapse blocks detected: {heur.timelapse_blocks} (ignored in metrics; {heur.timelapse_lines_skipped} lines skipped)."
            )
        # Add slicer time summary to notes for clarity
        if estimated_time_total_s or model_print_time_s:
            def _fmt_t(sec: int) -> str:
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                if h:
                    return f"{h}h {m}m {s}s"
                if m:
                    return f"{m}m {s}s"
                return f"{s}s"
            parts = []
            if estimated_time_total_s:
                parts.append(f"est total={_fmt_t(int(estimated_time_total_s))}")
            if model_print_time_s:
                parts.append(f"model={_fmt_t(int(model_print_time_s))}")
            notes.append("Slicer time: " + ", ".join(parts))

        summary = GcodeSummary(
            file=filename or Path("<stdin>"),
            filament_diameter_mm=self.filament_diameter_mm,
            inferred_material=inferred_material,
            material_confidence=confidence,
            estimated_length_mm=length_mm,
            estimated_volume_cm3=volume_cm3,
            estimated_mass_g=mass_g,
            density_used_g_cm3=density,
            temps=temps,
            retraction=retraction,
            speeds=speeds,
            fan=fan,
            flags=flags,
            notes=notes,
            printer_model=heur.printer_model,
            filament_brand=brand,
            filament_name=name,
            filament_color=color,
            filament_mvs_mm3_s=filament_mvs,
            heuristics=heur,
            extrusion=extrusion,
            slicer_length_mm=slicer_length_mm,
            slicer_volume_cm3=slicer_volume_cm3,
            slicer_mass_g=slicer_mass_g,
            total_layers=total_layers,
            estimated_time_total_s=estimated_time_total_s,
            model_print_time_s=model_print_time_s,
            max_z_height_mm=max_z_height_mm,
            avg_layer_move_mm=avg_layer_move_mm,
            first_layer_move_mm=first_layer_move_mm,
            last_layer_move_mm=last_layer_move_mm,
            avg_layer_time_s=avg_layer_time_s,
            first_layer_time_s=first_layer_time_s,
            last_layer_time_s=last_layer_time_s,
        )

        # Record total elapsed time observed during parse for plotting/reference
        try:
            if hasattr(self, "_cum_time_s"):
                summary.heuristics.total_time_s = float(getattr(self, "_cum_time_s", 0.0))
        except Exception:
            pass

        # Attach slicer estimate summary into notes for now to avoid struct churn
        if any(v is not None for v in (slicer_length_mm, slicer_volume_cm3, slicer_mass_g)):
            ests = []
            if slicer_length_mm is not None:
                ests.append(f"length≈{slicer_length_mm/1000.0:.2f} m")
            if slicer_volume_cm3 is not None:
                ests.append(f"volume≈{slicer_volume_cm3:.2f} cm^3")
            if slicer_mass_g is not None:
                ests.append(f"mass≈{slicer_mass_g:.1f} g")
            summary.notes.append("Slicer estimates: " + ", ".join(ests))

        # No extra note needed; printer prints in its own section
        
        return summary

    # ---------------------------- Helper Functions ---------------------------- #

    @staticmethod
    def _detect_printer_model(text: str) -> Optional[str]:
        t = text.lower()
        # Bambu A1 family
        for pat, key in PRN_PATS:
            if pat.search(t):
                return key
        return None

    @staticmethod
    def _detect_filament_brand(text: str) -> Optional[str]:
        t = (text or "").lower()
        brands = {
            "prusament": "Prusament",
            "bambu lab": "Bambu",
            "bambulab": "Bambu",
            "bambu": "Bambu",
            "polymaker": "Polymaker",
            "hatchbox": "HATCHBOX",
            "sunlu": "SUNLU",
            "esun": "eSUN",
            "overture": "OVERTURE",
            "e3d": "E3D",
            "colorfabb": "colorFabb",
            "fillamentum": "Fillamentum",
            "priline": "PRILINE",
            "sainsmart": "SainSmart",
            "eryone": "ERYONE",
            "geeetech": "Geeetech",
            "creality": "Creality",
            "amazonbasics": "AmazonBasics",
            "amolen": "AMOLEN",
            "3dxtech": "3DXTech",
            "atomic": "Atomic",
            "formfutura": "FormFutura",
            "fiberlogy": "Fiberlogy",
            "matterhackers": "MatterHackers",
            "proto-pasta": "Proto-pasta",
            "protopasta": "Proto-pasta",
        }
        for key, pretty in brands.items():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", t):
                return pretty
        return None

    

    @staticmethod
    def _infer_material(
        commented_material: Optional[str], temps: TempProfile, comment_kv: Dict[str, str], header_kv: Dict[str, object]
    ) -> Tuple[str, str, float]:
        # If slicer provided a material, trust it
        # Map to known densities (g/cm^3)
        densities = {
            "PLA": 1.24,
            "PETG": 1.27,
            "ABS": 1.04,
            "ASA": 1.07,
            "TPU": 1.20,
            "NYLON": 1.15,
            "PA": 1.15,
            "PC": 1.20,
            "HIPS": 1.04,
            "PET": 1.34,
        }

        # Helper to find material token in arbitrary string
        def find_token(text: str) -> Optional[str]:
            # Match material tokens even when separated by underscores or digits
            m = MATERIAL_RE.search(text)
            return m.group(1).upper() if m else None

        # Direct hints
        for key in (
            "filament_type",
            "material",
            "material_name",
            "filament_name",
        ):
            # Header (Bambu/Orca) may provide list per tool
            if key in header_kv:
                val = header_kv[key]
                if isinstance(val, list) and val:
                    mtxt = str(val[0]).strip()
                    upper = mtxt.upper()
                    if upper in densities:
                        return upper, "high (header)", densities[upper]
                    token = find_token(mtxt)
                    if token and token in densities:
                        return token, "medium (header substring)", densities[token]
                elif isinstance(val, str):
                    upper = val.strip().upper()
                    if upper in densities:
                        return upper, "high (header)", densities[upper]
                    token = find_token(val)
                    if token and token in densities:
                        return token, "medium (header substring)", densities[token]
            if key in comment_kv:
                txt = comment_kv[key].strip()
                upper = txt.upper()
                if upper in densities:
                    return upper, "high (comment)", densities[upper]
                token = find_token(txt)
                if token and token in densities:
                    return token, "medium (comment substring)", densities[token]

        if commented_material and commented_material in densities:
            return commented_material, "medium (comment)", densities[commented_material]

        # Temperature-based heuristic
        nozzle = None
        src_n = temps.nozzle_print_setpoints or [t for t in temps.nozzle_setpoints if t >= 150]
        if src_n:
            nozzle = sum(src_n) / len(src_n)
        bed = None
        src_b = temps.bed_print_setpoints or [t for t in temps.bed_setpoints if t >= 40]
        if src_b:
            bed = sum(src_b) / len(src_b)

        if nozzle is not None:
            if nozzle <= 215 and (bed is None or bed <= 65):
                return "PLA", "low (temp heuristic)", densities["PLA"]
            if 220 <= nozzle <= 250 and (bed is None or 60 <= bed <= 90):
                # PETG or ABS range; use bed to split
                if bed is not None and bed >= 90:
                    return "ABS", "low (temp heuristic)", densities["ABS"]
                return "PETG", "low (temp heuristic)", densities["PETG"]
            if 240 <= nozzle <= 270 and (bed is None or bed >= 70):
                return "NYLON", "low (temp heuristic)", densities["NYLON"]

        # Default to PLA
        return "PLA", "very low (default)", densities["PLA"]

    @staticmethod
    def _detect_flags(
        seen: Dict[str, bool],
        temps: TempProfile,
        retraction: RetractionStats,
        speeds: SpeedStats,
        fan: FanStats,
        material: str,
        length_mm: float,
        heur: HeuristicStats,
        extrusion: ExtrusionStats,
        filament_mvs_mm3_s: Optional[float] = None,
    ) -> List[str]:
        flags: List[str] = []

        # Start-up and safety
        if not seen["G28"]:
            flags.append("No homing (G28) found; printer may start un-homed.")
        if not (seen["M104"] or seen["M109"]):
            flags.append("No nozzle heat commands (M104/M109) detected.")
        if not (seen["M140"] or seen["M190"]):
            flags.append("No bed heat commands (M140/M190) detected.")
        if not (seen["G29"] or seen["M420"]):
            flags.append("No bed leveling/mesh use (G29/M420) detected; ensure manual leveling.")
        # Flag explicit disable only if leveling was disabled at start of print
        if heur.first_layer_found and heur.bed_leveling_enabled_at_first_layer is False:
            flags.append("Bed leveling explicitly disabled (M420 S0/G29.2 S0); ensure bed is trammed.")

        # Temperatures sanity vs material
        n = temps.first_layer_nozzle
        b = temps.first_layer_bed
        if n is not None:
            if material == "PLA" and n > 225:
                flags.append(f"Nozzle temp {n:.0f}C high for PLA; risk of stringing/gloss.")
            if material in {"PETG", "ABS"} and n < 220:
                flags.append(f"Nozzle temp {n:.0f}C low for {material}; risk of poor adhesion.")
        if b is not None:
            if material == "PLA" and b > 70:
                flags.append(f"Bed temp {b:.0f}C high for PLA; may cause elephant's foot.")
            if material in {"ABS"} and b < 90:
                flags.append("Bed temp low for ABS; risk of warping.")

        # Fan usage vs material
        if fan.avg_percent is not None:
            avg_fan = fan.avg_percent
            if material in {"PLA"} and avg_fan < 20:
                flags.append("Low fan for PLA; bridges/overhangs may sag.")
            if material in {"ABS", "ASA"} and avg_fan > 30:
                flags.append("High fan for ABS/ASA; may reduce layer adhesion/warp.")
            if material in {"PETG"} and avg_fan > 80:
                flags.append("Very high fan for PETG; may cause brittle layers.")

        # Retraction sanity
        high_flow_models = {"bambu_h2s", "bambu_h2d", "bambu_x1c", "bambu_p1s"}
        is_high_flow = heur.printer_model in high_flow_models
        if not retraction.samples:
            flags.append("No retraction detected; expect stringing and blobs.")
        else:
            sample_threshold = 0.2 if is_high_flow else 0.4
            macro_samples = [d for d in retraction.samples if d >= sample_threshold]
            avg_r = (sum(macro_samples) / len(macro_samples)) if macro_samples else (retraction.avg_distance or 0)
            low_ret_thresh = 0.15 if is_high_flow else 0.2
            max_r = retraction.max_distance or 0
            if max_r < low_ret_thresh:
                flags.append("Very low retraction distance; may cause stringing.")
            if avg_r > 8.0:
                flags.append("Very high retraction distance; risk of jams/wear.")
            if retraction.avg_speed_mms is not None:
                speed_thresh = 250 if is_high_flow else 60
                if retraction.avg_speed_mms > speed_thresh:
                    flags.append(f"High retraction speed (>{speed_thresh} mm/s); may grind filament.")
            if retraction.z_hops == 0:
                flags.append("No Z-hop detected during retractions; may cause scars on travel.")
            if retraction.avg_speed_mms is not None and retraction.avg_speed_mms < 10:
                flags.append("Low retraction speed (<10 mm/s); may cause stringing.")
            # Retraction frequency
            if length_mm > 0:
                macro_count = len(macro_samples)
                per100 = macro_count / (length_mm / 100.0)
                high_thresh = 80 if is_high_flow else 15
                med_thresh = 40 if is_high_flow else 8
                if per100 > high_thresh:
                    flags.append("Excessive retractions (>15 per 100 mm); risk of heat creep.")
                elif per100 > med_thresh:
                    flags.append("High retraction frequency; consider combing/avoid travel through perimeters.")

        # Speeds sanity
        extr_speed_thresh = 1000 if is_high_flow else 120
        if speeds.max_extrusion_mms is not None and speeds.max_extrusion_mms > extr_speed_thresh:
            flags.append(f"Very high extrusion speed (>{extr_speed_thresh} mm/s); risk of under-extrusion.")
        if speeds.avg_travel_mms is not None and speeds.avg_travel_mms > (500 if is_high_flow else 200):
            flags.append("Very high travel speed; may reduce positional accuracy.")

        # First layer speed heuristic (requires some feed data; we don't track Z/feeds precisely per layer here)
        # Provide a generic hint if overall average is quite fast
        if not heur.first_layer_found and speeds.avg_extrusion_mms is not None and speeds.avg_extrusion_mms > 60:
            flags.append("Average print speed high; ensure first layer is slow (<=30 mm/s).")

        # First-layer heuristics
        if heur.first_layer_found:
            # First-layer fan vs material
            if heur.first_layer_fan_max_percent is not None:
                flfan = heur.first_layer_fan_max_percent
                if material in {"PETG"} and flfan > 30:
                    flags.append("First layer fan high for PETG; reduce to improve adhesion.")
                if material in {"ABS", "ASA"} and flfan > 10:
                    flags.append("First layer fan enabled for ABS/ASA; may cause warping.")
            # First-layer speeds
            fl_extr_thresh = 1000 if is_high_flow else 35
            fl_travel_thresh = 1000 if is_high_flow else 100
            if heur.first_layer_extrusion_max_mms is not None and heur.first_layer_extrusion_max_mms > fl_extr_thresh:
                flags.append(f"First layer extrusion speed high (>{fl_extr_thresh} mm/s); risk of poor adhesion.")
            if heur.first_layer_travel_max_mms is not None and heur.first_layer_travel_max_mms > fl_travel_thresh:
                flags.append(f"First layer travel speed high (>{fl_travel_thresh} mm/s); risk of first-layer shifts.")
            # Skirt/Brim presence
            if not heur.skirt_brim_present:
                flags.append("No skirt/brim detected; first-layer priming may be insufficient.")

        # Bridge cooling
        if heur.bridge_segments > 0 and material in {"PLA", "PETG"}:
            if heur.bridge_low_fan_segments > 0:
                flags.append("Bridge moves with low fan; overhangs may sag.")

        # Travel without retraction
        if heur.travel_no_retract_count > 0:
            if heur.max_travel_no_retract_mm >= 20.0:
                flags.append("Long travels without retraction (>20 mm); expect stringing.")
            else:
                flags.append("Travels without retraction detected; may cause stringing.")

        # Extrusion mode toggles
        if heur.mode_switches > 1:
            flags.append("Extrusion mode toggled multiple times (M82/M83); may cause extrusion errors.")
        if extrusion.absolute_mode is None:
            flags.append("Extrusion mode (M82/M83) not set; relies on firmware default.")
        # Excessive E resets
        if extrusion.resets > 0:
            per1000 = extrusion.resets / max(1.0, (length_mm / 1000.0))
            if per1000 > 80:
                flags.append("Very frequent G92 E resets; may indicate slicer issue.")
            elif per1000 > 30:
                flags.append("Frequent G92 E resets; check slicer retraction/relative settings.")
        if retraction.samples and retraction.samples and retraction.avg_distance is not None and retraction.avg_distance > 0 and retraction.max_distance is not None and retraction.max_distance < 0.4:
            flags.append("Retractions very short (<0.4 mm); may not relieve pressure.")

        # Overrides
        if heur.feedrate_override is not None and int(heur.feedrate_override) != 100:
            flags.append(f"Feedrate override (M220) set to {int(heur.feedrate_override)}%; speeds changed from profile.")
        if heur.flow_override is not None and not (95 <= heur.flow_override <= 105):
            flags.append(f"Flow override (M221) set to {heur.flow_override:.0f}%; extrusion scaling active.")

        # Extrusion mode warning handled above; removed no-op placeholders.

        # Frequent temp changes (prefer print-phase temps), filter out purge/cooldown
        src = temps.nozzle_print_setpoints or [t for t in temps.nozzle_setpoints if t >= 150]
        unique_nozzle = len(set(int(t) for t in src)) if src else 0
        if unique_nozzle >= 4:
            flags.append("Frequent nozzle temp changes; can destabilize flow/adhesion.")
        # Bed temp swings during print
        bed_src = temps.bed_print_setpoints or [t for t in temps.bed_setpoints if t >= 40]
        if bed_src:
            bmin, bmax = min(bed_src), max(bed_src)
            # Keep a slightly higher threshold for Bambu family in general
            bed_swing_thresh = 25 if (heur.printer_model and heur.printer_model.startswith("bambu_")) else 15
            if (bmax - bmin) >= bed_swing_thresh:
                flags.append("Large bed temp swing (>=15C) during print; adhesion may vary.")

        # Pauses present
        if heur.pauses_detected > 0:
            flags.append("Pause commands present (M0/M1/M25); print will wait for user.")

        # Volumetric flow checks (mm^3/s)
        def vol_limits(mat: str, printer: Optional[str]) -> Tuple[float, float]:
            m = mat.upper()
            base_map = {
                "PLA": (14.0, 22.0),
                "PETG": (10.0, 16.0),
                "ABS": (12.0, 18.0),
                "ASA": (12.0, 18.0),
                "TPU": (6.0, 8.0),
                "NYLON": (12.0, 18.0),
                "PA": (12.0, 18.0),
                "PC": (12.0, 18.0),
            }
            base = base_map.get(m, (12.0, 20.0))

            # Printer/hotend specific caps
            if printer == "bambu_h2s":
                # Tuned for H2S 0.4 nozzle: warn/severe guardrails by material.
                h2s_map = {
                    "PLA": (22.0, 28.0),
                    "PETG": (12.0, 18.0),
                    "ABS": (15.0, 22.0),
                    "ASA": (14.0, 20.0),
                    "PC": (12.0, 18.0),
                    "NYLON": (12.0, 18.0),
                    "PA": (12.0, 18.0),
                    "TPU": (6.0, 9.0),
                }
                # Use modest uplift for unknowns
                return h2s_map.get(m, (base[0] * 1.2, base[1] * 1.2))

            # Generic multipliers for other printers (conservative)
            mult = {
                None: 1.0,
                "bambu_a1": 1.0,
                "bambu_p1s": 1.1,
                "bambu_x1c": 1.2,
                "bambu_h2d": 1.2,
            }.get(printer, 1.0)
            return base[0] * mult, base[1] * mult

        if heur.max_volumetric_mm3_s is not None:
            warn, severe = vol_limits(material, heur.printer_model)
            if heur.max_volumetric_mm3_s > severe:
                flags.append(f"Volumetric flow very high (~{heur.max_volumetric_mm3_s:.1f} mm^3/s); may exceed hotend capability.")
            elif heur.max_volumetric_mm3_s > warn:
                flags.append(f"Volumetric flow high (~{heur.max_volumetric_mm3_s:.1f} mm^3/s); watch for under-extrusion.")
            # Compare against typical filament safe limits (stricter guidance)
            try:
                safe_low, safe_high = _flow_safe_limits(material, heur.printer_model, filament_mvs_mm3_s)
                # Apply 5%/0.5 mm^3/s tolerance
                tol = max(0.5, safe_high * 0.05)
                if heur.max_volumetric_mm3_s > (safe_high + tol):
                    flags.append(
                        f"Flow above {('MVS' if safe_high and safe_high != 0 else 'typical')} {material} limit (~{safe_high:.0f} mm^3/s); risk of under-extruded or matte bands."
                    )
            except Exception:
                pass
        # First-layer volumetric threshold scaled by printer capability
        if heur.max_volumetric_first_layer_mm3_s is not None:
            if heur.printer_model == "bambu_h2s":
                # Tuned per-material first-layer caution thresholds for H2S
                fl_map = {
                    "PLA": 16.0,
                    "PETG": 9.0,
                    "ABS": 12.0,
                    "ASA": 12.0,
                    "PC": 11.0,
                    "NYLON": 11.0,
                    "PA": 11.0,
                    "TPU": 7.0,
                }
                first_layer_warn = fl_map.get(material.upper(), 12.0)
            else:
                # Reuse generic multipliers for others
                mult = {
                    None: 1.0,
                    "bambu_a1": 1.0,
                    "bambu_p1s": 1.1,
                    "bambu_x1c": 1.2,
                    "bambu_h2d": 1.2,
                }.get(heur.printer_model, 1.0)
                first_layer_warn = 12.0 * mult
            if heur.max_volumetric_first_layer_mm3_s > first_layer_warn:
                flags.append("First-layer volumetric flow high; reduce speed or increase temp/line width.")

        return flags


# ------------------------------ CLI Formatting ----------------------------- #


def fmt_float(val: Optional[float], unit: str = "", digits: int = 2) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}{unit}"


def print_report(summary: GcodeSummary) -> None:
    print(f"File: {summary.file}")
    print("")
    if summary.printer_model:
        pretty = _printer_pretty(summary.printer_model) or summary.printer_model
        print("Printer")
        print(f"- Detected: {pretty}")
        print("")
    # Print/job metadata
    if any(v is not None for v in (summary.total_layers, summary.estimated_time_total_s, summary.model_print_time_s, summary.max_z_height_mm)):
        def _fmt_time(seconds: Optional[int]) -> str:
            if not seconds:
                return "-"
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            if h:
                return f"{h}h {m}m {s}s"
            if m:
                return f"{m}m {s}s"
            return f"{s}s"
        print("Print")
        if summary.estimated_time_total_s is not None:
            print(f"- Est. total time: {_fmt_time(summary.estimated_time_total_s)}")
        if summary.model_print_time_s is not None:
            print(f"- Model time: {_fmt_time(summary.model_print_time_s)}")
        # Also show naive integrated time from G-code distances/feeds (typically underestimates actual time)
        try:
            t_obs = int(float(getattr(summary.heuristics, 'total_time_s', 0.0)))
        except Exception:
            t_obs = 0
        if t_obs > 0:
            print(f"- Observed time (integrated): {_fmt_time(t_obs)}")
        if summary.total_layers is not None:
            print(f"- Layers: {summary.total_layers}")
        if summary.max_z_height_mm is not None:
            print(f"- Max Z height: {summary.max_z_height_mm:.2f} mm")
        print("")
    print("Material")
    print(f"- Inferred: {summary.inferred_material} (confidence: {summary.material_confidence})")
    print(f"- Filament diameter: {summary.filament_diameter_mm:.2f} mm")
    print(f"- Density used: {summary.density_used_g_cm3:.2f} g/cm^3")
    if any([summary.filament_brand, summary.filament_name, summary.filament_color]):
        if summary.filament_brand:
            print(f"- Brand: {summary.filament_brand}")
        if summary.filament_name:
            print(f"- Spool/name: {summary.filament_name}")
        if summary.filament_color:
            print(f"- Color: {summary.filament_color}")
    print("")
    print("Usage")
    print(f"- Length: {summary.estimated_length_mm/1000.0:.2f} m")
    print(f"- Volume: {summary.estimated_volume_cm3:.2f} cm^3")
    print(f"- Mass: {summary.estimated_mass_g:.1f} g")
    # Compare with slicer estimates when available
    if any(v is not None for v in (summary.slicer_length_mm, summary.slicer_volume_cm3, summary.slicer_mass_g)):
        print(f"- Slicer length: {fmt_float((summary.slicer_length_mm or 0)/1000.0, ' m', 2)}")
        print(f"- Slicer volume: {fmt_float(summary.slicer_volume_cm3, ' cm^3', 2)}")
        print(f"- Slicer mass: {fmt_float(summary.slicer_mass_g, ' g', 1)}")
        # Difference percentages if both present
        if summary.slicer_length_mm and summary.estimated_length_mm:
            d = (summary.estimated_length_mm - summary.slicer_length_mm) / summary.slicer_length_mm * 100.0
            print(f"- Length delta: {d:+.1f}% vs slicer")
        if summary.slicer_volume_cm3 and summary.estimated_volume_cm3:
            d = (summary.estimated_volume_cm3 - summary.slicer_volume_cm3) / summary.slicer_volume_cm3 * 100.0
            print(f"- Volume delta: {d:+.1f}% vs slicer")
        if summary.slicer_mass_g and summary.estimated_mass_g:
            d = (summary.estimated_mass_g - summary.slicer_mass_g) / summary.slicer_mass_g * 100.0
            print(f"- Mass delta: {d:+.1f}% vs slicer")
    print("")
    print("Temperatures")
    n0 = fmt_float(summary.temps.first_layer_nozzle, "C", 0)
    b0 = fmt_float(summary.temps.first_layer_bed, "C", 0)
    c0 = fmt_float(summary.temps.first_layer_chamber, "C", 0)
    # Prefer averages during print phase if available
    # Prefer print-phase temps; otherwise, filter out purge/cooldown values
    nz_list = summary.temps.nozzle_print_setpoints or [t for t in summary.temps.nozzle_setpoints if t >= 150]
    bd_list = summary.temps.bed_print_setpoints or [t for t in summary.temps.bed_setpoints if t >= 40]
    ch_list = summary.temps.chamber_print_setpoints or [t for t in summary.temps.chamber_setpoints if t >= 0]
    navg = fmt_float((sum(nz_list) / len(nz_list)) if nz_list else None, "C", 0)
    bavg = fmt_float((sum(bd_list) / len(bd_list)) if bd_list else None, "C", 0)
    cavg = fmt_float((sum(ch_list) / len(ch_list)) if ch_list else None, "C", 0)
    print(f"- First layer nozzle: {n0}")
    print(f"- First layer bed: {b0}")
    if ch_list or summary.temps.first_layer_chamber is not None:
        print(f"- First layer chamber: {c0}")
    print(f"- Avg nozzle: {navg}")
    print(f"- Avg bed: {bavg}")
    if ch_list:
        print(f"- Avg chamber: {cavg}")
    print("")
    # Layer metrics (distance and time)
    if summary.heuristics and (summary.heuristics.layer_move_mm or summary.heuristics.layer_time_s):
        def _fmt_time_short(seconds: Optional[float]) -> str:
            if seconds is None:
                return "-"
            s = int(round(seconds))
            m = s // 60
            s = s % 60
            if m:
                return f"{m}m {s}s"
            return f"{s}s"
        print("Layer Metrics")
        if summary.avg_layer_move_mm is not None:
            print(f"- Avg movement/layer: {summary.avg_layer_move_mm:.1f} mm")
            print(f"- First layer movement: {fmt_float(summary.first_layer_move_mm, ' mm', 1)}")
            print(f"- Last layer movement: {fmt_float(summary.last_layer_move_mm, ' mm', 1)}")
        if summary.avg_layer_time_s is not None:
            print(f"- Avg time/layer: {_fmt_time_short(summary.avg_layer_time_s)}")
            print(f"- First layer time: {_fmt_time_short(summary.first_layer_time_s)}")
            print(f"- Last layer time: {_fmt_time_short(summary.last_layer_time_s)}")
        # Show significant consecutive deltas (distance/time) with robust thresholds
        def _median(xs: List[float]) -> float:
            if not xs:
                return 0.0
            s = sorted(xs)
            n = len(s)
            m = n // 2
            return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])
        def _mad(xs: List[float]) -> float:
            if not xs:
                return 0.0
            med = _median(xs)
            dev = [abs(x - med) for x in xs]
            return _median(dev)
        def _sig_jumps(keys: List[int], vals: List[float], abs_floor: float, pct_min: float, sigma_k: float, min_baseline: float, skip_first_n: int = 2) -> List[str]:
            if len(keys) < 2:
                return []
            diffs: List[float] = []
            for i in range(1, len(keys)):
                diffs.append(vals[i] - vals[i-1])
            med = _median(diffs)
            mad = _mad(diffs)
            scale = mad * 1.4826 if mad > 0 else 0.0
            out: List[str] = []
            for i in range(1, len(keys)):
                # Skip very early layers where startup behavior is noisy
                if i <= skip_first_n:
                    continue
                a, b = vals[i-1], vals[i]
                d = b - a
                if a <= min_baseline:
                    continue
                p = (abs(d) / a * 100.0) if a > 0 else 0.0
                z_ok = (scale > 0 and abs(d - med) >= sigma_k * scale)
                abs_ok = abs(d) >= abs_floor
                pct_ok = p >= pct_min
                # Oscillation filter: if the next diff reverses sign with similar magnitude, ignore
                if i < len(keys) - 1:
                    next_d = vals[i+1] - vals[i]
                    if (d > 0 and next_d < 0) or (d < 0 and next_d > 0):
                        if abs(next_d) >= 0.6 * abs(d):
                            continue
                # Require robust deviation and percent; or a strong absolute+percent change
                strong_abs = abs_ok and abs(d) >= 1.5 * abs_floor
                if (z_ok and pct_ok) or (strong_abs and pct_ok):
                    out.append((i, d, p))
            # Format top 10 by absolute delta
            out_sorted = sorted(out, key=lambda t: abs(t[1]), reverse=True)[:10]
            lines: List[str] = []
            for i, d, p in out_sorted:
                lines.append(f"L{keys[i-1]}→L{keys[i]}: {d:+.0f} mm ({p:.0f}%)")
            return lines

        lm = summary.heuristics.layer_move_mm
        lt = summary.heuristics.layer_time_s
        if lm:
            m_keys = sorted(lm.keys())
            m_vals = [lm[k] for k in m_keys]
            m_med = _median(m_vals)
            mv_lines = _sig_jumps(
                m_keys,
                m_vals,
                abs_floor=max(LM_ABS_FLOOR_MM_FACTOR * m_med, LM_ABS_FLOOR_MM_MIN),
                pct_min=LM_PCT_MIN,
                sigma_k=LM_SIGMA_K,
                min_baseline=LM_MIN_BASELINE_MM,
            )
            if mv_lines:
                print("- Significant movement jumps:")
                for s in mv_lines:
                    print(f"  • {s}")
        if lt:
            t_keys = sorted(lt.keys())
            t_vals = [lt[k] for k in t_keys]
            t_med = _median(t_vals)
            # Time is in seconds
            t_lines = _sig_jumps(
                t_keys,
                t_vals,
                abs_floor=max(LT_ABS_FLOOR_S_FACTOR * t_med, LT_ABS_FLOOR_S_MIN),
                pct_min=LT_PCT_MIN,
                sigma_k=LT_SIGMA_K,
                min_baseline=LT_MIN_BASELINE_S,
            )
            if t_lines:
                print("- Significant time jumps:")
                for s in t_lines:
                    # Replace unit for time lines
                    layer_pair, rest = s.split(": ", 1)
                    amt, pct = rest.split(" (", 1)
                    try:
                        amt_num = int(float(amt.split()[0]))
                    except Exception:
                        amt_num = 0
                    print(f"  • {layer_pair}: {amt_num:+d} s ({pct}")
        print("")
    print("Retraction")
    print(f"- Samples: {len(summary.retraction.samples)}")
    print(f"- Avg distance: {fmt_float(summary.retraction.avg_distance, ' mm', 2)}")
    print(f"- Max distance: {fmt_float(summary.retraction.max_distance, ' mm', 2)}")
    print(f"- Avg speed: {fmt_float(summary.retraction.avg_speed_mms, ' mm/s', 1)}")
    print(f"- Z-hops detected: {summary.retraction.z_hops}")
    # Frequency per 100 mm extruded
    if summary.estimated_length_mm > 0 and len(summary.retraction.samples) > 0:
        freq = len(summary.retraction.samples) / (summary.estimated_length_mm / 100.0)
        print(f"- Retracts per 100 mm: {freq:.1f}")
    print("")
    print("Speeds")
    print(f"- Avg extrusion: {fmt_float(summary.speeds.avg_extrusion_mms, ' mm/s', 1)}")
    print(f"- Max extrusion: {fmt_float(summary.speeds.max_extrusion_mms, ' mm/s', 1)}")
    print(f"- Avg travel: {fmt_float(summary.speeds.avg_travel_mms, ' mm/s', 1)}")
    print(f"- Max travel: {fmt_float(summary.speeds.max_travel_mms, ' mm/s', 1)}")
    # First layer speeds
    if summary.heuristics and summary.heuristics.first_layer_found:
        hl = summary.heuristics
        avg_fl_ext = (hl.first_layer_extrusion_sum_mms / hl.first_layer_extrusion_count) if hl.first_layer_extrusion_count else None
        avg_fl_trv = (hl.first_layer_travel_sum_mms / hl.first_layer_travel_count) if hl.first_layer_travel_count else None
        print(f"- 1st-layer avg extrusion: {fmt_float(avg_fl_ext, ' mm/s', 1)}")
        print(f"- 1st-layer max extrusion: {fmt_float(hl.first_layer_extrusion_max_mms, ' mm/s', 1)}")
        print(f"- 1st-layer avg travel: {fmt_float(avg_fl_trv, ' mm/s', 1)}")
        print(f"- 1st-layer max travel: {fmt_float(hl.first_layer_travel_max_mms, ' mm/s', 1)}")
        # First-layer dynamics
        if hl.fl_accel_print_max is not None or hl.fl_accel_travel_max is not None or any(
            v is not None for v in (hl.fl_jerk_x_max, hl.fl_jerk_y_max, hl.fl_jerk_z_max, hl.fl_jerk_e_max)
        ):
            print(f"- 1st-layer accel (print/travel): {fmt_float(hl.fl_accel_print_max, ' mm/s^2', 0)} / {fmt_float(hl.fl_accel_travel_max, ' mm/s^2', 0)}")
            jx = fmt_float(hl.fl_jerk_x_max, ' mm/s', 1)
            jy = fmt_float(hl.fl_jerk_y_max, ' mm/s', 1)
            jz = fmt_float(hl.fl_jerk_z_max, ' mm/s', 1)
            je = fmt_float(hl.fl_jerk_e_max, ' mm/s', 1)
            print(f"- 1st-layer jerk X/Y/Z/E: {jx} / {jy} / {jz} / {je}")
    print("")
    # Volumetric flow
    if summary.heuristics and (
        summary.heuristics.max_volumetric_mm3_s is not None or summary.heuristics.max_volumetric_first_layer_mm3_s is not None or summary.heuristics.avg_volumetric_mm3_s is not None
    ):
        print("Flow")
        if summary.heuristics.avg_volumetric_mm3_s is not None:
            print(f"- Avg volumetric: {summary.heuristics.avg_volumetric_mm3_s:.1f} mm^3/s")
        if summary.heuristics.max_volumetric_mm3_s is not None:
            print(f"- Max volumetric: {summary.heuristics.max_volumetric_mm3_s:.1f} mm^3/s")
        if summary.heuristics.avg_volumetric_first_layer_mm3_s is not None:
            print(f"- Avg first-layer flow: {summary.heuristics.avg_volumetric_first_layer_mm3_s:.1f} mm^3/s")
        if summary.heuristics.max_volumetric_first_layer_mm3_s is not None:
            print(f"- Max first-layer flow: {summary.heuristics.max_volumetric_first_layer_mm3_s:.1f} mm^3/s")
        print("")
    print("Cooling")
    print(f"- Avg fan: {fmt_float(summary.fan.avg_percent, ' %', 0)}")
    print(f"- Max fan: {fmt_float(summary.fan.max_percent, ' %', 0)}")
    print("")
    # Additional heuristics and overrides
    if summary.heuristics or summary.extrusion:
        print("Heuristics")
        if summary.heuristics:
            if summary.heuristics.first_layer_found:
                print("- First layer detected: yes")
            if summary.heuristics.first_layer_fan_max_percent is not None:
                print(f"- 1st-layer max fan: {summary.heuristics.first_layer_fan_max_percent:.0f} %")
            if summary.heuristics.bed_leveling_enabled_at_first_layer is not None:
                print(
                    f"- Mesh at layer 1: {'enabled' if summary.heuristics.bed_leveling_enabled_at_first_layer else 'disabled'}"
                )
            if summary.heuristics.travel_no_retract_count:
                print(f"- Travels without retract: {summary.heuristics.travel_no_retract_count}")
            if summary.heuristics.max_travel_no_retract_mm:
                print(f"- Longest travel w/o retract: {summary.heuristics.max_travel_no_retract_mm:.1f} mm")
            if summary.heuristics.skirt_brim_present:
                print("- Skirt/brim present: yes")
            if summary.heuristics.bridge_segments:
                low = summary.heuristics.bridge_low_fan_segments or 0
                print(f"- Bridge segments: {summary.heuristics.bridge_segments} (low fan: {low})")
            if summary.heuristics.pauses_detected:
                print(f"- Pauses detected: {summary.heuristics.pauses_detected}")
            if summary.heuristics.feedrate_override is not None:
                print(f"- Feedrate override (M220): {summary.heuristics.feedrate_override:.0f}%")
            if summary.heuristics.flow_override is not None:
                print(f"- Flow override (M221): {summary.heuristics.flow_override:.0f}%")
        if summary.extrusion:
            mode = (
                "absolute (M82)" if summary.extrusion.absolute_mode else ("relative (M83)" if summary.extrusion.absolute_mode is not None else "-")
            )
            print(f"- Extrusion mode: {mode}")
            print(f"- G92 E resets: {summary.extrusion.resets}")
            print(f"- Positive extrusions: {summary.extrusion.positive_extrusions}")
        print("")
    print("Potential Issues")
    if summary.flags:
        for f in summary.flags:
            print(f"- {f}")
    else:
        print("- None detected")
    # Extra notes (e.g., slicer estimates)
    if summary.notes:
        print("")
        print("Notes")
        for n in summary.notes:
            print(f"- {n}")


def _read_single_key() -> Optional[str]:
    """Read a single key without requiring Enter on Windows; fall back to input().
    Returns the character pressed, or None if unavailable.
    """
    try:
        import msvcrt  # type: ignore
        ch = msvcrt.getwch()
        return ch
    except Exception:
        try:
            s = input()
            return s[:1] if s else "\n"
        except Exception:
            return None


def _final_hold(prompt: str = "Press any key to exit...") -> None:
    """Block until a keypress using best available method to prevent window auto-close.

    Works both when a console is attached and when launched without a console
    (e.g., double-clicking/drag-dropping onto the .py on Windows).
    """
    # First, try Windows console without requiring Enter
    try:
        import msvcrt  # type: ignore
        try:
            print(prompt, end="", flush=True)
        except Exception:
            pass
        try:
            msvcrt.getwch()
        except Exception:
            try:
                msvcrt.getch()
            except Exception:
                # Fall through to other strategies
                raise
        try:
            print("")
        except Exception:
            pass
        return
    except Exception:
        # Not on Windows or no console available
        pass

    # Next, try standard input (console requiring Enter)
    try:
        try:
            # Ensure prompt is visible in typical consoles
            print(prompt)
        except Exception:
            pass
        _ = input()
        return
    except Exception:
        # No stdin or input() not possible (pythonw.exe)
        pass

    # Last resort: GUI dialog that requires user to close
    try:
        import tkinter as tk  # type: ignore
        from tkinter import messagebox  # type: ignore
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("G-code Inspector", "Close this window to exit.")
    except Exception:
        # If even this fails, nothing else we can do
        pass


def _should_hold() -> bool:
    """Heuristic: return True if likely launched without an interactive TTY.

    Used for non-error paths where we optionally hold to keep the window open.
    """
    try:
        return not (hasattr(sys, 'stdin') and sys.stdin and sys.stdin.isatty())
    except Exception:
        return True


def _plot_layer_metrics(summary: GcodeSummary, out_path: Optional[Path] = None) -> Optional[Path]:
    """Per-layer movement/time with jump markers; returns Path or None."""
    heur = summary.heuristics
    if not heur or (not heur.layer_move_mm and not heur.layer_time_s):
        return None
    # Ensure matplotlib is available; on ImportError, try one-time install
    plt = _ensure_matplotlib()
    if plt is None:
        # Could not import/install; record note and exit quietly
        summary.notes.append("Matplotlib not available; skipping plot.")
        return None

    # Prepare series
    layers = sorted(set(list(heur.layer_move_mm.keys()) + list(heur.layer_time_s.keys())))
    move = [heur.layer_move_mm.get(i, float('nan')) for i in layers]
    times = [heur.layer_time_s.get(i, float('nan')) for i in layers]

    # Compute significant jumps using a robust delta-based detector
    def _finite_series(vals: List[float]) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for i, v in enumerate(vals):
            if isinstance(v, float) and not math.isnan(v):
                out.append((i, float(v)))
        return out

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        if n % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _mad(xs: List[float]) -> float:
        # Median absolute deviation
        if not xs:
            return 0.0
        m = _median(xs)
        dev = [abs(x - m) for x in xs]
        return _median(dev)

    def robust_jump_mask(vals: List[float], abs_floor: float, pct_min: float, sigma_k: float = 3.0, min_baseline: float = 1e-6) -> List[bool]:
        mask = [False] * len(vals)
        finite = _finite_series(vals)
        if len(finite) < 2:
            return mask
        # Differences between consecutive finite points (aligned to the later index)
        diffs: List[Tuple[int, float, float]] = []  # (idx, diff, prev_val)
        for (i_prev, v_prev), (i_cur, v_cur) in zip(finite[:-1], finite[1:]):
            diffs.append((i_cur, v_cur - v_prev, v_prev))
        diff_vals = [d for (_, d, _) in diffs]
        med = _median(diff_vals)
        mad = _mad(diff_vals)
        # Scale MAD to approximate std-dev (consistency with normal dist.)
        # If MAD==0 (flat regions), fall back to abs_floor only.
        scale = mad * 1.4826 if mad > 0 else 0.0
        for idx, d, prev in diffs:
            if prev <= min_baseline:
                continue
            pct = (abs(d) / prev) * 100.0 if prev > 0 else 0.0
            # Two-part rule: robust z-score AND absolute floor AND percent change
            z_ok = (scale > 0 and abs(d - med) >= sigma_k * scale)
            abs_ok = abs(d) >= abs_floor
            pct_ok = pct >= pct_min
            if (z_ok or abs_ok) and pct_ok:
                mask[idx] = True
        return mask

    # Thresholds derived from robust central tendency
    mv_vals = [v for v in move if isinstance(v, float) and not math.isnan(v)]
    t_vals = [v for v in times if isinstance(v, float) and not math.isnan(v)]
    mv_med = _median(mv_vals)
    t_med = _median(t_vals)
    # Movement deltas must exceed either 8% of median movement or 60 mm, and 35% relative change
    mv_mask = robust_jump_mask(move, abs_floor=max(0.08 * mv_med, 60.0), pct_min=35.0, sigma_k=3.0, min_baseline=5.0)
    # Time deltas must exceed either 12% of median time or 12 s, and 35% relative change
    t_mask = robust_jump_mask(times, abs_floor=max(0.12 * t_med, 12.0), pct_min=35.0, sigma_k=3.0, min_baseline=5.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1, ax2 = axes
    ax1.plot(layers, move, label="Movement (mm)", color="#1f77b4")
    # Highlight significant points
    sig_layers = [layers[i] for i, m in enumerate(mv_mask) if m]
    sig_vals = [move[i] for i, m in enumerate(mv_mask) if m]
    if sig_layers:
        ax1.scatter(sig_layers, sig_vals, color="red", zorder=3, label="Significant jump")
    ax1.set_ylabel("Movement (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(layers, [v/60.0 if isinstance(v, float) and not math.isnan(v) else float('nan') for v in times],
             label="Time (min)", color="#2ca02c")
    sig_layers_t = [layers[i] for i, m in enumerate(t_mask) if m]
    sig_vals_t = [times[i]/60.0 for i, m in enumerate(t_mask) if m]
    if sig_layers_t:
        ax2.scatter(sig_layers_t, sig_vals_t, color="red", zorder=3, label="Significant jump")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Time (min)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    _apply_plot_style(fig,
                      title="Layer Metrics",
                      subtitle=_plot_subtitle(summary),
                      caption=(
                          "Red markers = significant layer-to-layer jumps.\n"
                          "Big per-layer deltas often align with feature/cooling changes; watch for artifacts/time spikes."
                      ))

    if out_path is None:
        out_path = summary.file.with_suffix('.layer_metrics.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # Attach note
    summary.notes.append(f"Saved layer metrics plot: {out_path.name}")
    return out_path


def _plot_e_per_mm(summary: GcodeSummary, out_path: Optional[Path] = None, window: Optional[int] = None) -> Optional[Path]:
    """E/mm per segment with rolling mean/std; returns Path or None."""
    heur = summary.heuristics
    if not heur or not heur.e_per_mm_samples:
        return None
    plt = _ensure_matplotlib()
    if plt is None:
        summary.notes.append("Matplotlib not available; skipping E/mm plot.")
        return None

    series = [v for v in heur.e_per_mm_samples if isinstance(v, float) and not math.isnan(v) and math.isfinite(v)]
    if not series:
        return None
    layers_opt = heur.e_per_mm_layers if (hasattr(heur, 'e_per_mm_layers') and heur.e_per_mm_layers and len(heur.e_per_mm_layers) == len(heur.e_per_mm_samples)) else None

    n = len(series)
    x = list(range(n))
    # Determine rolling window size if not provided: ~1% of segments, bounded [50, 1000]
    if window is None:
        w = max(50, min(1000, max(1, int(n * 0.01))))
    else:
        w = max(2, int(window))

    # Rolling standard deviation using cumulative sums for O(n)
    csum = [0.0]
    csum2 = [0.0]
    for v in series:
        csum.append(csum[-1] + v)
        csum2.append(csum2[-1] + v * v)

    def roll_std(idx: int) -> float:
        # centered trailing window [idx-w+1, idx]
        i0 = max(0, idx - w + 1)
        i1 = idx + 1
        k = i1 - i0
        if k <= 1:
            return 0.0
        s = csum[i1] - csum[i0]
        s2 = csum2[i1] - csum2[i0]
        mean = s / k
        var = max(0.0, (s2 / k) - (mean * mean))
        return math.sqrt(var)

    rolling_std = [roll_std(i) for i in range(n)]

    # Rolling mean for trend
    def roll_mean(idx: int) -> float:
        i0 = max(0, idx - w + 1)
        i1 = idx + 1
        k = i1 - i0
        if k <= 0:
            return series[idx]
        return (csum[i1] - csum[i0]) / k
    rolling_mean = [roll_mean(i) for i in range(n)]

    # Robust y-limits via percentiles
    def _percentile(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        pos = (len(s) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    p1 = _percentile(series, 0.01)
    p99 = _percentile(series, 0.99)

    # Spike/dip detection using rolling mean/std with percentile gating and prominence
    spikes: List[int] = []
    dips: List[int] = []
    for i, xval in enumerate(series):
        mu = rolling_mean[i]
        sd = max(1e-9, rolling_std[i])
        z = (xval - mu) / sd
        is_spike = (z >= 3.5) and (xval >= p99)
        is_dip = (z <= -3.5) and (xval <= p1)
        # Prominence relative to local mean
        prom = abs(xval - mu)
        prom_ok = prom >= max(0.01, 0.25 * max(mu, 1e-6))
        if is_spike and prom_ok:
            spikes.append(i)
        elif is_dip and prom_ok:
            dips.append(i)

    # Collapse contiguous detections to representative peaks (max |z|)
    def _collapse(idxs: List[int]) -> List[int]:
        out: List[int] = []
        if not idxs:
            return out
        run = [idxs[0]]
        for j in idxs[1:]:
            if j == run[-1] + 1:
                run.append(j)
            else:
                # pick best in run
                best = max(run, key=lambda k: abs((series[k] - rolling_mean[k]) / max(1e-9, rolling_std[k])))
                out.append(best)
                run = [j]
        best = max(run, key=lambda k: abs((series[k] - rolling_mean[k]) / max(1e-9, rolling_std[k])))
        out.append(best)
        return out

    spikes = _collapse(spikes)
    dips = _collapse(dips)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x, series, color="#1f77b4", linewidth=0.6, alpha=0.6, label="E/mm per segment")
    ax1.plot(x, rolling_mean, color="#2ca02c", linewidth=1.2, label=f"Rolling mean (w={w})")
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Extrusion per distance (E/mm)")
    ax1.grid(True, alpha=0.3)
    # Apply clipped limits with small padding
    if p99 > p1:
        pad = max(1e-6, (p99 - p1) * 0.1)
        ax1.set_ylim(p1 - pad, p99 + pad)

    # Secondary axis for rolling std-dev
    ax2 = ax1.twinx()
    ax2.plot(x, rolling_std, color="#d62728", linewidth=1.0, alpha=0.85, label=f"Rolling std-dev (w={w})")
    ax2.set_ylabel("Rolling std-dev (E/mm)")

    # Mark spikes/dips
    if spikes:
        ax1.scatter([x[i] for i in spikes], [series[i] for i in spikes], s=18, color="#ff7f0e", marker="^", label="Spikes")
    if dips:
        ax1.scatter([x[i] for i in dips], [series[i] for i in dips], s=18, color="#1f77b4", marker="v", label="Dips")

    # Build a combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    _apply_plot_style(fig,
                      title="Extrusion per Distance (E/mm)",
                      subtitle=_plot_subtitle(summary),
                      caption=(
                          "Blue = E/mm, Green = rolling mean, Red = rolling std (right axis). Triangles mark spikes/dips.\n"
                          "Stable E/mm => consistent line width; spikes/dips suggest flow/pressure issues (PA/temp/moisture/clogs)."
                      ))

    if out_path is None:
        out_path = summary.file.with_suffix('.e_per_mm.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # Add a brief anomalies summary to notes
    if spikes or dips:
        summary.notes.append(f"E/mm anomalies: spikes={len(spikes)}, dips={len(dips)}")
    if layers_opt and (spikes or dips):
        # Report top 3 by |z| with layer indices
        all_idxs = spikes + dips
        scored = sorted(all_idxs, key=lambda k: abs((series[k] - rolling_mean[k]) / max(1e-9, rolling_std[k])), reverse=True)[:3]
        parts = []
        for k in scored:
            z = (series[k] - rolling_mean[k]) / max(1e-9, rolling_std[k])
            layer = layers_opt[k] if k < len(layers_opt) else None
            parts.append(f"L{layer if layer is not None else '?'}: z={z:+.1f}")
        if parts:
            summary.notes.append("E/mm top anomalies: " + ", ".join(parts))
    summary.notes.append(f"Saved E/mm plot: {out_path.name}")
    return out_path

def _cooling_thresholds(material: str) -> Tuple[float, float]:
    m = (material or "").upper()
    # fan_low (%), short_time (s)
    if "PLA" in m:
        return 20.0, 20.0
    if "PETG" in m:
        return 10.0, 20.0
    if "ABS" in m or "ASA" in m:
        return 5.0, 15.0
    return 15.0, 20.0


def _plot_cooling_state(summary: GcodeSummary, out_path: Optional[Path] = None) -> Optional[Path]:
    """Layer time vs time-weighted fan%; returns Path or None."""
    heur = summary.heuristics
    if not heur or not heur.layer_time_s:
        return None
    plt = _ensure_matplotlib()
    if plt is None:
        summary.notes.append("Matplotlib not available; skipping cooling plot.")
        return None

    layers = sorted(heur.layer_time_s.keys())
    t_s = [heur.layer_time_s.get(i, 0.0) for i in layers]
    # Compute time-weighted average fan per layer
    fan_pct = []
    for i in layers:
        t = heur.layer_time_s.get(i, 0.0)
        ft = heur.layer_fan_time_255.get(i, 0.0)
        pct = (ft / t / 255.0 * 100.0) if t > 0 else 0.0
        fan_pct.append(pct)

    low_fan, short_time = _cooling_thresholds(summary.inferred_material)
    risk_mask = [(p <= low_fan and t <= short_time) for p, t in zip(fan_pct, t_s)]

    fig, ax = plt.subplots(figsize=(11, 6))
    # Plot non-risk points
    x = t_s
    y = fan_pct
    x_safe = [xv for xv, r in zip(x, risk_mask) if not r]
    y_safe = [yv for yv, r in zip(y, risk_mask) if not r]
    x_risk = [xv for xv, r in zip(x, risk_mask) if r]
    y_risk = [yv for yv, r in zip(y, risk_mask) if r]
    ax.scatter(x_safe, y_safe, s=18, alpha=0.7, label="Layers", color="#1f77b4")
    if x_risk:
        ax.scatter(x_risk, y_risk, s=32, alpha=0.9, label="High risk (low fan + short layer)", color="#d62728")

    # Risk region shading
    ax.axvspan(0, short_time, ymin=0.0, ymax=low_fan / 100.0, color="#d62728", alpha=0.12)
    ax.axhline(low_fan, color="#d62728", linestyle="--", linewidth=0.8)
    ax.axvline(short_time, color="#d62728", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Layer time (s)")
    ax.set_ylabel("Avg fan per layer (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    # Tidy limits with small padding
    if t_s:
        xmin, xmax = min(t_s), max(t_s)
        if xmax > xmin:
            pad = (xmax - xmin) * 0.08
            ax.set_xlim(max(0, xmin - pad), xmax + pad)
    if fan_pct:
        ymin, ymax = min(fan_pct), max(fan_pct)
        if ymax > ymin:
            pad = (ymax - ymin) * 0.08
            ax.set_ylim(max(0, ymin - pad), min(100.0, ymax + pad))
    _apply_plot_style(fig,
                      title="Cooling vs Layer Time",
                      subtitle=_plot_subtitle(summary),
                      caption=(
                          "Red shaded area = short layers with low fan.\n"
                          "Short layers need cooling; low fan here softens detail or warps."
                      ))

    if out_path is None:
        out_path = summary.file.with_suffix('.cooling.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    summary.notes.append(f"Saved cooling plot: {out_path.name}")
    return out_path

def _plot_corner_stress(summary: GcodeSummary, out_path: Optional[Path] = None) -> Optional[Path]:
    """Corner stress index over time; returns Path or None."""
    heur = summary.heuristics
    if not heur or not heur.corner_samples:
        return None
    plt = _ensure_matplotlib()
    if plt is None:
        summary.notes.append("Matplotlib not available; skipping corner stress plot.")
        return None

    samples = heur.corner_samples
    times_min = [t / 60.0 for (t, _, _, _, _, _) in samples]
    stress = [s for (_, s, _, _, _, _) in samples]

    fig, ax = plt.subplots(figsize=(12, 5))
    # Rolling mean for a smoother trendline
    n = len(stress)
    if n:
        w = max(10, min(1000, max(1, int(n * 0.01))))
        csum = [0.0]
        for v in stress:
            csum.append(csum[-1] + v)
        def rmean(i: int) -> float:
            i0 = max(0, i - w + 1)
            i1 = i + 1
            k = i1 - i0
            return (csum[i1] - csum[i0]) / max(1, k)
        smooth = [rmean(i) for i in range(n)]
    else:
        smooth = []
        w = 0
    ax.plot(times_min, stress, color="#9467bd", linewidth=0.6, alpha=0.6, label="Stress index")
    if smooth:
        ax.plot(times_min, smooth, color="#2ca02c", linewidth=1.4, label=f"Rolling mean (w={w})")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Stress index (dimensionless)")
    ax.grid(True, alpha=0.3)
    # Threshold guideline: 1.0 indicates incoming speed ≈ limit
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, label="Risk threshold")
    # Highlight exceedances
    ex_x = [tm for tm, s in zip(times_min, stress) if s >= 1.0]
    ex_y = [s for s in stress if s >= 1.0]
    if ex_x:
        ax.scatter(ex_x, ex_y, s=12, color="#d62728", alpha=0.9, marker="x", label=">= threshold")
    ax.legend(loc="upper right")
    _apply_plot_style(fig,
                      title="Corner Stress Index",
                      subtitle=_plot_subtitle(summary),
                      caption=(
                          "Dashed line = risk threshold (~1.0). Points above: corners where incoming speed approaches limits.\n"
                          "High stress correlates with ringing/overshoot/skips; useful for accel/JD/SCV tuning."
                      ))

    if out_path is None:
        out_path = summary.file.with_suffix('.corner_stress.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    summary.notes.append(f"Saved corner stress plot: {out_path.name}")
    return out_path

def _flow_safe_limits(material: str, printer: Optional[str] = None, mvs_override: Optional[float] = None) -> Tuple[float, float]:
    # If a filament-specific MVS is provided, prefer that (use a 20% band below it)
    if mvs_override and mvs_override > 0:
        high = float(mvs_override)
        low = max(0.0, high * 0.8)
        return low, high
    m = (material or "").upper()
    p = (printer or "").lower() if printer else None

    # Printer-specific adjustments: tuned H2S bands by material (safe shading band)
    if p == "bambu_h2s":
        if "PLA" in m:
            return 18.0, 25.0
        if "PETG" in m:
            return 9.0, 13.0
        if ("ABS" in m) or ("ASA" in m):
            return 11.0, 16.0
        if ("PA" in m) or ("NYLON" in m):
            return 10.0, 14.0
        if "PC" in m:
            return 10.0, 14.0
        if "TPU" in m:
            return 6.0, 8.0
        # Default modest uplift for unknowns
        return 8.0, 12.0

    # Material defaults (conservative) for non-H2S
    if "PETG" in m:
        return 6.0, 9.0
    if "ABS" in m or "ASA" in m:
        return 8.0, 12.0
    if "PLA" in m:
        return 8.0, 12.0
    if ("PA" in m) or ("NYLON" in m):
        return 8.0, 12.0
    if "PC" in m:
        return 8.0, 12.0
    # Default conservative range
    return 6.0, 10.0


def _plot_volumetric_flow(summary: GcodeSummary, out_path: Optional[Path] = None) -> Optional[Path]:
    """Volumetric flow (mm^3/s) vs time; returns Path or None.

    For long prints, splits the timeline into 15-minute windows so later
    intervals are visible rather than only the first segment.
    """
    heur = summary.heuristics
    if not heur or not heur.vol_samples:
        return None
    plt = _ensure_matplotlib()
    if plt is None:
        summary.notes.append("Matplotlib not available; skipping flow plot.")
        return None

    # Compute per-segment flow inputs (de, t, t_abs) then lightly condense by time-binning
    radius_mm = summary.filament_diameter_mm / 2.0
    area_mm2 = math.pi * radius_mm * radius_mm
    segs: List[Tuple[float, float, float]] = []  # (de, t, t_abs)
    cum = 0.0
    for sample in heur.vol_samples:
        try:
            de = float(sample[0])
            t = float(sample[1])
        except Exception:
            continue
        if t <= 0:
            continue
        # Prefer absolute end time if provided (index 3)
        t_abs: Optional[float] = None
        if len(sample) >= 4:
            try:
                t_abs = float(sample[3])
            except Exception:
                t_abs = None
        if t_abs is None or t_abs < 0:
            cum += t
            t_abs = cum
        segs.append((de, t, t_abs))

    if not segs:
        return None

    # Condense adjacent samples into small time windows to reduce noise/point count
    # Keep bins modest so peak detection still works; 50 ms works well for typical F values
    BIN_S = 0.050
    flows: List[float] = []
    times_s: List[float] = []
    over_count = 0
    low, high = _flow_safe_limits(summary.inferred_material, summary.printer_model, summary.filament_mvs_mm3_s)
    tol = max(0.5, high * 0.05) if high else 0.5

    if segs:
        # Ensure chronological order by t_abs (defensive)
        segs.sort(key=lambda x: x[2])
        bin_start = segs[0][2]
        acc_de = 0.0
        acc_t = 0.0
        last_t_abs = segs[0][2]
        for de, t, t_abs in segs:
            # Flush bin if we've crossed BIN_S in absolute time
            if (t_abs - bin_start) >= BIN_S and acc_t > 0.0:
                rate = (acc_de / acc_t) * area_mm2
                flows.append(rate)
                times_s.append(last_t_abs)
                if high and rate > (high + tol):
                    over_count += 1
                # Reset bin
                bin_start = t_abs
                acc_de = 0.0
                acc_t = 0.0
            # Accumulate current segment
            acc_de += de
            acc_t += t
            last_t_abs = t_abs
        # Flush any remainder
        if acc_t > 0.0:
            rate = (acc_de / acc_t) * area_mm2
            flows.append(rate)
            times_s.append(last_t_abs)
            if high and rate > (high + tol):
                over_count += 1

    if not flows:
        return None

    # Optionally scale the absolute time axis to slicer's model/estimated time to better match reality
    target_time_s = None
    try:
        if getattr(summary, 'model_print_time_s', None):
            target_time_s = float(summary.model_print_time_s)
        elif getattr(summary, 'estimated_time_total_s', None):
            target_time_s = float(summary.estimated_time_total_s)
    except Exception:
        target_time_s = None
    if target_time_s and times_s:
        observed_end = max(times_s)
        # Only stretch if the slicer time is meaningfully larger than our naive integration
        if observed_end > 0 and target_time_s > observed_end * 1.15:
            scale = target_time_s / observed_end
            times_s = [t * scale for t in times_s]

    t_min = [v / 60.0 for v in times_s]
    # Prepare smoothing once so we can reuse per window
    smooth = None
    n = len(flows)
    if n >= 5:
        w = max(10, min(1000, max(1, int(n * 0.01))))
        csum = [0.0]
        for v in flows:
            csum.append(csum[-1] + v)
        def rmean(i: int) -> float:
            i0 = max(0, i - w + 1)
            i1 = i + 1
            k = i1 - i0
            return (csum[i1] - csum[i0]) / max(1, k)
        smooth = [rmean(i) for i in range(n)]

    # If duration exceeds 1 hour, write multiple PNG segments (one per hour) for readability
    # Use observed end of print if larger than last sample time
    total_min = t_min[-1]
    try:
        # Prefer scaled/slicer time if present
        if target_time_s:
            total_min = max(total_min, float(target_time_s) / 60.0)
        elif hasattr(heur, 'total_time_s') and heur.total_time_s:
            total_min = max(total_min, float(heur.total_time_s) / 60.0)
    except Exception:
        pass
    # If duration exceeds 15 minutes, split into subplots within one figure
    window_min = 15.0
    if total_min > window_min + 1e-6:
        import math as _math
        windows = int(_math.ceil(total_min / window_min))
        fig, axes = plt.subplots(windows, 1, figsize=(12, max(5, 3.6 * windows)), sharey=True)
        # Matplotlib returns a numpy.ndarray of axes when windows>1; flatten safely
        try:
            axes = axes.ravel().tolist()  # type: ignore[attr-defined]
        except Exception:
            axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
        # Plot each 15-minute window into its own axis
        for i, ax in enumerate(axes):
            start = i * window_min
            end = min((i + 1) * window_min, total_min)
            mask = [(tm >= start) and (tm <= end if i == windows - 1 else tm < end) for tm in t_min]
            x = [tm for tm, m in zip(t_min, mask) if m]
            y = [fv for fv, m in zip(flows, mask) if m]
            if not x:
                # Still draw safe region to keep axis sizing
                ax.set_xlim(start, end)
                ax.set_ylabel("Flow (mm³/s)")
                ax.grid(True, alpha=0.3)
                ax.axhspan(low, high, color="#2ca02c", alpha=0.12)
                ax.axhline(high, color="#d62728", linestyle="--", linewidth=1.0)
                if i == windows - 1:
                    ax.set_xlabel("Time (min)")
                continue
            ax.plot(x, y, color="#1f77b4", linewidth=0.6, alpha=0.6)
            if smooth is not None:
                y2 = [sv for sv, m in zip(smooth, mask) if m]
                if y2:
                    ax.plot(x, y2, color="#2ca02c", linewidth=1.2)
            ax.set_xlim(start, end)
            ax.set_ylabel("Flow (mm³/s)")
            ax.grid(True, alpha=0.3)
            # Safe band and limit
            ax.axhspan(low, high, color="#2ca02c", alpha=0.12)
            ax.axhline(high, color="#d62728", linestyle="--", linewidth=1.0)
            # Exceedances in this window
            if high:
                ex_x = [tm for tm, fv, m in zip(t_min, flows, mask) if m and fv > (high + tol)]
                ex_y = [fv for fv, m in zip(flows, mask) if m and fv > (high + tol)]
                if ex_x:
                    ax.scatter(ex_x, ex_y, s=12, color="#d62728", alpha=0.9, marker="x")
            # Only label x-axis on last subplot to reduce clutter
            if i == windows - 1:
                ax.set_xlabel("Time (min)")

        # Apply common title/caption styling
        _apply_plot_style(
            fig,
            title="Volumetric Flow (mm³/s)",
            subtitle=_plot_subtitle(summary),
            caption=(
                "Split into 15-minute windows. Green band = typical safe range; dashed = limit; red × = exceedances.\n"
                "Above-limit flow risks under-extrusion/matte bands/weaker parts; use to set safe speeds/widths."
            ),
        )
    else:
        # Single-axes plot (short prints)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t_min, flows, color="#1f77b4", linewidth=0.6, alpha=0.6, label="Flow (mm³/s)")
        if smooth is not None:
            ax.plot(t_min, smooth, color="#2ca02c", linewidth=1.4, label="Rolling mean")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Flow (mm³/s)")
        ax.grid(True, alpha=0.3)
        # Safe range shading and threshold line
        ax.axhspan(low, high, color="#2ca02c", alpha=0.12, label=f"Safe {low:.0f}–{high:.0f}")
        ax.axhline(high, color="#d62728", linestyle="--", linewidth=1.0, label=f"Limit {high:.0f}")
        # Mark exceedances
        if high:
            exceed_x = [tm for tm, fv in zip(t_min, flows) if fv > (high + tol)]
            exceed_y = [fv for fv in flows if fv > (high + tol)]
            if exceed_x:
                ax.scatter(exceed_x, exceed_y, s=12, color="#d62728", alpha=0.9, marker="x", label="Exceedances")
        ax.legend(loc="upper right")
        _apply_plot_style(
            fig,
            title="Volumetric Flow (mm³/s)",
            subtitle=_plot_subtitle(summary),
            caption=(
                "Green band = typical safe range; dashed = limit; red × = exceedances.\n"
                "Above-limit flow risks under-extrusion/matte bands/weaker parts; use to set safe speeds/widths."
            ),
        )

    if out_path is None:
        out_path = summary.file.with_suffix('.flow.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    if over_count:
        summary.notes.append(f"Flow exceeded safe limit at {over_count} segments.")
    summary.notes.append(f"Saved volumetric flow plot: {out_path.name}")
    return out_path


def _ensure_matplotlib():  # returns pyplot module or None
    global _MPL_TRIED_INSTALL
    try:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        # Try a one-time install
        if _MPL_TRIED_INSTALL:
            return None
        _MPL_TRIED_INSTALL = True
        try:
            import subprocess, sys, importlib
            # Best-effort user install without polluting system site-packages
            cmd = [sys.executable, "-m", "pip", "install", "--user", "matplotlib"]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Re-try import
            importlib.invalidate_caches()
            import matplotlib
            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass
            import matplotlib.pyplot as plt  # type: ignore
            return plt
        except Exception:
            return None


# ------------------------------ Plot Utilities ----------------------------- #


def _printer_pretty(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    return {
        "bambu_a1": "Bambu A1/A1 mini",
        "bambu_p1s": "Bambu P1S/P1P",
        "bambu_x1c": "Bambu X1C/X1E",
        "bambu_h2d": "Bambu H2D hotend",
        "bambu_h2s": "Bambu H2S hotend",
    }.get(model, model)


def _plot_subtitle(summary: GcodeSummary) -> str:
    parts: List[str] = [summary.file.stem]
    pm = _printer_pretty(summary.printer_model)
    if pm:
        parts.append(pm)
    if summary.inferred_material:
        parts.append(f"{summary.inferred_material}")
    try:
        parts.append(f"Ø{summary.filament_diameter_mm:.2f} mm")
    except Exception:
        pass
    # First-layer temps if available
    fln = getattr(summary.temps, 'first_layer_nozzle', None)
    flb = getattr(summary.temps, 'first_layer_bed', None)
    if isinstance(fln, (int, float)) and isinstance(flb, (int, float)) and fln > 0 and flb > 0:
        parts.append(f"1st layer {int(round(fln))}/{int(round(flb))}C")
    # Layers count
    layers = summary.total_layers
    if not layers:
        # Fallback: infer from heuristics
        heur = summary.heuristics
        if heur and (heur.layer_move_mm or heur.layer_time_s):
            keys = list((heur.layer_move_mm or {}).keys()) + list((heur.layer_time_s or {}).keys())
            if keys:
                layers = max(keys) + 1
    if layers:
        parts.append(f"Layers: {layers}")
    # Slicer time (estimated/model) if available
    try:
        et = summary.estimated_time_total_s
        mt = summary.model_print_time_s
        def _fmt_t(sec: Optional[int]) -> Optional[str]:
            if sec is None:
                return None
            h = int(sec) // 3600
            m = (int(sec) % 3600) // 60
            s = int(sec) % 60
            if h:
                return f"{h}h {m}m {s}s"
            if m:
                return f"{m}m {s}s"
            return f"{s}s"
        et_str = _fmt_t(et)
        mt_str = _fmt_t(mt)
        if et_str or mt_str:
            if et_str and mt_str:
                parts.append(f"Slicer: est {et_str}, model {mt_str}")
            elif et_str:
                parts.append(f"Slicer: est {et_str}")
            elif mt_str:
                parts.append(f"Slicer: model {mt_str}")
    except Exception:
        pass
    return " • ".join(parts)


def _apply_plot_style(fig, title: str, subtitle: Optional[str] = None, caption: Optional[str] = None) -> None:
    try:
        import matplotlib as mpl  # type: ignore
    except Exception:
        # If mpl is not importable at this point, bail; caller already handles
        return
    # Lightly unify aesthetics
    try:
        rc = mpl.rcParams
        rc.setdefault('axes.titlesize', 12)
        rc.setdefault('axes.labelsize', 10)
        rc.setdefault('xtick.labelsize', 9)
        rc.setdefault('ytick.labelsize', 9)
        rc.setdefault('legend.fontsize', 9)
        rc.setdefault('grid.linestyle', '-')
        rc.setdefault('grid.alpha', 0.3)
    except Exception:
        pass
    # Title + subtitle
    suptitle = title
    if subtitle:
        suptitle = f"{title}\n{subtitle}"
    try:
        fig.suptitle(suptitle)
    except Exception:
        pass
    # Caption at bottom center
    if caption:
        try:
            # Slightly higher y to allow two lines without clipping
            fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=9, alpha=0.9)
        except Exception:
            pass
    try:
        # Reserve a bit more bottom space for multi-line captions
        fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    except Exception:
        pass


def _gcode_file_list_from_arg(path: Path, recursive: bool = False) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".gcode", ".gco", ".gc", ".nc"}
    if recursive:
        return [p for p in path.rglob("*") if p.suffix.lower() in exts]
    else:
        return [p for p in path.glob("*") if p.suffix.lower() in exts]


def _analyze_single_file(path: Path, filament_diameter: float, printer_override: Optional[str]) -> str:
    return _analyze_single_file_with_opts(path, filament_diameter, printer_override, plot=False, show_progress=True)


def _show_options_dialog(default_filament: float = 1.75) -> Tuple[bool, Dict[str, object]]:
    """Show a Tkinter dialog to select plotting and analysis options.
    Returns (ok, options_dict). On cancel, returns (False, {}).
    options keys: plot, flow, e_per_mm, corner_stress, cooling (bools),
                  filament_diameter (float), printer_override (str or None).
    """
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore

    result: Dict[str, object] = {}
    ok_pressed = {"ok": False}

    root = tk.Tk()
    root.title("G-code Inspector Options")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")

    # Plot options (use IntVar with explicit on/off to avoid tri-state '-')
    plot_var = tk.IntVar(value=0)
    flow_var = tk.IntVar(value=0)
    epermm_var = tk.IntVar(value=0)
    corner_var = tk.IntVar(value=0)
    cooling_var = tk.IntVar(value=0)

    ttk.Label(frm, text="Plots:").grid(row=0, column=0, sticky="w")
    cb_plot = ttk.Checkbutton(frm, text="Layer metrics", variable=plot_var, onvalue=1, offvalue=0)
    cb_flow = ttk.Checkbutton(frm, text="Volumetric flow", variable=flow_var, onvalue=1, offvalue=0)
    cb_eper = ttk.Checkbutton(frm, text="E/mm + rolling std", variable=epermm_var, onvalue=1, offvalue=0)
    cb_corner = ttk.Checkbutton(frm, text="Corner stress", variable=corner_var, onvalue=1, offvalue=0)
    cb_cool = ttk.Checkbutton(frm, text="Cooling (fan vs layer time)", variable=cooling_var, onvalue=1, offvalue=0)
    cb_plot.grid(row=1, column=0, sticky="w")
    cb_flow.grid(row=2, column=0, sticky="w")
    cb_eper.grid(row=3, column=0, sticky="w")
    cb_corner.grid(row=4, column=0, sticky="w")
    cb_cool.grid(row=5, column=0, sticky="w")
    # Ensure no tri-state 'alternate' is active at start and vars are 0
    try:
        for v, cb in ((plot_var, cb_plot), (flow_var, cb_flow), (epermm_var, cb_eper), (corner_var, cb_corner), (cooling_var, cb_cool)):
            v.set(0)
            cb.state(["!alternate"])  # force out of tri-state
    except Exception:
        pass

    # Filament diameter
    ttk.Label(frm, text="Filament diameter (mm):").grid(row=0, column=1, padx=(15, 0), sticky="w")
    fd_var = tk.StringVar(value=f"{default_filament:.2f}")
    fd_entry = ttk.Entry(frm, textvariable=fd_var, width=8)
    fd_entry.grid(row=1, column=1, padx=(15, 0), sticky="w")

    # Printer override
    ttk.Label(frm, text="Printer/hotend:").grid(row=2, column=1, padx=(15, 0), sticky="w")
    choices = ["Auto", "A1", "P1S", "X1C", "H2D", "H2S"]
    printer_var = tk.StringVar(value=choices[0])
    ttk.OptionMenu(frm, printer_var, choices[0], *choices).grid(row=3, column=1, padx=(15, 0), sticky="w")

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=2, pady=(10, 0), sticky="e")

    def _on_ok():
        ok_pressed["ok"] = True
        try:
            fd_text = fd_var.get().strip()
            fd_val = float(fd_text)
        except Exception:
            fd_val = default_filament
        result["filament_diameter"] = fd_val
        result["plot"] = bool(int(plot_var.get() or 0))
        result["flow"] = bool(int(flow_var.get() or 0))
        result["e_per_mm"] = bool(int(epermm_var.get() or 0))
        result["corner_stress"] = bool(int(corner_var.get() or 0))
        result["cooling"] = bool(int(cooling_var.get() or 0))
        # Map selection to internal printer key
        sel = (printer_var.get() or "Auto").strip().upper()
        mapping = {
            "AUTO": None,
            "A1": "bambu_a1",
            "P1S": "bambu_p1s",
            "X1C": "bambu_x1c",
            "H2D": "bambu_h2d",
            "H2S": "bambu_h2s",
        }
        result["printer_override"] = mapping.get(sel, None)
        root.destroy()

    def _on_cancel():
        ok_pressed["ok"] = False
        root.destroy()

    ttk.Button(btns, text="Cancel", command=_on_cancel).grid(row=0, column=0, padx=5)
    ttk.Button(btns, text="Run", command=_on_ok).grid(row=0, column=1, padx=5)

    # Center the window a bit
    try:
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (w // 2)
        y = (root.winfo_screenheight() // 2) - (h // 2)
        root.geometry(f"+{x}+{y}")
    except Exception:
        pass

    root.mainloop()
    return ok_pressed["ok"], (result if ok_pressed["ok"] else {})


def _analyze_single_file_with_opts(path: Path, filament_diameter: float, printer_override: Optional[str], plot: bool = False, show_progress: bool = True, e_per_mm_plot: bool = False, flow_plot: bool = False, corner_stress_plot: bool = False, cooling_plot: bool = False, flow_limit_override: Optional[float] = None) -> str:
    from io import StringIO
    buf = StringIO()
    inspector = GcodeInspector(filament_diameter_mm=filament_diameter, printer_override=printer_override, flow_limit_override_mm3_s=flow_limit_override)
    # Wrap the file iterator with a lightweight progress bar when appropriate
    total_bytes = None
    try:
        total_bytes = path.stat().st_size
    except Exception:
        total_bytes = None

    def _progress_wrap(fh):
        import sys as _sys
        import time as _time
        import itertools as _it

        enabled = bool(show_progress) and hasattr(_sys.stderr, "isatty") and _sys.stderr.isatty()
        if not enabled:
            for _line in fh:
                yield _line
            return

        spinner = _it.cycle("|/-\\")
        last = 0.0

        def _render(pct: Optional[float]):
            try:
                if pct is not None and total_bytes:
                    pct = max(0.0, min(1.0, pct))
                    width = 30
                    filled = int(width * pct)
                    bar = "#" * filled + "-" * (width - filled)
                    _sys.stderr.write(f"\rAnalyzing [{bar}] {int(pct*100):3d}%")
                else:
                    _sys.stderr.write(f"\rAnalyzing {next(spinner)}")
                _sys.stderr.flush()
            except Exception:
                pass

        for _line in fh:
            yield _line
            now = _time.monotonic()
            if now - last >= 0.1:
                last = now
                pos = None
                try:
                    # Prefer byte position from the underlying buffer for accurate %
                    pos = fh.buffer.tell() if hasattr(fh, "buffer") else fh.tell()
                except Exception:
                    pos = None
                _render((pos / float(total_bytes)) if (pos is not None and total_bytes) else None)

        # Finish bar at 100%
        try:
            _render(1.0 if total_bytes else None)
            _sys.stderr.write("\n")
            _sys.stderr.flush()
        except Exception:
            pass

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        summary = inspector.inspect(_progress_wrap(fh), filename=path)
    # Optional plotting of per-layer metrics
    if plot:
        try:
            _plot_layer_metrics(summary, out_path=path.with_suffix('.layer_metrics.png'))
        except Exception as e:
            # Non-fatal; include a note
            summary.notes.append(f"Plotting failed: {e}")
    # Optional plotting of E/mm with rolling std-dev
    if e_per_mm_plot:
        try:
            _plot_e_per_mm(summary, out_path=path.with_suffix('.e_per_mm.png'))
        except Exception as e:
            summary.notes.append(f"E/mm plotting failed: {e}")
    # Optional plotting of volumetric flow (mm^3/s)
    if flow_plot:
        try:
            _plot_volumetric_flow(summary, out_path=path.with_suffix('.flow.png'))
        except Exception as e:
            summary.notes.append(f"Flow plotting failed: {e}")
    # Optional plotting of corner stress index
    if corner_stress_plot:
        try:
            _plot_corner_stress(summary, out_path=path.with_suffix('.corner_stress.png'))
        except Exception as e:
            summary.notes.append(f"Corner stress plotting failed: {e}")
    # Optional cooling plot (fan vs layer time)
    if cooling_plot:
        try:
            _plot_cooling_state(summary, out_path=path.with_suffix('.cooling.png'))
        except Exception as e:
            summary.notes.append(f"Cooling plotting failed: {e}")
    # Capture report into a string
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        print_report(summary)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def main(argv: List[str]) -> int:
    used_dialog = False
    show_post_prompt = True
    paths: List[Path] = []
    # Defaults for CLI-related options so they exist regardless of branch
    filament_diameter = 1.75
    printer_override: Optional[str] = None
    jobs: Optional[int] = None
    recursive = False
    plot = False
    e_per_mm_plot = False
    flow_plot = False
    corner_stress_plot = False
    cooling_plot = False
    interactive_prompt = False
    flow_limit_override: Optional[float] = None
    # Whether to suppress the final hold/pause screen
    no_hold = False
    # Lightweight arg parsing: first non-flag is file path
    if len(argv) < 2:
        # Try interactive file picker (helps when double-clicking on Windows)
        try:
            import tkinter as tk  # type: ignore
            from tkinter import filedialog, messagebox  # type: ignore

            root = tk.Tk()
            root.withdraw()
            used_dialog = True  # mark early so we can perform a hold on cancel
            sel = filedialog.askopenfilename(
                title="Select a G-code file",
                filetypes=[
                    ("G-code", "*.gcode *.gco *.gc *.nc"),
                    ("All files", "*.*"),
                ],
            )
            if not sel:
                # Inform and hold so double-click doesn't close instantly
                try:
                    messagebox.showinfo("G-code Inspector", "No file selected.")
                except Exception:
                    pass
                _final_hold()
                return 2
            # Ensure we don't keep a hidden root window alive
            try:
                root.update_idletasks(); root.destroy()
            except Exception:
                pass
            paths = [Path(sel)]
            # Show an options dialog to select plots and overrides
            try:
                ok, opts = _show_options_dialog(default_filament=filament_diameter)
            except Exception:
                ok, opts = True, {}
            if not ok:
                _final_hold()
                return 2
            try:
                fd = opts.get("filament_diameter")
                if isinstance(fd, (int, float)):
                    filament_diameter = float(fd)
            except Exception:
                pass
            po = opts.get("printer_override")
            if isinstance(po, str) and po:
                printer_override = po
            plot = bool(opts.get("plot", plot))
            flow_plot = bool(opts.get("flow", flow_plot))
            e_per_mm_plot = bool(opts.get("e_per_mm", e_per_mm_plot))
            corner_stress_plot = bool(opts.get("corner_stress", corner_stress_plot))
            cooling_plot = bool(opts.get("cooling", cooling_plot))
            # Suppress extra post-run prompt when options were shown
            show_post_prompt = False
        except Exception:
            print(USAGE)
            _final_hold()
            return 2
    else:
        args = argv[1:]
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "--filament-diameter" and i + 1 < len(args):
                try:
                    filament_diameter = float(args[i + 1])
                except Exception:
                    print("Invalid --filament-diameter value; using default 1.75 mm")
                i += 2
                continue
            if tok == "--plot":
                plot = True
                i += 1
                continue
            if tok == "--e-per-mm":
                e_per_mm_plot = True
                i += 1
                continue
            if tok == "--flow":
                flow_plot = True
                i += 1
                continue
            if tok == "--corner-stress":
                corner_stress_plot = True
                i += 1
                continue
            if tok == "--cooling":
                cooling_plot = True
                i += 1
                continue
            if tok == "--interactive":
                interactive_prompt = True
                i += 1
                continue
            if tok == "--no-hold":
                no_hold = True
                i += 1
                continue
            if tok == "--printer" and i + 1 < len(args):
                try:
                    pv = args[i + 1]
                    key = GcodeInspector._detect_printer_model(pv)
                    if key is None:
                        print(f"Warning: unknown --printer '{pv}', ignoring.")
                    else:
                        printer_override = key
                except Exception:
                    print("Invalid --printer value; expected one of: A1, P1S, X1C, H2D, H2S")
                i += 2
                continue
            if tok == "--flow-limit" and i + 1 < len(args):
                try:
                    flow_limit_override = float(args[i + 1])
                except Exception:
                    print("Invalid --flow-limit value; expected mm^3/s number")
                i += 2
                continue
            if tok in {"-j", "--jobs"} and i + 1 < len(args):
                try:
                    jobs = max(1, int(args[i + 1]))
                except Exception:
                    print("Invalid --jobs value; using default")
                i += 2
                continue
            if tok in {"-r", "--recursive"}:
                recursive = True
                i += 1
                continue
            if not tok.startswith("-"):
                paths.append(Path(tok))
            i += 1
        # Heuristic: if no usable paths parsed but there are non-flag tokens, try joining them
        if not paths:
            nonflags = [t for t in args if not t.startswith("-")]
            if nonflags:
                joined = " ".join(nonflags).strip().strip('"')
                if joined:
                    p = Path(joined)
                    if p.exists():
                        paths.append(p)
    # Expand directories and validate paths
    if not paths:
        print("Error: no files provided")
        if _should_hold():
            _final_hold()
        return 2
    expanded: List[Path] = []
    for p in paths:
        if not p.exists():
            print(f"Warning: path not found: {p}")
            continue
        expanded.extend(_gcode_file_list_from_arg(p, recursive=recursive))
    if not expanded:
        print("Error: no G-code files to analyze")
        if _should_hold():
            _final_hold()
        return 2

    # Single file path → preserve existing behavior
    if len(expanded) == 1:
        target = expanded[0]
        # Always print the target early so drag-and-drop users see something immediately
        try:
            print(f"Analyzing: {target}", flush=True)
        except Exception:
            pass
        try:
            out = _analyze_single_file_with_opts(target, filament_diameter, printer_override, plot=plot, e_per_mm_plot=e_per_mm_plot, flow_plot=flow_plot, corner_stress_plot=corner_stress_plot, cooling_plot=cooling_plot, flow_limit_override=flow_limit_override)
            print(out, end="")
        except Exception as e:
            msg = f"Error analyzing file: {e}"
            print(msg)
            if used_dialog:
                try:
                    import tkinter as tk  # type: ignore
                    from tkinter import messagebox  # type: ignore
                    root = tk.Tk(); root.withdraw()
                    messagebox.showerror("G-code Inspector", msg)
                except Exception:
                    pass
            _final_hold()
            return 1
        # Show optional interactive prompt either when launched via picker or when --interactive is passed
        if show_post_prompt and (used_dialog or interactive_prompt):
            handled = False
            # In picker mode, prefer a GUI prompt so double-clicking .py doesn't close immediately
            if used_dialog:
                try:
                    import tkinter as tk  # type: ignore
                    from tkinter import messagebox  # type: ignore
                    root = tk.Tk()
                    root.withdraw()
                    if messagebox.askyesno("G-code Inspector", "Save layer plot (PNG) now?"):
                        inspector = GcodeInspector(filament_diameter_mm=filament_diameter, printer_override=printer_override)
                        with target.open("r", encoding="utf-8", errors="ignore") as fh:
                            summary = inspector.inspect(fh, filename=target)
                        try:
                            out_path = _plot_layer_metrics(summary, out_path=target.with_suffix('.layer_metrics.png'))
                            if out_path:
                                messagebox.showinfo("G-code Inspector", f"Saved plot to:\n{out_path}")
                            else:
                                messagebox.showinfo("G-code Inspector", "No per-layer data to plot.")
                        except Exception as e:
                            messagebox.showerror("G-code Inspector", f"Plotting failed:\n{e}")
                    # Always show a final dismiss to prevent immediate close
                    messagebox.showinfo("G-code Inspector", "Done.")
                    handled = True
                except Exception:
                    handled = False
            if not handled:
                # Fallback to console prompt (works in terminals or when --interactive is passed)
                try:
                    print("\nHotkeys — L: layer, F: flow, E: E/mm, K: corner, O: cooling, any other to quit", flush=True)
                    while True:
                        print("Select plot (L/F/E/K/O) or any other key to exit: ", end="", flush=True)
                        choice = _read_single_key()
                        print("")
                        ch = (choice or "").strip().lower()
                        if ch not in {"l", "f", "e", "k", "o"}:
                            break
                        inspector = GcodeInspector(filament_diameter_mm=filament_diameter, printer_override=printer_override)
                        with target.open("r", encoding="utf-8", errors="ignore") as fh:
                            summary = inspector.inspect(fh, filename=target)
                        try:
                            if ch == "l":
                                out_path = _plot_layer_metrics(summary, out_path=target.with_suffix('.layer_metrics.png'))
                            elif ch == "f":
                                out_path = _plot_volumetric_flow(summary, out_path=target.with_suffix('.flow.png'))
                            elif ch == "e":
                                out_path = _plot_e_per_mm(summary, out_path=target.with_suffix('.e_per_mm.png'))
                            elif ch == "k":
                                out_path = _plot_corner_stress(summary, out_path=target.with_suffix('.corner_stress.png'))
                            elif ch == "o":
                                out_path = _plot_cooling_state(summary, out_path=target.with_suffix('.cooling.png'))
                            else:
                                out_path = None
                            if out_path:
                                print(f"Saved: {out_path}")
                            else:
                                print("Nothing to plot or failed.")
                        except Exception as e:
                            print(f"Plotting failed: {e}")
                except Exception:
                    pass
            # In picker mode, add a final hold to prevent the console from closing immediately
            if used_dialog and not no_hold:
                _final_hold()
        # If launched via options dialog (used_dialog True) and we suppressed the post prompt,
        # show a simple GUI summary of any plots saved, then hold the window.
        elif used_dialog and not show_post_prompt:
            try:
                import tkinter as tk  # type: ignore
                from tkinter import messagebox  # type: ignore
                root = tk.Tk(); root.withdraw()
                saved_msgs: List[str] = []
                if plot and target.with_suffix('.layer_metrics.png').exists():
                    saved_msgs.append(f"Layer metrics → {target.with_suffix('.layer_metrics.png').name}")
                if flow_plot and target.with_suffix('.flow.png').exists():
                    saved_msgs.append(f"Volumetric flow → {target.with_suffix('.flow.png').name}")
                if e_per_mm_plot and target.with_suffix('.e_per_mm.png').exists():
                    saved_msgs.append(f"E/mm → {target.with_suffix('.e_per_mm.png').name}")
                if corner_stress_plot and target.with_suffix('.corner_stress.png').exists():
                    saved_msgs.append(f"Corner stress → {target.with_suffix('.corner_stress.png').name}")
                if cooling_plot and target.with_suffix('.cooling.png').exists():
                    saved_msgs.append(f"Cooling → {target.with_suffix('.cooling.png').name}")
                if saved_msgs:
                    messagebox.showinfo("G-code Inspector", "Saved plots:\n\n" + "\n".join(saved_msgs))
                else:
                    messagebox.showinfo("G-code Inspector", "Done.")
            except Exception:
                pass
            _final_hold()
        # If launched by drag-and-drop onto .py (no picker, no --interactive), the console window
        # may close immediately; add a final hold when not attached to a TTY.
        if not used_dialog and not interactive_prompt and _should_hold() and not no_hold:
            _final_hold()
        return 0

    # Multi-file: analyze concurrently using threads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    max_workers = jobs or min(32, (os.cpu_count() or 4))
    results: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_analyze_single_file_with_opts, p, filament_diameter, printer_override, plot, False, e_per_mm_plot, flow_plot, corner_stress_plot, cooling_plot, flow_limit_override): p for p in expanded}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                s = fut.result()
            except Exception as e:
                s = f"File: {p}\n\nError: {e}\n"
            results.append((p, s))
    # Print outputs sorted by filename for determinism
    for _, s in sorted(results, key=lambda t: t[0].name.lower()):
        print(s)
    if _should_hold() and not no_hold:
        _final_hold()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main(sys.argv)
    except SystemExit as e:
        # Preserve explicit SystemExit behavior
        raise
    except Exception as e:
        # Catch any unexpected errors, print traceback, and pause before exit
        print(f"Fatal error: {e}")
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        try:
            _final_hold()
        except Exception:
            pass
        exit_code = 1
    raise SystemExit(exit_code)
