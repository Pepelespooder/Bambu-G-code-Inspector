G-code Inspector

Overview
- Command-line tool to analyze 3D printer G-code, estimate material usage, infer likely filament material, and flag common issues.

Quick Start
- Windows
  - Double-click `gcode_inspector.bat` and drag‑and‑drop a `.gcode` file onto it, or
  - Open PowerShell in this folder and run: `py -3 gcode_inspector.py .\file.gcode`
  - Tip: Double‑clicking `gcode_inspector.py` now opens a file picker and pauses on exit.
- macOS/Linux
  - Run: `python3 gcode_inspector.py /path/to/file.gcode`
- Optional: `--filament-diameter 1.75` to override diameter parsed from comments.
- Auto-detects printer/hotend from Bambu/Orca headers and comments.
- Optional: `--printer A1|P1S|X1C|H2D|H2S` to override detection and apply specific volumetric limits.

Batch/Multithreaded Analysis
- You can pass multiple files or directories; the tool analyzes files concurrently using a thread pool.
- Examples:
  - `python3 gcode_inspector.py file1.gcode file2.gcode -j 8`
  - `python3 gcode_inspector.py /path/to/folder --recursive --jobs 8`
- Flags:
  - `-j, --jobs <N>`: number of worker threads (default: CPU count)
  - `-r, --recursive`: when given a directory, search recursively for G-code files

What It Reports
- Material: inferred type (PLA, PETG, ABS, etc.), confidence level, density used.
- Usage: estimated filament length, volume, and mass based on extruded E values.
- Slicer Comparison: optional slicer-reported length/volume/mass and % delta.
- Temps: first-layer and average nozzle/bed temperatures (M104/M109, M140/M190) and chamber setpoints when available (M141/M191 or header keys).
- Retraction: sample count, average/max distance, average speed, z-hop detections.
- Speeds: average/max extrusion and travel speeds (from F values).
- First Layer: average and max extrusion/travel speeds on layer 1.
- Cooling: average/max fan duty (M106/M107).
- Flow: average and maximum volumetric flow (mm^3/s), plus first-layer averages and maxima when detectable.
- Print: total estimated time (from header), model time, total layers, and max Z height when available.
- Heuristics: extrusion mode (M82/M83), G92 resets, travels without retraction and longest segment, skirt/brim presence, bridge segments and low-fan bridges, pauses (M0/M1/M25), feedrate/flow overrides (M220/M221), and mesh state at first layer.
- Potential Issues: richer heuristics, for example:
  - Missing homing/heat/mesh (G28/M104/M109/M140/M190/G29/M420).
  - Bed leveling explicitly disabled (M420 S0/G29.2 S0).
  - Temps mismatched to material (first-layer aware; ignores purge temps).
  - Bed temp swings during print (>=15C).
  - Retraction anomalies: none/very low/very high distance, low/high speed, frequency per 100 mm.
  - Travels without retraction (stringing risk) and long travel-without-retraction segments.
  - First-layer checks: high speed, fan too high for PETG/ABS, missing skirt/brim.
  - Bridge segments with low fan (when detected via comments).
  - Very high speeds and frequent nozzle temp changes.
  - Volumetric flow too high for material/hotend (mm^3/s), incl. first layer.
  - Pause commands present (M0/M1/M25), feedrate/flow overrides (M220/M221).
  - Extrusion mode not set (no M82/M83), frequent G92 E resets.

Notes
- Material inference prefers explicit comments (e.g., `; filament_type = PLA`) and falls back to temperature heuristics.
- Extrusion computation supports absolute (M82) and relative (M83) E modes and G92 resets.
- Density defaults per common materials (PLA 1.24 g/cc). If material is unknown, PLA is assumed.
- Bambu Studio/OrcaSlicer: parses header/config blocks, first-layer temps from header when present, and uses `; layer num/total_layer_count: 1/N` to mark layer 1 so purge temps don’t pollute first-layer readings.
- Chamber temperature: parsed from `M141`/`M191` or header keys like `chamber_temperature`/`chamber_temperatures`. Zero values (disabled) are hidden.
- When present, header keys like `total estimated time`, `model printing time`, `total layer number`, and `max_z_height` are parsed for the Print section.
- Heuristics are slicer/firmware dependent and best-effort; always validate with your printer profile.

Hotend/Printer limits
- Volumetric flow limits are scaled by printer/hotend when detected in the file or passed via `--printer`.
- High‑flow models (relaxed speed/retraction heuristics): H2S, H2D, X1C/X1E, P1S/P1P.
- Flow‑limited: A1/A1 mini (stricter default heuristics).
- Mappings are conservative heuristics; override with `--printer` if needed.
- Limits are conservative heuristics; you can ignore or adjust as needed. Ask if you want a config file to customize per‑printer limits.

Examples
- `python gcode_inspector.py Benchy.gcode`
- `python gcode_inspector.py Part.gcode --filament-diameter 2.85`
