## Report

### What worked
- **Metres instead of pixels.** Calibrating on the zebra crossing turned raw pixel distances into real metres; close-call false alarms in calm traffic fell from 2,328 to 32 in 10 s.
- **Physics-based risk.** Time-to-collision needs no crash training data and every alarm can be explained.
- **Low alarm rate on unseen videos.** Rules tuned on one video only (C3905) gave 0.2–0.6 alarms per minute on the three videos they never saw.
- **Inside the time budget.** All four sample videos finish at 2.6–2.7 × their length on a T4 (limit 3 ×); `evaluate.py --validate-only` passes with 0 errors.

### What did not work
- **Pixel distances** flagged thousands of fake conflicts because perspective squashes the far lanes.
- **Raw speeds** of partly hidden cars in queues were impossible (8–15 m/s while standing); we had to skip hidden objects and use median speeds.
- **Only 2 of 14 classes.** We had no labelled examples to validate the other rules in time, so we chose not to predict them.
- **Tracker ID swaps** when people and riders overlap, and the far road (> 25 m) is too inaccurate for risk.

### What we would do next
1. Label the sample videos to measure Score A and B and tune thresholds on numbers, not by eye.
2. Add `jaywalking`, `wrong_way` and red-light rules using the scene map (lanes, directions, stop line, crosswalks, signal).
3. More calibration points for the far road; oriented boxes for buses and trucks.
4. GPU video decoding (NVDEC), which frees time for denser sampling.
5. A small temporal model trained on public crash datasets (DoTA, CCD) to complement the rules.

### Why it matters
Tashkent has many traffic cameras whose footage is usually watched only after an incident. Running this on existing feeds could speed up emergency response, clear stopped vehicles faster and give city planners near-miss statistics, all from one ordinary camera.
