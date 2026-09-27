## Problem
A fixed CCTV camera watches one intersection. For any video we return every traffic event as `[start_sec, end_sec, label]` (14 classes) and, frame by frame, the probability that an accident starts within 5 seconds.

## Pipeline
```
video ─► frame sampling ─► YOLO detector ─► ByteTrack tracker ─► tracks
                                                                   │
                         scene map (lanes, stop line, crosswalks) ─┤
                                                                   ▼
                                               rule engine (14 classes)
                                                                   │
                                        post-processing (merge gaps, drop blips)
                                                                   ▼
                                                 events [start, end, label]
tracks ─► time-to-collision ─► risk score (Part B, causal)
```

## What is learned and what is rule-based
| Part | Learned or rules | Why |
|---|---|---|
| Detecting cars, buses, people, bikes | **Learned** (pretrained YOLO) | Robust off-the-shelf, no labels needed |
| Tracking | Algorithm (ByteTrack) | Fast, no training |
| Scene map | **Hand-drawn once** on a 4K frame | The camera never moves |
| Event classes | **Rules** on tracks + scene map | No labelled data for this camera; rules are explainable |
| Accident risk | Rules (time-to-collision) | Works before any crash happens, no crash training data needed |

## Scene map
Drawn once on the camera's 3840×2160 frame: road area, two lane groups with their legal driving direction, stop line, three crosswalks, the solid line along the median, the signal head and the queue zone. The rules use it to decide *where* something happens (see the image below).
