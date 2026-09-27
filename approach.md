## Problem
A fixed 4K CCTV camera watches one intersection in Tashkent. For any video we return traffic events as `[start_sec, end_sec, label]` (Part A) and, frame by frame, the probability that an accident starts within the next 5 seconds (Part B).

## Pipeline
```
frames ─► sample every 0.1 s ─► YOLO11m detection ─► ByteTrack tracking
                                                        │
                    clean-up: riders, hidden objects, picture edge, road area
                                                        │
                    feet point ─► bird's-eye view (metres) ─► speed & direction
                                   │                                  │
                  time-to-collision for every pair            stopped-vehicle rule
                                   │                                  │
                     Part B: risk score per frame          Part A: stopped_vehicle
                                   │
                        Part A: near_miss (risk ≥ 0.5)
```

## What is learned and what is rule-based
| Component | Type | Details |
|---|---|---|
| Object detector | **Learned** (open weights, no fine-tuning) | YOLO11m (COCO) at 1280 px: people, bicycles, cars, motorcycles, buses, trucks |
| Tracker | Algorithm, no training | ByteTrack keeps one ID per road user |
| Scene geometry | **Set up once by hand** | Road-area polygon + a perspective transform from the main zebra crossing (26.2 m × 4.5 m, measured on satellite imagery) into real metres |
| Accident risk (Part B) | Rules from physics | Smallest confirmed time-to-collision (TTC) → score = 1 / (1 + e^(2·(TTC − 1 s))). TTC 1 s = 0.5 = alarm |
| Events (Part A) | Rules | `near_miss`: risk ≥ 0.5 (padded, merged). `stopped_vehicle`: a vehicle that was driving stands still ≥ 10 s and is not in a queue |

## Why this design
The hidden test set comes from the **same camera and angle**, so an explicit scene model transfers directly, needs no labelled training footage, and every alarm is explainable: which two road users, how far apart, how fast they approach, how many seconds are left.

## Why only 2 classes
A class we predict that never occurs in the test set scores **zero** in the macro-F1. So we only predict classes whose rule we could validate on the samples: `near_miss` and `stopped_vehicle`. Next in line are `jaywalking`, `wrong_way` and red-light events, using the scene map below.

## Scene map
Drawn once on the 3840 × 2160 frame: road area, two carriageways with their legal driving direction, stop line, three crosswalks, the solid line along the median, the signal head and the queue zone.
