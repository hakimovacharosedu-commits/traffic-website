"""Scene-map helpers for the rule engine. Loads scene.json (drawn on the 3840x2160 frame).

    from scene import Scene
    sc = Scene("scene.json")
    sc.zone_of(x, y)              -> e.g. ["road", "lane_1", "crosswalk_2"]
    sc.in_zone("crosswalk", x, y) -> True if inside ANY crosswalk
    sc.crossed("stop_line", p_prev, p_now) -> True if the track crossed it
    sc.lane_direction(1)          -> unit vector cars in lane 1 should move along
Pass pixel coordinates of the video; if the video is not 3840x2160,
call sc.rescale(width, height) once.
"""
import json
import math


def _point_in_poly(x, y, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


class Scene:
    def __init__(self, path="scene.json"):
        with open(path) as f:
            self.data = json.load(f)
        self.width, self.height = self.data["width"], self.data["height"]

    def rescale(self, width, height):
        sx, sy = width / self.width, height / self.height
        for items in self.data.values():
            if not isinstance(items, list):
                continue
            for it in items:
                it["points"] = [[x * sx, y * sy] for x, y in it["points"]]
                if "direction" in it:
                    it["direction"] = [[x * sx, y * sy] for x, y in it["direction"]]
        self.width, self.height = width, height
        self.data["width"], self.data["height"] = width, height

    def items(self, name):
        return self.data.get(name, [])

    def in_zone(self, name, x, y):
        return any(_point_in_poly(x, y, it["points"]) for it in self.items(name)
                   if it["type"] == "polygon")

    def zone_of(self, x, y):
        hits = []
        for name, items in self.data.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if it["type"] == "polygon" and _point_in_poly(x, y, it["points"]):
                    hits.append(name if name == "road" else f"{name}_{it['id']}")
        return hits

    def lane_of(self, x, y):
        for it in self.items("lane"):
            if _point_in_poly(x, y, it["points"]):
                return it["id"]
        return None

    def lane_direction(self, lane_id):
        for it in self.items("lane"):
            if it["id"] == lane_id:
                (x1, y1), (x2, y2) = it["direction"]
                n = math.hypot(x2 - x1, y2 - y1) or 1.0
                return ((x2 - x1) / n, (y2 - y1) / n)
        return None

    def crossed(self, name, p_prev, p_now):
        for it in self.items(name):
            pts = it["points"]
            for a, b in zip(pts, pts[1:]):
                if _segments_cross(p_prev, p_now, a, b):
                    return True
        return False
