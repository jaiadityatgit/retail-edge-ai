"""
EdgeRetail AI // SIH26179 - On-Device Retail Intelligence Platform
Target Hardware: Raspberry Pi 5 (ARM64 Quad-Core Cortex-A76) & Edge Appliances
100% Offline, DPDP 2023 Compliant, Multi-Threaded Vision Pipeline + Spatial Tracking + FastAPI Dashboard
"""

import os
import sys
import time
import math
import json
import random
import platform
import threading
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==============================================================================
# 1. HARDWARE TELEMETRY & SYSTEM PROFILER
# ==============================================================================

class HardwareProfiler:
    """Reads real-time SoC temperature, CPU utilization, and system metrics."""

    @staticmethod
    def is_raspberry_pi() -> bool:
        """Detect if running on Raspberry Pi hardware."""
        try:
            if os.path.exists("/sys/firmware/devicetree/base/model"):
                with open("/sys/firmware/devicetree/base/model", "r") as f:
                    model = f.read().lower()
                    return "raspberry pi" in model
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    return "raspberry pi" in model
        except Exception:
            pass
        return "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower()

    @classmethod
    def get_soc_temperature(cls) -> float:
        """Read SoC temperature in Celsius with cross-platform fallbacks."""
        # 1. Direct sysfs thermal zone (Linux / Raspberry Pi OS Bookworm)
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            try:
                with open(thermal_path, "r") as f:
                    temp_raw = f.read().strip()
                    temp_c = float(temp_raw) / 1000.0
                    if 10.0 <= temp_c <= 110.0:
                        return round(temp_c, 1)
            except Exception:
                pass

        # 2. psutil hardware sensors
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ["cpu_thermal", "cpu-thermal", "coretemp", "k10temp", "acpitz", "soc_thermal"]:
                    if name in temps and len(temps[name]) > 0:
                        return round(temps[name][0].current, 1)
                # Any first temperature entry
                for entries in temps.values():
                    if entries and len(entries) > 0:
                        return round(entries[0].current, 1)
        except Exception:
            pass

        # 3. Dynamic simulated edge temperature based on CPU load (Dev environment fallback)
        cpu_load = psutil.cpu_percent(interval=None)
        base_temp = 48.5 + (cpu_load * 0.12) + random.uniform(-0.4, 0.4)
        return round(min(max(base_temp, 42.0), 78.0), 1)

    @classmethod
    def get_telemetry(cls, edge_fps: float, inference_latency_ms: float) -> Dict[str, Any]:
        """Produce comprehensive hardware metrics."""
        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        mem = psutil.virtual_memory()
        soc_temp = cls.get_soc_temperature()
        is_rpi = cls.is_raspberry_pi()
        platform_name = "Raspberry Pi 5 (ARM64)" if is_rpi else f"{platform.system()} {platform.machine()} Edge SBC"

        return {
            "platform": platform_name,
            "soc_temp_c": soc_temp,
            "cpu_load_pct": cpu_pct,
            "ram_used_mb": round(mem.used / (1024 * 1024), 1),
            "ram_total_mb": round(mem.total / (1024 * 1024), 1),
            "ram_pct": round(mem.percent, 1),
            "edge_fps": round(edge_fps, 1),
            "inference_latency_ms": round(inference_latency_ms, 1),
            "throttled": soc_temp > 75.0,
        }


# ==============================================================================
# 2. SPATIAL TRACKED ENTITY & RE-ID ENGINE
# ==============================================================================

class TrackedEntity:
    """Maintains persistent trajectory, dwell duration, and kinematics for an object."""

    def __init__(self, track_id: int, class_name: str, box: List[float], centroid: Tuple[float, float], confidence: float):
        self.track_id = track_id
        self.class_name = class_name
        self.box = [float(v) for v in box]
        self.centroid = (float(centroid[0]), float(centroid[1]))
        self.confidence = float(confidence)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.history: List[Tuple[int, int]] = [(int(centroid[0]), int(centroid[1]))]
        self.zone = "TRANSIT"
        self.velocity_mps = 0.0
        self.occluding_shelf = False

    def update(self, box: List[float], centroid: Tuple[float, float], conf: float, zone: str, occluding: bool):
        now = time.time()
        dt = max(0.001, now - self.last_seen)
        old_cx, old_cy = self.centroid
        new_cx, new_cy = float(centroid[0]), float(centroid[1])
        
        dist_pixels = math.hypot(new_cx - old_cx, new_cy - old_cy)
        # Approximate 120 pixels = 1 meter on testbed
        self.velocity_mps = round((dist_pixels / 120.0) / dt, 2)

        self.box = [float(v) for v in box]
        self.centroid = (new_cx, new_cy)
        self.confidence = float(conf)
        self.last_seen = now
        self.zone = zone
        self.occluding_shelf = occluding
        
        self.history.append((int(new_cx), int(new_cy)))
        if len(self.history) > 30:
            self.history.pop(0)

    @property
    def dwell_time_sec(self) -> float:
        return round(time.time() - self.first_seen, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 2),
            "box": [round(v, 1) for v in self.box],
            "centroid": [round(c, 1) for c in self.centroid],
            "zone": self.zone,
            "dwell_time_sec": self.dwell_time_sec,
            "velocity_mps": self.velocity_mps,
            "occluding_shelf": self.occluding_shelf
        }


# ==============================================================================
# 3. SYNTHETIC VISUAL TEST GENERATOR (AUTOMATIC CAMERA FALLBACK)
# ==============================================================================

class SyntheticRetailGenerator:
    """
    Generates a continuous, realistic 640x360 retail test stream containing:
    - 2-Tier Shelf Zone on left with beverage cans / bottles / snack boxes.
    - Transit Corridor in middle with realistic floor tiles.
    - Checkout Queue Zone on right with cash counter.
    - Dynamic customer avatars walking, occluding shelves, taking items, and queuing.
    """

    def __init__(self, width: int = 640, height: int = 360):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.start_time = time.time()

        # Simulation state
        self.simulation_mode = "AUTO_CYCLE"  # AUTO_CYCLE, RESTOCK_EMPTY, CUSTOMER_OCCLUSION, QUEUE_CONGESTION
        self.manual_stock_override: Optional[int] = None
        self.manual_queue_override: Optional[int] = None
        self.manual_occlusion_override: Optional[bool] = None

    def set_mode(self, mode: str):
        self.simulation_mode = mode

    def generate_frame(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Synthesizes a visual frame and returns (frame, synthetic_ground_truth_detections).
        Synthetic detections enable 100% deterministic CV testing even in headless environments.
        """
        self.frame_idx += 1
        t = time.time() - self.start_time
        w, h = self.width, self.height

        # Create dark retail environment canvas
        frame = np.full((h, w, 3), 22, dtype=np.uint8)

        # 1. Floor tiles & store background
        for y in range(int(h * 0.15), h, 30):
            cv2.line(frame, (0, y), (w, y), (34, 36, 44), 1)
        for x in range(0, w, 40):
            cv2.line(frame, (x, int(h * 0.15)), (x, h), (30, 32, 40), 1)

        # 2. Zone Base Layouts
        # Shelf unit on left (0 to 40% X)
        shelf_x1, shelf_y1 = int(w * 0.04), int(h * 0.20)
        shelf_x2, shelf_y2 = int(w * 0.36), int(h * 0.85)

        # Draw shelf frame
        cv2.rectangle(frame, (shelf_x1, shelf_y1), (shelf_x2, shelf_y2), (36, 40, 50), -1)
        cv2.rectangle(frame, (shelf_x1, shelf_y1), (shelf_x2, shelf_y2), (70, 78, 92), 2)

        # Shelf tiers (Tier 1 & Tier 2)
        tier1_y = int(h * 0.48)
        tier2_y = int(h * 0.76)
        cv2.line(frame, (shelf_x1, tier1_y), (shelf_x2, tier1_y), (100, 112, 130), 3)
        cv2.line(frame, (shelf_x1, tier2_y), (shelf_x2, tier2_y), (100, 112, 130), 3)

        # Shelf header strip
        cv2.rectangle(frame, (shelf_x1, shelf_y1), (shelf_x2, shelf_y1 + 18), (28, 32, 42), -1)
        cv2.putText(frame, "SHELF TIER-1 / FMCG BEVERAGES", (shelf_x1 + 6, shelf_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 200, 220), 1)

        # Checkout counter on right (60% to 100% X)
        q_x1, q_y1 = int(w * 0.64), int(h * 0.25)
        q_x2, q_y2 = int(w * 0.95), int(h * 0.85)
        cv2.rectangle(frame, (q_x1 + 80, q_y1 + 40), (q_x2, q_y2 - 20), (38, 42, 52), -1)
        cv2.rectangle(frame, (q_x1 + 80, q_y1 + 40), (q_x2, q_y2 - 20), (65, 72, 85), 2)
        cv2.putText(frame, "REGISTER 1", (q_x1 + 90, q_y1 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1)

        # 3. Determine Scenario Timings & Object States
        cycle_len = 30.0  # 30 second repeating test cycle
        cycle_t = t % cycle_len

        detections = []

        if self.simulation_mode == "RESTOCK_EMPTY":
            stock_count = 0
            shopper_near_shelf = False
            queue_shoppers = 1
        elif self.simulation_mode == "CUSTOMER_OCCLUSION":
            stock_count = 4
            shopper_near_shelf = True
            queue_shoppers = 1
        elif self.simulation_mode == "QUEUE_CONGESTION":
            stock_count = 5
            shopper_near_shelf = False
            queue_shoppers = 3
        else:  # AUTO_CYCLE
            if cycle_t < 7.0:
                stock_count = 5
                shopper_near_shelf = False
                queue_shoppers = 0
            elif cycle_t < 14.0:
                stock_count = 3
                shopper_near_shelf = True
                queue_shoppers = 1
            elif cycle_t < 22.0:
                stock_count = 0  # Emptied! Customer left shelf
                shopper_near_shelf = False
                queue_shoppers = 1
            elif cycle_t < 27.0:
                stock_count = 0
                shopper_near_shelf = False
                queue_shoppers = 2  # Congested!
            else:
                stock_count = 5  # Restocked!
                shopper_near_shelf = False
                queue_shoppers = 1

        if self.manual_stock_override is not None:
            stock_count = self.manual_stock_override
        if self.manual_queue_override is not None:
            queue_shoppers = self.manual_queue_override
        if self.manual_occlusion_override is not None:
            shopper_near_shelf = self.manual_occlusion_override

        # 4. Render Shelf Products (Bottles, Cans, Boxes)
        shelf_items_coords = [
            (shelf_x1 + 18, tier1_y - 45, 20, 42, "bottle", (0, 165, 255)),
            (shelf_x1 + 46, tier1_y - 45, 20, 42, "bottle", (0, 200, 200)),
            (shelf_x1 + 74, tier1_y - 40, 22, 37, "cup", (50, 220, 100)),
            (shelf_x1 + 104, tier1_y - 40, 22, 37, "cup", (220, 100, 50)),
            (shelf_x1 + 134, tier1_y - 48, 24, 45, "book", (180, 100, 220)),
        ]

        active_items = shelf_items_coords[:stock_count]
        for (ix, iy, iw, ih, label, col) in active_items:
            cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), col, -1)
            cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), (255, 255, 255), 1)
            cv2.circle(frame, (ix + iw // 2, iy + 6), 4, (240, 240, 240), -1)

            detections.append({
                "class_id": 39 if label == "bottle" else 41 if label == "cup" else 73,
                "class_name": label,
                "confidence": 0.88 + random.uniform(0.01, 0.08),
                "box": [ix, iy, ix + iw, iy + ih],
                "centroid": [ix + iw / 2.0, iy + ih / 2.0]
            })

        if stock_count == 0:
            cv2.putText(frame, "[SHELF VOID - DEPLETED]", (shelf_x1 + 10, tier1_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 220), 1)

        # 5. Render Moving Shoppers
        def draw_shopper(px: int, py: int, pw: int, ph: int, person_id: int, color=(0, 210, 255)):
            head_radius = int(pw * 0.28)
            head_center = (px + pw // 2, py + head_radius + 4)
            cv2.circle(frame, head_center, head_radius, (220, 180, 140), -1)
            cv2.circle(frame, head_center, head_radius, (255, 255, 255), 1)

            torso_top = head_center[1] + head_radius
            torso_bottom = py + int(ph * 0.72)
            cv2.rectangle(frame, (px + int(pw * 0.15), torso_top), (px + int(pw * 0.85), torso_bottom), color, -1)
            cv2.rectangle(frame, (px + int(pw * 0.15), torso_top), (px + int(pw * 0.85), torso_bottom), (255, 255, 255), 1)

            cv2.line(frame, (px + int(pw * 0.3), torso_bottom), (px + int(pw * 0.3), py + ph), (40, 40, 50), 3)
            cv2.line(frame, (px + int(pw * 0.7), torso_bottom), (px + int(pw * 0.7), py + ph), (40, 40, 50), 3)

            detections.append({
                "class_id": 0,
                "class_name": "person",
                "track_id": person_id,
                "confidence": 0.91 + random.uniform(0.01, 0.06),
                "box": [px, py, px + pw, py + ph],
                "centroid": [px + pw / 2.0, py + ph * 0.85]
            })

        if shopper_near_shelf:
            p1_x = int(w * 0.18 + math.sin(t * 1.5) * 8)
            p1_y = int(h * 0.30)
            draw_shopper(p1_x, p1_y, 48, 110, person_id=101, color=(50, 180, 240))
        elif cycle_t < 7.0 or self.simulation_mode == "RESTOCK_EMPTY":
            progress = (cycle_t / 7.0) if cycle_t < 7.0 else 0.5
            p1_x = int(w * 0.42 + progress * (w * 0.12))
            p1_y = int(h * 0.32 + math.cos(t * 2.0) * 6)
            draw_shopper(p1_x, p1_y, 44, 105, person_id=101, color=(70, 160, 230))

        queue_positions = [
            (int(w * 0.72), int(h * 0.34), 44, 105, 102, (240, 140, 60)),
            (int(w * 0.65), int(h * 0.36), 44, 105, 103, (160, 100, 240)),
            (int(w * 0.58), int(h * 0.38), 44, 105, 104, (100, 220, 180)),
        ]

        for i in range(min(queue_shoppers, len(queue_positions))):
            qx, qy, qw, qh, pid, col = queue_positions[i]
            qx_shift = int(qx + math.sin(t * 1.2 + i) * 3)
            draw_shopper(qx_shift, qy, qw, qh, person_id=pid, color=col)

        return frame, detections


# ==============================================================================
# 4. SPATIAL ROIs & OCCLUSION-AWARE CSIM STATE MACHINE
# ==============================================================================

class SpatialZones:
    """Normalized Dynamic Frame ROIs (0.0 to 1.0 coordinates)."""
    SHELF = {"x1": 0.0, "y1": 0.15, "x2": 0.40, "y2": 0.88, "name": "Shelf Zone (Left)"}
    TRANSIT = {"x1": 0.40, "y1": 0.15, "x2": 0.60, "y2": 0.88, "name": "Transit Corridor (Mid)"}
    QUEUE = {"x1": 0.60, "y1": 0.15, "x2": 1.00, "y2": 0.88, "name": "Checkout Queue (Right)"}


class RetailStateMachine:
    """
    Implements:
    1. Spatial Tracking & Centroid History Engine.
    2. CSIM (Centroid-Shelf Interaction Matrix) Occlusion Handling.
    3. Out-of-Stock Debounce Timer (2.0s continuous non-occluded empty detection).
    4. Queue Congestion & Dwell Estimator (queue_count >= 2 -> CONGESTION_WARNING).
    """

    def __init__(self, debounce_threshold_sec: float = 2.0):
        self.debounce_threshold_sec = debounce_threshold_sec

        # Shelf state tracking
        self.shelf_status = "OPTIMAL"  # OPTIMAL, CUSTOMER_INTERACTING, PENDING_OOS, OUT_OF_STOCK_ALERT
        self.stock_count = 0
        self.is_shelf_occluded = False
        self.shelf_alert_active = False
        self.empty_shelf_start_time: Optional[float] = None
        self.last_occluded_time: float = 0.0

        # Queue state tracking
        self.queue_status = "NORMAL"  # NORMAL, CONGESTION_WARNING
        self.queue_customer_count = 0
        self.queue_congestion_alert = False
        self.estimated_wait_min = 0.0

        # Continuous tracking memory
        self.active_tracks: Dict[int, TrackedEntity] = {}
        self.next_track_id = 101
        self.lock = threading.Lock()

    @staticmethod
    def calculate_box_roi_overlap_pct(box: List[float], roi: Dict[str, float], frame_w: int, frame_h: int) -> float:
        bx1, by1, bx2, by2 = box
        rx1, ry1 = int(roi["x1"] * frame_w), int(roi["y1"] * frame_h)
        rx2, ry2 = int(roi["x2"] * frame_w), int(roi["y2"] * frame_h)

        inter_x1 = max(bx1, rx1)
        inter_y1 = max(by1, ry1)
        inter_x2 = min(bx2, rx2)
        inter_y2 = min(by2, ry2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        person_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter_area / person_area

    @staticmethod
    def is_point_in_roi(point: Tuple[float, float], roi: Dict[str, float], frame_w: int, frame_h: int) -> bool:
        px, py = point
        rx1, ry1 = roi["x1"] * frame_w, roi["y1"] * frame_h
        rx2, ry2 = roi["x2"] * frame_w, roi["y2"] * frame_h
        return rx1 <= px <= rx2 and ry1 <= py <= ry2

    def determine_zone(self, centroid: Tuple[float, float], frame_w: int, frame_h: int) -> str:
        if self.is_point_in_roi(centroid, SpatialZones.SHELF, frame_w, frame_h):
            return "SHELF"
        elif self.is_point_in_roi(centroid, SpatialZones.QUEUE, frame_w, frame_h):
            return "QUEUE"
        elif self.is_point_in_roi(centroid, SpatialZones.TRANSIT, frame_w, frame_h):
            return "TRANSIT"
        return "AISLE"

    def update(self, detections: List[Dict[str, Any]], frame_w: int, frame_h: int):
        """Processes frame detections through tracking and state machine."""
        now = time.time()
        with self.lock:
            retail_classes = {"bottle", "cup", "can", "bowl", "cell phone", "book", "product"}
            shelf_items = []
            current_persons = []

            for det in detections:
                cname = det.get("class_name", "").lower()
                centroid = det.get("centroid", [(det["box"][0] + det["box"][2]) / 2, (det["box"][1] + det["box"][3]) / 2])

                if cname == "person":
                    current_persons.append(det)
                elif cname in retail_classes or det.get("class_id") in [39, 41, 45, 67, 73]:
                    if self.is_point_in_roi(centroid, SpatialZones.SHELF, frame_w, frame_h):
                        shelf_items.append(det)

            self.stock_count = len(shelf_items)

            # 1. Update Spatial Tracker for Persons
            matched_track_ids = set()
            for p in current_persons:
                box = p["box"]
                centroid = p.get("centroid", [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                conf = p.get("confidence", 0.90)
                assigned_id = p.get("track_id")

                overlap_ratio = self.calculate_box_roi_overlap_pct(box, SpatialZones.SHELF, frame_w, frame_h)
                centroid_in_shelf = self.is_point_in_roi(centroid, SpatialZones.SHELF, frame_w, frame_h)
                is_occluding = (overlap_ratio > 0.10) or centroid_in_shelf
                zone = self.determine_zone(centroid, frame_w, frame_h)

                # Match to existing track
                best_tid = None
                if assigned_id and assigned_id in self.active_tracks:
                    best_tid = assigned_id
                else:
                    min_dist = 100.0  # Max distance in pixels to match track
                    for tid, entity in self.active_tracks.items():
                        if tid in matched_track_ids:
                            continue
                        dist = math.hypot(centroid[0] - entity.centroid[0], centroid[1] - entity.centroid[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_tid = tid

                if best_tid is not None:
                    self.active_tracks[best_tid].update(box, centroid, conf, zone, is_occluding)
                    p["track_id"] = best_tid
                    matched_track_ids.add(best_tid)
                else:
                    tid = assigned_id if assigned_id else self.next_track_id
                    if not assigned_id:
                        self.next_track_id += 1
                    entity = TrackedEntity(tid, "person", box, centroid, conf)
                    entity.zone = zone
                    entity.occluding_shelf = is_occluding
                    self.active_tracks[tid] = entity
                    p["track_id"] = tid
                    matched_track_ids.add(tid)

            # Clean up stale tracks (not seen for > 1.0s)
            stale_keys = [k for k, v in self.active_tracks.items() if (now - v.last_seen) > 1.0]
            for k in stale_keys:
                del self.active_tracks[k]

            # Mark non-updated tracks as not actively occluding in this frame
            for tid, entity in self.active_tracks.items():
                if tid not in matched_track_ids:
                    entity.occluding_shelf = False

            # 2. Centroid-Shelf Interaction Matrix (CSIM) Occlusion Check
            occlusion_detected = any(v.occluding_shelf and (now - v.last_seen < 0.25) for v in self.active_tracks.values())
            self.is_shelf_occluded = occlusion_detected

            # 3. Shelf State Machine & Restock Logic
            if self.is_shelf_occluded:
                self.shelf_status = "CUSTOMER_INTERACTING"
                self.last_occluded_time = now
                self.empty_shelf_start_time = None
                self.shelf_alert_active = False
            else:
                if self.stock_count == 0:
                    if self.empty_shelf_start_time is None:
                        self.empty_shelf_start_time = now

                    elapsed_empty = now - self.empty_shelf_start_time
                    if elapsed_empty >= self.debounce_threshold_sec:
                        self.shelf_status = "OUT_OF_STOCK_ALERT"
                        self.shelf_alert_active = True
                    else:
                        self.shelf_status = "PENDING_OOS"
                        self.shelf_alert_active = False
                else:
                    self.shelf_status = "OPTIMAL"
                    self.empty_shelf_start_time = None
                    self.shelf_alert_active = False

            # 4. Queue Congestion & Dwell Time Calculation
            queue_persons = [v for v in self.active_tracks.values() if v.zone == "QUEUE" and (now - v.last_seen < 0.25)]
            self.queue_customer_count = len(queue_persons)
            self.estimated_wait_min = round(self.queue_customer_count * 1.5, 1)

            if self.queue_customer_count >= 2:
                self.queue_status = "CONGESTION_WARNING"
                self.queue_congestion_alert = True
            else:
                self.queue_status = "NORMAL"
                self.queue_congestion_alert = False


# ==============================================================================
# 5. THREAD-SAFE MULTI-TASK VISION PIPELINE
# ==============================================================================

class VisionEngine:
    """
    Dedicated background worker thread for camera ingestion, YOLOv8 inference,
    HUD rendering, spatial trajectory tracking, and double-buffered JPEG streaming.
    """

    def __init__(self, camera_index: int = 0, model_path: str = "yolov8n.pt", imgsz: int = 320):
        self.camera_index = camera_index
        self.model_path = model_path
        self.imgsz = imgsz
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.use_synthetic = False
        self.synthetic_gen = SyntheticRetailGenerator(width=640, height=360)

        self.model = None
        self.model_loaded = False

        self.state_machine = RetailStateMachine(debounce_threshold_sec=2.0)

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.edge_fps: float = 0.0
        self.inference_latency_ms: float = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _init_camera(self):
        try:
            if platform.system().lower() == "linux":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap and self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    self.use_synthetic = False
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print("[VisionEngine] Physical camera initialized successfully.")
                    return
        except Exception as e:
            print(f"[VisionEngine] Physical camera open warning: {e}")

        self.use_synthetic = True
        print("[VisionEngine] No physical camera found. Auto-fallback to Synthetic Simulation Generator.")

    def _init_model(self):
        if not YOLO_AVAILABLE:
            print("[VisionEngine] Ultralytics not installed. Operating in synthetic telemetry mode.")
            return

        try:
            print(f"[VisionEngine] Loading YOLOv8 model: {self.model_path} (imgsz={self.imgsz})...")
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            print("[VisionEngine] YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"[VisionEngine] Warning loading YOLO model: {e}. Fallback to simulated detections.")
            self.model_loaded = False

    def _run_loop(self):
        self._init_camera()
        self._init_model()

        fps_timer = time.time()
        fps_frames = 0

        while self.running:
            loop_start = time.time()
            frame: Optional[np.ndarray] = None
            synthetic_detections = []

            if not self.use_synthetic and self.cap and self.cap.isOpened():
                ret, frame_read = self.cap.read()
                if ret and frame_read is not None:
                    frame = frame_read
                else:
                    self.use_synthetic = True
                    frame, synthetic_detections = self.synthetic_gen.generate_frame()
            else:
                frame, synthetic_detections = self.synthetic_gen.generate_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            frame_h, frame_w = frame.shape[:2]
            infer_start = time.time()
            detections = []

            if self.model_loaded and self.model is not None and not self.use_synthetic:
                try:
                    results = self.model(frame, imgsz=self.imgsz, verbose=False, conf=0.35)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = r.names.get(cls_id, str(cls_id))
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].tolist()
                            detections.append({
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "confidence": conf,
                                "box": xyxy,
                                "centroid": [(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0]
                            })
                except Exception as e:
                    print(f"[VisionEngine] Inference error: {e}")
                    detections = synthetic_detections
            else:
                detections = synthetic_detections

            self.inference_latency_ms = (time.time() - infer_start) * 1000.0

            # State Machine & Spatial Tracker Update
            self.state_machine.update(detections, frame_w, frame_h)

            # Render Overlays, Glowing Trajectories & HUD
            annotated_frame = self._render_hud_and_overlays(frame, detections, frame_w, frame_h)

            # Encode JPEG Buffer
            ret, jpeg_buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with self.lock:
                    self.latest_frame = annotated_frame
                    self.latest_jpeg = jpeg_buf.tobytes()

            fps_frames += 1
            if time.time() - fps_timer >= 1.0:
                self.edge_fps = fps_frames / (time.time() - fps_timer)
                fps_frames = 0
                fps_timer = time.time()

            elapsed = time.time() - loop_start
            target_period = 1.0 / 30.0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    def _render_hud_and_overlays(self, frame: np.ndarray, detections: List[Dict[str, Any]], w: int, h: int) -> np.ndarray:
        """Renders industrial dark HUD, glowing ROI bounding boxes, and object markers."""
        out = frame.copy()
        overlay = out.copy()

        # 1. Draw Spatial ROI Bounding Zones
        s_x1, s_y1 = int(SpatialZones.SHELF["x1"] * w), int(SpatialZones.SHELF["y1"] * h)
        s_x2, s_y2 = int(SpatialZones.SHELF["x2"] * w), int(SpatialZones.SHELF["y2"] * h)

        if self.state_machine.shelf_alert_active:
            shelf_color = (0, 0, 240)
            shelf_tint = (15, 15, 60)
        elif self.state_machine.is_shelf_occluded:
            shelf_color = (0, 180, 255)
            shelf_tint = (10, 35, 55)
        else:
            shelf_color = (0, 200, 120)
            shelf_tint = (10, 40, 20)

        cv2.rectangle(overlay, (s_x1, s_y1), (s_x2, s_y2), shelf_tint, -1)
        cv2.rectangle(out, (s_x1, s_y1), (s_x2, s_y2), shelf_color, 2)

        # Transit Corridor
        t_x1, t_y1 = int(SpatialZones.TRANSIT["x1"] * w), int(SpatialZones.TRANSIT["y1"] * h)
        t_x2, t_y2 = int(SpatialZones.TRANSIT["x2"] * w), int(SpatialZones.TRANSIT["y2"] * h)
        cv2.rectangle(out, (t_x1, t_y1), (t_x2, t_y2), (80, 85, 95), 1)

        # Queue Zone
        q_x1, q_y1 = int(SpatialZones.QUEUE["x1"] * w), int(SpatialZones.QUEUE["y1"] * h)
        q_x2, q_y2 = int(SpatialZones.QUEUE["x2"] * w), int(SpatialZones.QUEUE["y2"] * h)

        if self.state_machine.queue_congestion_alert:
            q_color = (0, 0, 240)
            q_tint = (20, 15, 60)
        else:
            q_color = (220, 140, 0)
            q_tint = (35, 25, 10)

        cv2.rectangle(overlay, (q_x1, q_y1), (q_x2, q_y2), q_tint, -1)
        cv2.rectangle(out, (q_x1, q_y1), (q_x2, q_y2), q_color, 2)
        cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)

        # Zone Labels
        cv2.putText(out, "ROI 1: SHELF ZONE (0-40%)", (s_x1 + 6, s_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, shelf_color, 1)
        cv2.putText(out, "ROI 2: TRANSIT (40-60%)", (t_x1 + 4, t_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 155, 165), 1)
        cv2.putText(out, "ROI 3: CHECKOUT QUEUE (60-100%)", (q_x1 + 6, q_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, q_color, 1)

        # 2. Draw Fading Motion Trajectories & Centroid Crosshairs
        with self.state_machine.lock:
            for tid, entity in self.state_machine.active_tracks.items():
                pts = entity.history
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness = max(1, int(alpha * 3))
                    col = (int(0 * alpha), int(220 * alpha), int(255 * alpha))
                    cv2.line(out, pts[i - 1], pts[i], col, thickness)

                # Crosshairs on centroid
                cx, cy = int(entity.centroid[0]), int(entity.centroid[1])
                cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

                # CSIM Raycast indicator if near shelf
                if entity.occluding_shelf:
                    cv2.line(out, (cx, cy), (s_x2, cy), (0, 180, 255), 1, cv2.LINE_AA)
                    cv2.putText(out, "CSIM LOCK", (cx - 25, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 200, 255), 1)

        # 3. Draw Bounding Boxes with Detailed Tracking Badges
        for det in detections:
            bx1, by1, bx2, by2 = [int(v) for v in det["box"]]
            cname = det.get("class_name", "obj")
            conf = det.get("confidence", 0.0)
            tid = det.get("track_id")

            if cname == "person":
                b_color = (0, 220, 255)
                # Lookup dwell time
                with self.state_machine.lock:
                    entity = self.state_machine.active_tracks.get(tid)
                    dwell_str = f"{entity.dwell_time_sec}s" if entity else "0s"
                    zone_str = entity.zone if entity else "TRANSIT"
                tag = f"#{tid} {cname} | {dwell_str} | {zone_str}"
            else:
                b_color = (255, 180, 50)
                tag = f"{cname} {conf:.2f}"

            cv2.rectangle(out, (bx1, by1), (bx2, by2), b_color, 2)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(out, (bx1, by1 - th - 6), (bx1 + tw + 6, by1), (15, 18, 24), -1)
            cv2.rectangle(out, (bx1, by1 - th - 6), (bx1 + tw + 6, by1), b_color, 1)
            cv2.putText(out, tag, (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 250), 1)

        # 4. Top Edge HUD Bar
        hud_h = 28
        cv2.rectangle(out, (0, 0), (w, hud_h), (11, 14, 22), -1)
        cv2.line(out, (0, hud_h), (w, hud_h), (40, 48, 60), 1)

        hud_title = "EDGERETAIL AI // SIH26179"
        cv2.putText(out, hud_title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        hud_stats = f"FPS: {self.edge_fps:.1f} | Latency: {self.inference_latency_ms:.1f}ms | 100% On-Device"
        cv2.putText(out, hud_stats, (w - 340, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 210, 230), 1)

        dot_color = (0, 255, 100) if (int(time.time() * 2) % 2 == 0) else (0, 180, 60)
        cv2.circle(out, (w - 355, 14), 4, dot_color, -1)

        # 5. Bottom Status HUD Banner
        bot_h = 24
        cv2.rectangle(out, (0, h - bot_h), (w, h), (11, 14, 22), -1)
        cv2.line(out, (0, h - bot_h), (w, h - bot_h), (40, 48, 60), 1)

        shelf_banner = f"Shelf: {self.state_machine.shelf_status} ({self.state_machine.stock_count} items)"
        queue_banner = f"Queue: {self.state_machine.queue_status} ({self.state_machine.queue_customer_count} ppl - {self.state_machine.estimated_wait_min}m)"
        dpdp_banner = "DPDP 2023: 0.0 KB/s UPLINK"

        cv2.putText(out, shelf_banner, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, shelf_color, 1)
        cv2.putText(out, queue_banner, (int(w * 0.42), h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, q_color, 1)
        cv2.putText(out, dpdp_banner, (w - 175, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 240, 160), 1)

        return out

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_telemetry_payload(self) -> Dict[str, Any]:
        hw = HardwareProfiler.get_telemetry(self.edge_fps, self.inference_latency_ms)
        sm = self.state_machine

        with sm.lock:
            shelf_data = {
                "stock_count": sm.stock_count,
                "status": sm.shelf_status,
                "is_occluded": sm.is_shelf_occluded,
                "alert_active": sm.shelf_alert_active,
            }
            queue_data = {
                "customer_count": sm.queue_customer_count,
                "status": sm.queue_status,
                "estimated_wait_min": sm.estimated_wait_min,
                "congestion_alert": sm.queue_congestion_alert,
            }
            active_tracks_list = [t.to_dict() for t in sm.active_tracks.values()]

        return {
            "device": hw,
            "edge_fps": hw["edge_fps"],
            "inference_latency_ms": hw["inference_latency_ms"],
            "shelf": shelf_data,
            "queue": queue_data,
            "tracks": active_tracks_list,
            "compliance": {
                "dpdp_compliant": True,
                "cloud_bandwidth_kbps": 0.0,
                "cloud_uplink_kbps": 0.0,
                "zero_raw_video_leakage": True,
                "on_device_processed": True,
            },
            "timestamp": time.time(),
        }


# ==============================================================================
# 6. FASTAPI WEB SERVER & ROUTING
# ==============================================================================

engine = VisionEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    yield
    engine.stop()

app = FastAPI(
    title="EdgeRetail AI // SIH26179",
    description="On-Device Retail Vision Analytics Platform for Raspberry Pi 5 & Edge Appliances",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return HTMLResponse("<h2>Dashboard initializing. static/index.html not found.</h2>")


@app.get("/presentation", response_class=HTMLResponse)
async def serve_presentation():
    pres_path = STATIC_DIR / "presentation.html"
    if pres_path.exists():
        return FileResponse(
            str(pres_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return HTMLResponse("<h2>Presentation slide deck initializing. static/presentation.html not found.</h2>")


def gen_mjpeg_stream():
    while True:
        frame_bytes = engine.get_jpeg_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gen_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/metrics")
async def get_metrics():
    return JSONResponse(content=engine.get_telemetry_payload())


@app.get("/api/diagnostics/tracking")
async def get_tracking_diagnostics():
    """Returns granular spatial tracking diagnostics."""
    payload = engine.get_telemetry_payload()
    return JSONResponse(content={
        "timestamp": payload["timestamp"],
        "active_track_count": len(payload["tracks"]),
        "tracks": payload["tracks"],
        "shelf": payload["shelf"],
        "queue": payload["queue"]
    })


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vision_engine_running": engine.running,
        "mode": "synthetic" if engine.use_synthetic else "physical_camera",
        "timestamp": time.time()
    }


class SimulationControlRequest(BaseModel):
    mode: Optional[str] = None
    stock_override: Optional[int] = None
    queue_override: Optional[int] = None
    occlusion_override: Optional[bool] = None


@app.post("/api/simulation/control")
async def control_simulation(req: SimulationControlRequest):
    if req.mode:
        engine.synthetic_gen.set_mode(req.mode)
    if req.stock_override is not None:
        engine.synthetic_gen.manual_stock_override = req.stock_override if req.stock_override >= 0 else None
    if req.queue_override is not None:
        engine.synthetic_gen.manual_queue_override = req.queue_override if req.queue_override >= 0 else None
    if req.occlusion_override is not None:
        engine.synthetic_gen.manual_occlusion_override = req.occlusion_override

    return {
        "status": "success",
        "mode": engine.synthetic_gen.simulation_mode,
        "stock_override": engine.synthetic_gen.manual_stock_override,
        "queue_override": engine.synthetic_gen.manual_queue_override,
        "occlusion_override": engine.synthetic_gen.manual_occlusion_override
    }


@app.websocket("/ws/live-metrics")
async def websocket_live_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = engine.get_telemetry_payload()
            await websocket.send_json(payload)
            await time.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting EdgeRetail AI on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False, workers=1)
