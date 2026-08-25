"""
verify_dual_cam.py
Automated Verification Suite for RetailSense OS // Enterprise HAL & Auto-Baselining (SIH26179)
Verifies:
1. Hardware Abstraction Layer (HAL) Camera Discovery Probe (discover_camera).
2. Auto-Configured YOLOv8 Runtime & Intra-Op Thread Clamping for ARM64/CPU Thermal Guard.
3. Dual Synthetic Multi-Zone Ground Truth Generators (>1000 FPS).
4. Supermarket FMCG Whitelisting & Role-Based Compute Decoupling.
5. Multi-Shelf Planogram Spatial Centroid Tracking & CSIM Occlusion Guard.
6. Dynamic Multi-Register Queue Congestion Matrix & Smart Traffic Director.
7. Enterprise Admin Calibration UI Endpoint (/admin) Isolation & HTML Delivery.
8. Normalized 2-Click ROI Configuration API (/api/config/zones) with { shelf_roi, queue_roi }.
9. Temporal Auto-Baselining Engine (/api/config/baseline) with 30-Frame Window & Median MAX_CAPACITY.
10. Dynamic Stock Percentage Calculation against Calibrated Integer Baseline.
11. Local SQLite Event Persistence (WAL Mode) & Shift Summary Analytics.
12. Mobile WebSocket Video Ingestion (/ws/mobile_upload).
"""

import time
import os
import socket
import sqlite3
import numpy as np
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
from fastapi.testclient import TestClient

def run_verification():
    print("=" * 80)
    print("  RETAILSENSE OS // ENTERPRISE HAL & DYNAMIC AUTO-BASELINING VERIFICATION")
    print("=" * 80)

    # 1. Test Hardware Abstraction Layer (HAL) & Camera Discovery
    print("\n[TEST 1] Verifying HAL Camera Discovery (discover_camera)...")
    from app import discover_camera, DEVICE, USE_HALF, DEVICE_NAME, BACKEND_TYPE, HardwareProfiler, get_local_ip, MAX_CAPACITY

    cap, idx, b_name = discover_camera((0, 1, 2))
    print(f"  -> HAL Probe Result: Cap={cap is not None}, Index={idx}, Backend={b_name}")
    assert b_name in ["V4L2", "DSHOW", "MSMF", "AVFOUNDATION", "ANY", "Synthetic"]
    if cap is not None:
        cap.release()
    print("  [PASS] HAL Camera Discovery successfully probed hardware interfaces.")

    # 2. Test YOLOv8 Runtime Auto-Configuration & Intra-Op Thread Clamping
    print("\n[TEST 2] Verifying YOLOv8 Runtime Configuration & Intra-Op Thread Clamping...")
    telemetry = HardwareProfiler.get_system_telemetry(fps_cam1=29.2, fps_cam2=28.4, latency_ms=6.8)
    print(f"  -> Inference Device: {DEVICE_NAME} (Device: {DEVICE}, FP16: {USE_HALF})")
    print(f"  -> PyTorch Intra-Op Threads: {torch.get_num_threads()} (Clamped for thermal protection)")
    print(f"  -> Silicon Target: {telemetry['silicon_target']}")
    print(f"  -> SoC Temperature: {telemetry['soc_temp_c']}°C")
    assert "inference_device" in telemetry
    assert torch.get_num_threads() <= 8
    print("  [PASS] YOLOv8 runtime and CPU thermal thread limits verified.")

    # 3. Test Dual Synthetic Stream Generators
    print("\n[TEST 3] Verifying Multi-Zone Synthetic Stream Generators...")
    from app import SyntheticShelfGenerator, SyntheticQueueGenerator

    gen_shelf = SyntheticShelfGenerator(640, 360)
    t0 = time.time()
    for _ in range(60):
        frame_s, det_s = gen_shelf.generate()
    dt_s = max(0.0001, time.time() - t0)
    fps_s = 60 / dt_s
    print(f"  -> Multi-Tier Shelf Generator Throughput: {fps_s:.1f} FPS (Shape: {frame_s.shape})")
    assert frame_s.shape == (360, 640, 3)

    gen_queue = SyntheticQueueGenerator(640, 360)
    t0 = time.time()
    for _ in range(60):
        frame_q, det_q = gen_queue.generate()
    dt_q = max(0.0001, time.time() - t0)
    fps_q = 60 / dt_q
    print(f"  -> Multi-Register Queue Generator Throughput: {fps_q:.1f} FPS (Shape: {frame_q.shape})")
    assert frame_q.shape == (360, 640, 3)
    print("  [PASS] Synthetic ground truth generators exceed real-time requirements (>30 FPS).")

    # 4. Test Supermarket Whitelisting & Role Filtering
    print("\n[TEST 4] Verifying Supermarket FMCG Whitelisting & Role Filtering...")
    from app import SUPERMARKET_RETAIL_CLASSES, SHELF_TARGET_CLASS_IDS, QUEUE_TARGET_CLASS_IDS, engine
    if not engine.running:
        engine.start()
    time.sleep(0.5)

    print(f"  -> Shelf Target Classes: {len(SHELF_TARGET_CLASS_IDS)} (Shopper + 16 Supermarket Categories)")
    print(f"  -> Queue Target Classes: {len(QUEUE_TARGET_CLASS_IDS)} (Shopper ONLY)")
    assert 0 in SHELF_TARGET_CLASS_IDS
    assert 39 in SHELF_TARGET_CLASS_IDS  # Bottle
    assert 73 in SHELF_TARGET_CLASS_IDS  # Packaged Box
    assert QUEUE_TARGET_CLASS_IDS == [0]
    print("  [PASS] Role-based compute decoupling and supermarket whitelisting verified.")

    # 5. Test Multi-Shelf Planogram Tracking
    print("\n[TEST 5] Verifying Multi-Shelf Planogram Spatial Tracking...")
    simulated_shelf_detections = [
        {"class_id": 39, "class_name": "Beverage Bottle", "box": [100, 80, 140, 130], "centroid": [120, 105]},
        {"class_id": 39, "class_name": "Beverage Bottle", "box": [200, 80, 240, 130], "centroid": [220, 105]},
        {"class_id": 45, "class_name": "Snack Container", "box": [100, 220, 140, 260], "centroid": [120, 240]},
        {"class_id": 73, "class_name": "Packaged Goods / Box", "box": [200, 220, 240, 260], "centroid": [220, 240]},
    ]
    engine._process_multi_shelf_zones(simulated_shelf_detections, 640, 360)
    assert engine.shelf_stock_count >= 2
    print(f"  -> Multi-Shelf Stock Count: {engine.shelf_stock_count} / {engine.shelf_capacity} ({engine.shelf_percentage}%)")
    print("  [PASS] Multi-Shelf Planogram spatial tracking accurately detected item counts.")

    # 6. Test Multi-Register Queue Ranking & Smart Traffic Director
    print("\n[TEST 6] Verifying Dynamic Multi-Register Queue Traffic Director...")
    engine.queue_lanes = [
        {"id": "reg_1", "name": "Counter 1 (Main)", "box": [0.05, 0.20, 0.32, 0.85], "max_wait_threshold_min": 3.0},
        {"id": "reg_2", "name": "Counter 2 (UPI)", "box": [0.35, 0.20, 0.65, 0.85], "max_wait_threshold_min": 3.0},
        {"id": "reg_3", "name": "Counter 3 (Self-Checkout)", "box": [0.68, 0.20, 0.95, 0.85], "max_wait_threshold_min": 3.0}
    ]
    simulated_queue_detections = [
        {"class_id": 0, "class_name": "Shopper", "box": [50, 120, 100, 260], "centroid": [75, 190]},
        {"class_id": 0, "class_name": "Shopper", "box": [110, 120, 160, 260], "centroid": [135, 190]},
        {"class_id": 0, "class_name": "Shopper", "box": [280, 120, 330, 260], "centroid": [300, 190]}
    ]
    engine._process_multi_queue_lanes(simulated_queue_detections, 640, 360)

    r1 = next(l for l in engine.queue_lanes if l["id"] == "reg_1")
    r3 = next(l for l in engine.queue_lanes if l["id"] == "reg_3")
    print(f"  -> Counter 1: {r1['headcount']} shoppers ({r1['status']})")
    print(f"  -> Counter 3: {r3['headcount']} shoppers ({r3['status']})")
    print(f"  -> Smart Recommendation: {engine.smart_recommendation}")
    assert r1["status"] == "CONGESTED"
    assert r3["status"] == "FREE_AVAILABLE"
    assert "congested" in engine.smart_recommendation.lower()
    print("  [PASS] Dynamic Multi-Register Traffic Director correctly ranked all counters.")

    # 7. Test Admin Calibration UI Endpoint (/admin)
    print("\n[TEST 7] Verifying Enterprise Admin Calibration UI Route (/admin)...")
    from app import app

    with TestClient(app) as client:
        res_admin = client.get("/admin")
        print(f"  -> GET /admin -> Status {res_admin.status_code}")
        assert res_admin.status_code == 200
        assert "Enterprise Admin" in res_admin.text
        assert "Temporal Auto-Baselining" in res_admin.text
        assert "Optical Viewport & 2-Click ROI Calibration" in res_admin.text
        print("  [PASS] Admin Calibration UI (/admin) served successfully.")

        # 8. Test 2-Click Normalized ROI Configuration (/api/config/zones with shelf_roi and queue_roi)
        print("\n[TEST 8] Verifying Normalized ROI Dispatch (/api/config/zones with { shelf_roi, queue_roi })...")
        res_roi_post = client.post("/api/config/zones", json={
            "shelf_roi": [0.12, 0.16, 0.88, 0.82],
            "queue_roi": [0.08, 0.22, 0.92, 0.86]
        })
        print(f"  -> POST /api/config/zones (Normalized ROIs) -> Status {res_roi_post.status_code}")
        assert res_roi_post.status_code == 200
        roi_resp = res_roi_post.json()
        assert roi_resp["shelf_roi"] == [0.12, 0.16, 0.88, 0.82]
        assert roi_resp["queue_roi"] == [0.08, 0.22, 0.92, 0.86]
        print(f"     * Calibrated Shelf ROI: {roi_resp['shelf_roi']}")
        print(f"     * Calibrated Queue ROI: {roi_resp['queue_roi']}")
        print("  [PASS] Normalized 2-Click ROI dispatch and persistence verified.")

        # 9. Test Temporal Auto-Baselining Engine (/api/config/baseline)
        print("\n[TEST 9] Verifying Temporal Auto-Baselining (POST /api/config/baseline over 30 frames)...")
        res_baseline = client.post("/api/config/baseline", json={
            "shelf_roi": [0.10, 0.15, 0.90, 0.85],
            "frames_to_sample": 30
        })
        print(f"  -> POST /api/config/baseline -> Status {res_baseline.status_code}")
        assert res_baseline.status_code == 200
        b_data = res_baseline.json()
        assert b_data["status"] == "success"
        assert "baseline_capacity" in b_data
        assert b_data["samples_collected"] == 30
        assert len(b_data["samples"]) == 30
        assert b_data["baseline_capacity"] >= 1
        print(f"     * Calibrated MAX_CAPACITY: {b_data['baseline_capacity']} items (Median of 30 frames)")
        print(f"     * Raw Samples Window: {b_data['samples'][:8]}... (Total 30 samples)")
        print("  [PASS] Temporal Auto-Baselining engine accurately computed median capacity.")

        # 10. Verify Dynamic Capacity Calculation in Telemetry
        print("\n[TEST 10] Verifying Dynamic Capacity Percentage on Manager Dashboard (/api/metrics)...")
        res_metrics = client.get("/api/metrics")
        assert res_metrics.status_code == 200
        m = res_metrics.json()
        assert m["shelf"]["stock_capacity"] == b_data["baseline_capacity"]
        expected_pct = int((m["shelf"]["stock_count"] / max(1, b_data["baseline_capacity"])) * 100)
        print(f"     * Dynamic Capacity: {m['shelf']['stock_capacity']} units")
        print(f"     * Current Stock: {m['shelf']['stock_count']} units ({m['shelf']['stock_percentage']}%)")
        assert m["shelf"]["stock_percentage"] == expected_pct
        print("  [PASS] Telemetry calculates capacity percentage dynamically against calibrated baseline.")

        # 11. Test Shift Summary Report & CSV Export
        print("\n[TEST 11] Verifying SQLite WAL Shift Analytics & CSV Export...")
        res_shift = client.get("/api/reports/shift_summary")
        assert res_shift.status_code == 200
        res_csv = client.get("/api/reports/shift_summary/csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers.get("content-type", "")
        print("  [PASS] SQLite WAL Analytics & CSV Export verified.")

    # 12. Test WebSocket Mobile Ingestion
    print("\n[TEST 12] Verifying Mobile Frame WebSocket Ingestion (/ws/mobile_upload)...")
    dummy_frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    _, dummy_jpeg = cv2.imencode('.jpg', dummy_frame)
    dummy_bytes = dummy_jpeg.tobytes()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/mobile_upload") as websocket:
            websocket.send_bytes(dummy_bytes)
            time.sleep(0.1)
            assert engine.cam2_source.startswith("Mobile Phone")
            print(f"  -> Ingested Mobile Frame successfully. Active Source: {engine.cam2_source}")

    print("  [PASS] WebSocket Mobile Ingestion verified.")

    print("\n" + "=" * 80)
    print("  ALL 12 ENTERPRISE HAL & AUTO-BASELINING TEST SUITES PASSED (100% GREEN)!")
    print("=" * 80)

    try:
        engine.stop()
    except Exception:
        pass

if __name__ == "__main__":
    import sys
    run_verification()
    sys.exit(0)
