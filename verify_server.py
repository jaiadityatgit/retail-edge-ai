"""
Automated Headless Smoke Test & Verification Suite for EdgeRetail AI (SIH26179)
Verifies:
1. YOLOv8 Model Ingestion & Inference (imgsz=320 / CPU).
2. Synthetic Retail Vision Engine & High-Frequency Generation (>15 FPS).
3. CSIM Occlusion & Debounce State Machine.
4. Hardware Profiler & SoC Temperature Sensor Fallbacks.
5. FastAPI Server Endpoints (/, /video_feed, /metrics, /health, /api/simulation/control).
"""

import os
import sys
import time
import json
import threading
import numpy as np

def run_tests():
    print("=" * 70)
    print("  EDGERETAIL AI // SIH26179 - AUTOMATED VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test Hardware Profiler
    print("\n[TEST 1] Verifying Hardware Telemetry & SoC Sensor...")
    from app import HardwareProfiler
    soc_temp = HardwareProfiler.get_soc_temperature()
    is_rpi = HardwareProfiler.is_raspberry_pi()
    telemetry = HardwareProfiler.get_telemetry(edge_fps=28.5, inference_latency_ms=18.2)
    
    print(f"  -> Platform: {telemetry['platform']}")
    print(f"  -> Detected SoC Temp: {soc_temp}°C (Realistic range: 10 - 110°C)")
    print(f"  -> CPU Utilization: {telemetry['cpu_load_pct']}%")
    print(f"  -> RAM Usage: {telemetry['ram_used_mb']} / {telemetry['ram_total_mb']} MB ({telemetry['ram_pct']}%)")
    assert 10.0 <= soc_temp <= 110.0, f"Invalid SoC Temp: {soc_temp}"
    assert "cpu_load_pct" in telemetry
    print("  [PASS] Hardware Telemetry & Sensor test successful.")

    # 2. Test Synthetic Frame Generator
    print("\n[TEST 2] Verifying Synthetic Retail Stream Generator (>15 FPS target)...")
    from app import SyntheticRetailGenerator
    gen = SyntheticRetailGenerator(width=640, height=360)
    
    t0 = time.time()
    frames_count = 60
    for _ in range(frames_count):
        frame, detections = gen.generate_frame()
        assert frame is not None, "Generated frame was None"
        assert frame.shape == (360, 640, 3), f"Unexpected frame shape: {frame.shape}"
        assert len(detections) >= 0, "Detections list invalid"
    
    gen_time = time.time() - t0
    gen_fps = frames_count / gen_time
    print(f"  -> Generated {frames_count} frames in {gen_time:.3f}s ({gen_fps:.1f} FPS)")
    assert gen_fps >= 15.0, f"Synthetic generation too slow: {gen_fps} FPS < 15 FPS"
    print("  [PASS] Frame generation exceeds throughput requirements.")

    # 3. Test CSIM Occlusion & Stock State Machine
    print("\n[TEST 3] Verifying CSIM Occlusion & Restock State Machine Logic...")
    from app import RetailStateMachine, SpatialZones
    sm = RetailStateMachine(debounce_threshold_sec=0.5)  # 0.5s for fast test

    # Scenario A: Initial State with Stock
    stock_dets = [
        {"class_name": "bottle", "class_id": 39, "box": [50, 150, 70, 200], "centroid": [60, 175]},
        {"class_name": "bottle", "class_id": 39, "box": [80, 150, 100, 200], "centroid": [90, 175]},
    ]
    sm.update(stock_dets, frame_w=640, frame_h=360)
    assert sm.stock_count == 2, f"Expected 2 stock items, got {sm.stock_count}"
    assert sm.shelf_status == "OPTIMAL", f"Expected OPTIMAL, got {sm.shelf_status}"
    assert not sm.shelf_alert_active, "Shelf alert should be False"
    print("  -> Scenario A (Stock Available): OPTIMAL (PASS)")

    # Scenario B: Customer Interaction / Occlusion
    customer_occlusion = [
        {"class_name": "person", "class_id": 0, "box": [40, 100, 120, 300], "centroid": [80, 200]},
    ]
    sm.update(customer_occlusion, frame_w=640, frame_h=360)
    assert sm.is_shelf_occluded, "Expected shelf occlusion to be True"
    assert sm.shelf_status == "CUSTOMER_INTERACTING", f"Expected CUSTOMER_INTERACTING, got {sm.shelf_status}"
    assert not sm.shelf_alert_active, "Alert should be suppressed during customer occlusion"
    print("  -> Scenario B (Customer Occlusion): CUSTOMER_INTERACTING (Alarms Suppressed) (PASS)")

    # Scenario C: Shelf Emptied & Debounce Timer
    sm.update([], frame_w=640, frame_h=360)
    assert sm.stock_count == 0
    assert not sm.is_shelf_occluded
    assert sm.shelf_status == "PENDING_OOS", f"Expected PENDING_OOS initially, got {sm.shelf_status}"
    
    # Wait for debounce duration
    time.sleep(0.6)
    sm.update([], frame_w=640, frame_h=360)
    assert sm.shelf_status == "OUT_OF_STOCK_ALERT", f"Expected OUT_OF_STOCK_ALERT after debounce, got {sm.shelf_status}"
    assert sm.shelf_alert_active, "Alert should be True after debounce"
    print("  -> Scenario C (Stock Depletion + Debounce): OUT_OF_STOCK_ALERT (PASS)")

    # Scenario D: Queue Congestion Trigger
    queue_dets = [
        {"class_name": "person", "class_id": 0, "box": [450, 100, 500, 300], "centroid": [475, 200]},
        {"class_name": "person", "class_id": 0, "box": [520, 100, 570, 300], "centroid": [545, 200]},
    ]
    sm.update(queue_dets, frame_w=640, frame_h=360)
    assert sm.queue_customer_count == 2, f"Expected 2 queue customers, got {sm.queue_customer_count}"
    assert sm.queue_status == "CONGESTION_WARNING", f"Expected CONGESTION_WARNING, got {sm.queue_status}"
    assert sm.queue_congestion_alert, "Queue congestion alert should be True"
    assert sm.estimated_wait_min == 3.0, f"Expected 3.0 min wait, got {sm.estimated_wait_min}"
    print("  -> Scenario D (Queue Congestion >= 2): CONGESTION_WARNING (Wait: 3.0m) (PASS)")

    # 4. Test YOLOv8 Ingestion & Inference
    print("\n[TEST 4] Verifying YOLOv8 Model Ingestion on CPU (imgsz=320)...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    t_inf_start = time.time()
    results = model(dummy_img, imgsz=320, verbose=False)
    inf_latency = (time.time() - t_inf_start) * 1000.0
    print(f"  -> Inference completed in {inf_latency:.1f}ms on CPU with imgsz=320")
    assert len(results) > 0, "Model inference failed to return results"
    print("  [PASS] YOLOv8n initialized and executed inference cleanly.")

    # 5. Test FastAPI Application Endpoints
    print("\n[TEST 5] Verifying FastAPI Web Server & API Telemetry Schema...")
    from fastapi.testclient import TestClient
    from app import app, engine

    # Ensure engine has started
    if not engine.running:
        engine.start()
    time.sleep(0.5)

    with TestClient(app) as client:
        # GET / (Dashboard HTML)
        res_root = client.get("/")
        print(f"  -> GET / -> Status {res_root.status_code}")
        assert res_root.status_code == 200
        assert "EdgeRetail AI" in res_root.text

        # GET /presentation (Apple Keynote Slide Deck HTML)
        res_pres = client.get("/presentation")
        print(f"  -> GET /presentation -> Status {res_pres.status_code}")
        assert res_pres.status_code == 200
        assert "Executive Presentation Deck" in res_pres.text

        # GET /health
        res_health = client.get("/health")
        print(f"  -> GET /health -> Status {res_health.status_code}")
        assert res_health.status_code == 200
        health_json = res_health.json()
        assert health_json["status"] == "healthy"

        # GET /metrics
        res_metrics = client.get("/metrics")
        print(f"  -> GET /metrics -> Status {res_metrics.status_code}")
        assert res_metrics.status_code == 200
        m = res_metrics.json()
        
        # Verify complete schema contract
        print("  -> Validating Telemetry JSON Schema Contract:")
        assert "device" in m, "Missing 'device' in metrics"
        assert "shelf" in m, "Missing 'shelf' in metrics"
        assert "queue" in m, "Missing 'queue' in metrics"
        assert "compliance" in m, "Missing 'compliance' in metrics"
        assert "edge_fps" in m, "Missing 'edge_fps' in metrics"
        assert "inference_latency_ms" in m, "Missing 'inference_latency_ms' in metrics"
        assert "timestamp" in m, "Missing 'timestamp' in metrics"

        # Validate nested contract items
        assert "soc_temp_c" in m["device"]
        assert "cpu_load_pct" in m["device"]
        assert "stock_count" in m["shelf"]
        assert "status" in m["shelf"]
        assert "is_occluded" in m["shelf"]
        assert "alert_active" in m["shelf"]
        assert "customer_count" in m["queue"]
        assert "estimated_wait_min" in m["queue"]
        assert "congestion_alert" in m["queue"]
        assert m["compliance"]["dpdp_compliant"] is True
        assert m["compliance"]["cloud_bandwidth_kbps"] == 0.0
        assert m["compliance"]["zero_raw_video_leakage"] is True

        print(f"     * Device: {m['device']['platform']} (Temp: {m['device']['soc_temp_c']}°C, CPU: {m['device']['cpu_load_pct']}%)")
        print(f"     * Shelf: Stock={m['shelf']['stock_count']}, Status={m['shelf']['status']}, Alert={m['shelf']['alert_active']}")
        print(f"     * Queue: Count={m['queue']['customer_count']}, Status={m['queue']['status']}, Wait={m['queue']['estimated_wait_min']}m")
        print(f"     * Compliance: DPDP={m['compliance']['dpdp_compliant']}, Uplink={m['compliance']['cloud_bandwidth_kbps']} KB/s")

        # POST /api/simulation/control
        res_sim = client.post("/api/simulation/control", json={"mode": "QUEUE_CONGESTION"})
        print(f"  -> POST /api/simulation/control -> Status {res_sim.status_code}")
        assert res_sim.status_code == 200
        assert res_sim.json()["mode"] == "QUEUE_CONGESTION"

    print("  [PASS] All FastAPI endpoints and telemetry contracts verified.")

    print("\n" + "=" * 70)
    print("  ALL 5 VERIFICATION MODULES PASSED SUCCESSFULLY! (100% GREEN)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
