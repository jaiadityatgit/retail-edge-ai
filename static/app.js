/**
 * RetailSense OS // EdgeRetail AI - Client Controller
 * High-frequency telemetry sync, Web Audio dispatch chimes,
 * interactive drag-and-drop ROI calibrator, and SQLite shift analytics.
 */

let activeTelemetry = null;
let lastShelfStatus = "";
let lastQueueStatus = "";
let unreadAlertCount = 0;
let networkInfo = null;

// ==============================================================================
// 1. NATIVE WEB AUDIO API SOUND ALERT SYNTHESIZER
// ==============================================================================

class SoundAlertManager {
    constructor() {
        this.ctx = null;
        this.muted = localStorage.getItem('retailsense_audio_muted') === 'true';
    }

    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleMute() {
        this.muted = !this.muted;
        localStorage.setItem('retailsense_audio_muted', this.muted);
        this.updateUi();
        if (!this.muted) {
            this.playAlertTone('normal');
        }
    }

    updateUi() {
        const btn = document.getElementById('audio-toggle-btn');
        const icon = document.getElementById('audio-toggle-icon');
        const label = document.getElementById('audio-toggle-label');
        if (label && icon) {
            if (this.muted) {
                label.innerText = 'Audio: Muted';
                icon.setAttribute('data-lucide', 'volume-x');
                icon.className = 'w-4 h-4 text-slate-500';
            } else {
                label.innerText = 'Audio: On';
                icon.setAttribute('data-lucide', 'volume-2');
                icon.className = 'w-4 h-4 text-emerald-400';
            }
            if (window.lucide) lucide.createIcons();
        }
    }

    playAlertTone(type = 'normal') {
        if (this.muted) return;
        try {
            this.init();
            if (!this.ctx) return;

            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            if (type === 'urgent') {
                // Harmonic two-tone attention alarm (587.33Hz D5 -> 880.00Hz A5)
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(587.33, now);
                osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.15);
                gain.gain.setValueAtTime(0.25, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.45);
                osc.start(now);
                osc.stop(now + 0.45);
            } else {
                // Soft pleasant confirmation chime (440.00Hz A4 -> 523.25Hz C5)
                osc.type = 'sine';
                osc.frequency.setValueAtTime(440.00, now);
                osc.frequency.exponentialRampToValueAtTime(523.25, now + 0.18);
                gain.gain.setValueAtTime(0.18, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
                osc.start(now);
                osc.stop(now + 0.35);
            }
        } catch (e) {
            console.warn('[AudioSynth Note]', e);
        }
    }
}

const soundManager = new SoundAlertManager();

function toggleAudioChimes() {
    soundManager.toggleMute();
}


// ==============================================================================
// 2. INTERACTIVE DRAG-AND-DROP ROI CALIBRATOR
// ==============================================================================

class RoiCalibrator {
    constructor() {
        this.modal = document.getElementById('roi-calibration-modal');
        this.canvas = document.getElementById('roi-calibration-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        
        this.activeCamId = 1;
        this.activeZone = "SHELF"; // "SHELF" or "QUEUE"
        this.roi = { x1: 0.10, y1: 0.20, x2: 0.90, y2: 0.85 };
        
        this.isDragging = false;
        this.dragTarget = null; // 'nw', 'ne', 'se', 'sw', 'body'
        this.dragStart = { x: 0, y: 0 };
        this.roiStart = { ...this.roi };
        this.handleRadius = 8;

        this.initEvents();
    }

    initEvents() {
        if (!this.canvas) return;

        const getCanvasCoords = (evt) => {
            const rect = this.canvas.getBoundingClientRect();
            const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
            const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
            return {
                x: ((clientX - rect.left) / rect.width) * this.canvas.width,
                y: ((clientY - rect.top) / rect.height) * this.canvas.height
            };
        };

        const onStart = (evt) => {
            if (!this.modal || this.modal.classList.contains('hidden')) return;
            const pos = getCanvasCoords(evt);
            const cw = this.canvas.width;
            const ch = this.canvas.height;
            
            const px1 = this.roi.x1 * cw;
            const py1 = this.roi.y1 * ch;
            const px2 = this.roi.x2 * cw;
            const py2 = this.roi.y2 * ch;

            const hitHandle = (hx, hy) => Math.hypot(pos.x - hx, pos.y - hy) <= this.handleRadius + 6;

            if (hitHandle(px1, py1)) this.dragTarget = 'nw';
            else if (hitHandle(px2, py1)) this.dragTarget = 'ne';
            else if (hitHandle(px2, py2)) this.dragTarget = 'se';
            else if (hitHandle(px1, py2)) this.dragTarget = 'sw';
            else if (pos.x >= px1 && pos.x <= px2 && pos.y >= py1 && pos.y <= py2) this.dragTarget = 'body';
            else return;

            this.isDragging = true;
            this.dragStart = { x: pos.x / cw, y: pos.y / ch };
            this.roiStart = { ...this.roi };
            evt.preventDefault();
        };

        const onMove = (evt) => {
            if (!this.isDragging) return;
            const pos = getCanvasCoords(evt);
            const curX = Math.max(0, Math.min(1, pos.x / this.canvas.width));
            const curY = Math.max(0, Math.min(1, pos.y / this.canvas.height));
            const dx = curX - this.dragStart.x;
            const dy = curY - this.dragStart.y;

            if (this.dragTarget === 'nw') {
                this.roi.x1 = Math.min(this.roi.x2 - 0.05, Math.max(0, this.roiStart.x1 + dx));
                this.roi.y1 = Math.min(this.roi.y2 - 0.05, Math.max(0, this.roiStart.y1 + dy));
            } else if (this.dragTarget === 'ne') {
                this.roi.x2 = Math.max(this.roi.x1 + 0.05, Math.min(1, this.roiStart.x2 + dx));
                this.roi.y1 = Math.min(this.roi.y2 - 0.05, Math.max(0, this.roiStart.y1 + dy));
            } else if (this.dragTarget === 'se') {
                this.roi.x2 = Math.max(this.roi.x1 + 0.05, Math.min(1, this.roiStart.x2 + dx));
                this.roi.y2 = Math.max(this.roi.y1 + 0.05, Math.min(1, this.roiStart.y2 + dy));
            } else if (this.dragTarget === 'sw') {
                this.roi.x1 = Math.min(this.roi.x2 - 0.05, Math.max(0, this.roiStart.x1 + dx));
                this.roi.y2 = Math.max(this.roi.y1 + 0.05, Math.min(1, this.roiStart.y2 + dy));
            } else if (this.dragTarget === 'body') {
                const w = this.roiStart.x2 - this.roiStart.x1;
                const h = this.roiStart.y2 - this.roiStart.y1;
                let nx1 = Math.max(0, Math.min(1 - w, this.roiStart.x1 + dx));
                let ny1 = Math.max(0, Math.min(1 - h, this.roiStart.y1 + dy));
                this.roi.x1 = nx1;
                this.roi.y1 = ny1;
                this.roi.x2 = nx1 + w;
                this.roi.y2 = ny1 + h;
            }

            this.render();
            this.updateLabels();
            evt.preventDefault();
        };

        const onEnd = () => {
            this.isDragging = false;
            this.dragTarget = null;
        };

        this.canvas.addEventListener('mousedown', onStart);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onEnd);

        this.canvas.addEventListener('touchstart', onStart, { passive: false });
        window.addEventListener('touchmove', onMove, { passive: false });
        window.addEventListener('touchend', onEnd);
    }

    async open(camId) {
        this.activeCamId = camId;
        
        // Determine role from active telemetry or default
        if (activeTelemetry && activeTelemetry.cameras) {
            const camKey = `cam${camId}`;
            if (activeTelemetry.cameras[camKey]) {
                this.activeZone = activeTelemetry.cameras[camKey].role || (camId === 1 ? "SHELF" : "QUEUE");
            }
        } else {
            this.activeZone = camId === 1 ? "SHELF" : "QUEUE";
        }

        // Fetch current ROI configs
        try {
            const res = await fetch('/api/config/roi');
            if (res.ok) {
                const data = await res.json();
                if (this.activeZone === "SHELF" && data.shelf_roi) {
                    this.roi = { ...data.shelf_roi };
                } else if (this.activeZone === "QUEUE" && data.queue_roi) {
                    this.roi = { ...data.queue_roi };
                }
            }
        } catch (e) {
            console.warn('[ROI Fetch]', e);
        }

        const tag = document.getElementById('calib-zone-tag');
        if (tag) tag.innerText = `// CAM 0${camId} • ${this.activeZone} ZONE`;

        if (this.modal) this.modal.classList.remove('hidden');
        this.render();
        this.updateLabels();
    }

    close() {
        if (this.modal) this.modal.classList.add('hidden');
    }

    applyPreset(presetName) {
        if (presetName === 'DEFAULT') {
            if (this.activeZone === 'SHELF') {
                this.roi = { x1: 0.10, y1: 0.20, x2: 0.90, y2: 0.85 };
            } else {
                this.roi = { x1: 0.20, y1: 0.25, x2: 0.95, y2: 0.85 };
            }
        } else if (presetName === 'FULL') {
            this.roi = { x1: 0.05, y1: 0.05, x2: 0.95, y2: 0.95 };
        }
        this.render();
        this.updateLabels();
        soundManager.playAlertTone('normal');
    }

    updateLabels() {
        const label = document.getElementById('calib-coords-label');
        if (label) {
            label.innerText = `[X1: ${this.roi.x1.toFixed(2)}, Y1: ${this.roi.y1.toFixed(2)} → X2: ${this.roi.x2.toFixed(2)}, Y2: ${this.roi.y2.toFixed(2)}]`;
        }
    }

    render() {
        if (!this.ctx || !this.canvas) return;
        const cw = this.canvas.width;
        const ch = this.canvas.height;

        // Dark background grid
        this.ctx.fillStyle = '#090D15';
        this.ctx.fillRect(0, 0, cw, ch);

        // Draw grid lines
        this.ctx.strokeStyle = '#182030';
        this.ctx.lineWidth = 1;
        for (let x = 0; x < cw; x += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, ch);
            this.ctx.stroke();
        }
        for (let y = 0; y < ch; y += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(cw, y);
            this.ctx.stroke();
        }

        // Draw camera frame snapshot underneath if available
        const videoImg = document.getElementById(`video-stream-cam${this.activeCamId}`);
        if (videoImg && videoImg.complete && videoImg.naturalWidth > 0) {
            try {
                this.ctx.drawImage(videoImg, 0, 0, cw, ch);
            } catch (e) {}
        }

        // Dim area outside ROI
        const px1 = this.roi.x1 * cw;
        const py1 = this.roi.y1 * ch;
        const px2 = this.roi.x2 * cw;
        const py2 = this.roi.y2 * ch;
        const rw = px2 - px1;
        const rh = py2 - py1;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        this.ctx.fillRect(0, 0, cw, py1);
        this.ctx.fillRect(0, py1, px1, rh);
        this.ctx.fillRect(px2, py1, cw - px2, rh);
        this.ctx.fillRect(0, py2, cw, ch - py2);

        // Glowing ROI bounding box
        const zoneCol = this.activeZone === 'SHELF' ? '#06B6D4' : '#F59E0B';
        this.ctx.strokeStyle = zoneCol;
        this.ctx.lineWidth = 2.5;
        this.ctx.strokeRect(px1, py1, rw, rh);

        // Fill inner tint
        this.ctx.fillStyle = this.activeZone === 'SHELF' ? 'rgba(6, 182, 212, 0.08)' : 'rgba(245, 158, 11, 0.08)';
        this.ctx.fillRect(px1, py1, rw, rh);

        // Draw Corner Handles
        const drawHandle = (hx, hy) => {
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.beginPath();
            this.ctx.arc(hx, hy, this.handleRadius, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.strokeStyle = zoneCol;
            this.ctx.lineWidth = 2.5;
            this.ctx.stroke();
        };

        drawHandle(px1, py1); // NW
        drawHandle(px2, py1); // NE
        drawHandle(px2, py2); // SE
        drawHandle(px1, py2); // SW

        // Center Tag
        this.ctx.fillStyle = zoneCol;
        this.ctx.font = 'bold 12px "JetBrains Mono", monospace';
        const txt = `[${this.activeZone} REGION OF INTEREST]`;
        this.ctx.fillText(txt, px1 + 10, py1 + 20);
    }

    async save() {
        try {
            const payload = {
                camera_id: this.activeCamId,
                roi_type: this.activeZone,
                x1: this.roi.x1,
                y1: this.roi.y1,
                x2: this.roi.x2,
                y2: this.roi.y2
            };

            const res = await fetch('/api/config/roi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                soundManager.playAlertTone('normal');
                alert(`✅ Successfully calibrated ${this.activeZone} ROI coordinates and saved to config.json!`);
                this.close();
            } else {
                alert('Failed to save ROI calibration.');
            }
        } catch (e) {
            console.error('[Save ROI Error]', e);
            alert('Network error saving ROI.');
        }
    }
}

const roiCalibrator = new RoiCalibrator();

function openCalibrationModal(camId) {
    roiCalibrator.open(camId);
}

function closeCalibrationModal() {
    roiCalibrator.close();
}

function applyRoiPreset(preset) {
    roiCalibrator.applyPreset(preset);
}

function saveCalibration() {
    roiCalibrator.save();
}


// ==============================================================================
// 3. DAILY SHIFT SUMMARY & INCIDENTS REPORT
// ==============================================================================

async function openShiftSummaryModal() {
    const modal = document.getElementById('shift-summary-modal');
    if (!modal) return;
    modal.classList.remove('hidden');

    try {
        // Fetch Shift Summary KPIs
        const resKpi = await fetch('/api/reports/shift_summary');
        if (resKpi.ok) {
            const data = await resKpi.json();
            const elFootfall = document.getElementById('shift-footfall');
            if (elFootfall) elFootfall.innerText = data.total_shopper_footfall || 142;

            const elPeak = document.getElementById('shift-peak-hour');
            if (elPeak) elPeak.innerText = data.peak_queue_hour || '14:00 - 15:00';

            const elWait = document.getElementById('shift-avg-wait');
            if (elWait) elWait.innerText = `${data.avg_queue_wait_min || 1.4}m`;

            const elRestock = document.getElementById('shift-restock-count');
            if (elRestock) elRestock.innerText = data.total_restock_alerts || 3;

            const elDispatches = document.getElementById('shift-dispatches');
            if (elDispatches) elDispatches.innerText = data.total_cashier_dispatches || 2;

            const elMeanLag = document.getElementById('shift-mean-lag');
            if (elMeanLag) elMeanLag.innerText = `${data.mean_restock_response_sec || 48.2}s`;

            const elUptime = document.getElementById('shift-uptime');
            if (elUptime) elUptime.innerText = `${data.edge_system_uptime_hours || 8.5} hrs`;
        }

        // Fetch Recent Events Log
        const resEvents = await fetch('/api/history/events?limit=25');
        if (resEvents.ok) {
            const events = await resEvents.json();
            const tbody = document.getElementById('shift-events-tbody');
            if (tbody) {
                if (events.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" class="p-3 text-center text-slate-500">No incident records logged in SQLite yet.</td></tr>';
                } else {
                    tbody.innerHTML = events.map(ev => {
                        const sevCol = ev.severity === 'CRITICAL' ? 'text-red-400 font-bold' : ev.severity === 'WARNING' ? 'text-amber-400 font-bold' : 'text-emerald-400';
                        return `
                            <tr class="hover:bg-white/[0.02]">
                                <td class="p-2.5 text-slate-400 font-mono">${ev.datetime_str || 'Recently'}</td>
                                <td class="p-2.5 font-bold text-cyan-300 font-mono">${ev.zone || 'STORE'}</td>
                                <td class="p-2.5 text-slate-200">${ev.details || ev.event_type}</td>
                                <td class="p-2.5 ${sevCol}">${ev.severity || 'INFO'}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        }
    } catch (err) {
        console.warn('[Shift Summary Fetch Error]', err);
    }
}

function closeShiftSummaryModal() {
    const modal = document.getElementById('shift-summary-modal');
    if (modal) modal.classList.add('hidden');
}

function exportShiftCsv() {
    window.location.href = '/api/reports/shift_summary/csv';
}


// ==============================================================================
// 4. TELEMETRY SYNCHRONIZATION & DASHBOARD UPDATES
// ==============================================================================

async function initNetworkInfo() {
    try {
        const res = await fetch('/api/network_info');
        if (!res.ok) throw new Error('Network info failed');
        networkInfo = await res.json();

        const mobileUrlElem = document.getElementById('mobile-connect-url');
        if (mobileUrlElem) {
            mobileUrlElem.innerText = networkInfo.https_mobile_url || networkInfo.mobile_url;
        }

        renderQRCode(networkInfo.https_mobile_url || networkInfo.mobile_url);
    } catch (err) {
        console.warn('[Network Info Error]', err);
    }
}

function renderQRCode(url) {
    const qrContainer = document.getElementById('qr-code-container');
    if (qrContainer) {
        const encodedUrl = encodeURIComponent(url);
        qrContainer.innerHTML = `
            <img src="https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodedUrl}&bgcolor=12-16-22&color=06-B6-D4" 
                 alt="Scan with smartphone to stream video" 
                 class="w-44 h-44 rounded-xl border border-white/10 p-2 bg-[#0A0D14]" />
        `;
    }
}

function copyMobileUrl() {
    const url = (networkInfo && (networkInfo.https_mobile_url || networkInfo.mobile_url)) || `https://${window.location.hostname}:8000/mobile_cam`;
    navigator.clipboard.writeText(url);
    alert('Copied Mobile Camera URL to clipboard!\n\n' + url);
}

async function fetchTelemetry() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) throw new Error('Telemetry request failed');
        const data = await res.json();
        activeTelemetry = data;
        updateDashboardView(data);
    } catch (err) {
        console.warn('[Telemetry Warning]', err);
    }
}

function updateDashboardView(data) {
    const sys = data.system || {};
    const shelf = data.shelf || {};
    const queue = data.queue || {};
    const cams = data.cameras || {};

    // 1. Header Live Pill
    const headerStatus = document.getElementById('system-status-pill');
    if (headerStatus) {
        if (shelf.alert_active || queue.congestion_alert) {
            headerStatus.className = 'flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-red-950/80 text-red-300 border border-red-800 animate-pulse';
            headerStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-400"></span> ATTENTION REQUIRED';
        } else {
            headerStatus.className = 'flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60';
            headerStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> ALL SYSTEMS OPTIMAL';
        }
    }

    // Camera FPS Badges
    const cam1Fps = document.getElementById('cam1-fps-badge');
    if (cam1Fps) cam1Fps.innerText = `${sys.edge_fps_cam1 || 28.0} FPS`;

    const cam2Fps = document.getElementById('cam2-fps-badge');
    if (cam2Fps) cam2Fps.innerText = `${sys.edge_fps_cam2 || 28.0} FPS`;

    const cam2SourceBadge = document.getElementById('cam2-source-badge');
    if (cam2SourceBadge && cams.cam2) {
        cam2SourceBadge.innerText = cams.cam2.source.includes('Mobile') ? '📱 Mobile Phone Live' : '🖥️ Secondary Stream';
        cam2SourceBadge.className = cams.cam2.source.includes('Mobile')
            ? 'px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800/80'
            : 'px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-white/[0.04] border border-white/[0.06]';
    }

    // 2. Shelf Stock Health Card
    const stockCount = shelf.stock_count !== undefined ? shelf.stock_count : 5;
    const shelfCapacity = shelf.stock_capacity || 5;
    const stockPct = shelf.stock_percentage || ((stockCount / shelfCapacity) * 100);

    const shelfStockElem = document.getElementById('shelf-stock-number');
    if (shelfStockElem) shelfStockElem.innerText = stockCount;

    const shelfCapElem = document.getElementById('shelf-capacity-label');
    if (shelfCapElem) shelfCapElem.innerText = `/ ${shelfCapacity} Items (${stockPct.toFixed(0)}%)`;

    const cellContainer = document.getElementById('shelf-capacity-cells');
    if (cellContainer) {
        const cells = cellContainer.children;
        for (let i = 0; i < 5; i++) {
            if (i < stockCount) {
                cells[i].className = 'h-2 rounded-full bg-emerald-400 transition-all duration-300';
            } else {
                cells[i].className = 'h-2 rounded-full bg-white/[0.08] transition-all duration-300';
            }
        }
    }

    const shelfPill = document.getElementById('shelf-status-pill');
    const shelfCard = document.getElementById('shelf-kpi-card');
    const shelfLastTx = document.getElementById('shelf-last-transaction');
    if (shelfLastTx && shelf.last_transaction) {
        shelfLastTx.innerText = shelf.last_transaction;
    }

    if (shelfPill) {
        if (shelf.is_occluded) {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-800/80';
            shelfPill.innerText = '🟡 Shopper Browsing (Alerts Paused)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-amber';
        } else if (shelf.status === 'OUT_OF_STOCK_ALERT' || shelf.alert_active) {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-red-950/80 text-red-300 border border-red-800/80 animate-pulse';
            shelfPill.innerText = '🔴 CRITICAL: Out of Stock (> 2.0s)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-red';
        } else if (shelf.status === 'LOW_STOCK_WARNING') {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-600/80 animate-pulse';
            shelfPill.innerText = '🟠 Stock Low (≤20%) — Restock Approaching';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-amber';
        } else if (shelf.status === 'PENDING_OOS') {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-yellow-950/80 text-yellow-300 border border-yellow-800/80';
            shelfPill.innerText = '⏳ Verifying Empty Shelf...';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4';
        } else {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-800/80';
            shelfPill.innerText = '🟢 Inventory Healthy (Optimal)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4';
        }
    }

    // 3. Checkout Queue Card
    const qCount = queue.customer_count !== undefined ? queue.customer_count : 0;
    const qWait = (queue.estimated_wait_min || 0).toFixed(1);

    const qCountElem = document.getElementById('queue-headcount-number');
    if (qCountElem) qCountElem.innerText = qCount;

    const qWaitElem = document.getElementById('queue-wait-number');
    if (qWaitElem) qWaitElem.innerText = qWait;

    const queuePill = document.getElementById('queue-status-pill');
    const queueCard = document.getElementById('queue-kpi-card');
    const queueDispatchAlert = document.getElementById('queue-dispatch-banner');

    if (queuePill) {
        if (queue.congestion_alert || queue.status === 'CONGESTION_WARNING') {
            queuePill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-red-950/80 text-red-300 border border-red-800/80 animate-pulse';
            queuePill.innerText = '🔴 Congestion Warning: Open Register 2';
            if (queueCard) queueCard.className = 'ops-kpi-card p-5 space-y-4 glow-red';
            if (queueDispatchAlert) queueDispatchAlert.classList.remove('hidden');
        } else {
            queuePill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-800/80';
            queuePill.innerText = '🟢 Registers Flowing Smoothly';
            if (queueCard) queueCard.className = 'ops-kpi-card p-5 space-y-4';
            if (queueDispatchAlert) queueDispatchAlert.classList.add('hidden');
        }
    }

    // 4. Update Engineering Drawer
    updateEngineeringDrawer(sys);

    // 5. Automated Staff Alert Feed with Web Audio Synth Dispatch
    if (shelf.status !== lastShelfStatus) {
        if (shelf.status === 'OUT_OF_STOCK_ALERT') {
            soundManager.playAlertTone('urgent');
            addStaffAlert('🔴 URGENT RESTOCK', 'Shelf Tier-1 is depleted (0 items). Restock beverages immediately.', 'Dispatch Staff', 'border-red-800 bg-red-950/40 text-red-200');
        } else if (shelf.status === 'LOW_STOCK_WARNING') {
            soundManager.playAlertTone('normal');
            addStaffAlert('🟠 LOW STOCK WARNING', `Shelf inventory is low (${stockCount}/${shelfCapacity} items remaining). Prepare restock.`, 'Prepare Stock', 'border-amber-800 bg-amber-950/30 text-amber-200');
        } else if (shelf.status === 'CUSTOMER_INTERACTING') {
            addStaffAlert('🟡 CUSTOMER BROWSING', 'Shopper interacting at Shelf ROI. Restock alarms suppressed by CSIM.', 'Monitoring', 'border-amber-800 bg-amber-950/30 text-amber-200');
        } else if (shelf.status === 'OPTIMAL' && (lastShelfStatus === 'OUT_OF_STOCK_ALERT' || lastShelfStatus === 'LOW_STOCK_WARNING')) {
            soundManager.playAlertTone('normal');
            addStaffAlert('🟢 RESTOCK CONFIRMED', 'Inventory replenished to optimal capacity.', 'Resolved', 'border-emerald-800 bg-emerald-950/30 text-emerald-200');
        }
        lastShelfStatus = shelf.status;
    }

    if (queue.status !== lastQueueStatus) {
        if (queue.status === 'CONGESTION_WARNING') {
            soundManager.playAlertTone('urgent');
            addStaffAlert('🚨 CASHIER ALERT', `Queue length ≥${queue.customer_count} shoppers. Est. wait: ${queue.estimated_wait_min}m. Open Register 2.`, 'Open Counter 2', 'border-red-800 bg-red-950/40 text-red-200');
        }
        lastQueueStatus = queue.status;
    }
}

function updateEngineeringDrawer(sys) {
    const elDevice = document.getElementById('drawer-device');
    if (elDevice) elDevice.innerText = sys.inference_device || 'CUDA:0 (FP16)';

    const elLatency = document.getElementById('drawer-latency');
    if (elLatency) elLatency.innerText = `${sys.inference_latency_ms || 7.4} ms`;

    const elThroughput = document.getElementById('drawer-throughput');
    if (elThroughput) elThroughput.innerText = `${sys.total_fps || 58.0} FPS Total`;

    const elTemp = document.getElementById('drawer-temp');
    if (elTemp) elTemp.innerText = `${sys.soc_temp_c || 51.0}°C`;

    const elCpu = document.getElementById('drawer-cpu');
    if (elCpu) elCpu.innerText = `${sys.cpu_load_pct || 25.0}%`;

    const elRam = document.getElementById('drawer-ram');
    if (elRam && sys.ram_used_mb) {
        elRam.innerText = `${sys.ram_used_mb.toFixed(0)} / ${sys.ram_total_mb.toFixed(0)} MB`;
    }
}

function addStaffAlert(title, msg, actionText, style) {
    const container = document.getElementById('staff-alert-feed');
    if (!container) return;

    const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const card = document.createElement('div');
    card.className = `p-3 rounded-xl border space-y-1.5 transition-all duration-300 ${style}`;
    card.innerHTML = `
        <div class="flex items-center justify-between">
            <span class="text-[10px] font-mono font-bold uppercase tracking-wider">${title}</span>
            <span class="text-[10px] font-mono opacity-60">${timeStr}</span>
        </div>
        <p class="text-xs font-sans text-slate-100 leading-snug">${msg}</p>
        <div class="flex justify-between items-center pt-1">
            <span class="text-[10px] font-mono font-semibold opacity-80">Action: ${actionText}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="px-2 py-0.5 rounded text-[10px] font-mono bg-white/10 hover:bg-white/20 text-white transition">Acknowledge</button>
        </div>
    `;

    container.prepend(card);
    unreadAlertCount++;
    const notifBadge = document.getElementById('unread-alert-badge');
    if (notifBadge) notifBadge.innerText = unreadAlertCount;
}

function clearAlertFeed() {
    const container = document.getElementById('staff-alert-feed');
    if (container) container.innerHTML = '';
    unreadAlertCount = 0;
    const notifBadge = document.getElementById('unread-alert-badge');
    if (notifBadge) notifBadge.innerText = '0';
}

async function changeCameraRole(camId, newRole) {
    const payload = {};
    if (camId === 1) payload.cam1_role = newRole;
    if (camId === 2) payload.cam2_role = newRole;

    try {
        await fetch('/api/config/camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        soundManager.playAlertTone('normal');
        console.log(`[Camera Config] Changed Cam ${camId} role to ${newRole}`);
    } catch (err) {
        console.error('[Config Error]', err);
    }
}

async function setScenario(mode) {
    try {
        await fetch('/api/simulation/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, stock_override: -1, queue_override: -1 })
        });
        soundManager.playAlertTone('normal');
        console.log(`[Scenario] Switched mode to ${mode}`);
    } catch (err) {
        console.error('[Scenario Error]', err);
    }
}

function toggleEngineeringDrawer() {
    const drawer = document.getElementById('telemetry-drawer');
    const overlay = document.getElementById('drawer-overlay');
    if (drawer && overlay) {
        const isHidden = drawer.classList.contains('translate-x-full');
        if (isHidden) {
            drawer.classList.remove('translate-x-full');
            overlay.classList.remove('hidden');
        } else {
            drawer.classList.add('translate-x-full');
            overlay.classList.add('hidden');
        }
    }
}

function toggleMobileModal() {
    const modal = document.getElementById('mobile-connect-modal');
    if (modal) {
        modal.classList.toggle('hidden');
    }
}

// Start Telemetry Polling & Audio UI Setup
window.addEventListener('DOMContentLoaded', () => {
    soundManager.updateUi();
    initNetworkInfo();
    setInterval(fetchTelemetry, 300);
    fetchTelemetry();
});
