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

        const feedBtn = document.getElementById('feed-audio-toggle-btn');
        const feedIcon = document.getElementById('feed-audio-toggle-icon');
        const feedLabel = document.getElementById('feed-audio-toggle-label');

        if (label && icon) {
            if (this.muted) {
                label.innerText = 'Audio: Muted';
                icon.setAttribute('data-lucide', 'volume-x');
                icon.className = 'w-4 h-4 text-red-400';
                if (btn) btn.className = 'flex items-center gap-1.5 px-3 py-2 rounded-xl bg-red-950/40 hover:bg-red-900/60 border border-red-800/80 text-xs font-mono font-bold text-red-300 transition tactile-btn';
            } else {
                label.innerText = 'Audio: On';
                icon.setAttribute('data-lucide', 'volume-2');
                icon.className = 'w-4 h-4 text-emerald-400';
                if (btn) btn.className = 'flex items-center gap-1.5 px-3 py-2 rounded-xl bg-surfaceInner hover:bg-slate-800 border border-white/[0.08] text-xs font-mono font-bold text-slate-200 transition tactile-btn';
            }
        }

        if (feedLabel && feedIcon) {
            if (this.muted) {
                feedLabel.innerText = 'Unmute Sounds';
                feedIcon.setAttribute('data-lucide', 'volume-x');
                feedIcon.className = 'w-3.5 h-3.5 text-red-400';
                if (feedBtn) feedBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-lg bg-red-950/50 hover:bg-red-900 border border-red-800 text-[10px] font-mono font-bold text-red-300 transition';
            } else {
                feedLabel.innerText = 'Mute Sounds';
                feedIcon.setAttribute('data-lucide', 'volume-2');
                feedIcon.className = 'w-3.5 h-3.5 text-emerald-400';
                if (feedBtn) feedBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surfaceInner hover:bg-slate-800 border border-white/[0.08] text-[10px] font-mono font-bold text-slate-200 transition';
            }
        }

        if (window.lucide) lucide.createIcons();
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
        this.zonesConfig = { shelf_zones: [], queue_lanes: [] };
        
        // Determine role from active telemetry or default
        if (activeTelemetry && activeTelemetry.cameras) {
            const camKey = `cam${camId}`;
            if (activeTelemetry.cameras[camKey]) {
                this.activeZone = activeTelemetry.cameras[camKey].role || (camId === 1 ? "SHELF" : "QUEUE");
            }
        } else {
            this.activeZone = camId === 1 ? "SHELF" : "QUEUE";
        }

        // Fetch current Multi-Zone configs
        try {
            const res = await fetch('/api/config/zones');
            if (res.ok) {
                this.zonesConfig = await res.json();
            }
        } catch (e) {
            console.warn('[Zones Fetch]', e);
        }

        if (!this.zonesConfig.shelf_zones || this.zonesConfig.shelf_zones.length === 0) {
            this.zonesConfig.shelf_zones = [
                { id: 'shelf_tier_1', name: 'Tier 1 - Soft Drinks & Beverages', box: [0.10, 0.18, 0.90, 0.48], capacity: 6, low_stock_threshold: 2 },
                { id: 'shelf_tier_2', name: 'Tier 2 - Snacks & Packaged Goods', box: [0.10, 0.52, 0.90, 0.85], capacity: 8, low_stock_threshold: 2 }
            ];
        }

        if (!this.zonesConfig.queue_lanes || this.zonesConfig.queue_lanes.length === 0) {
            this.zonesConfig.queue_lanes = [
                { id: 'reg_1', name: 'Counter 1 (General)', box: [0.10, 0.20, 0.48, 0.85], max_wait_threshold_min: 3.0 },
                { id: 'reg_2', name: 'Counter 2 (Express)', box: [0.52, 0.20, 0.90, 0.85], max_wait_threshold_min: 3.0 }
            ];
        }

        const defaultZone = this.activeZone === "SHELF" 
            ? (this.zonesConfig.shelf_zones[0] ? this.zonesConfig.shelf_zones[0].id : 'shelf_tier_1')
            : (this.zonesConfig.queue_lanes[0] ? this.zonesConfig.queue_lanes[0].id : 'reg_1');

        this.selectZone(defaultZone);

        if (this.modal) this.modal.classList.remove('hidden');
    }

    renderZoneSelectorButtons() {
        const container = document.getElementById('calib-zone-selector');
        if (!container) return;

        let html = '';
        if (this.zonesConfig.shelf_zones) {
            this.zonesConfig.shelf_zones.forEach(z => {
                const isActive = z.id === this.activeZoneId;
                const cls = isActive 
                    ? 'px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/60 font-bold transition flex items-center gap-1.5'
                    : 'px-3 py-1.5 rounded-lg bg-white/[0.04] text-slate-400 border border-white/[0.06] font-bold transition flex items-center gap-1.5 hover:bg-white/[0.08]';
                html += `<button onclick="selectCalibZone('${z.id}')" id="btn-zone-${z.id}" class="${cls}">
                    <i data-lucide="package" class="w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-500'}"></i>
                    <span>${z.name}</span>
                </button>`;
            });
        }

        if (this.zonesConfig.queue_lanes) {
            this.zonesConfig.queue_lanes.forEach(l => {
                const isActive = l.id === this.activeZoneId;
                const cls = isActive 
                    ? 'px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/60 font-bold transition flex items-center gap-1.5'
                    : 'px-3 py-1.5 rounded-lg bg-white/[0.04] text-slate-400 border border-white/[0.06] font-bold transition flex items-center gap-1.5 hover:bg-white/[0.08]';
                html += `<button onclick="selectCalibZone('${l.id}')" id="btn-zone-${l.id}" class="${cls}">
                    <i data-lucide="users" class="w-3.5 h-3.5 ${isActive ? 'text-amber-400' : 'text-slate-500'}"></i>
                    <span>${l.name}</span>
                </button>`;
            });
        }

        container.innerHTML = html;
        if (window.lucide) lucide.createIcons();
    }

    selectZone(zoneId) {
        this.activeZoneId = zoneId;

        // Find target zone
        let target = null;
        let isShelf = false;
        if (this.zonesConfig.shelf_zones) {
            target = this.zonesConfig.shelf_zones.find(z => z.id === zoneId);
            if (target) isShelf = true;
        }
        if (!target && this.zonesConfig.queue_lanes) {
            target = this.zonesConfig.queue_lanes.find(l => l.id === zoneId);
            if (target) isShelf = false;
        }

        if (target) {
            if (target.box) {
                this.roi = { x1: target.box[0], y1: target.box[1], x2: target.box[2], y2: target.box[3] };
            }
            const tag = document.getElementById('calib-zone-tag');
            if (tag) tag.innerText = `// ${target.name.toUpperCase()}`;

            const nameInput = document.getElementById('calib-zone-name-input');
            if (nameInput) nameInput.value = target.name || '';

            const paramLabel = document.getElementById('calib-zone-param-label');
            const paramInput = document.getElementById('calib-zone-param-input');
            if (paramLabel && paramInput) {
                if (isShelf) {
                    paramLabel.innerText = 'Capacity:';
                    paramInput.value = target.capacity || 6;
                } else {
                    paramLabel.innerText = 'Max Wait (m):';
                    paramInput.value = target.max_wait_threshold_min || 3.0;
                }
            }
        }

        this.renderZoneSelectorButtons();
        this.render();
        this.updateLabels();
    }

    addNewQueueLane() {
        const nextIdx = (this.zonesConfig.queue_lanes ? this.zonesConfig.queue_lanes.length : 0) + 1;
        const newId = `reg_${Date.now()}`;
        const newLane = {
            id: newId,
            name: `Counter ${nextIdx} (Billing Lane)`,
            box: [0.15 + (nextIdx % 3) * 0.25, 0.20, 0.38 + (nextIdx % 3) * 0.25, 0.85],
            max_wait_threshold_min: 3.0
        };

        if (!this.zonesConfig.queue_lanes) this.zonesConfig.queue_lanes = [];
        this.zonesConfig.queue_lanes.push(newLane);
        this.selectZone(newId);
        soundManager.playAlertTone('normal');
    }

    addNewShelfTier() {
        const nextIdx = (this.zonesConfig.shelf_zones ? this.zonesConfig.shelf_zones.length : 0) + 1;
        const newId = `shelf_tier_${Date.now()}`;
        const newTier = {
            id: newId,
            name: `Tier ${nextIdx} - Product Section`,
            box: [0.10, 0.15 + (nextIdx % 4) * 0.20, 0.90, 0.32 + (nextIdx % 4) * 0.20],
            capacity: 6,
            low_stock_threshold: 2
        };

        if (!this.zonesConfig.shelf_zones) this.zonesConfig.shelf_zones = [];
        this.zonesConfig.shelf_zones.push(newTier);
        this.selectZone(newId);
        soundManager.playAlertTone('normal');
    }

    deleteActiveZone() {
        if (!this.activeZoneId) return;

        let deleted = false;
        if (this.zonesConfig.shelf_zones && this.zonesConfig.shelf_zones.some(z => z.id === this.activeZoneId)) {
            if (this.zonesConfig.shelf_zones.length <= 1) {
                alert('⚠️ Store must maintain at least 1 configured shelf tier.');
                return;
            }
            this.zonesConfig.shelf_zones = this.zonesConfig.shelf_zones.filter(z => z.id !== this.activeZoneId);
            deleted = true;
        } else if (this.zonesConfig.queue_lanes && this.zonesConfig.queue_lanes.some(l => l.id === this.activeZoneId)) {
            if (this.zonesConfig.queue_lanes.length <= 1) {
                alert('⚠️ Store must maintain at least 1 configured checkout register.');
                return;
            }
            this.zonesConfig.queue_lanes = this.zonesConfig.queue_lanes.filter(l => l.id !== this.activeZoneId);
            deleted = true;
        }

        if (deleted) {
            const nextZone = (this.zonesConfig.queue_lanes && this.zonesConfig.queue_lanes.length > 0)
                ? this.zonesConfig.queue_lanes[0].id
                : (this.zonesConfig.shelf_zones && this.zonesConfig.shelf_zones.length > 0 ? this.zonesConfig.shelf_zones[0].id : null);
            if (nextZone) {
                this.selectZone(nextZone);
            }
            soundManager.playAlertTone('normal');
        }
    }

    updateActiveZoneName(val) {
        if (!this.activeZoneId) return;
        let target = null;
        if (this.zonesConfig.shelf_zones) target = this.zonesConfig.shelf_zones.find(z => z.id === this.activeZoneId);
        if (!target && this.zonesConfig.queue_lanes) target = this.zonesConfig.queue_lanes.find(l => l.id === this.activeZoneId);
        if (target) {
            target.name = val;
            const tag = document.getElementById('calib-zone-tag');
            if (tag) tag.innerText = `// ${val.toUpperCase()}`;
            this.renderZoneSelectorButtons();
            this.render();
        }
    }

    updateActiveZoneParam(val) {
        if (!this.activeZoneId) return;
        let target = null;
        let isShelf = false;
        if (this.zonesConfig.shelf_zones) {
            target = this.zonesConfig.shelf_zones.find(z => z.id === this.activeZoneId);
            if (target) isShelf = true;
        }
        if (!target && this.zonesConfig.queue_lanes) {
            target = this.zonesConfig.queue_lanes.find(l => l.id === this.activeZoneId);
            if (target) isShelf = false;
        }

        if (target) {
            if (isShelf) {
                target.capacity = Math.max(1, parseInt(val) || 6);
            } else {
                target.max_wait_threshold_min = Math.max(0.5, parseFloat(val) || 3.0);
            }
        }
    }

    close() {
        if (this.modal) this.modal.classList.add('hidden');
    }

    applyPreset(presetName) {
        if (presetName === 'LEFT_HALF') {
            this.roi = { x1: 0.05, y1: 0.15, x2: 0.48, y2: 0.85 };
        } else if (presetName === 'RIGHT_HALF') {
            this.roi = { x1: 0.52, y1: 0.15, x2: 0.95, y2: 0.85 };
        } else if (presetName === 'CENTER') {
            this.roi = { x1: 0.25, y1: 0.20, x2: 0.75, y2: 0.80 };
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

        // Draw all inactive zones faintly in background
        const allZones = [...(this.zonesConfig.shelf_zones || []), ...(this.zonesConfig.queue_lanes || [])];
        allZones.forEach(z => {
            if (z.id !== this.activeZoneId && z.box) {
                const zx1 = z.box[0] * cw;
                const zy1 = z.box[1] * ch;
                const zw = (z.box[2] - z.box[0]) * cw;
                const zh = (z.box[3] - z.box[1]) * ch;
                const zcol = z.id.startsWith('shelf') ? 'rgba(6, 182, 212, 0.4)' : 'rgba(245, 158, 11, 0.4)';

                this.ctx.setLineDash([4, 4]);
                this.ctx.strokeStyle = zcol;
                this.ctx.lineWidth = 1.5;
                this.ctx.strokeRect(zx1, zy1, zw, zh);
                this.ctx.setLineDash([]);

                this.ctx.fillStyle = zcol;
                this.ctx.font = '10px "JetBrains Mono", monospace';
                this.ctx.fillText(z.name || z.id, zx1 + 4, zy1 + 12);
            }
        });

        // Dim area outside active selected ROI
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

        // Glowing Active ROI bounding box
        const zoneCol = this.activeZoneId.startsWith('shelf') ? '#06B6D4' : '#F59E0B';
        this.ctx.strokeStyle = zoneCol;
        this.ctx.lineWidth = 2.5;
        this.ctx.strokeRect(px1, py1, rw, rh);

        // Fill inner tint
        this.ctx.fillStyle = this.activeZoneId.startsWith('shelf') ? 'rgba(6, 182, 212, 0.12)' : 'rgba(245, 158, 11, 0.12)';
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
        const txt = `[SELECTED: ${this.activeZoneId.toUpperCase()}]`;
        this.ctx.fillText(txt, px1 + 10, py1 + 20);
    }

    async save() {
        try {
            // Reset drag state
            this.isDragging = false;
            this.dragTarget = null;

            // Update the box of active zone in zonesConfig
            const updatedBox = [this.roi.x1, this.roi.y1, this.roi.x2, this.roi.y2];
            if (this.zonesConfig.shelf_zones) {
                const z = this.zonesConfig.shelf_zones.find(z => z.id === this.activeZoneId);
                if (z) z.box = updatedBox;
            }
            if (this.zonesConfig.queue_lanes) {
                const l = this.zonesConfig.queue_lanes.find(l => l.id === this.activeZoneId);
                if (l) l.box = updatedBox;
            }

            const res = await fetch('/api/config/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.zonesConfig)
            });

            if (res.ok) {
                soundManager.playAlertTone('normal');
                this.close();
                showToastNotification('✅ Planogram & Queue zones saved & applied!', 'success');
            } else {
                showToastNotification('⚠️ Failed to save planogram calibration.', 'error');
            }
        } catch (e) {
            console.error('[Save ROI Error]', e);
            showToastNotification(`Network error saving planogram: ${e.message}`, 'error');
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

function selectCalibZone(zoneId) {
    roiCalibrator.selectZone(zoneId);
}

function addNewQueueLane() {
    roiCalibrator.addNewQueueLane();
}

function addNewShelfTier() {
    roiCalibrator.addNewShelfTier();
}

function deleteActiveZone() {
    roiCalibrator.deleteActiveZone();
}

function updateActiveZoneName(val) {
    roiCalibrator.updateActiveZoneName(val);
}

function updateActiveZoneParam(val) {
    roiCalibrator.updateActiveZoneParam(val);
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

    // 1. Header Live Pill & NPU Acceleration Pill
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

    const npuPill = document.getElementById('npu-status-pill');
    const npuText = document.getElementById('npu-status-text');
    if (npuPill && npuText) {
        npuPill.classList.remove('hidden');
        npuText.innerText = `${sys.inference_device || '⚡ NPU / DirectML Active'} (${sys.inference_latency_ms || 6.8}ms)`;
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

    // 2. Multi-Shelf Planogram Matrix Card
    const shelfSections = shelf.sections || data.shelf_sections || [
        { id: 'shelf_tier_1', name: 'Tier 1 - Soft Drinks & Beverages', current_count: 5, capacity: 6, status: 'OPTIMAL' },
        { id: 'shelf_tier_2', name: 'Tier 2 - Snacks & Packaged Goods', current_count: 6, capacity: 8, status: 'OPTIMAL' }
    ];

    const shelfContainer = document.getElementById('multi-shelf-sections-container');
    if (shelfContainer) {
        shelfContainer.innerHTML = shelfSections.map(sec => {
            const count = sec.current_count !== undefined ? sec.current_count : 5;
            const cap = sec.capacity || 6;
            const pct = Math.min(100, Math.round((count / Math.max(1, cap)) * 100));
            const status = sec.status || 'OPTIMAL';
            let statusBadge = '🟢 Optimal';
            let barCol = 'bg-emerald-400';
            if (status === 'CRITICAL_EMPTY' || count === 0) {
                statusBadge = '🔴 Critical Empty';
                barCol = 'bg-red-500';
            } else if (status === 'LOW_STOCK_WARNING') {
                statusBadge = '🟠 Low Stock (≤20%)';
                barCol = 'bg-amber-400';
            } else if (status === 'CUSTOMER_BROWSING' || sec.is_occluded) {
                statusBadge = '🟡 Shopper Browsing';
                barCol = 'bg-cyan-400';
            }

            return `
                <div class="p-2.5 rounded-xl bg-surfaceInner border border-white/[0.04] space-y-1.5">
                    <div class="flex items-center justify-between text-xs font-mono">
                        <span class="font-bold text-slate-200">${sec.name}</span>
                        <span class="text-[11px] font-bold ${status.includes('EMPTY') ? 'text-red-400' : status.includes('LOW') ? 'text-amber-400' : 'text-emerald-400'}">${statusBadge}</span>
                    </div>
                    <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <span>Stock: <strong class="text-white font-bold">${count}</strong> / ${cap} items</span>
                        <span>${pct}% Capacity</span>
                    </div>
                    <div class="w-full bg-white/[0.06] rounded-full h-1.5 overflow-hidden">
                        <div class="${barCol} h-full rounded-full transition-all duration-300" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }).join('');
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
            shelfPill.innerText = '🔴 CRITICAL: Out of Stock';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-red';
        } else if (shelf.status === 'LOW_STOCK_WARNING') {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-600/80 animate-pulse';
            shelfPill.innerText = '🟠 Stock Low (≤20%)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-amber';
        } else {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-800/80';
            shelfPill.innerText = '🟢 Inventory Healthy (Optimal)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4';
        }
    }

    // 3. Multi-Register Queue Card
    const registers = queue.registers || data.queue_registers || [
        { id: 'reg_1', name: 'Counter 1 (General)', headcount: 0, est_wait_min: 0.0, status: 'FREE_AVAILABLE' },
        { id: 'reg_2', name: 'Counter 2 (Express)', headcount: 0, est_wait_min: 0.0, status: 'FREE_AVAILABLE' }
    ];

    const queueContainer = document.getElementById('multi-queue-lanes-container');
    if (queueContainer) {
        queueContainer.innerHTML = registers.map(reg => {
            const cnt = reg.headcount !== undefined ? reg.headcount : 0;
            const wait = (reg.est_wait_min !== undefined ? reg.est_wait_min : 0.0).toFixed(1);
            const isCongested = reg.status === 'CONGESTED' || cnt >= 2;
            const borderCol = isCongested ? 'border-red-600/80 bg-red-950/30' : cnt === 1 ? 'border-amber-600/60 bg-amber-950/20' : 'border-white/[0.04] bg-surfaceInner';
            const badge = isCongested ? '🔴 Congested' : cnt === 1 ? '🟡 1 Shopper' : '🟢 Open & Free';

            return `
                <div class="p-3 rounded-xl ${borderCol} border space-y-2">
                    <div class="flex items-center justify-between text-xs font-mono">
                        <span class="font-bold text-slate-200 truncate">${reg.name}</span>
                    </div>
                    <div class="flex items-baseline justify-between font-mono">
                        <div>
                            <span class="text-2xl font-extrabold text-white tabular-nums">${cnt}</span>
                            <span class="text-[10px] text-slate-400 ml-1">queued</span>
                        </div>
                        <span class="text-xs font-bold text-cyan-300 tabular-nums">${wait}m wait</span>
                    </div>
                    <div class="text-[10px] font-mono font-bold ${isCongested ? 'text-red-300 animate-pulse' : 'text-emerald-300'}">
                        ${badge}
                    </div>
                </div>
            `;
        }).join('');
    }

    const queuePill = document.getElementById('queue-status-pill');
    const queueCard = document.getElementById('queue-kpi-card');
    const smartRecText = document.getElementById('smart-traffic-text');

    if (smartRecText) {
        smartRecText.innerText = data.smart_recommendation || queue.smart_recommendation || '🟢 All checkout registers are flowing smoothly.';
    }

    if (queuePill) {
        if (queue.congestion_alert || queue.status === 'CONGESTION_WARNING') {
            queuePill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-red-950/80 text-red-300 border border-red-800/80 animate-pulse';
            queuePill.innerText = '🔴 Congestion Warning';
            if (queueCard) queueCard.className = 'ops-kpi-card p-5 space-y-4 glow-red';
        } else {
            queuePill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-800/80';
            queuePill.innerText = '🟢 Flowing Smoothly';
            if (queueCard) queueCard.className = 'ops-kpi-card p-5 space-y-4';
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

// Sleek In-App Toast Notification
function showToastNotification(message, type = 'info') {
    let toast = document.getElementById('app-toast-banner');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'app-toast-banner';
        toast.className = 'fixed top-5 left-1/2 -translate-x-1/2 z-[9999] px-4 py-2.5 rounded-2xl shadow-2xl font-mono text-xs font-bold transition-all duration-300 flex items-center gap-2 border';
        document.body.appendChild(toast);
    }

    if (type === 'success') {
        toast.className = 'fixed top-5 left-1/2 -translate-x-1/2 z-[9999] px-4 py-2.5 rounded-2xl shadow-2xl font-mono text-xs font-bold transition-all duration-300 flex items-center gap-2 border bg-emerald-950/90 text-emerald-200 border-emerald-500/80 shadow-emerald-950/50 scale-100 opacity-100';
    } else if (type === 'error') {
        toast.className = 'fixed top-5 left-1/2 -translate-x-1/2 z-[9999] px-4 py-2.5 rounded-2xl shadow-2xl font-mono text-xs font-bold transition-all duration-300 flex items-center gap-2 border bg-rose-950/90 text-rose-200 border-rose-500/80 shadow-rose-950/50 scale-100 opacity-100';
    } else {
        toast.className = 'fixed top-5 left-1/2 -translate-x-1/2 z-[9999] px-4 py-2.5 rounded-2xl shadow-2xl font-mono text-xs font-bold transition-all duration-300 flex items-center gap-2 border bg-cyan-950/90 text-cyan-200 border-cyan-500/80 shadow-cyan-950/50 scale-100 opacity-100';
    }

    toast.innerText = message;

    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.className += ' opacity-0 -translate-y-4 pointer-events-none';
    }, 3200);
}

// Global Keyboard Handler (Escape closes any open modal)
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeCalibrationModal();
        closeShiftSummaryModal();
        const mobModal = document.getElementById('mobile-connect-modal');
        if (mobModal && !mobModal.classList.contains('hidden')) {
            toggleMobileModal();
        }
    }
});

// Start Telemetry Polling & Audio UI Setup
window.addEventListener('DOMContentLoaded', () => {
    soundManager.updateUi();
    initNetworkInfo();
    setInterval(fetchTelemetry, 300);
    fetchTelemetry();
});
