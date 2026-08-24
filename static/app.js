/**
 * RetailSense OS // EdgeRetail AI - Client Controller
 * High-frequency telemetry sync, dual-stream controls, QR code generation, and telemetry drawer.
 */

let activeTelemetry = null;
let lastShelfStatus = "";
let lastQueueStatus = "";
let unreadAlertCount = 0;
let networkInfo = null;

// Initialize Lucide Icons
if (window.lucide) {
    lucide.createIcons();
}

// 1. Fetch Local Network Info & Generate QR Code
async function initNetworkInfo() {
    try {
        const res = await fetch('/api/network_info');
        if (!res.ok) throw new Error('Network info failed');
        networkInfo = await res.json();

        // Update mobile URL text
        const mobileUrlElem = document.getElementById('mobile-connect-url');
        if (mobileUrlElem) {
            mobileUrlElem.innerText = networkInfo.mobile_url;
            mobileUrlElem.href = networkInfo.mobile_url;
        }

        // Render QR Code for Mobile Ingest
        renderQRCode(networkInfo.mobile_url);
    } catch (err) {
        console.warn('[Network Info Error]', err);
    }
}

// Simple dynamic QR Code renderer using Google Chart API / QR Server
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

// Copy Mobile URL to Clipboard
function copyMobileUrl() {
    if (networkInfo && networkInfo.mobile_url) {
        navigator.clipboard.writeText(networkInfo.mobile_url);
        alert('Copied Mobile Camera URL to clipboard!\n\nOpen this link on your smartphone browser on the same Wi-Fi:\n' + networkInfo.mobile_url);
    }
}

// 2. High-Frequency Telemetry Synchronization (300ms)
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

    // 1. Header & Live Status
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

    // 5-Cell Visual Capacity Meter
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
    if (shelfPill) {
        if (shelf.is_occluded) {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-800/80';
            shelfPill.innerText = '🟡 Customer Browsing (Alerts Paused)';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-amber';
        } else if (shelf.status === 'OUT_OF_STOCK_ALERT' || shelf.alert_active) {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-red-950/80 text-red-300 border border-red-800/80 animate-pulse';
            shelfPill.innerText = '🔴 URGENT: Stock Depleted';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4 glow-red';
        } else if (shelf.status === 'PENDING_OOS') {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-yellow-950/80 text-yellow-300 border border-yellow-800/80';
            shelfPill.innerText = '⏳ Verifying Empty Shelf...';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4';
        } else {
            shelfPill.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-800/80';
            shelfPill.innerText = '🟢 Fully Stocked & Optimal';
            if (shelfCard) shelfCard.className = 'ops-kpi-card p-5 space-y-4';
        }
    }

    // 3. Checkout Queue Operations Card
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

    // 4. Update Qualcomm Engineering Drawer
    updateEngineeringDrawer(sys);

    // 5. Automated Staff Alert Feed
    if (shelf.status !== lastShelfStatus) {
        if (shelf.status === 'OUT_OF_STOCK_ALERT') {
            addStaffAlert('🔴 URGENT RESTOCK', 'Shelf Tier-1 is depleted (0 items). Restock beverages immediately.', 'Dispatch Staff', 'border-red-800 bg-red-950/40 text-red-200');
        } else if (shelf.status === 'CUSTOMER_INTERACTING') {
            addStaffAlert('🟡 CUSTOMER BROWSING', 'Customer interacting at Shelf ROI. Restock alarms suppressed by CSIM.', 'Monitoring', 'border-amber-800 bg-amber-950/30 text-amber-200');
        } else if (shelf.status === 'OPTIMAL' && lastShelfStatus === 'OUT_OF_STOCK_ALERT') {
            addStaffAlert('🟢 RESTOCK CONFIRMED', 'Inventory replenished to optimal capacity.', 'Resolved', 'border-emerald-800 bg-emerald-950/30 text-emerald-200');
        }
        lastShelfStatus = shelf.status;
    }

    if (queue.status !== lastQueueStatus) {
        if (queue.status === 'CONGESTION_WARNING') {
            addStaffAlert('🚨 CASHIER ALERT', `Queue length ≥${queue.customer_count} persons. Est. wait: ${queue.estimated_wait_min}m. Open Register 2.`, 'Open Counter 2', 'border-red-800 bg-red-950/40 text-red-200');
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

// 3. Dynamic Camera Role Swapping
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
        console.log(`[Camera Config] Changed Cam ${camId} role to ${newRole}`);
    } catch (err) {
        console.error('[Config Error]', err);
    }
}

// 4. One-Click Scenario Demonstrator
async function setScenario(mode) {
    try {
        await fetch('/api/simulation/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, stock_override: -1, queue_override: -1 })
        });
        console.log(`[Scenario] Switched mode to ${mode}`);
    } catch (err) {
        console.error('[Scenario Error]', err);
    }
}

// 5. Drawer & Modal Toggles
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

// Video Reconnect Handlers
function reloadVideoStreams() {
    const v1 = document.getElementById('video-stream-cam1');
    const v2 = document.getElementById('video-stream-cam2');
    const t = new Date().getTime();
    if (v1) v1.src = `/video_feed/cam1?t=${t}`;
    if (v2) v2.src = `/video_feed/cam2?t=${t}`;
}

// Start Telemetry Polling Loop
window.addEventListener('DOMContentLoaded', () => {
    initNetworkInfo();
    setInterval(fetchTelemetry, 300);
    fetchTelemetry();
});
