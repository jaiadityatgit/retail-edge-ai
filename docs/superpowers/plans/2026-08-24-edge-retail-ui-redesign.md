# EdgeRetail AI // Apple-Grade Design & Taste Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Edge AI Retail platform (`static/index.html` and `static/presentation.html`) into an Apple-grade, Linear-tier Command Center using concentric double-bezels, optical alignment, balanced typography, tactile island physics, and high-density telemetry.

**Architecture:** Single-page application built on a Deep Obsidian matte surface (`#07090E`), employing concentric squircle geometry (`outer radius = inner radius + padding`), Plus Jakarta Sans & JetBrains Mono with tabular numerals, custom SVG circular meters, and sub-300ms telemetry synchronization via Vanilla JS.

**Tech Stack:** HTML5, Tailwind CSS, Plus Jakarta Sans & JetBrains Mono (Google Fonts), Lucide Icons (light stroke), Chart.js, Vanilla ES6+ JS, FastAPI.

## Global Constraints & Design Rules

- **Zero AI Slop:** No neon rainbow gradients, no harsh pure-black boxes, no generic 1px gray borders, no heavy dark drop shadows.
- **Typography:** `Plus Jakarta Sans` for titles and UI labels (`text-wrap: balance` on headers, `text-wrap: pretty` on descriptions), and `JetBrains Mono` with `font-variant-numeric: tabular-nums` for all telemetry.
- **Concentric Radius:** All nested cards must obey `outer_radius = inner_radius + padding` (e.g., `rounded-[1.75rem]` outer with `p-1.5` -> `rounded-[calc(1.75rem-0.375rem)]` inner).
- **Tactile Island Buttons:** Button-in-button nested icons with explicit property transitions (`transition-property: transform, background-color, border-color, box-shadow`) and `active:scale-[0.97]`. No `transition: all`.
- **Optical Centering & Hit Areas:** Interactive controls maintain minimum 40x40px (target 44x44px) touch targets with optically centered icons.
- **Cache Control:** Strict `Cache-Control: no-cache, no-store, must-revalidate` headers in `app.py` and HTML meta tags.
- **Verification Gate:** `verify_server.py` must pass 100% green before committing.

---

### Task 1: Design System Foundations, CSS Custom Properties & Noise Canvas

**Files:**
- Modify: `static/index.html:1-120`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: Google Fonts API, Tailwind CDN, Lucide CDN.
- Produces: CSS custom properties (`--ease-spring`, `--ease-apple`), concentric double-bezel utilities, font smoothing (`-webkit-font-smoothing: antialiased`), and subtle film-grain background overlay.

- [ ] **Step 1: Setup Apple-grade Typography, Palette and Anti-Aliasing**
- [ ] **Step 2: Configure Concentric Double-Bezel and Glow Utility Rules**
- [ ] **Step 3: Run verify_server.py to confirm server integrity**
- [ ] **Step 4: Commit design system foundations**

---

### Task 2: Floating Detached Island Header with Concentric Hardware Chips

**Files:**
- Modify: `static/index.html:120-220`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `GET /metrics` (`device.soc_temp_c`, `device.edge_fps`, `compliance.dpdp_compliant`).
- Produces: Floating glass pill top bar, pulsing emerald live beacon, tabular-num SoC/FPS chips, Inspector modal trigger, and Keynote pitch deck button.

- [ ] **Step 1: Build Floating Detached Island Top Bar**
- [ ] **Step 2: Add Real-Time Hardware Chips & Button-in-Button Pitch Deck Link**
- [ ] **Step 3: Run verify_server.py**
- [ ] **Step 4: Commit header component**

---

### Task 3: Cinematic 16:9 Vision Stage with Spatial ROI HUD & Overlay Controls

**Files:**
- Modify: `static/index.html:220-380`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `GET /video_feed` (MJPEG stream).
- Produces: 16:9 video frame with chamfered concentric double-bezel, live spatial zone tags (Shelf 0-40%, Transit 40-60%, Queue 60-100%), floating resolution/latency HUD tag, and stream refresh toolbar.

- [ ] **Step 1: Architect 16:9 Cinematic Video Viewport with Concentric Borders**
- [ ] **Step 2: Implement Live Viewport Floating HUD Tag & Stream Controls**
- [ ] **Step 3: Run verify_server.py**
- [ ] **Step 4: Commit video stage component**

---

### Task 4: Live Spatial Tracking Diagnostics Matrix & Entity Table

**Files:**
- Modify: `static/index.html:380-480`
- Modify: `app.py:500-580`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `GET /metrics` (`tracks: [{track_id, class_name, zone, dwell_time_sec, velocity_mps, occluding_shelf}]`).
- Produces: Live tracking data table with active track count badge, zone pills, dwell timers, speed kinematics, and CSIM state indicators.

- [ ] **Step 1: Build Diagnostic Table DOM in Left Bento Column**
- [ ] **Step 2: Implement Javascript Dynamic Table Synchronization**
- [ ] **Step 3: Run verify_server.py**
- [ ] **Step 4: Commit tracking matrix component**

---

### Task 5: Interactive Simulation Studio with Tactile Physical Switch Cards

**Files:**
- Modify: `static/index.html:480-590`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `POST /api/simulation/control` (`{mode, stock_override, queue_override}`).
- Produces: 4 tactile scenario cards (*Auto Cycle*, *Customer Occlusion*, *Stock Depletion*, *Queue Congestion*) with button-in-button nested icons and fine-tuning sliders.

- [ ] **Step 1: Implement Tactile Scenario Switcher Cards**
- [ ] **Step 2: Add Fine Sliders for Physical Stock & Queue Count**
- [ ] **Step 3: Run verify_server.py**
- [ ] **Step 4: Commit simulation studio component**

---

### Task 6: High-Density KPI Cards with 5-Cell Capacity Meters & RPi 5 Diagnostics

**Files:**
- Modify: `static/index.html:590-760`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `GET /metrics` (`shelf`, `queue`, `device`).
- Produces:
  1. Shelf Inventory KPI Card with 5 illuminated stock cell pills and glowing status badge.
  2. Checkout Queue KPI Card with customer count, wait-time calculation, and flashing Register 2 dispatch banner.
  3. Raspberry Pi 5 Hardware Telemetry Card with gradient progress bars for SoC Temp, CPU load, and LPDDR4X DMA RAM.
  4. Real-time Chart.js Latency & FPS curve.

- [ ] **Step 1: Build Shelf Inventory Telemetry Card**
- [ ] **Step 2: Build Checkout Queue Intelligence Card**
- [ ] **Step 3: Build RPi 5 Hardware & DPDP Compliance Card**
- [ ] **Step 4: Build Real-Time Performance Line Graph (Chart.js)**
- [ ] **Step 5: Run verify_server.py**
- [ ] **Step 6: Commit KPI cards and telemetry graphs**

---

### Task 7: Raw JSON Telemetry Inspector Modal & Keynote Slide Deck Polish

**Files:**
- Modify: `static/index.html:760-920`
- Modify: `static/presentation.html`
- Test: `verify_server.py`

**Interfaces:**
- Consumes: `GET /metrics`.
- Produces: Modal dialog with macOS-style window controls, syntax-highlighted JSON viewer, copy-to-clipboard functionality, and upgraded Keynote slide deck.

- [ ] **Step 1: Implement Inspector Modal DOM & Styling**
- [ ] **Step 2: Add Javascript Handlers for Inspector**
- [ ] **Step 3: Elevate presentation.html to Match Apple Design Standards**
- [ ] **Step 4: Run verify_server.py**
- [ ] **Step 5: Commit inspector modal and slide deck updates**
