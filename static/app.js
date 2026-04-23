/* ── InspectAI Frontend ─────────────────────────────────────────────────────── */

'use strict';

// ── Arm positions (SVG translate X for gantry carriage) ──────────────────────
const ARM_X = { PASS: 55, REWORK: 155, QUARANTINE: 260, SCRAP: 362, IDLE: 210 };
const COLORS = {
  PASS:       '#00e676',
  REWORK:     '#ffd600',
  QUARANTINE: '#ff6d00',
  SCRAP:      '#ff1744',
};
const COLORS_RGBA = {
  PASS:       'rgba(0,230,118,0.9)',
  REWORK:     'rgba(255,214,0,0.9)',
  QUARANTINE: 'rgba(255,109,0,0.9)',
  SCRAP:      'rgba(255,23,68,0.9)',
  ANALYZING:  'rgba(100,100,180,0.9)',
};

// ── State ─────────────────────────────────────────────────────────────────────
let statsData      = { total: 0, PASS: 0, REWORK: 0, QUARANTINE: 0, SCRAP: 0, pass_rate: 0 };
let currentVerdict = null;
let sending        = false;
let logCount       = 0;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const webcam        = $('webcam');
const overlayCanvas = $('overlay-canvas');
const overlayCtx    = overlayCanvas.getContext('2d');
const carriage      = $('arm-carriage');
const armState      = $('arm-state');
const verdictBadge  = $('verdict-badge');
const confBar       = $('conf-bar');
const confValue     = $('conf-value');
const metaObject    = $('meta-object');
const metaSeverity  = $('meta-severity');
const metaDefect    = $('meta-defect');
const metaTime      = $('meta-time');
const reasoningText = $('reasoning-text');
const objectTag     = $('object-tag');
const statusPill    = $('status-pill');
const statusText    = $('status-text');
const logTbody      = $('log-tbody');
const logCountEl    = $('log-count');
const pieCanvas     = $('pie-chart');
const pieCtx        = pieCanvas.getContext('2d');

// ── Webcam initialisation ─────────────────────────────────────────────────────
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    webcam.srcObject = stream;
    await new Promise(resolve => { webcam.onloadedmetadata = resolve; });
    webcam.play();
    setStatus('LIVE', true);
    startFrameLoop();
  } catch (err) {
    setStatus('CAM ERROR');
    reasoningText.textContent = `Camera access denied: ${err.message}. Allow camera in browser settings and refresh.`;
  }
}

// ── Frame capture & send loop ─────────────────────────────────────────────────
const _captureCanvas = document.createElement('canvas');
const _captureCtx    = _captureCanvas.getContext('2d');

async function sendFrame() {
  if (sending || !webcam.videoWidth) return;
  sending = true;

  try {
    _captureCanvas.width  = webcam.videoWidth;
    _captureCanvas.height = webcam.videoHeight;
    _captureCtx.drawImage(webcam, 0, 0);
    const dataURL = _captureCanvas.toDataURL('image/jpeg', 0.75);
    const base64  = dataURL.split(',')[1];

    const res  = await fetch('/process_frame', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: base64 }),
    });
    const data = await res.json();

    if (data.status !== 'ok') return;

    // Sync overlay canvas size to displayed video size
    overlayCanvas.width  = webcam.offsetWidth;
    overlayCanvas.height = webcam.offsetHeight;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Scale factors: video natural → display size
    const scaleX = overlayCanvas.width  / (data.frame_w || webcam.videoWidth);
    const scaleY = overlayCanvas.height / (data.frame_h || webcam.videoHeight);

    // Draw YOLO bounding boxes
    if (data.detections && data.detections.length) {
      const verdictKey = data.verdict?.verdict ?? 'ANALYZING';
      const boxColor   = COLORS_RGBA[verdictKey] ?? COLORS_RGBA.ANALYZING;

      data.detections.forEach(det => {
        const x  = det.x1 * scaleX;
        const y  = det.y1 * scaleY;
        const w  = (det.x2 - det.x1) * scaleX;
        const h  = (det.y2 - det.y1) * scaleY;
        const lbl = `${det.class_name}  ${Math.round(det.conf * 100)}%`;

        overlayCtx.strokeStyle = boxColor;
        overlayCtx.lineWidth   = 2;
        overlayCtx.strokeRect(x, y, w, h);

        overlayCtx.font         = '13px "Share Tech Mono", monospace';
        const tw = overlayCtx.measureText(lbl).width;
        overlayCtx.fillStyle    = boxColor;
        overlayCtx.fillRect(x, y - 22, tw + 10, 20);
        overlayCtx.fillStyle    = '#000';
        overlayCtx.fillText(lbl, x + 5, y - 7);
      });

      objectTag.textContent = data.detections[0].class_name +
        (data.verdict?.verdict ? `  ·  ${data.verdict.verdict}` : '  ·  ANALYZING');
    } else {
      objectTag.textContent = '— scanning —';
      moveArm('IDLE');
    }

    if (data.verdict?.verdict) {
      applyVerdict(data.verdict);
      setStatus('ANALYZING', true);
    } else if (!data.detections?.length) {
      setStatus('SCANNING');
    }
  } catch {
    setStatus('OFFLINE');
  } finally {
    sending = false;
  }
}

function startFrameLoop() {
  // Send a frame every 500 ms (~2 fps) — enough for quality inspection
  setInterval(sendFrame, 500);
}

// ── Robot arm movement ────────────────────────────────────────────────────────
function moveArm(verdict) {
  const x = ARM_X[verdict] ?? ARM_X.IDLE;
  carriage.style.transform = `translate(${x}px, 0)`;
  armState.textContent     = verdict === 'IDLE' ? 'IDLE' : `→ ${verdict}`;
  armState.className       = 'arm-state' + (verdict !== 'IDLE' ? ' moving' : '');

  ['PASS', 'REWORK', 'QUARANTINE', 'SCRAP'].forEach(v => {
    const el = $(`bin-${v}`);
    if (!el) return;
    el.className.baseVal = verdict === v ? `bin-active-${v}` : '';
  });

  if (verdict !== 'IDLE') {
    animateGripper('close');
    setTimeout(() => animateGripper('open'), 1200);
  }
}

function animateGripper(state) {
  const jawL = $('jaw-left');
  const jawR = $('jaw-right');
  if (!jawL || !jawR) return;
  jawL.style.transition = jawR.style.transition = 'transform 0.4s ease';
  if (state === 'close') {
    jawL.style.transform = 'translateX(8px)';
    jawR.style.transform = 'translateX(-8px)';
  } else {
    jawL.style.transform = '';
    jawR.style.transform = '';
  }
}

// ── Update analysis UI ────────────────────────────────────────────────────────
function applyVerdict(data) {
  const v = data.verdict;
  if (!v) return;
  currentVerdict = v;

  verdictBadge.textContent = v;
  verdictBadge.className   = `verdict-badge flash ${v}`;

  const pct = Math.round((data.confidence ?? 0) * 100);
  confBar.style.width      = `${pct}%`;
  confBar.style.background = `linear-gradient(90deg, ${COLORS[v]}44, ${COLORS[v]})`;
  confValue.textContent    = `${pct}%`;

  metaObject.textContent   = data.class_name ?? '—';
  const sev = (data.severity ?? '').toLowerCase();
  metaSeverity.textContent = data.severity ?? '—';
  metaSeverity.className   = `meta-val severity ${sev}`;
  metaDefect.textContent   = data.defect_type ?? '—';
  metaTime.textContent     = data.timestamp ?? '—';

  reasoningText.style.opacity = '0';
  setTimeout(() => {
    reasoningText.textContent  = data.reasoning ?? '—';
    reasoningText.style.opacity = '1';
    reasoningText.style.transition = 'opacity 0.4s';
  }, 150);

  moveArm(v);
}

// ── Status pill ───────────────────────────────────────────────────────────────
function setStatus(label, active = false) {
  statusText.textContent = label;
  statusPill.className   = `status-pill${active ? ' active' : ''}`;
}

// ── Pie chart ─────────────────────────────────────────────────────────────────
function drawPie() {
  const s = statsData;
  const total = s.PASS + s.REWORK + s.QUARANTINE + s.SCRAP || 1;
  const slices = [
    { val: s.PASS,       color: COLORS.PASS },
    { val: s.REWORK,     color: COLORS.REWORK },
    { val: s.QUARANTINE, color: COLORS.QUARANTINE },
    { val: s.SCRAP,      color: COLORS.SCRAP },
  ];

  const cx = 55, cy = 55, r = 48;
  pieCtx.clearRect(0, 0, 110, 110);

  let start = -Math.PI / 2;
  slices.forEach(({ val, color }) => {
    if (!val) return;
    const sweep = (val / total) * Math.PI * 2;
    pieCtx.beginPath();
    pieCtx.moveTo(cx, cy);
    pieCtx.arc(cx, cy, r, start, start + sweep);
    pieCtx.closePath();
    pieCtx.fillStyle = color;
    pieCtx.fill();
    start += sweep;
  });

  pieCtx.beginPath();
  pieCtx.arc(cx, cy, r * 0.52, 0, Math.PI * 2);
  pieCtx.fillStyle = '#0d1c2e';
  pieCtx.fill();

  pieCtx.fillStyle    = '#c8ddf0';
  pieCtx.font         = 'bold 18px Orbitron, sans-serif';
  pieCtx.textAlign    = 'center';
  pieCtx.textBaseline = 'middle';
  pieCtx.fillText(s.total, cx, cy);
}

// ── Stats & log polling ───────────────────────────────────────────────────────
function updateStats() {
  $('stat-total').textContent      = statsData.total;
  $('stat-pass-rate').textContent  = `${statsData.pass_rate}%`;
  $('stat-rework').textContent     = statsData.REWORK;
  $('stat-quarantine').textContent = statsData.QUARANTINE;
  $('stat-scrap').textContent      = statsData.SCRAP;
  drawPie();
}

async function pollStats() {
  try {
    const res = await fetch('/stats');
    statsData = await res.json();
    updateStats();
  } catch { /* ignore */ }
}

async function pollLog() {
  try {
    const res     = await fetch('/log');
    const entries = await res.json();
    if (!entries.length) return;
    logCount = entries.length;
    logCountEl.textContent = `${logCount} record${logCount !== 1 ? 's' : ''}`;
    const sevClass = { low: 'sev-low', medium: 'sev-med', high: 'sev-high' };
    logTbody.innerHTML = entries.map(e => `
      <tr>
        <td class="mono">${e.time}</td>
        <td>${e.object}</td>
        <td><span class="verdict-chip ${e.verdict}">${e.verdict}</span></td>
        <td class="${sevClass[e.severity] ?? ''}">${e.severity}</td>
        <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis">${e.defect}</td>
      </tr>
    `).join('');
  } catch { /* ignore */ }
}

// ── Initialise ────────────────────────────────────────────────────────────────
(function init() {
  carriage.style.transform = `translate(${ARM_X.IDLE}px, 0)`;
  setStatus('INITIALIZING');
  drawPie();

  startWebcam();

  setInterval(pollStats, 2000);
  setInterval(pollLog,   3000);
})();
