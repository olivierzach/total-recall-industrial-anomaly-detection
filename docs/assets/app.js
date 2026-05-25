const queryCanvas = document.querySelector("#query");
const heatCanvas = document.querySelector("#heatmap");
const queryCtx = queryCanvas.getContext("2d");
const heatCtx = heatCanvas.getContext("2d");

const intensity = document.querySelector("#intensity");
const threshold = document.querySelector("#threshold");
const match = document.querySelector("#match");
const stateLabel = document.querySelector("#state-label");
const stateDetail = document.querySelector("#state-detail");
const queryScore = document.querySelector("#query-score");
const supportScore = document.querySelector("#support-score");
const supports = document.querySelector("#supports");

function drawPart(ctx, heat = false) {
  ctx.clearRect(0, 0, 520, 360);
  const grd = ctx.createLinearGradient(0, 0, 520, 360);
  grd.addColorStop(0, "#121827");
  grd.addColorStop(1, "#07101b");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 520, 360);

  ctx.save();
  ctx.translate(260, 180);
  ctx.fillStyle = "#1b2638";
  ctx.strokeStyle = "rgba(231,231,234,0.28)";
  ctx.lineWidth = 2;
  roundedRect(ctx, -150, -96, 300, 192, 26);
  ctx.fill();
  ctx.stroke();

  for (let i = -120; i <= 120; i += 30) {
    ctx.strokeStyle = "rgba(125,211,252,0.18)";
    ctx.beginPath();
    ctx.moveTo(i, -78);
    ctx.lineTo(i + 30, 78);
    ctx.stroke();
  }

  const strength = Number(intensity.value) / 100;
  const spots = [
    [64, -22, 34, 0.95],
    [88, 36, 21, 0.62],
    [-72, 48, 18, 0.45],
  ];
  for (const [x, y, r, a] of spots) {
    const g = ctx.createRadialGradient(x, y, 2, x, y, r * (1 + strength));
    g.addColorStop(0, heat ? `rgba(244,114,182,${a * strength})` : `rgba(245,158,11,${0.34 * strength})`);
    g.addColorStop(0.52, heat ? `rgba(244,114,182,${0.42 * strength})` : `rgba(125,211,252,${0.12 * strength})`);
    g.addColorStop(1, "rgba(244,114,182,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r * 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function renderSupports(score) {
  const items = [
    ["scratch / edge wear", 0.91],
    ["surface inclusion", 0.84],
    ["novel anomaly", 0.43],
  ];
  supports.innerHTML = items
    .map(([name, base], i) => {
      const v = Math.min(0.99, Math.max(0.05, base * score + (i === 2 ? 0.18 : 0)));
      return `<div class="support-card">
        <strong>${name}</strong>
        <span>support distance ${(1 - v).toFixed(2)}</span>
        <span>margin ${(1 + v * 0.44).toFixed(2)}</span>
        <div class="spark"></div>
      </div>`;
    })
    .join("");
}

function update() {
  const anomaly = Number(intensity.value) / 100;
  const gate = Number(threshold.value) / 100;
  const matchScore = Number(match.value) / 100;
  const nominalDistance = 0.28 + anomaly * 0.82;
  const margin = 1 + matchScore * 0.44;
  const accepted = nominalDistance > gate && margin > 1.2;

  stateLabel.textContent = nominalDistance <= gate ? "normal" : accepted ? "known failure" : "unknown anomaly";
  stateLabel.style.color = nominalDistance <= gate ? "var(--cyan)" : accepted ? "var(--mint)" : "var(--amber)";
  stateDetail.textContent = `nominal distance ${nominalDistance.toFixed(2)} / failure margin ${margin.toFixed(2)}`;
  queryScore.textContent = nominalDistance.toFixed(2);
  supportScore.textContent = accepted ? "3 hits" : "reject path";
  drawPart(queryCtx, false);
  drawPart(heatCtx, true);
  renderSupports(matchScore);
}

[intensity, threshold, match].forEach((el) => el.addEventListener("input", update));
update();
