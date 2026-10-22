function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(260, Math.round(rect.width || 340));
  const height = Math.max(150, Math.round(rect.height || 170));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

export function drawHistory(canvas, history) {
  const { context, width, height } = setupCanvas(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fffdfd";
  context.fillRect(0, 0, width, height);
  const points = (history?.points || []).filter((point) => Number.isFinite(point.value));
  if (!points.length) {
    context.fillStyle = "#715f67";
    context.font = "12px Segoe UI";
    context.textAlign = "center";
    context.fillText("No valid observations for this measure", width / 2, height / 2);
    return;
  }

  const padding = { left: 42, right: 12, top: 14, bottom: 27 };
  const values = points.map((point) => point.value);
  const baselines = points.map((point) => point.baseline).filter(Number.isFinite);
  const minimum = Math.min(...values, ...baselines);
  const maximum = Math.max(...values, ...baselines);
  const range = Math.max(1e-9, maximum - minimum);
  const x = (index) => padding.left + index * (width - padding.left - padding.right) / Math.max(1, points.length - 1);
  const y = (value) => padding.top + (maximum - value) / range * (height - padding.top - padding.bottom);

  context.strokeStyle = "rgba(112,53,65,.12)";
  context.lineWidth = 1;
  for (let index = 0; index <= 3; index += 1) {
    const gridY = padding.top + index * (height - padding.top - padding.bottom) / 3;
    context.beginPath();
    context.moveTo(padding.left, gridY);
    context.lineTo(width - padding.right, gridY);
    context.stroke();
  }

  const baseline = baselines[0];
  if (Number.isFinite(baseline)) {
    context.save();
    context.setLineDash([5, 4]);
    context.strokeStyle = "#267a74";
    context.beginPath();
    context.moveTo(padding.left, y(baseline));
    context.lineTo(width - padding.right, y(baseline));
    context.stroke();
    context.restore();
  }

  context.strokeStyle = "#a7354d";
  context.lineWidth = 2.2;
  context.lineJoin = "round";
  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0 || point.gap_before) context.moveTo(x(index), y(point.value));
    else context.lineTo(x(index), y(point.value));
  });
  context.stroke();

  context.fillStyle = "#715f67";
  context.font = "10px Segoe UI";
  context.textAlign = "right";
  context.fillText(maximum.toFixed(1), padding.left - 7, padding.top + 3);
  context.fillText(minimum.toFixed(1), padding.left - 7, height - padding.bottom + 3);
  context.textAlign = "left";
  context.fillText(points[0].label.slice(0, 16), padding.left, height - 8);
  context.textAlign = "right";
  context.fillText(points.at(-1).label.slice(0, 16), width - padding.right, height - 8);
  canvas.setAttribute("aria-label", `${history.metric} history for ${history.region_id} ${history.entity_id}; ${points.length} valid points`);
}
