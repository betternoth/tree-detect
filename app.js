const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startButton = document.getElementById("startButton");
const captureButton = document.getElementById("captureButton");
const resetButton = document.getElementById("resetButton");
const statusEl = document.getElementById("status");
const treeCountEl = document.getElementById("treeCount");
const objectCountEl = document.getElementById("objectCount");

let session = null;
let stream = null;
let lastDetections = [];

// แก้ให้เหมาะกับ class ที่ trained ในโมเดลของคุณ
const classNames = ["tree"];
const treeLabels = new Set(["tree", "potted plant", "plant"]);

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "var(--danger)" : "var(--muted)";
}

async function loadModel() {
  try {
    setStatus("Loading YOLOv8 ONNX model…");
    session = await ort.InferenceSession.create("tree_detect_model/treedetect.onnx");
    startButton.disabled = false;
    setStatus("Model loaded. Click 'Start camera' to begin.");
  } catch (error) {
    console.error(error);
    setStatus("Failed to load YOLOv8 model. ตรวจสอบ tree_detect_model/treedetect.onnx", true);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera not supported in this browser.", true);
    return;
  }

  try {
    startButton.disabled = true;
    setStatus("Requesting camera access…");

    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = stream;

    await video.play();

    captureButton.disabled = false;
    resetButton.disabled = false;
    setStatus("Ready. Take a photo to detect trees.");
  } catch (error) {
    console.error(error);
    setStatus("Unable to access camera. Check permissions.", true);
    startButton.disabled = false;
  }
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
}

function drawBoxes(detections) {
  const ctx = overlay.getContext("2d");
  const width = (overlay.width = video.videoWidth);
  const height = (overlay.height = video.videoHeight);

  ctx.clearRect(0, 0, width, height);

  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    ctx.strokeStyle = det.tree ? "#38bdf8" : "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, w, h);

    ctx.fillStyle = det.tree ? "rgba(56, 189, 248, 0.9)" : "rgba(0, 0, 0, 0.65)";
    ctx.font = "18px system-ui, sans-serif";
    ctx.fillText(`${det.class} ${(det.score * 100).toFixed(0)}%`, x1 + 6, y1 + 22);
  });
}

function nms(boxes, scores, iouThreshold = 0.45) {
  const idxs = scores
    .map((s, i) => ({ s, i }))
    .sort((a, b) => b.s - a.s)
    .map((v) => v.i);

  const out = [];

  while (idxs.length) {
    const i = idxs.shift();
    out.push(i);

    for (let j = idxs.length - 1; j >= 0; j--) {
      const k = idxs[j];
      const [x1, y1, x2, y2] = boxes[i];
      const [x3, y3, x4, y4] = boxes[k];

      const xx1 = Math.max(x1, x3);
      const yy1 = Math.max(y1, y3);
      const xx2 = Math.min(x2, x4);
      const yy2 = Math.min(y2, y4);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter;
      const iou = union === 0 ? 0 : inter / union;

      if (iou > iouThreshold) idxs.splice(j, 1);
    }
  }

  return out;
}

function postprocessOutput(raw, imageWidth, imageHeight, confThreshold = 0.25) {
  const [batch, numDet, cols] = raw.dims;
  const data = raw.data;
  const detections = [];

  for (let i = 0; i < numDet; i++) {
    const base = i * cols;
    const x = data[base + 0];
    const y = data[base + 1];
    const w = data[base + 2];
    const h = data[base + 3];
    const objConfidence = data[base + 4];

    let bestClass = -1;
    let bestScore = 0;

    for (let c = 5; c < cols; c++) {
      const clsScore = data[base + c];
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestClass = c - 5;
      }
    }

    const score = objConfidence * bestScore;
    if (score < confThreshold) continue;

    // coordinates in model are relative to input size (0..1) or absolute pixels (onnx export may vary). adjust if necessary
    const x1 = Math.max(0, (x - w / 2) * imageWidth);
    const y1 = Math.max(0, (y - h / 2) * imageHeight);
    const x2 = Math.min(imageWidth, (x + w / 2) * imageWidth);
    const y2 = Math.min(imageHeight, (y + h / 2) * imageHeight);

    detections.push({
      bbox: [x1, y1, x2, y2],
      score,
      classId: bestClass,
      class: classNames[bestClass] ?? `class ${bestClass}`,
    });
  }

  if (detections.length === 0) return [];

  const boxes = detections.map((d) => d.bbox);
  const scores = detections.map((d) => d.score);
  const keep = nms(boxes, scores, 0.45);

  return keep.map((i) => detections[i]);
}

function countTrees(detections) {
  const treeDetections = detections.filter((det) => {
    const label = det.class?.toLowerCase() ?? "";
    const isTree = treeLabels.has(label) || label.includes("tree") || label.includes("plant");
    det.tree = isTree;
    return isTree;
  });

  treeCountEl.textContent = treeDetections.length.toString();
  objectCountEl.textContent = detections.length.toString();
  drawBoxes(detections);
}

async function capturePhoto() {
  if (!session) {
    setStatus("Model not loaded yet.", true);
    return;
  }

  return res.json();
}

function normalizeBbox(bbox) {
  // Many YOLO outputs are [x1, y1, x2, y2]. Convert to [x, y, w, h].
  if (!Array.isArray(bbox) || bbox.length < 4) return [0, 0, 0, 0];

  const [x0, y0, x1, y1] = bbox;

  // If values exceed the video dimensions, it’s likely already in [x,y,w,h].
  if (x1 > video.videoWidth || y1 > video.videoHeight) {
    return [x0, y0, x1, y1];
  }

  const width = x1 - x0;
  const height = y1 - y0;

  if (width <= 0 || height <= 0) {
    return [x0, y0, x1, y1];
  }

  return [x0, y0, width, height];
}

function parseYoloResponse(resp) {
  // This attempts to normalize common Ultralytics YOLO response shapes.
  const detections = [];

  const addPrediction = (pred) => {
    const label = pred.label ?? pred.name ?? pred.class ?? pred.cls;
    const score = pred.confidence ?? pred.score ?? pred.conf ?? 0;
    const bbox = normalizeBbox(pred.bbox ?? pred.box ?? pred.xyxy ?? pred.xywh ?? []);

    if (label) {
      detections.push({ class: label.toString(), score: Number(score) ?? 0, bbox });
    }
  };

  // Case A: response.predictions is a flat array
  if (Array.isArray(resp.predictions)) {
    resp.predictions.forEach(addPrediction);
    return detections;
  }

  // Case B: response.results is an array of objects
  if (Array.isArray(resp.results)) {
    resp.results.forEach((r) => {
      if (Array.isArray(r.predictions)) {
        r.predictions.forEach(addPrediction);
      }

      // Some responses use a `boxes` object with parallel arrays
      if (r.boxes && Array.isArray(r.boxes.xyxy)) {
        const xyxy = r.boxes.xyxy;
        const conf = r.boxes.conf || [];
        const cls = r.boxes.cls || [];

        for (let i = 0; i < xyxy.length; i++) {
          const label = (Array.isArray(cls) ? cls[i] : null) ?? "";
          const score = (Array.isArray(conf) ? conf[i] : null) ?? 0;
          detections.push({ class: label.toString(), score: Number(score) ?? 0, bbox: normalizeBbox(xyxy[i]) });
        }
      }
    });

    return detections;
  }

  // Fallback: unknown shape, just return empty
  return detections;
}

async function capturePhoto() {
  if (video.readyState < 2) {
    setStatus("Video not ready. Please wait a moment.", true);
    return;
  }

  setStatus("Capturing photo…");
  const imageBase64 = captureFrame();

  setStatus("Detecting objects (YOLOv8)…");

  try {
    // prepare frame
    const inputSize = 640;
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = inputSize;
    tmpCanvas.height = inputSize;
    const tmpCtx = tmpCanvas.getContext("2d");
    tmpCtx.drawImage(video, 0, 0, inputSize, inputSize);

    const imageData = tmpCtx.getImageData(0, 0, inputSize, inputSize);
    const data = imageData.data;
    const floatData = new Float32Array(3 * inputSize * inputSize);

    // normalize and reorder to CHW
    for (let y = 0; y < inputSize; y++) {
      for (let x = 0; x < inputSize; x++) {
        const i = (y * inputSize + x) * 4;
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;
        const idx = y * inputSize + x;
        floatData[idx] = r;
        floatData[inputSize * inputSize + idx] = g;
        floatData[2 * inputSize * inputSize + idx] = b;
      }
    }

    const tensor = new ort.Tensor("float32", floatData, [1, 3, inputSize, inputSize]);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const results = await session.run({ [inputName]: tensor });
    const rawOutput = results[outputName];

    const detections = postprocessOutput(rawOutput, video.videoWidth, video.videoHeight);
    lastDetections = detections;

    if (!detections.length) {
      setStatus("No objects detected.");
    } else {
      setStatus(`Detected ${detections.length} objects.`);
    }

    countTrees(detections);
  } catch (error) {
    console.error(error);
    setStatus("Detection failed. See console for details.", true);
  }
}

function reset() {
  stopCamera();
  captureButton.disabled = true;
  resetButton.disabled = true;
  startButton.disabled = false;
  lastDetections = [];
  treeCountEl.textContent = "0";
  objectCountEl.textContent = "0";
  setStatus("Ready. Click 'Start camera' to begin.");

  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

startButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", capturePhoto);
resetButton.addEventListener("click", reset);

setStatus("Ready. Click 'Start camera' to begin.");
startButton.disabled = false;
