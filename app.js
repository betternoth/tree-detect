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

async function setStatus(text, isError = false) {
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
  if (!raw || !raw.dims || !raw.data) return [];
  console.log("ONNX raw output dims", raw.dims);

  const detections = [];

  if (raw.dims.length === 3 && raw.dims[2] >= 6) {
    const [batch, numDet, cols] = raw.dims;
    const data = raw.data;

    for (let i = 0; i < numDet; i++) {
      const base = i * cols;
      const x = data[base + 0];
      const y = data[base + 1];
      const w = data[base + 2];
      const h = data[base + 3];
      const score = data[base + 4];

      if (cols === 6) {
        // format: [x1, y1, x2, y2, score, class]
        if (score < confThreshold) continue;

        const classId = Math.round(data[base + 5]);
        const x1 = Math.max(0, x);
        const y1 = Math.max(0, y);
        const x2 = Math.min(imageWidth, w);
        const y2 = Math.min(imageHeight, h);

        detections.push({
          bbox: [x1, y1, x2, y2],
          score,
          classId,
          class: classNames[classId] ?? `class ${classId}`,
        });

        continue;
      }

      // format: [x_center, y_center, width, height, obj_conf, class0, class1, ...]
      let bestClass = -1;
      let bestScore = 0;

      for (let c = 5; c < cols; c++) {
        const clsScore = data[base + c];
        if (clsScore > bestScore) {
          bestScore = clsScore;
          bestClass = c - 5;
        }
      }

      const confidence = score * bestScore;
      if (confidence < confThreshold) continue;

      const isRelative = x <= 1 && y <= 1 && w <= 1 && h <= 1;
      let x1, y1, x2, y2;

      if (isRelative) {
        x1 = Math.max(0, (x - w / 2) * imageWidth);
        y1 = Math.max(0, (y - h / 2) * imageHeight);
        x2 = Math.min(imageWidth, (x + w / 2) * imageWidth);
        y2 = Math.min(imageHeight, (y + h / 2) * imageHeight);
      } else {
        x1 = Math.max(0, x - w / 2);
        y1 = Math.max(0, y - h / 2);
        x2 = Math.min(imageWidth, x + w / 2);
        y2 = Math.min(imageHeight, y + h / 2);
      }

      detections.push({
        bbox: [x1, y1, x2, y2],
        score: confidence,
        classId: bestClass,
        class: classNames[bestClass] ?? `class ${bestClass}`,
      });
    }

    if (raw.dims[1] > 100) {
      const boxes = detections.map((d) => d.bbox);
      const scores = detections.map((d) => d.score);
      const keep = nms(boxes, scores, 0.45);
      return keep.map((i) => detections[i]);
    }

    return detections;
  }

  if (raw.dims.length === 2 && raw.dims[1] >= 6) {
    const [numDet, cols] = raw.dims;
    const data = raw.data;

    for (let i = 0; i < numDet; i++) {
      const base = i * cols;
      const x1 = Math.max(0, data[base + 0]);
      const y1 = Math.max(0, data[base + 1]);
      const x2 = Math.min(imageWidth, data[base + 2]);
      const y2 = Math.min(imageHeight, data[base + 3]);
      const score = data[base + 4];
      const classId = Math.round(data[base + 5]);
      if (score < confThreshold) continue;

      detections.push({
        bbox: [x1, y1, x2, y2],
        score,
        classId,
        class: classNames[classId] ?? `class ${classId}`,
      });
    }

    return detections;
  }

  console.warn("Unknown ONNX output format", raw.dims);
  return [];
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

  if (video.readyState < 2) {
    setStatus("Video not ready. Please wait a moment.", true);
    return;
  }

  setStatus("Detecting objects…");

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

loadModel();
