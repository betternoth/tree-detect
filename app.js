const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startButton = document.getElementById("startButton");
const captureButton = document.getElementById("captureButton");
const resetButton = document.getElementById("resetButton");
const statusEl = document.getElementById("status");
const treeCountEl = document.getElementById("treeCount");
const objectCountEl = document.getElementById("objectCount");

// Your Ultralytics API key (keep it private in a real app)
const YOLO_API_KEY = "ul_13eca71e947c220d049eada5f7fab1940fd1e543";

let stream = null;
let lastDetections = [];

const treeLabels = new Set(["tree", "potted plant", "plant"]);

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "var(--danger)" : "var(--muted)";
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
  const width = overlay.width = video.videoWidth;
  const height = overlay.height = video.videoHeight;

  ctx.clearRect(0, 0, width, height);

  detections.forEach((det) => {
    const [x, y, w, h] = det.bbox;
    ctx.strokeStyle = det.tree ? "#38bdf8" : "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = det.tree ? "rgba(56, 189, 248, 0.9)" : "rgba(0, 0, 0, 0.65)";
    ctx.font = "18px system-ui, sans-serif";
    ctx.fillText(`${det.class} (${(det.score * 100).toFixed(0)}%)`, x + 6, y + 22);
  });
}

function countTrees(detections) {
  const treeDetections = detections.filter((det) => {
    const label = det.class?.toLowerCase?.() ?? "";
    const isTree = treeLabels.has(label) || label.includes("tree") || label.includes("plant");
    det.tree = isTree;
    return isTree;
  });

  treeCountEl.textContent = treeDetections.length.toString();
  objectCountEl.textContent = detections.length.toString();
  drawBoxes(detections);
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg");
}

async function inferWithYOLO(imageBase64) {
  const res = await fetch("https://api.ultralytics.com/v1/infer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${YOLO_API_KEY}`,
    },
    body: JSON.stringify({
      model: "yolov8n",
      input: imageBase64,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`YOLO inference failed (${res.status}): ${text}`);
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
    const response = await inferWithYOLO(imageBase64);
    console.log("YOLO response:", response);

    const detections = parseYoloResponse(response);
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
