const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startButton = document.getElementById("startButton");
const captureButton = document.getElementById("captureButton");
const resetButton = document.getElementById("resetButton");
const statusEl = document.getElementById("status");
const treeCountEl = document.getElementById("treeCount");
const objectCountEl = document.getElementById("objectCount");

let model = null;
let stream = null;
let lastDetections = [];

const treeLabels = new Set(["tree", "potted plant", "plant"]);

async function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "var(--danger)" : "var(--muted)";
}

async function loadModel() {
  try {
    setStatus("Loading detection model…");
    model = await cocoSsd.load();
    setStatus("Model loaded. Click 'Start camera' to begin.");
    startButton.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("Failed to load model. Check console for details.", true);
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

async function capturePhoto() {
  if (!model) {
    setStatus("Model not loaded yet.", true);
    return;
  }

  if (video.readyState < 2) {
    setStatus("Video not ready. Please wait a moment.", true);
    return;
  }

  setStatus("Detecting objects…");

  try {
    const detections = await model.detect(video);
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
