// FaST-ISL demo client: webcam -> JPEG frames over WebSocket -> landmarks + captions back.
const $ = (id) => document.getElementById(id);
const FPS = 10;
let ws = null, stream = null, timer = null, waiting = false, modelLoaded = false;
let sent = 0, fpsT0 = performance.now(), sentences = 0;

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
}));

fetch("/api/status").then((r) => r.json()).then((s) => {
  modelLoaded = s.model_loaded;
  $("banner").hidden = modelLoaded;
});

function setPill(state, label) {
  const p = $("pill");
  p.className = "pill " + state;
  p.textContent = label;
}

function speak(text) {
  if (!$("speak").checked || !text || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  const v = speechSynthesis.getVoices().find((v) => v.lang === "en-IN") ||
            speechSynthesis.getVoices().find((v) => v.lang.startsWith("en"));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

function addLine(list, text, meta) {
  const hint = list.querySelector(".hint");
  if (hint) hint.remove();
  const li = document.createElement("li");
  li.textContent = text;
  const s = document.createElement("small");
  s.textContent = meta;
  li.appendChild(s);
  list.appendChild(li);
  list.scrollTop = list.scrollHeight;
}

const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
const HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

function draw(lm) {
  const c = $("overlay"), v = $("cam");
  c.width = v.videoWidth || 640;
  c.height = v.videoHeight || 480;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const P = (p) => [p[0] * c.width, p[1] * c.height];
  const lines = (pts, pairs, color) => {
    g.strokeStyle = color; g.lineWidth = 3;
    for (const [a, b] of pairs) {
      if (!pts[a] || !pts[b]) continue;
      const [x1, y1] = P(pts[a]), [x2, y2] = P(pts[b]);
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    }
  };
  if (lm.pose) lines(lm.pose, BONES, "rgba(255,255,255,0.7)");
  if (lm.left_hand) lines(lm.left_hand, HAND, "#2dd4bf");
  if (lm.right_hand) lines(lm.right_hand, HAND, "#fb923c");
  if (lm.face) {
    g.fillStyle = "rgba(250,204,21,0.9)";
    for (const p of lm.face) { const [x, y] = P(p); g.fillRect(x - 1, y - 1, 2, 2); }
  }
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (e) {
    $("stageEmpty").querySelector("p").textContent = "Camera permission was refused: " + e.message;
    return;
  }
  $("cam").srcObject = stream;
  $("stageEmpty").hidden = true;
  $("startBtn").disabled = true;
  $("stopBtn").disabled = false;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "frame") {
      waiting = false;
      draw(m.landmarks);
      $("energy").style.width = Math.min(100, m.energy * 800) + "%";
      if (!$("pill").classList.contains("translating")) setPill(m.signing ? "signing" : "idle", m.signing ? "Signing…" : "Idle");
    } else if (m.type === "translating") {
      setPill("translating", "Translating…");
    } else if (m.type === "caption") {
      sentences += 1;
      $("count").textContent = sentences;
      const text = m.trained ? (m.text || "(no output)") : `Sentence ${sentences} detected`;
      addLine($("transcript"), text, `${m.duration_s}s of signing` + (m.trained ? ` · ${m.latency_ms} ms` : ""));
      if (m.trained) { $("lat").textContent = m.latency_ms; speak(m.text); }
      setPill("idle", "Idle");
    }
  };
  ws.onclose = () => stop();
  ws.onopen = () => { timer = setInterval(sendFrame, 1000 / FPS); };
}

const grab = document.createElement("canvas");
function sendFrame() {
  if (!ws || ws.readyState !== 1 || waiting) return;   // wait for the server: no frame pile-up
  const v = $("cam");
  if (!v.videoWidth) return;
  grab.width = 480;
  grab.height = Math.round(480 * v.videoHeight / v.videoWidth);
  grab.getContext("2d").drawImage(v, 0, 0, grab.width, grab.height);  // NOT mirrored: model expects raw view
  grab.toBlob(async (blob) => {
    if (!blob || !ws || ws.readyState !== 1) return;
    const head = new DataView(new ArrayBuffer(8));
    head.setFloat64(0, performance.now() / 1000, true);
    ws.send(await new Blob([head.buffer, blob]).arrayBuffer());
    waiting = true;
    sent += 1;
    const now = performance.now();
    if (now - fpsT0 > 1000) { $("fps").textContent = (sent * 1000 / (now - fpsT0)).toFixed(1); sent = 0; fpsT0 = now; }
  }, "image/jpeg", 0.7);
}

function stop() {
  clearInterval(timer);
  if (ws && ws.readyState <= 1) ws.close();
  ws = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("startBtn").disabled = false;
  $("stopBtn").disabled = true;
  $("stageEmpty").hidden = false;
  setPill("idle", "Idle");
}

$("startBtn").onclick = start;
$("stopBtn").onclick = stop;
$("file").onchange = () => { $("uploadBtn").disabled = !$("file").files.length; };
$("uploadBtn").onclick = async () => {
  const f = $("file").files[0];
  if (!f) return;
  $("uploadBtn").disabled = true;
  $("vresult").innerHTML = "";
  addLine($("vresult"), "Processing…", f.name);
  const body = new FormData();
  body.append("file", f);
  try {
    const r = await (await fetch("/api/video", { method: "POST", body })).json();
    $("vresult").innerHTML = "";
    if (!r.sentences.length) addLine($("vresult"), "No signing detected.", `${r.frames} frames`);
    r.sentences.forEach((s, i) => addLine($("vresult"),
      s.trained ? (s.text || "(no output)") : `Sentence ${i + 1} detected`,
      `${s.start_s}–${s.end_s}s` + (s.trained ? ` · ${s.latency_ms} ms` : "")));
    const said = r.sentences.filter((s) => s.trained).map((s) => s.text).join(". ");
    speak(said);
  } catch (e) {
    $("vresult").innerHTML = "";
    addLine($("vresult"), "Failed: " + e.message, "");
  }
  $("uploadBtn").disabled = false;
};
