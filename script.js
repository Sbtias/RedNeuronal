class NeuralNetwork {
  constructor() {
    this.learningRate = 0.7;
    this.reset();
  }

  reset() {
    this.weightsInputHidden = Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => Math.random() * 2 - 1));
    this.biasHidden = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
    this.weightsHiddenOutput = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
    this.biasOutput = Math.random() * 2 - 1;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));
  }

  forward(inputs) {
    const hidden = this.biasHidden.map((b, j) => this.sigmoid(b + inputs.reduce((sum, value, i) => sum + value * this.weightsInputHidden[i][j], 0)));
    const output = this.sigmoid(this.biasOutput + hidden.reduce((sum, value, j) => sum + value * this.weightsHiddenOutput[j], 0));
    return { hidden, output };
  }

  train(inputs, expected) {
    const { hidden, output } = this.forward(inputs);
    const outputDelta = (expected - output) * output * (1 - output);
    const hiddenDelta = hidden.map((value, j) => value * (1 - value) * this.weightsHiddenOutput[j] * outputDelta);

    for (let j = 0; j < 4; j++) this.weightsHiddenOutput[j] += hidden[j] * outputDelta * this.learningRate;
    this.biasOutput += outputDelta * this.learningRate;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) this.weightsInputHidden[i][j] += inputs[i] * hiddenDelta[j] * this.learningRate;
    for (let j = 0; j < 4; j++) this.biasHidden[j] += hiddenDelta[j] * this.learningRate;

    return (expected - output) ** 2;
  }

  predict(inputs) {
    return this.forward(inputs).output;
  }

  serialize() {
    return JSON.stringify({
      weightsInputHidden: this.weightsInputHidden,
      biasHidden: this.biasHidden,
      weightsHiddenOutput: this.weightsHiddenOutput,
      biasOutput: this.biasOutput
    });
  }

  restore(serialized) {
    const data = typeof serialized === "string" ? JSON.parse(serialized) : serialized;
    if (!data || !Array.isArray(data.weightsInputHidden) || data.weightsInputHidden.length !== 2) throw new Error("Estado de red inválido");
    if (!Array.isArray(data.biasHidden) || data.biasHidden.length !== 4) throw new Error("Estado de red inválido");
    if (!Array.isArray(data.weightsHiddenOutput) || data.weightsHiddenOutput.length !== 4) throw new Error("Estado de red inválido");
    this.weightsInputHidden = data.weightsInputHidden;
    this.biasHidden = data.biasHidden;
    this.weightsHiddenOutput = data.weightsHiddenOutput;
    this.biasOutput = Number(data.biasOutput) || 0;
  }
}

const trainingData = [
  [[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]
];

const brain = new NeuralNetwork();
const $ = id => document.getElementById(id);
const SUPABASE_URL = "https://otdfvaufiqwlwmaqsljs.supabase.co";
const SUPABASE_KEY = "sb_publishable_jC98NCK4bQeRRYmlZ9CVw_tALEihhJ";
const supabaseClient = window.supabase?.createClient(SUPABASE_URL, SUPABASE_KEY) || null;
const MEMORY_KEY = "neural3d_chat_memory_v5";
const STATE_QUESTION = "__NEURAL_NETWORK_STATE_V1__";

let learnedMemory = loadLocalMemory();
let globalMemory = [];
let epoch = 0;
let loss = 1;
let trained = false;
let saving = false;
let angleX = 8, angleY = -20, zoom = 1, dragging = false, lastX = 0, lastY = 0;

const layers = [
  { name: "ENTRADA", count: 2, labels: ["x₁", "x₂"] },
  { name: "OCULTA", count: 4, labels: ["h₁", "h₂", "h₃", "h₄"] },
  { name: "SALIDA", count: 1, labels: ["y"] }
];

function loadLocalMemory() {
  try {
    const data = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]");
    return Array.isArray(data) ? data.slice(-100) : [];
  } catch {
    return [];
  }
}

function saveLocalMemory() {
  try { localStorage.setItem(MEMORY_KEY, JSON.stringify(learnedMemory.slice(-100))); } catch {}
}

function normalize(text) {
  return String(text).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9ñ?¿! ]/g, "").replace(/\s+/g, " ").trim();
}

function setState(text, error = false) {
  $("dbState").textContent = text;
  $("dbState").classList.toggle("error", error);
}

async function loadFromDatabase() {
  if (!supabaseClient) { setState("modo local", true); return; }
  try {
    const { data, error } = await supabaseClient.from("neural_memory").select("question,answer,created_at").order("created_at", { ascending: false }).limit(500);
    if (error) throw error;
    globalMemory = Array.isArray(data) ? data.filter(row => row.question !== STATE_QUESTION) : [];
    const stateRow = Array.isArray(data) ? data.find(row => row.question === STATE_QUESTION) : null;
    if (stateRow) {
      try {
        const state = JSON.parse(stateRow.answer);
        brain.restore(state.network);
        epoch = Number(state.epoch) || 0;
        loss = Number(state.loss) || 1;
        trained = Boolean(state.trained);
      } catch (e) { console.warn("Estado de red inválido:", e); }
    }
    setState(`base de datos · ${globalMemory.length} recuerdos`);
    updateAll();
  } catch (e) {
    setState("base de datos no disponible", true);
    console.warn("Supabase:", e.message);
  }
}

async function saveNetworkState() {
  if (!supabaseClient || saving) return;
  saving = true;
  try {
    const payload = JSON.stringify({ network: JSON.parse(brain.serialize()), epoch, loss, trained, updatedAt: new Date().toISOString() });
    const { data: existing, error: readError } = await supabaseClient.from("neural_memory").select("question").eq("question", STATE_QUESTION).limit(1);
    if (readError) throw readError;
    if (existing?.length) {
      const { error } = await supabaseClient.from("neural_memory").update({ answer: payload }).eq("question", STATE_QUESTION);
      if (error) throw error;
    } else {
      const { error } = await supabaseClient.from("neural_memory").insert({ question: STATE_QUESTION, answer: payload });
      if (error) throw error;
    }
    setState(`guardado · ${globalMemory.length} recuerdos`);
  } catch (e) {
    setState("guardado local · revisa Supabase", true);
    console.warn("Guardado de red:", e.message);
  } finally { saving = false; }
}

async function saveGlobalMemory(question, answer) {
  const q = question.trim().slice(0, 300), a = answer.trim().slice(0, 600);
  if (!q || !a || !supabaseClient) return;
  try {
    const { error } = await supabaseClient.from("neural_memory").insert({ question: q, answer: a });
    if (error) throw error;
    globalMemory.unshift({ question: q, answer: a });
    globalMemory = globalMemory.slice(0, 500);
    setState(`base de datos · ${globalMemory.length} recuerdos`);
  } catch (e) { console.warn("Memoria global:", e.message); }
}

function rememberConversation(q, a) {
  const normalized = normalize(q);
  const old = learnedMemory.find(x => normalize(x.question) === normalized);
  if (old) { old.answer = a; old.uses = (old.uses || 0) + 1; }
  else learnedMemory.push({ question: q.trim(), answer: a, uses: 1 });
  learnedMemory = learnedMemory.slice(-100);
  saveLocalMemory();
}

function findLearnedAnswer(text) {
  const q = normalize(text), all = [...globalMemory, ...learnedMemory];
  const exact = all.find(x => normalize(x.question) === q);
  if (exact) return exact.answer;
  const words = q.split(" ").filter(w => w.length > 3);
  let best = null, bestScore = 0;
  for (const item of all) {
    const known = new Set(normalize(item.question).split(" "));
    const score = words.length ? words.filter(w => known.has(w)).length / words.length : 0;
    if (score > bestScore && score >= .6) { best = item.answer; bestScore = score; }
  }
  return best;
}

function nodeY(i, count) { return (i - (count - 1) / 2) * 92; }
function weight(fromLayer, from, to) { return fromLayer === 0 ? brain.weightsInputHidden[from][to] : brain.weightsHiddenOutput[from]; }

function createNetwork3D() {
  const nodes = $("nodes"), connections = $("connections");
  nodes.innerHTML = ""; connections.innerHTML = "";
  const positions = [];
  const x = [-220, 0, 220];

  layers.forEach((layer, li) => {
    const title = document.createElement("div");
    title.className = "layerLabel";
    title.textContent = layer.name;
    title.style.transform = `translate3d(${x[li] - 34}px,${-((layer.count - 1) * 92) / 2 - 82}px,0)`;
    nodes.appendChild(title);
    const layerPositions = [];
    for (let i = 0; i < layer.count; i++) {
      const y = nodeY(i, layer.count), z = li === 1 ? (i % 2 ? -30 : 30) : 0;
      const node = document.createElement("button");
      node.type = "button"; node.className = `node layer-${li}`; node.dataset.layer = li; node.dataset.index = i;
      node.textContent = layer.labels[i]; node.title = `${layer.name} · ${layer.labels[i]}`;
      node.style.transform = `translate3d(${x[li]}px,${y}px,${z}px)`;
      node.addEventListener("pointerdown", e => e.stopPropagation());
      node.addEventListener("click", e => { e.stopPropagation(); selectNode(li, i); });
      nodes.appendChild(node);
      layerPositions.push({ x: x[li], y, z });
    }
    positions.push(layerPositions);
  });

  for (let l = 0; l < positions.length - 1; l++) {
    for (let a = 0; a < positions[l].length; a++) {
      for (let b = 0; b < positions[l + 1].length; b++) {
        const line = document.createElement("div");
        line.className = "connection";
        line.dataset.fromLayer = l; line.dataset.from = a; line.dataset.to = b;
        connections.appendChild(line);
        positionConnection(line, positions[l][a], positions[l + 1][b], weight(l, a, b));
      }
    }
  }
  window._networkPositions = positions;
}

function positionConnection(line, a, b, w) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const length = Math.hypot(dx, dy, dz), angle = Math.atan2(dy, dx) * 180 / Math.PI, tilt = Math.atan2(dz, Math.hypot(dx, dy)) * 180 / Math.PI;
  const strength = Math.min(1, Math.abs(w));
  line.style.width = `${length}px`; line.style.height = `${1.2 + strength * 2.8}px`; line.style.opacity = `${.14 + strength * .65}`;
  line.style.transform = `translate3d(${a.x}px,${a.y}px,${a.z}px) rotateZ(${angle}deg) rotateY(${-tilt}deg)`;
  line.classList.toggle("positive", w >= 0); line.classList.toggle("negative", w < 0);
}

function updateNetworkVisuals() {
  if (!window._networkPositions) return;
  document.querySelectorAll(".connection").forEach(line => {
    const l = +line.dataset.fromLayer, a = +line.dataset.from, b = +line.dataset.to;
    positionConnection(line, window._networkPositions[l][a], window._networkPositions[l + 1][b], weight(l, a, b));
  });
  draw2D();
}

function selectNode(layer, index) {
  document.querySelectorAll(".node").forEach(n => n.classList.remove("selected"));
  const node = document.querySelector(`.node[data-layer="${layer}"][data-index="${index}"]`);
  if (node) node.classList.add("selected");
  const label = layers[layer].labels[index];
  const bias = layer === 1 ? brain.biasHidden[index] : layer === 2 ? brain.biasOutput : 0;
  $("nodeInfo").textContent = `${label} · ${layers[layer].name.toLowerCase()} · sesgo ${bias.toFixed(3)}`;
}

function render() { $("network3d").style.transform = `translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`; }

function draw2D() {
  const canvas = $("networkCanvas"), rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.floor(rect.width)), height = Math.max(300, Math.floor(rect.height));
  canvas.width = width * dpr; canvas.height = height * dpr;
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const centers = [width * .17, width * .5, width * .83];
  const colors = { line: "rgba(190,190,190,.32)", text: "#8a8a8a", node: "#242424", border: "#777" };
  const positions = layers.map((layer, li) => Array.from({ length: layer.count }, (_, i) => ({ x: centers[li], y: height / 2 + nodeY(i, layer.count) * Math.min(1, height / 480), z: 0 })));

  ctx.lineWidth = 1;
  for (let l = 0; l < positions.length - 1; l++) for (let a = 0; a < positions[l].length; a++) for (let b = 0; b < positions[l + 1].length; b++) {
    const w = weight(l, a, b), p = positions[l][a], q = positions[l + 1][b];
    ctx.globalAlpha = .12 + Math.min(1, Math.abs(w)) * .55; ctx.strokeStyle = w >= 0 ? "#aaa" : "#555"; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  layers.forEach((layer, li) => {
    positions[li].forEach((p, i) => {
      const radius = li === 2 ? 24 : li === 1 ? 20 : 18;
      const g = ctx.createRadialGradient(p.x - radius * .3, p.y - radius * .3, 2, p.x, p.y, radius);
      g.addColorStop(0, "#777"); g.addColorStop(.55, "#292929"); g.addColorStop(1, "#0d0d0d");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = colors.border; ctx.stroke();
      ctx.fillStyle = "#ddd"; ctx.font = "600 11px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(layer.labels[i], p.x, p.y);
    });
    ctx.fillStyle = colors.text; ctx.font = "10px system-ui"; ctx.letterSpacing = "2px"; ctx.textAlign = "center"; ctx.fillText(layer.name, centers[li], 24);
  });
}

function updateStats(sample = trainingData[1]) {
  $("epoch").textContent = epoch.toLocaleString(); $("loss").textContent = loss.toFixed(4); $("prediction").textContent = brain.predict(sample[0]).toFixed(3);
  $("architectureValue").textContent = "2 · 4 · 1";
}

function updateAll() { updateStats(); updateNetworkVisuals(); render(); $("state").textContent = trained ? "Entrenada · estado restaurado" : "Lista para aprender XOR"; $("result").textContent = trained ? "La red recuperó lo aprendido." : "La red todavía no ha aprendido."; }

function trainBrain() {
  const trainButton = $("train"); trainButton.disabled = true; $("reset").disabled = true;
  $("state").textContent = "Aprendiendo y guardando..."; setState("entrenando");
  let frames = 0;
  const run = () => {
    for (let i = 0; i < 100; i++) { let total = 0; for (const [x, y] of trainingData) { total += brain.train(x, y); epoch++; } loss = total / 4; }
    trained = true; updateStats(trainingData[epoch % 4]); updateNetworkVisuals();
    frames++;
    if (frames < 20) return requestAnimationFrame(run);
    $("state").textContent = "Entrenada · aprendiendo continuamente"; $("result").textContent = "La red aprendió XOR y guardó su estado."; $("infoText").textContent = "El lienzo 2D muestra la misma red y la base de datos conserva los pesos y recuerdos.";
    $("scene").classList.add("trained"); trainButton.disabled = false; $("reset").disabled = false; saveNetworkState();
  };
  run();
}

function neuronReply(text) {
  const m = normalize(text);
  if (!trained) return "Todavía no estoy entrenada. Pulsa «Entrenar red» primero. 🧠";
  const learned = findLearnedAnswer(text); if (learned) return learned;
  if (m.includes("hola") || m.includes("buenas")) return "Hola. Mis conexiones están activas. 🧠";
  if (m.includes("quien eres")) return "Soy una red neuronal artificial 2·4·1 entrenada con XOR.";
  if (m.includes("que aprendiste") || m.includes("que sabes")) return `Aprendí XOR. Error actual: ${loss.toFixed(4)}.`;
  if (m.includes("que puedes hacer")) return "Puedo entrenar, visualizar pesos, guardar mi estado y conservar ejemplos en la base de datos.";
  if (m.endsWith("?")) return "Todavía no tengo una respuesta para eso. Puedes enseñarme un ejemplo y lo guardaré en memoria.";
  return "He recibido esa información. Puede convertirse en un ejemplo para mi memoria.";
}

$("train").addEventListener("click", trainBrain);
$("reset").addEventListener("click", async () => {
  brain.reset(); epoch = 0; loss = 1; trained = false; learnedMemory = []; saveLocalMemory();
  $("state").textContent = "Lista para aprender XOR"; $("result").textContent = "La red fue reiniciada."; $("infoText").textContent = "Entrénala para volver a modificar los pesos."; $("scene").classList.remove("trained"); $("messages").innerHTML = '<div class="message neuronMessage">Red reiniciada. La memoria de conversaciones de la base de datos permanece. 🧠</div>';
  createNetwork3D(); updateAll(); await saveNetworkState();
});

$("chatForm").addEventListener("submit", async e => {
  e.preventDefault(); const input = $("chatInput"), text = input.value.trim(); if (!text || !trained) return;
  addMessage(text, "user"); input.value = ""; input.disabled = true; $("send").disabled = true; setState("guardando aprendizaje...");
  await new Promise(resolve => setTimeout(resolve, 180)); const reply = neuronReply(text); addMessage(reply, "neuron"); rememberConversation(text, reply); await saveGlobalMemory(text, reply); input.disabled = false; $("send").disabled = false; setState(`base de datos · ${globalMemory.length} recuerdos`); input.focus();
});

function addMessage(text, type) { const m = document.createElement("div"); m.className = `message ${type === "user" ? "userMessage" : "neuronMessage"}`; m.textContent = text; $("messages").appendChild(m); $("messages").scrollTop = $("messages").scrollHeight; }

for (const button of document.querySelectorAll("[data-prompt]")) button.addEventListener("click", () => { $("chatInput").value = button.dataset.prompt; $("chatInput").focus(); });

const scene = $("scene");
scene.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; scene.setPointerCapture(e.pointerId); });
scene.addEventListener("pointermove", e => { if (!dragging) return; angleY += (e.clientX - lastX) * .4; angleX -= (e.clientY - lastY) * .3; angleX = Math.max(-70, Math.min(70, angleX)); lastX = e.clientX; lastY = e.clientY; render(); });
scene.addEventListener("pointerup", () => dragging = false); scene.addEventListener("pointercancel", () => dragging = false);
scene.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(.55, Math.min(1.5, zoom - e.deltaY * .001)); render(); }, { passive: false });
window.addEventListener("resize", draw2D);

createNetwork3D(); updateAll(); draw2D(); loadFromDatabase();
