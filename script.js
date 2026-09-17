class NeuralNetwork {
  constructor() {
    this.learningRate = 0.7;
    this.reset();
  }

  reset() {
    this.weightsInputHidden = Array.from({ length: 2 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 2 - 1)
    );
    this.biasHidden = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
    this.weightsHiddenOutput = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
    this.biasOutput = Math.random() * 2 - 1;
  }

  sigmoid(x) {
    const safe = Math.max(-60, Math.min(60, x));
    return 1 / (1 + Math.exp(-safe));
  }

  forward(inputs) {
    const hidden = this.biasHidden.map((bias, j) => {
      let sum = bias;
      for (let i = 0; i < 2; i++) sum += inputs[i] * this.weightsInputHidden[i][j];
      return this.sigmoid(sum);
    });

    let outputSum = this.biasOutput;
    for (let j = 0; j < 4; j++) outputSum += hidden[j] * this.weightsHiddenOutput[j];

    return { hidden, output: this.sigmoid(outputSum) };
  }

  train(inputs, expected) {
    const { hidden, output } = this.forward(inputs);
    const outputDelta = (expected - output) * output * (1 - output);
    const hiddenDelta = hidden.map(
      (value, j) => value * (1 - value) * this.weightsHiddenOutput[j] * outputDelta
    );

    for (let j = 0; j < 4; j++) {
      this.weightsHiddenOutput[j] += hidden[j] * outputDelta * this.learningRate;
    }
    this.biasOutput += outputDelta * this.learningRate;

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 4; j++) {
        this.weightsInputHidden[i][j] += inputs[i] * hiddenDelta[j] * this.learningRate;
      }
    }

    for (let j = 0; j < 4; j++) {
      this.biasHidden[j] += hiddenDelta[j] * this.learningRate;
    }

    return (expected - output) ** 2;
  }

  predict(inputs) {
    return this.forward(inputs).output;
  }
}

const trainingData = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];

const brain = new NeuralNetwork();
const $ = (id) => document.getElementById(id);

const SUPABASE_URL = "https://otdfvaufiqwlwmaqsljs.supabase.co";
const SUPABASE_KEY = "sb_publishable_jC98NCK4bQeRRYmlZ9CVw_tALEihhJ";
const supabaseClient = window.supabase?.createClient
  ? window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY)
  : null;

const MEMORY_KEY = "neural3d_chat_memory_v4";
let learnedMemory = loadLocalMemory();
let globalMemory = [];
let epoch = 0;
let loss = 1;
let trained = false;
let angleX = 8;
let angleY = -20;
let zoom = 1;
let dragging = false;
let lastX = 0;
let lastY = 0;

const layerConfig = [
  { name: "ENTRADA", count: 2, x: -220, labels: ["x₁", "x₂"] },
  { name: "OCULTA", count: 4, x: 0, labels: ["h₁", "h₂", "h₃", "h₄"] },
  { name: "SALIDA", count: 1, x: 220, labels: ["y"] }
];

function loadLocalMemory() {
  try {
    const stored = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]");
    return Array.isArray(stored) ? stored.slice(-100) : [];
  } catch {
    return [];
  }
}

function saveLocalMemory() {
  try {
    localStorage.setItem(MEMORY_KEY, JSON.stringify(learnedMemory.slice(-100)));
  } catch (error) {
    console.warn("No se pudo guardar la memoria local:", error);
  }
}

function normalize(text) {
  return String(text)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9ñ?¿! ]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function findLearnedAnswer(text) {
  const question = normalize(text);
  const all = [...globalMemory, ...learnedMemory];
  const exact = all.find((item) => normalize(item.question) === question);
  if (exact) return exact.answer;

  const words = question.split(" ").filter((word) => word.length > 3);
  let best = null;
  let bestScore = 0;

  for (const item of all) {
    const known = new Set(normalize(item.question).split(" "));
    const score = words.length
      ? words.filter((word) => known.has(word)).length / words.length
      : 0;
    if (score > bestScore && score >= 0.6) {
      best = item.answer;
      bestScore = score;
    }
  }

  return best;
}

async function loadGlobalMemory() {
  if (!supabaseClient) {
    $("chatState").textContent = trained ? "modo local" : "sin entrenar";
    return;
  }

  try {
    const { data, error } = await supabaseClient
      .from("neural_memory")
      .select("question,answer")
      .order("created_at", { ascending: false })
      .limit(500);

    if (error) throw error;
    globalMemory = Array.isArray(data) ? data : [];
    $("chatState").textContent = trained ? "memoria conectada" : "sin entrenar";
  } catch (error) {
    globalMemory = [];
    $("chatState").textContent = trained ? "modo local" : "sin entrenar";
    console.warn("Memoria global:", error.message);
  }
}

async function saveGlobalMemory(question, answer) {
  const q = question.trim().slice(0, 300);
  const a = answer.trim().slice(0, 600);
  if (!q || !a) return;

  if (!supabaseClient) return;

  try {
    const { error } = await supabaseClient
      .from("neural_memory")
      .insert({ question: q, answer: a });

    if (error) throw error;
    globalMemory.unshift({ question: q, answer: a });
    globalMemory = globalMemory.slice(0, 500);
  } catch (error) {
    console.warn("No se pudo guardar en memoria global:", error.message);
  }
}

function rememberConversation(question, answer) {
  const normalized = normalize(question);
  const old = learnedMemory.find((item) => normalize(item.question) === normalized);

  if (old) {
    old.answer = answer;
    old.uses = (old.uses || 0) + 1;
  } else {
    learnedMemory.push({ question: question.trim(), answer, uses: 1 });
  }

  learnedMemory = learnedMemory.slice(-100);
  saveLocalMemory();
}

function nodeY(index, count) {
  return (index - (count - 1) / 2) * 92;
}

function getWeight(layer, from, to) {
  if (layer === 0) return brain.weightsInputHidden[from][to];
  return brain.weightsHiddenOutput[from];
}

function createNetwork() {
  const nodes = $("nodes");
  const connections = $("connections");
  if (!nodes || !connections) return;

  nodes.innerHTML = "";
  connections.innerHTML = "";
  const positions = [];

  layerConfig.forEach((layer, layerIndex) => {
    const title = document.createElement("div");
    title.className = "layerLabel";
    title.textContent = layer.name;
    title.style.transform = `translate3d(${layer.x - 34}px, ${-((layer.count - 1) * 92) / 2 - 82}px, 0)`;
    nodes.appendChild(title);

    const layerPositions = [];

    for (let index = 0; index < layer.count; index++) {
      const y = nodeY(index, layer.count);
      const z = layerIndex === 1 ? (index % 2 ? -30 : 30) : 0;
      const node = document.createElement("button");

      node.type = "button";
      node.className = `node layer-${layerIndex}`;
      node.dataset.layer = layerIndex;
      node.dataset.index = index;
      node.textContent = layer.labels[index];
      node.title = `${layer.name} · ${layer.labels[index]}`;
      node.style.transform = `translate3d(${layer.x}px, ${y}px, ${z}px)`;
      node.addEventListener("click", () => selectNode(layerIndex, index));

      nodes.appendChild(node);
      layerPositions.push({ x: layer.x, y, z, element: node });
    }

    positions.push(layerPositions);
  });

  for (let layer = 0; layer < positions.length - 1; layer++) {
    for (let from = 0; from < positions[layer].length; from++) {
      for (let to = 0; to < positions[layer + 1].length; to++) {
        const line = document.createElement("div");
        line.className = "connection";
        line.dataset.fromLayer = layer;
        line.dataset.from = from;
        line.dataset.to = to;
        connections.appendChild(line);
        positionConnection(
          line,
          positions[layer][from],
          positions[layer + 1][to],
          getWeight(layer, from, to)
        );
      }
    }
  }

  window._networkPositions = positions;
}

function positionConnection(line, a, b, weight) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dz = b.z - a.z;
  const length = Math.hypot(dx, dy, dz);
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  const tilt = Math.atan2(dz, Math.hypot(dx, dy)) * 180 / Math.PI;
  const strength = Math.min(1, Math.abs(weight));

  line.style.width = `${length}px`;
  line.style.height = `${1.2 + strength * 2.8}px`;
  line.style.opacity = `${0.14 + strength * 0.65}`;
  line.style.transform = `translate3d(${a.x}px, ${a.y}px, ${a.z}px) rotateZ(${angle}deg) rotateY(${-tilt}deg)`;
  line.classList.toggle("positive", weight >= 0);
  line.classList.toggle("negative", weight < 0);
}

function updateNetworkVisuals() {
  const positions = window._networkPositions;
  if (!positions) return;

  document.querySelectorAll(".connection").forEach((line) => {
    const layer = Number(line.dataset.fromLayer);
    const from = Number(line.dataset.from);
    const to = Number(line.dataset.to);
    positionConnection(
      line,
      positions[layer][from],
      positions[layer + 1][to],
      getWeight(layer, from, to)
    );
  });
}

function selectNode(layer, index) {
  document.querySelectorAll(".node").forEach((node) => node.classList.remove("selected"));
  const node = document.querySelector(`.node[data-layer="${layer}"][data-index="${index}"]`);
  if (node) node.classList.add("selected");

  const label = layerConfig[layer].labels[index];
  $("nodeInfo").textContent = layer === 0
    ? `${label} · entrada`
    : layer === 2
      ? `${label} · salida`
      : `${label} · neurona oculta`;
}

function render() {
  $("network3d").style.transform =
    `translate(-50%, -50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
}

function updateStats(sample = trainingData[1]) {
  $("epoch").textContent = epoch.toLocaleString();
  $("loss").textContent = loss.toFixed(4);
  $("prediction").textContent = brain.predict(sample[0]).toFixed(3);
}

function trainBrain() {
  const trainButton = $("train");
  const resetButton = $("reset");
  trainButton.disabled = true;
  resetButton.disabled = true;
  $("state").textContent = "Aprendiendo XOR...";
  $("chatState").textContent = "entrenando";

  let frame = 0;

  const run = () => {
    for (let i = 0; i < 100; i++) {
      let total = 0;
      for (const [inputs, expected] of trainingData) {
        total += brain.train(inputs, expected);
      }
      loss = total / trainingData.length;
      epoch++;
    }

    updateStats(trainingData[epoch % trainingData.length]);
    updateNetworkVisuals();
    frame++;

    if (frame < 20) {
      requestAnimationFrame(run);
      return;
    }

    trained = true;
    $("state").textContent = "Entrenada · XOR aprendido";
    $("chatState").textContent = "memoria conectada";
    $("result").textContent = "Red 2 → 4 → 1 entrenada.";
    $("infoText").textContent = "Selecciona una neurona para inspeccionarla. Las conexiones muestran la fuerza de sus pesos.";
    $("scene").classList.add("trained");
    trainButton.disabled = false;
    resetButton.disabled = false;
    loadGlobalMemory();
    $("chatInput").focus();
  };

  run();
}

function addMessage(text, type) {
  const message = document.createElement("div");
  message.className = `message ${type === "user" ? "userMessage" : "neuronMessage"}`;
  message.textContent = text;
  $("messages").appendChild(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function neuronReply(text) {
  const normalized = normalize(text);
  if (!trained) return "Todavía no estoy entrenada. Pulsa «Entrenar red» primero. 🧠";

  const learned = findLearnedAnswer(text);
  if (learned) return learned;
  if (normalized.includes("hola") || normalized.includes("buenas")) return "Hola. Mis conexiones están activas. 🧠";
  if (normalized.includes("quien eres")) return "Soy una red neuronal artificial 2·4·1 entrenada con XOR.";
  if (normalized.includes("que aprendiste") || normalized.includes("que sabes")) {
    return `Aprendí XOR. Error actual: ${loss.toFixed(4)}.`;
  }
  if (normalized.includes("que puedes hacer")) {
    return "Puedo entrenar, visualizar pesos, seleccionar neuronas y guardar ejemplos en memoria.";
  }
  if (normalized.endsWith("?")) {
    return "Todavía no tengo una respuesta para eso. Puedes enseñarme un ejemplo y lo guardaré en memoria.";
  }
  return "He recibido esa información. Puede convertirse en un ejemplo para mi memoria.";
}

$("train").addEventListener("click", trainBrain);

$("reset").addEventListener("click", () => {
  brain.reset();
  epoch = 0;
  loss = 1;
  trained = false;
  learnedMemory = [];

  try {
    localStorage.removeItem(MEMORY_KEY);
  } catch (error) {
    console.warn("No se pudo limpiar la memoria local:", error);
  }

  $("state").textContent = "Lista para aprender XOR";
  $("chatState").textContent = "sin entrenar";
  $("result").textContent = "La red todavía no ha aprendido.";
  $("infoText").textContent = "Entrénala para observar cómo cambian sus conexiones.";
  $("nodeInfo").textContent = "Ninguno";
  $("scene").classList.remove("trained");
  $("messages").innerHTML = '<div class="message neuronMessage">Memoria local reiniciada. La memoria global permanece. 🧠</div>';
  createNetwork();
  updateStats();
  render();
});

$("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  const input = $("chatInput");
  const text = input.value.trim();
  if (!text || !trained) return;

  addMessage(text, "user");
  input.value = "";
  input.disabled = true;
  $("send").disabled = true;
  $("chatState").textContent = "pensando...";

  await new Promise((resolve) => setTimeout(resolve, 180));

  const reply = neuronReply(text);
  addMessage(reply, "neuron");
  rememberConversation(text, reply);
  await saveGlobalMemory(text, reply);

  input.disabled = false;
  $("send").disabled = false;
  $("chatState").textContent = globalMemory.length ? "memoria conectada" : "modo local";
  input.focus();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    $("chatInput").value = button.dataset.prompt;
    $("chatInput").focus();
  });
});

const scene = $("scene");
scene.addEventListener("pointerdown", (event) => {
  if (event.target.closest("button")) return;
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  scene.setPointerCapture(event.pointerId);
});

scene.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  angleY += (event.clientX - lastX) * 0.4;
  angleX -= (event.clientY - lastY) * 0.3;
  angleX = Math.max(-70, Math.min(70, angleX));
  lastX = event.clientX;
  lastY = event.clientY;
  render();
});

scene.addEventListener("pointerup", () => { dragging = false; });
scene.addEventListener("pointercancel", () => { dragging = false; });
scene.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(0.55, Math.min(1.5, zoom - event.deltaY * 0.001));
  render();
}, { passive: false });

createNetwork();
updateStats();
render();
loadGlobalMemory();
