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

  sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x)))); }
  sigmoidDerivative(x) { return x * (1 - x); }

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
    const outputError = expected - output;
    const outputDelta = outputError * this.sigmoidDerivative(output);

    const hiddenDeltas = hidden.map((value, j) =>
      this.sigmoidDerivative(value) * this.weightsHiddenOutput[j] * outputDelta
    );

    for (let j = 0; j < 4; j++) {
      this.weightsHiddenOutput[j] += hidden[j] * outputDelta * this.learningRate;
    }
    this.biasOutput += outputDelta * this.learningRate;

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 4; j++) {
        this.weightsInputHidden[i][j] += inputs[i] * hiddenDeltas[j] * this.learningRate;
      }
    }
    for (let j = 0; j < 4; j++) this.biasHidden[j] += hiddenDeltas[j] * this.learningRate;

    return Math.pow(expected - output, 2);
  }

  predict(inputs) { return this.forward(inputs).output; }
}

const trainingData = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];

const brain = new NeuralNetwork();
const $ = id => document.getElementById(id);

const SUPABASE_URL = "https://otdfvaufiqwlwmaqsljs.supabase.co";
const SUPABASE_KEY = "sb_publishable_jC98NCK4bQeRRYmlZ9CVw_tALEihhJ";
const supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
const MEMORY_KEY = "neural3d_chat_memory_v3";

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
let nodePositions = [];

const layerConfig = [
  { name: "ENTRADA", count: 2, x: -220, labels: ["x₁", "x₂"] },
  { name: "OCULTA", count: 4, x: 0, labels: ["h₁", "h₂", "h₃", "h₄"] },
  { name: "SALIDA", count: 1, x: 220, labels: ["y"] }
];

function loadLocalMemory() {
  try {
    const saved = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.slice(-100) : [];
  } catch { return []; }
}

function saveLocalMemory() {
  localStorage.setItem(MEMORY_KEY, JSON.stringify(learnedMemory.slice(-100)));
}

function normalize(text) {
  return text.toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9ñ?¿! ]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function findLearnedAnswer(text) {
  const q = normalize(text);
  const all = [...globalMemory, ...learnedMemory];
  const exact = all.find(item => normalize(item.question) === q);
  if (exact) return exact.answer;

  const words = q.split(" ").filter(word => word.length > 3);
  if (words.length < 2) return null;

  let best = null;
  let bestScore = 0;
  for (const item of all) {
    const knownWords = new Set(normalize(item.question).split(" "));
    const score = words.filter(word => knownWords.has(word)).length / words.length;
    if (score > bestScore && score >= 0.6) {
      best = item.answer;
      bestScore = score;
    }
  }
  return best;
}

async function loadGlobalMemory() {
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
    console.warn("Memoria global no disponible:", error.message);
    globalMemory = [];
    $("chatState").textContent = trained ? "modo local" : "sin entrenar";
  }
}

async function saveGlobalMemory(question, answer) {
  const cleanQuestion = question.trim().slice(0, 300);
  const cleanAnswer = answer.trim().slice(0, 600);
  if (!cleanQuestion || !cleanAnswer) return;
  try {
    const { error } = await supabaseClient.from("neural_memory").insert({ question: cleanQuestion, answer: cleanAnswer });
    if (error) throw error;
    globalMemory.unshift({ question: cleanQuestion, answer: cleanAnswer });
    globalMemory = globalMemory.slice(0, 500);
  } catch (error) {
    console.warn("No se pudo guardar en memoria global:", error.message);
  }
}

function rememberConversation(question, answer) {
  const normalizedQuestion = normalize(question);
  const existing = learnedMemory.find(item => normalize(item.question) === normalizedQuestion);
  if (existing) {
    existing.answer = answer;
    existing.uses = (existing.uses || 0) + 1;
  } else {
    learnedMemory.push({ question: question.trim(), answer, uses: 1 });
  }
  learnedMemory = learnedMemory.slice(-100);
  saveLocalMemory();
}

function nodeY(index, count) {
  const gap = 92;
  return (index - (count - 1) / 2) * gap;
}

function createNetwork() {
  const nodes = $("nodes");
  const connections = $("connections");
  nodes.innerHTML = "";
  connections.innerHTML = "";
  nodePositions = [];

  layerConfig.forEach((layer, layerIndex) => {
    const label = document.createElement("div");
    label.className = "layerLabel";
    label.textContent = layer.name;
    label.style.transform = `translate3d(${layer.x - 34}px, ${-((layer.count - 1) * 92) / 2 - 82}px, 0)`;
    nodes.appendChild(label);

    const layerPositions = [];
    for (let i = 0; i < layer.count; i++) {
      const y = nodeY(i, layer.count);
      const z = layerIndex === 1 ? (i % 2 === 0 ? 30 : -30) : 0;
      const node = document.createElement("div");
      node.className = `node layer-${layerIndex}`;
      node.dataset.layer = layerIndex;
      node.dataset.index = i;
      node.textContent = layer.labels[i];
      node.style.transform = `translate3d(${layer.x}px, ${y}px, ${z}px)`;
      nodes.appendChild(node);
      layerPositions.push({ x: layer.x, y, z, element: node });
    }
    nodePositions.push(layerPositions);
  });

  const connectionGroups = [
    [0, 1],
    [1, 2]
  ];

  connectionGroups.forEach(([fromLayer, toLayer]) => {
    nodePositions[fromLayer].forEach((from, fromIndex) => {
      nodePositions[toLayer].forEach((to, toIndex) => {
        const line = document.createElement("div");
        line.className = "connection";
        line.dataset.fromLayer = fromLayer;
        line.dataset.from = fromIndex;
        line.dataset.toLayer = toLayer;
        line.dataset.to = toIndex;
        connections.appendChild(line);
        positionConnection(line, from, to, getWeight(fromLayer, fromIndex, toIndex));
      });
    });
  });
}

function getWeight(fromLayer, fromIndex, toIndex) {
  if (fromLayer === 0) return brain.weightsInputHidden[fromIndex][toIndex];
  return brain.weightsHiddenOutput[fromIndex];
}

function positionConnection(line, from, to, weight) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dz = to.z - from.z;
  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  const tilt = Math.atan2(dz, Math.sqrt(dx * dx + dy * dy)) * 180 / Math.PI;
  const strength = Math.min(1, Math.abs(weight));
  line.style.width = `${length}px`;
  line.style.height = `${1.2 + strength * 2.2}px`;
  line.style.opacity = `${0.12 + strength * 0.55}`;
  line.style.transform = `translate3d(${from.x}px, ${from.y}px, ${from.z}px) rotateZ(${angle}deg) rotateY(${-tilt}deg)`;
  line.classList.toggle("positive", weight >= 0);
  line.classList.toggle("negative", weight < 0);
}

function updateNetworkVisuals() {
  const lines = document.querySelectorAll(".connection");
  lines.forEach(line => {
    const fromLayer = Number(line.dataset.fromLayer);
    const fromIndex = Number(line.dataset.from);
    const toIndex = Number(line.dataset.to);
    positionConnection(line, nodePositions[fromLayer][fromIndex], nodePositions[fromLayer + 1][toIndex], getWeight(fromLayer, fromIndex, toIndex));
  });
}

function render() {
  $("network3d").style.transform = `translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
}

function updateStats(sample = trainingData[1]) {
  const prediction = brain.predict(sample[0]);
  $("epoch").textContent = epoch.toLocaleString();
  $("loss").textContent = loss.toFixed(4);
  $("prediction").textContent = prediction.toFixed(3);
}

function trainBrain() {
  const button = $("train");
  button.disabled = true;
  $("reset").disabled = true;
  $("state").textContent = "Aprendiendo XOR...";
  $("chatState").textContent = "entrenando";

  let frames = 0;
  const run = () => {
    for (let i = 0; i < 80; i++) {
      let total = 0;
      for (const [inputs, expected] of trainingData) total += brain.train(inputs, expected);
      loss = total / trainingData.length;
      epoch++;
    }

    updateStats(trainingData[epoch % trainingData.length]);
    updateNetworkVisuals();
    frames++;

    if (frames < 18) {
      requestAnimationFrame(run);
      return;
    }

    trained = true;
    $("state").textContent = "Entrenada · XOR aprendido";
    $("chatState").textContent = "memoria conectada";
    $("result").textContent = "La red aprendió una función XOR con una capa oculta de 4 neuronas.";
    $("infoText").textContent = "Cada línea representa un peso. El grosor y la intensidad muestran su fuerza; los pesos negativos se muestran como conexiones diferentes.";
    $("scene").classList.add("trained");
    button.disabled = false;
    $("reset").disabled = false;
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
  const message = normalize(text);
  if (!trained) return "Todavía no estoy entrenada. Pulsa «Entrenar red» primero. 🧠";

  const learned = findLearnedAnswer(text);
  if (learned) return learned;

  if (message.includes("hola") || message.includes("buenas")) return "Hola. Mis conexiones están activas. 🧠";
  if (message.includes("quien eres")) return "Soy una red neuronal artificial pequeña: 2 entradas, 4 neuronas ocultas y 1 salida.";
  if (message.includes("como estas")) return "Funcionando. No tengo emociones, pero mis pesos están haciendo su pequeño trabajo matemático.";
  if (message.includes("que aprendiste") || message.includes("que sabes")) return `Aprendí XOR. Ahora mi error está alrededor de ${loss.toFixed(4)} y tengo ${epoch.toLocaleString()} épocas registradas.`;
  if (message.includes("entrenar") || message.includes("aprend")) return "El entrenamiento ajustó mis pesos mediante retropropagación del error.";
  if (message.includes("que puedes hacer")) return "Puedo entrenar una red 2·4·1, visualizar sus conexiones y guardar ejemplos de conversación en memoria.";
  if (message.includes("adios")) return "Hasta luego. Mis neuronas seguirán calculando cosas sin que nadie se lo haya pedido.";
  if (message.endsWith("?")) return "Todavía no tengo una respuesta para eso. Puedes enseñarme una respuesta útil y quedará como ejemplo de memoria.";
  return "He recibido esa información. Podemos convertir conversaciones en ejemplos para mi memoria.";
}

$("train").addEventListener("click", trainBrain);

$("reset").addEventListener("click", () => {
  brain.reset();
  epoch = 0;
  loss = 1;
  trained = false;
  learnedMemory = [];
  localStorage.removeItem(MEMORY_KEY);
  angleX = 8;
  angleY = -20;
  zoom = 1;
  $("state").textContent = "Lista para aprender XOR";
  $("chatState").textContent = "sin entrenar";
  $("result").textContent = "La red todavía no ha aprendido.";
  $("infoText").textContent = "Entrénala para que aprenda el problema XOR y observa cómo cambian sus conexiones.";
  $("scene").classList.remove("trained");
  $("messages").innerHTML = '<div class="message neuronMessage">Memoria local reiniciada. La memoria global permanece para los demás usuarios. 🧠</div>';
  createNetwork();
  updateStats();
  render();
});

$("chatForm").addEventListener("submit", async event => {
  event.preventDefault();
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text || !trained) return;

  addMessage(text, "user");
  input.value = "";
  input.disabled = true;
  $("send").disabled = true;
  $("chatState").textContent = "pensando...";

  await new Promise(resolve => setTimeout(resolve, 220));
  const reply = neuronReply(text);
  addMessage(reply, "neuron");
  rememberConversation(text, reply);
  await saveGlobalMemory(text, reply);

  input.disabled = false;
  $("send").disabled = false;
  $("chatState").textContent = globalMemory.length ? "memoria conectada" : "modo local";
  input.focus();
});

for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    const input = $("chatInput");
    input.value = button.dataset.prompt || "";
    input.focus();
  });
}

const scene = $("scene");
scene.addEventListener("pointerdown", event => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  scene.setPointerCapture(event.pointerId);
});

scene.addEventListener("pointermove", event => {
  if (!dragging) return;
  angleY += (event.clientX - lastX) * 0.4;
  angleX -= (event.clientY - lastY) * 0.3;
  angleX = Math.max(-70, Math.min(70, angleX));
  lastX = event.clientX;
  lastY = event.clientY;
  render();
});

scene.addEventListener("pointerup", () => dragging = false);
scene.addEventListener("pointercancel", () => dragging = false);
scene.addEventListener("wheel", event => {
  event.preventDefault();
  zoom = Math.max(0.55, Math.min(1.5, zoom - event.deltaY * 0.001));
  render();
}, { passive: false });

createNetwork();
updateStats();
render();
loadGlobalMemory();
