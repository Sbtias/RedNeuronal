class NeuralNetwork {
  constructor(learningRate = 0.15) {
    this.learningRate = learningRate;
    this.reset();
  }

  reset() {
    this.weights = Array.from({ length: 2 }, () => Math.random() * 2 - 1);
    this.bias = Math.random() * 2 - 1;
  }

  sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  predict(inputs) {
    return this.sigmoid(inputs[0] * this.weights[0] + inputs[1] * this.weights[1] + this.bias);
  }

  train(inputs, expected) {
    const output = this.predict(inputs);
    const error = expected - output;
    const delta = error * output * (1 - output);
    this.weights[0] += inputs[0] * delta * this.learningRate;
    this.weights[1] += inputs[1] * delta * this.learningRate;
    this.bias += delta * this.learningRate;
  }
}

const trainingData = [
  [[0, 0], 0], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]
];

const brain = new NeuralNetwork();
const $ = id => document.getElementById(id);

const SUPABASE_URL = "https://otdfvaufiqwlwmaqsljs.supabase.co";
const SUPABASE_KEY = "sb_publishable_jC98NCK4bQeRRYmlZL9CVw_tALEihhJ";
const supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

const MEMORY_KEY = "neural3d_chat_memory_v2";
let learnedMemory = loadLocalMemory();
let globalMemory = [];
let epoch = 0;
let angleX = 12;
let angleY = -18;
let zoom = 1;
let dragging = false;
let lastX = 0;
let lastY = 0;
let trained = false;

function loadLocalMemory() {
  try {
    const saved = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.slice(-100) : [];
  } catch {
    return [];
  }
}

function saveLocalMemory() {
  localStorage.setItem(MEMORY_KEY, JSON.stringify(learnedMemory.slice(-100)));
}

function normalize(text) {
  return text
    .toLowerCase()
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
    const { error } = await supabaseClient.from("neural_memory").insert({
      question: cleanQuestion,
      answer: cleanAnswer
    });
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

function render() {
  const neuron = $("neuron3d");
  if (!neuron) return;
  neuron.style.transform = `translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
}

function trainBrain() {
  const button = $("train");
  button.disabled = true;
  $("state").textContent = "Aprendiendo...";
  $("chatState").textContent = "aprendiendo";

  let steps = 0;
  const learn = () => {
    for (let i = 0; i < 250; i++) {
      for (const [inputs, expected] of trainingData) brain.train(inputs, expected);
      epoch++;
    }
    $("epoch").textContent = epoch.toLocaleString();

    steps++;
    if (steps < 12) {
      requestAnimationFrame(learn);
      return;
    }

    trained = true;
    $("state").textContent = "Entrenada";
    $("chatState").textContent = "memoria conectada";
    $("result").textContent = "La neurona está lista para aprender del chat.";
    $("infoText").textContent = "Lo que aprende se guarda localmente y, si Supabase está conectado, también en la memoria global.";
    $("scene").classList.add("trained");
    button.disabled = false;
    loadGlobalMemory();
    $("chatInput").focus();
  };

  learn();
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
  if (!trained) return "Todavía no estoy entrenada. Pulsa «Entrenar neurona» primero. 🧠";

  const learned = findLearnedAnswer(text);
  if (learned) return learned;

  if (message.includes("hola") || message.includes("buenas")) return "Hola. Mis conexiones están activas. 🧠";
  if (message.includes("quien eres")) return "Soy una pequeña neurona artificial. Tengo pesos, un sesgo, una función de activación y una memoria compartida.";
  if (message.includes("como estas")) return "Funcionando. No tengo emociones, pero mis cálculos están despiertos.";
  if (message.includes("entrenar") || message.includes("aprend")) return "El entrenamiento ajustó mis pesos. Además, cada conversación puede dejar un ejemplo en mi memoria.";
  if (message.includes("que puedes hacer")) return "Puedo procesar mensajes, recordar ejemplos y compartirlos con otros usuarios mediante la memoria global.";
  if (message.includes("que sabes") || message.includes("sabes sobre")) return "Conozco lo programado y los ejemplos que están guardados en mi memoria local o global.";
  if (message.includes("adios")) return "Hasta luego. Mis dendritas seguirán aquí, dramáticamente inmóviles.";
  if (message.endsWith("?")) return "Todavía no tengo una respuesta para eso. Si conversamos y aparece una respuesta útil, podré guardarla como ejemplo. 🧠";
  return "He recibido esa información. La conversación puede convertirse en un nuevo ejemplo para mi memoria.";
}

$("train").addEventListener("click", trainBrain);

$("reset").addEventListener("click", () => {
  brain.reset();
  epoch = 0;
  trained = false;
  learnedMemory = [];
  localStorage.removeItem(MEMORY_KEY);
  angleX = 12;
  angleY = -18;
  zoom = 1;
  $("epoch").textContent = "0";
  $("state").textContent = "Lista para aprender";
  $("chatState").textContent = "sin entrenar";
  $("result").textContent = "La neurona todavía no ha aprendido.";
  $("infoText").textContent = "Entrénala una vez y después conversa con ella.";
  $("scene").classList.remove("trained");
  $("messages").innerHTML = '<div class="message neuronMessage">Memoria local reiniciada. La memoria global permanece para los demás usuarios. 🧠</div>';
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

render();
loadGlobalMemory();
