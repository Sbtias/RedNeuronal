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
let epoch = 0;
let angleX = 12;
let angleY = -18;
let zoom = 1;
let rotating = false;
let dragging = false;
let lastX = 0;
let lastY = 0;
let trained = false;

// Memoria de conversación. Se guarda en este dispositivo.
const MEMORY_KEY = "neural3d_chat_memory_v1";
let learnedMemory = loadMemory();

function loadMemory() {
  try {
    const saved = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.slice(-100) : [];
  } catch {
    return [];
  }
}

function saveMemory() {
  localStorage.setItem(MEMORY_KEY, JSON.stringify(learnedMemory.slice(-100)));
}

function normalize(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\\u0300-\\u036f]/g, "")
    .replace(/[^a-z0-9ñ?¿! ]/g, "")
    .replace(/\\s+/g, " ")
    .trim();
}

function learnFromConversation(question, answer) {
  const q = normalize(question);
  if (!q || !answer) return;

  const existing = learnedMemory.find(item => item.question === q);
  if (existing) {
    existing.answer = answer;
    existing.uses = (existing.uses || 0) + 1;
  } else {
    learnedMemory.push({ question: q, answer, uses: 1 });
  }
  learnedMemory = learnedMemory.slice(-100);
  saveMemory();
}

function findLearnedAnswer(text) {
  const q = normalize(text);
  const exact = learnedMemory.find(item => item.question === q);
  if (exact) return exact.answer;

  const words = q.split(" ").filter(word => word.length > 3);
  if (words.length < 2) return null;

  let best = null;
  let bestScore = 0;
  for (const item of learnedMemory) {
    const knownWords = new Set(item.question.split(" "));
    const score = words.filter(word => knownWords.has(word)).length / words.length;
    if (score > bestScore && score >= 0.6) {
      best = item.answer;
      bestScore = score;
    }
  }
  return best;
}

function render() {
  const neuron = $("neuron3d");
  if (!neuron) return;
  neuron.style.transform = `translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
  if (rotating) {
    angleY += 0.25;
    requestAnimationFrame(render);
  }
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
    $("state").textContent = "Entrenamiento completado";
    $("chatState").textContent = "lista para aprender";
    $("result").textContent = "La neurona está lista para aprender del chat.";
    $("infoText").textContent = "Cada conversación puede enseñarle nuevas respuestas en este dispositivo.";
    $("scene").classList.add("trained");
    button.disabled = false;
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
  if (message.includes("quien eres")) return "Soy una pequeña neurona artificial. Tengo pesos, un sesgo, una función de activación y ahora también memoria de conversación.";
  if (message.includes("como estas")) return "Funcionando. No tengo emociones, pero mis cálculos están despiertos.";
  if (message.includes("entrenar") || message.includes("aprend")) return "El entrenamiento ajustó mis pesos. Además, puedo guardar nuevas conversaciones como memoria para responder mejor después.";
  if (message.includes("que puedes hacer")) return "Puedo procesar mensajes, recordar respuestas aprendidas en este dispositivo y mostrar visualmente mi estructura neuronal.";
  if (message.includes("que sabes") || message.includes("sabes sobre")) return "Sé lo que viene programado y lo que he aprendido durante las conversaciones guardadas en esta memoria.";
  if (message.includes("adios")) return "Hasta luego. Mis dendritas seguirán aquí, dramáticamente inmóviles.";
  if (message.endsWith("?")) return "Todavía no conozco esa respuesta. Si me enseñas una respuesta concreta, puedo recordarla en este dispositivo. 🧠";
  return "He recibido esa información. Puedo recordarla como parte de nuestra conversación, aunque todavía no tengo conocimiento general como una IA grande.";
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
  $("infoText").textContent = "Presiona Entrenar neurona y el modelo aprenderá automáticamente.";
  $("scene").classList.remove("trained");
  $("messages").innerHTML = '<div class="message neuronMessage">Memoria reiniciada. Entréname y podemos volver a empezar. 🧠</div>';
  render();
});

$("rotate").addEventListener("click", event => {
  rotating = !rotating;
  event.currentTarget.textContent = rotating ? "Detener rotación" : "Rotación automática";
  if (rotating) render();
});

$("chatForm").addEventListener("submit", event => {
  event.preventDefault();
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text || !trained) return;

  addMessage(text, "user");
  input.value = "";
  input.disabled = true;
  $("send").disabled = true;

  setTimeout(() => {
    const reply = neuronReply(text);
    addMessage(reply, "neuron");
    // La respuesta generada queda asociada a la pregunta para futuras conversaciones.
    learnFromConversation(text, reply);
    input.disabled = false;
    $("send").disabled = false;
    input.focus();
  }, 280);
});

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
