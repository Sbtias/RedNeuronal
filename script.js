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

function render() {
  const neuron = $("neuron3d");
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
    $("chatState").textContent = "lista para hablar";
    $("result").textContent = "La neurona ha aprendido el patrón.";
    $("infoText").textContent = "Ahora puedes escribirle en el chat.";
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
  const message = text.toLowerCase().trim();
  if (!trained) return "Todavía no estoy entrenada. Pulsa «Entrenar neurona» primero. 🧠";
  if (message.includes("hola") || message.includes("buenas")) return "Hola. Mis conexiones están activas. 🧠";
  if (message.includes("quien eres") || message.includes("quién eres")) return "Soy una pequeña neurona artificial. Tengo pesos, un sesgo y una función de activación.";
  if (message.includes("como estas") || message.includes("cómo estás")) return "Funcionando. No tengo emociones, pero mis cálculos están perfectamente despiertos.";
  if (message.includes("entrenar") || message.includes("aprend")) return "El entrenamiento ajustó mis pesos para reconocer el patrón que me enseñaron.";
  if (message.includes("que puedes hacer") || message.includes("qué puedes hacer")) return "Puedo responder mensajes sencillos y mostrar visualmente mi estructura neuronal.";
  if (message.includes("adios") || message.includes("adiós")) return "Hasta luego. Mis dendritas seguirán aquí, dramáticamente inmóviles.";
  if (message.endsWith("?")) return "Buena pregunta. Por ahora soy una neurona pequeña, así que todavía estoy aprendiendo a responder cosas más complejas.";
  return "Procesando... Esa entrada pasó por mis conexiones. Todavía estoy aprendiendo a conversar mejor. 🧠";
}

$("train").addEventListener("click", trainBrain);

$("reset").addEventListener("click", () => {
  brain.reset();
  epoch = 0;
  trained = false;
  angleX = 12;
  angleY = -18;
  zoom = 1;
  $("epoch").textContent = "0";
  $("state").textContent = "Lista para aprender";
  $("chatState").textContent = "sin entrenar";
  $("result").textContent = "La neurona todavía no ha aprendido.";
  $("infoText").textContent = "Presiona Entrenar neurona y el modelo aprenderá automáticamente.";
  $("scene").classList.remove("trained");
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
  if (!text) return;
  addMessage(text, "user");
  input.value = "";
  setTimeout(() => addMessage(neuronReply(text), "neuron"), 280);
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
