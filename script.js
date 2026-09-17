class NeuralNetwork {
  constructor(learningRate = 0.1) {
    this.weights = [Math.random() * 2 - 1, Math.random() * 2 - 1];
    this.bias = Math.random() * 2 - 1;
    this.learningRate = learningRate;
    this.trained = false;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(inputs) {
    const total = inputs[0] * this.weights[0] + inputs[1] * this.weights[1] + this.bias;
    return this.sigmoid(total);
  }

  train(inputs, expected) {
    const prediction = this.predict(inputs);
    const error = expected - prediction;
    const adjustment = error * prediction * (1 - prediction);

    this.weights[0] += inputs[0] * adjustment * this.learningRate;
    this.weights[1] += inputs[1] * adjustment * this.learningRate;
    this.bias += adjustment * this.learningRate;
  }

  fit(data, epochs = 10000) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const [inputs, expected] of data) this.train(inputs, expected);
    }
    this.trained = true;
  }
}

const data = [
  [[0, 0], 0],
  [[0, 1], 0],
  [[1, 0], 0],
  [[1, 1], 1]
];

const brain = new NeuralNetwork();
const $ = (id) => document.getElementById(id);

function updateNetwork() {
  const x1 = Number($("x1").value) ? 1 : 0;
  const x2 = Number($("x2").value) ? 1 : 0;
  const prediction = brain.predict([x1, x2]);

  $("input1").textContent = x1;
  $("input2").textContent = x2;
  $("output").textContent = prediction.toFixed(3);
  $("prediction").textContent = prediction.toFixed(3);
  $("classification").textContent = prediction >= 0.5 ? "Resultado: 1" : "Resultado: 0";
  $("barFill").style.width = `${prediction * 100}%`;
}

function renderTable() {
  $("table").innerHTML = `
    <table>
      <thead><tr><th>Entrada 1</th><th>Entrada 2</th><th>Esperado</th><th>Predicción</th></tr></thead>
      <tbody>
        ${data.map(([inputs, expected]) => {
          const prediction = brain.predict(inputs);
          return `<tr><td>${inputs[0]}</td><td>${inputs[1]}</td><td>${expected}</td><td>${prediction.toFixed(3)}</td></tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

$("predict").addEventListener("click", updateNetwork);
$("train").addEventListener("click", () => {
  brain.fit(data, 10000);
  $("status").textContent = "Red entrenada con 10,000 épocas";
  renderTable();
  updateNetwork();
});

$("x1").addEventListener("input", updateNetwork);
$("x2").addEventListener("input", updateNetwork);

renderTable();
updateNetwork();
