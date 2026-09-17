class NeuralNetwork {
  constructor(learningRate = 0.1) {
    this.learningRate = learningRate;
    this.reset();
  }
  reset() {
    this.weights = Array.from({length: 8}, () => Math.random() * 2 - 1);
    this.outWeights = Array.from({length: 4}, () => Math.random() * 2 - 1);
    this.bias = Math.random() * 2 - 1;
    this.outBias = Math.random() * 2 - 1;
  }
  sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  predict(inputs) {
    const hidden = [];
    for (let h = 0; h < 4; h++) hidden[h] = this.sigmoid(inputs[0] * this.weights[h * 2] + inputs[1] * this.weights[h * 2 + 1] + this.bias);
    return this.sigmoid(hidden.reduce((s, v, i) => s + v * this.outWeights[i], this.outBias));
  }
  train(inputs, expected) {
    const hidden = [];
    for (let h = 0; h < 4; h++) hidden[h] = this.sigmoid(inputs[0] * this.weights[h * 2] + inputs[1] * this.weights[h * 2 + 1] + this.bias);
    const output = this.sigmoid(hidden.reduce((s, v, i) => s + v * this.outWeights[i], this.outBias));
    const error = expected - output;
    const deltaOut = error * output * (1 - output);
    for (let h = 0; h < 4; h++) {
      this.outWeights[h] += hidden[h] * deltaOut * this.learningRate;
      const deltaHidden = deltaOut * this.outWeights[h] * hidden[h] * (1 - hidden[h]);
      this.weights[h * 2] += inputs[0] * deltaHidden * this.learningRate;
      this.weights[h * 2 + 1] += inputs[1] * deltaHidden * this.learningRate;
    }
    this.outBias += deltaOut * this.learningRate;
    this.bias += deltaOut * this.learningRate * 0.5;
  }
  fit(data, epochs = 1000) { for (let e = 0; e < epochs; e++) for (const [x, y] of data) this.train(x, y); }
}

const data = [[[0,0],0],[[0,1],0],[[1,0],0],[[1,1],1]];
const brain = new NeuralNetwork();
const $ = id => document.getElementById(id);
let epoch = 0, angleX = 8, angleY = -18, zoom = 1, rotating = false;

function updatePrediction() {
  const x1 = Number($("x1").value) ? 1 : 0, x2 = Number($("x2").value) ? 1 : 0;
  const p = brain.predict([x1, x2]);
  document.querySelectorAll('.inputs .neuron').forEach((n,i) => n.textContent = i ? x2 : x1);
  $("prediction").textContent = p.toFixed(3);
  $("classification").textContent = p >= .5 ? "Resultado: 1" : "Resultado: 0";
  document.querySelector('.outputs .neuron').textContent = p.toFixed(3);
}
function render() {
  const net = $("network3d");
  net.style.transform = `translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
  if (rotating) { angleY += .35; requestAnimationFrame(render); }
}
$("train").addEventListener("click", () => { brain.fit(data, 1000); epoch += 1000; $("epoch").textContent = epoch.toLocaleString(); updatePrediction(); render(); });
$("reset").addEventListener("click", () => { brain.reset(); epoch=0; $("epoch").textContent='0'; angleX=8; angleY=-18; zoom=1; updatePrediction(); render(); });
$("rotate").addEventListener("click", e => { rotating=!rotating; e.currentTarget.textContent=rotating?'Detener rotación':'Rotación automática'; if(rotating) render(); });
$("predict").addEventListener("click", updatePrediction);
$("x1").addEventListener("input", updatePrediction); $("x2").addEventListener("input", updatePrediction);
const scene=$("scene"); let dragging=false,lastX=0,lastY=0;
scene.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;scene.setPointerCapture(e.pointerId);});
scene.addEventListener('pointermove',e=>{if(!dragging)return;angleY+=(e.clientX-lastX)*.45;angleX-=(e.clientY-lastY)*.35;angleX=Math.max(-70,Math.min(70,angleX));lastX=e.clientX;lastY=e.clientY;render();});
scene.addEventListener('pointerup',()=>dragging=false); scene.addEventListener('pointercancel',()=>dragging=false);
scene.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.55,Math.min(1.5,zoom-e.deltaY*.001));render();},{passive:false});
updatePrediction(); render();
