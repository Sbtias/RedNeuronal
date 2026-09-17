class NeuralNetwork {
  constructor() { this.learningRate = 0.7; this.reset(); }

  reset() {
    this.weightsInputHidden = Array.from({length:2}, () => Array.from({length:4}, () => Math.random()*2-1));
    this.biasHidden = Array.from({length:4}, () => Math.random()*2-1);
    this.weightsHiddenOutput = Array.from({length:4}, () => Math.random()*2-1);
    this.biasOutput = Math.random()*2-1;
  }
  sigmoid(x) { return 1/(1+Math.exp(-Math.max(-60,Math.min(60,x)))); }
  forward(inputs) {
    const hidden=this.biasHidden.map((b,j)=>this.sigmoid(b+inputs.reduce((s,v,i)=>s+v*this.weightsInputHidden[i][j],0)));
    const output=this.sigmoid(this.biasOutput+hidden.reduce((s,v,j)=>s+v*this.weightsHiddenOutput[j],0));
    return {hidden,output};
  }
  train(inputs,expected) {
    const {hidden,output}=this.forward(inputs);
    const od=(expected-output)*output*(1-output);
    const hd=hidden.map((v,j)=>v*(1-v)*this.weightsHiddenOutput[j]*od);
    for(let j=0;j<4;j++) this.weightsHiddenOutput[j]+=hidden[j]*od*this.learningRate;
    this.biasOutput+=od*this.learningRate;
    for(let i=0;i<2;i++) for(let j=0;j<4;j++) this.weightsInputHidden[i][j]+=inputs[i]*hd[j]*this.learningRate;
    for(let j=0;j<4;j++) this.biasHidden[j]+=hd[j]*this.learningRate;
    return (expected-output)**2;
  }
  predict(inputs){return this.forward(inputs).output;}
}

const trainingData=[[[0,0],0],[[0,1],1],[[1,0],1],[[1,1],0]];
const brain=new NeuralNetwork();
const $=id=>document.getElementById(id);
const SUPABASE_URL="https://otdfvaufiqwlwmaqsljs.supabase.co";
const SUPABASE_KEY="sb_publishable_jC98NCK4bQeRRYmlZ9CVw_tALEihhJ";
const supabaseClient=window.supabase.createClient(SUPABASE_URL,SUPABASE_KEY);
const MEMORY_KEY="neural3d_chat_memory_v4";
let learnedMemory=loadLocalMemory(),globalMemory=[],epoch=0,loss=1,trained=false;
let angleX=8,angleY=-20,zoom=1,dragging=false,lastX=0,lastY=0;

const layerConfig=[
  {name:"ENTRADA",count:2,x:-220,labels:["x₁","x₂"]},
  {name:"OCULTA",count:4,x:0,labels:["h₁","h₂","h₃","h₄"]},
  {name:"SALIDA",count:1,x:220,labels:["y"]}
];

function loadLocalMemory(){try{const s=JSON.parse(localStorage.getItem(MEMORY_KEY)||"[]");return Array.isArray(s)?s.slice(-100):[]}catch{return[]}}
function saveLocalMemory(){localStorage.setItem(MEMORY_KEY,JSON.stringify(learnedMemory.slice(-100)))}
function normalize(t){return t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/[^a-z0-9ñ?¿! ]/g,"").replace(/\s+/g," ").trim()}
function findLearnedAnswer(text){const q=normalize(text),all=[...globalMemory,...learnedMemory],exact=all.find(x=>normalize(x.question)===q);if(exact)return exact.answer;const words=q.split(" ").filter(w=>w.length>3);let best=null,scoreBest=0;for(const item of all){const known=new Set(normalize(item.question).split(" "));const score=words.length?words.filter(w=>known.has(w)).length/words.length:0;if(score>scoreBest&&score>=.6){best=item.answer;scoreBest=score}}return best}
async function loadGlobalMemory(){try{const {data,error}=await supabaseClient.from("neural_memory").select("question,answer").order("created_at",{ascending:false}).limit(500);if(error)throw error;globalMemory=Array.isArray(data)?data:[];$('chatState').textContent=trained?"memoria conectada":"sin entrenar"}catch(e){globalMemory=[];$('chatState').textContent=trained?"modo local":"sin entrenar"}}
async function saveGlobalMemory(question,answer){try{const q=question.trim().slice(0,300),a=answer.trim().slice(0,600);if(!q||!a)return;const {error}=await supabaseClient.from("neural_memory").insert({question:q,answer:a});if(error)throw error;globalMemory.unshift({question:q,answer:a});globalMemory=globalMemory.slice(0,500)}catch(e){console.warn("Memoria global:",e.message)}}
function rememberConversation(q,a){const n=normalize(q),old=learnedMemory.find(x=>normalize(x.question)===n);if(old){old.answer=a;old.uses=(old.uses||0)+1}else learnedMemory.push({question:q.trim(),answer:a,uses:1});learnedMemory=learnedMemory.slice(-100);saveLocalMemory()}

function nodeY(i,count){return (i-(count-1)/2)*92}
function getWeight(fl,fi,ti){return fl===0?brain.weightsInputHidden[fi][ti]:brain.weightsHiddenOutput[fi]}
function createNetwork(){
  const nodes=$("nodes"),connections=$("connections");nodes.innerHTML="";connections.innerHTML="";
  const positions=[];
  layerConfig.forEach((layer,li)=>{
    const title=document.createElement("div");title.className="layerLabel";title.textContent=layer.name;title.style.transform=`translate3d(${layer.x-34}px,${-((layer.count-1)*92)/2-82}px,0)`;nodes.appendChild(title);
    const lp=[];
    for(let i=0;i<layer.count;i++){
      const y=nodeY(i,layer.count),z=li===1?(i%2? -30:30):0,n=document.createElement("button");
      n.className=`node layer-${li}`;n.dataset.layer=li;n.dataset.index=i;n.textContent=layer.labels[i];n.style.transform=`translate3d(${layer.x}px,${y}px,${z}px)`;
      n.type="button";n.title=`${layer.name} · ${layer.labels[i]}`;n.addEventListener("click",()=>selectNode(li,i));nodes.appendChild(n);lp.push({x:layer.x,y,z,element:n});
    }positions.push(lp);
  });
  for(let l=0;l<positions.length-1;l++)for(let a=0;a<positions[l].length;a++)for(let b=0;b<positions[l+1].length;b++){
    const line=document.createElement("div");line.className="connection";line.dataset.fromLayer=l;line.dataset.from=a;line.dataset.to=b;connections.appendChild(line);positionConnection(line,positions[l][a],positions[l+1][b],getWeight(l,a,b));
  }
  window._networkPositions=positions;
}
function positionConnection(line,a,b,w){const dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z,len=Math.hypot(dx,dy,dz),ang=Math.atan2(dy,dx)*180/Math.PI,tilt=Math.atan2(dz,Math.hypot(dx,dy))*180/Math.PI,s=Math.min(1,Math.abs(w));line.style.width=`${len}px`;line.style.height=`${1.2+s*2.8}px`;line.style.opacity=`${.14+s*.65}`;line.style.transform=`translate3d(${a.x}px,${a.y}px,${a.z}px) rotateZ(${ang}deg) rotateY(${-tilt}deg)`;line.classList.toggle("positive",w>=0);line.classList.toggle("negative",w<0)}
function updateNetworkVisuals(){document.querySelectorAll(".connection").forEach(line=>{const l=+line.dataset.fromLayer,a=+line.dataset.from,b=+line.dataset.to;positionConnection(line,window._networkPositions[l][a],window._networkPositions[l+1][b],getWeight(l,a,b))})}
function selectNode(layer,index){document.querySelectorAll(".node").forEach(n=>n.classList.remove("selected"));const node=document.querySelector(`.node[data-layer="${layer}"][data-index="${index}"]`);if(node)node.classList.add("selected");const label=layerConfig[layer].labels[index];$('nodeInfo').textContent=layer===0?`${label} · entrada`:layer===2?`${label} · salida`:`${label} · neurona oculta`;}
function render(){$("network3d").style.transform=`translate(-50%,-50%) scale(${zoom}) rotateX(${angleX}deg) rotateY(${angleY}deg)`}
function updateStats(sample=trainingData[1]){$("epoch").textContent=epoch.toLocaleString();$("loss").textContent=loss.toFixed(4);$("prediction").textContent=brain.predict(sample[0]).toFixed(3)}
function trainBrain(){const btn=$("train");btn.disabled=true;$("reset").disabled=true;$("state").textContent="Aprendiendo XOR...";$("chatState").textContent="entrenando";let frame=0;const run=()=>{for(let i=0;i<100;i++){let total=0;for(const [x,y] of trainingData)total+=brain.train(x,y);loss=total/4;epoch++}updateStats(trainingData[epoch%4]);updateNetworkVisuals();frame++;if(frame<20)return requestAnimationFrame(run);trained=true;$("state").textContent="Entrenada · XOR aprendido";$("chatState").textContent="memoria conectada";$("result").textContent="Red 2 → 4 → 1 entrenada.";$("infoText").textContent="Selecciona una neurona para inspeccionarla. Las conexiones muestran la fuerza de sus pesos.";$("scene").classList.add("trained");btn.disabled=false;$("reset").disabled=false;loadGlobalMemory();$("chatInput").focus()};run()}
function addMessage(text,type){const m=document.createElement("div");m.className=`message ${type==="user"?"userMessage":"neuronMessage"}`;m.textContent=text;$("messages").appendChild(m);$("messages").scrollTop=$("messages").scrollHeight}
function neuronReply(text){const m=normalize(text);if(!trained)return"Todavía no estoy entrenada. Pulsa «Entrenar red» primero. 🧠";const learned=findLearnedAnswer(text);if(learned)return learned;if(m.includes("hola")||m.includes("buenas"))return"Hola. Mis conexiones están activas. 🧠";if(m.includes("quien eres"))return"Soy una red neuronal artificial 2·4·1 entrenada con XOR.";if(m.includes("que aprendiste")||m.includes("que sabes"))return`Aprendí XOR. Error actual: ${loss.toFixed(4)}.`;if(m.includes("que puedes hacer"))return"Puedo entrenar, visualizar pesos, seleccionar neuronas y guardar ejemplos en memoria.";if(m.endsWith("?"))return"Todavía no tengo una respuesta para eso. Puedes enseñarme un ejemplo y lo guardaré en memoria.";return"He recibido esa información. Puede convertirse en un ejemplo para mi memoria."}

$("train").addEventListener("click",trainBrain);
$("reset").addEventListener("click",()=>{brain.reset();epoch=0;loss=1;trained=false;learnedMemory=[];localStorage.removeItem(MEMORY_KEY);$("state").textContent="Lista para aprender XOR";$("chatState").textContent="sin entrenar";$("result").textContent="La red todavía no ha aprendido.";$("infoText").textContent="Entrénala para observar cómo cambian sus conexiones.";$("scene").classList.remove("trained");$("messages").innerHTML='<div class="message neuronMessage">Memoria local reiniciada. La memoria global permanece. 🧠</div>';createNetwork();updateStats();render()});
$("chatForm").addEventListener("submit",async e=>{e.preventDefault();const input=$("chatInput"),text=input.value.trim();if(!text||!trained)return;addMessage(text,"user");input.value="";input.disabled=true;$("send").disabled=true;$("chatState").textContent="pensando...";await new Promise(r=>setTimeout(r,180));const reply=neuronReply(text);addMessage(reply,"neuron");rememberConversation(text,reply);await saveGlobalMemory(text,reply);input.disabled=false;$("send").disabled=false;$("chatState").textContent=globalMemory.length?"memoria conectada":"modo local";input.focus()});
for(const b of document.querySelectorAll("[data-prompt]"))b.addEventListener("click",()=>{$("chatInput").value=b.dataset.prompt;$('chatInput').focus()});
const scene=$("scene");scene.addEventListener("pointerdown",e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;scene.setPointerCapture(e.pointerId)});scene.addEventListener("pointermove",e=>{if(!dragging)return;angleY+=(e.clientX-lastX)*.4;angleX-=(e.clientY-lastY)*.3;angleX=Math.max(-70,Math.min(70,angleX));lastX=e.clientX;lastY=e.clientY;render()});scene.addEventListener("pointerup",()=>dragging=false);scene.addEventListener("pointercancel",()=>dragging=false);scene.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.55,Math.min(1.5,zoom-e.deltaY*.001));render()},{passive:false});
createNetwork();updateStats();render();loadGlobalMemory();
