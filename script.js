/* RDio Neural Core v10.0 · aprendizaje real + respuestas dinámicas */
'use strict';

const $=id=>document.getElementById(id);
const SUPABASE_URL='https://otdfvaufiqwlwmaqsljs.supabase.co';
const SUPABASE_KEY='sb_publishable_f0BM6a9YHx-jT1Rn1f9h6A_hj1BQ-yO';
const STATE_KEY='__RDIO_STATE_V7__';
const LOCAL_MEMORY_KEY='rdio_memory_v7';

class NeuralNetwork{
  constructor(layers=[2,48,32,24,12,1]){this.layers=layers.slice();this.learningRate=.14;this.reset()}
  reset(){this.weights=[];this.biases=[];for(let l=0;l<this.layers.length-1;l++){const input=this.layers[l],output=this.layers[l+1],scale=Math.sqrt(2/input);this.weights.push(Array.from({length:output},()=>Array.from({length:input},()=>((Math.random()*2-1)*scale))));this.biases.push(Array(output).fill(0))}}
  sigmoid(x){const z=Math.max(-40,Math.min(40,x));return 1/(1+Math.exp(-z))}
  forward(input,trace=false){let a=input.map(Number),acts=[a.slice()];for(let l=0;l<this.weights.length;l++){a=this.weights[l].map((row,j)=>this.sigmoid(this.biases[l][j]+row.reduce((sum,w,i)=>sum+w*a[i],0)));acts.push(a.slice())}return trace?acts:a[0]}
  train(input,target){const acts=this.forward(input,true),y=Number(Array.isArray(target)?target[0]:target),last=acts.length-1,d=Array.from({length:this.weights.length},()=>[]);d[last-1]=acts[last].map(v=>(y-v)*v*(1-v));for(let l=this.weights.length-2;l>=0;l--)d[l]=acts[l+1].map((v,j)=>{let e=0;for(let k=0;k<this.weights[l+1].length;k++)e+=this.weights[l+1][k][j]*d[l+1][k];return e*v*(1-v)});for(let l=0;l<this.weights.length;l++)for(let j=0;j<this.weights[l].length;j++){for(let i=0;i<this.weights[l][j].length;i++)this.weights[l][j][i]+=this.learningRate*d[l][j]*acts[l][i];this.biases[l][j]+=this.learningRate*d[l][j]}return(y-acts[last][0])**2}
  predict(input){return this.forward(input)}
  serialize(){return{layers:this.layers,weights:this.weights,biases:this.biases,learningRate:this.learningRate}}
  restore(s){if(!s?.layers||!s?.weights||!s?.biases)throw Error('Estado inválido');this.layers=s.layers.map(Number);this.weights=s.weights;this.biases=s.biases;this.learningRate=Number(s.learningRate)||.1}
}

const XOR=[{input:[0,0],target:[0]},{input:[0,1],target:[1]},{input:[1,0],target:[1]},{input:[1,1],target:[0]}];

const KNOWLEDGE=[
 ['hola',['Hola. 🧠','Hola, aquí estoy.','Buenas. Soy RDio. 🧠']],
 ['que eres',['Soy RDio, una red neuronal artificial con memoria persistente.','Soy RDio, un sistema neuronal experimental que puede aprender.','Soy RDio. Tengo memoria y un pequeño modelo neuronal.']],
 ['quien eres',['Soy RDio. 🧠','RDio. Una red neuronal experimental con memoria persistente.','Me llamo RDio y estoy aquí para responderte.']],
 ['que es una red neuronal',['Una red neuronal es un modelo matemático que aprende patrones ajustando sus pesos.','Es un sistema de capas que transforma datos y modifica sus pesos durante el entrenamiento.']],
 ['como funciona una red neuronal',['Los datos pasan por varias capas y el entrenamiento ajusta los pesos para reducir el error.','Recibo entradas, proceso señales por mis capas y ajusto mis pesos para aprender patrones.']],
 ['que es javascript',['JavaScript es un lenguaje usado para crear aplicaciones e interfaces web interactivas.','JavaScript permite programar comportamiento e interactividad en páginas y aplicaciones web.']],
 ['que es html',['HTML define la estructura y el contenido de una página web.','HTML organiza los elementos que forman una página web.']],
 ['que es css',['CSS controla la apariencia y distribución visual de una página web.','CSS se encarga del diseño, estilos y presentación de los elementos web.']],
 ['que es python',['Python es un lenguaje de programación de propósito general conocido por su sintaxis sencilla.','Python es un lenguaje versátil usado en aplicaciones, automatización, datos e inteligencia artificial.']],
 ['que es supabase',['Supabase ofrece PostgreSQL, autenticación y APIs para aplicaciones.','Supabase es una plataforma que proporciona servicios backend basados en PostgreSQL.']],
 ['que puedes hacer',['Puedo conversar, recordar conocimientos que me enseñes y aprender patrones.','Puedo responder, consultar mi memoria y mejorar mi modelo con entrenamiento.']],
 ['como estas',['Funcionando correctamente. 🧠','Activa y lista para trabajar.','Todo en orden por aquí. 🧠']],
 ['gracias',['De nada. 🧠','Con gusto.','Para eso estoy.']],
 ['adios',['Hasta luego. 🧠','Nos vemos.','Hasta la próxima.']]
];

const FALLBACKS=[
 'Todavía no conozco esa respuesta.',
 'No tengo suficiente conocimiento sobre eso todavía.',
 'Esa pregunta está fuera de lo que conozco por ahora.',
 'No tengo una respuesta fiable para eso todavía.'
];

const brain=new NeuralNetwork();
let db=null,memory=[],epoch=0,loss=1,trained=false,learningEvents=0,isTraining=false,saveTimer=null,selectedNode=null;
let recentReplies=[];

function normalize(text){
 return String(text??'').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'')
 .replace(/\bplus\b/g,'+').replace(/\bmas\b/g,'+').replace(/\bmenos\b/g,'-')
 .replace(/\bpor\b/g,'*').replace(/\bveces\b/g,'*').replace(/\bdividido\s+entre\b/g,'/')
 .replace(/[^a-z0-9ñ+*/().?\- ]/g,' ').replace(/\s+/g,' ').trim()
}

function words(text){return normalize(text).split(' ').filter(w=>w.length>1)}

function similarity(a,b){
 const A=words(a),B=words(b);if(!A.length||!B.length)return 0;
 const sa=new Set(A),sb=new Set(B);let common=0;for(const w of sa)if(sb.has(w))common++;
 const na=normalize(a),nb=normalize(b),exact=na===nb?1:0,phrase=(na.includes(nb)||nb.includes(na))?.3:0;
 const coverage=common/Math.max(sa.size,sb.size),jaccard=common/Math.max(1,sa.size+sb.size-common);
 return Math.min(1,coverage*.45+jaccard*.25+common/(sa.size+sb.size)*.1+exact*.12+phrase)
}

function findMemory(query){
 let best=null,bestScore=0;
 for(const item of memory){
  if(item.learned!==true)continue;
  const s=similarity(query,item.question);
  if(s>bestScore){bestScore=s;best=item}
 }
 if(best&&bestScore>=.42){
  best.hits=(Number(best.hits)||0)+1;saveLocal();return best
 }
 return null
}

function loadLocal(){try{const v=JSON.parse(localStorage.getItem(LOCAL_MEMORY_KEY)||'[]');return Array.isArray(v)?v.slice(0,2000):[]}catch{return[]}}
function saveLocal(){try{localStorage.setItem(LOCAL_MEMORY_KEY,JSON.stringify(memory.slice(0,2000)))}catch{}}
function setStatus(text){if($('dbState'))$('dbState').textContent=text;if($('chatState'))$('chatState').textContent=text}

function update(progress=0){
 const set=(id,v)=>{const e=$(id);if(e)e.textContent=v};
 set('epoch',epoch.toLocaleString());set('loss',Number(loss).toFixed(4));set('prediction',brain.predict([0,1]).toFixed(3));
 set('memoryCount',memory.length.toLocaleString());set('learningEvents',learningEvents.toLocaleString());set('drawerMemoryCount',memory.length.toLocaleString());set('drawerLearningEvents',learningEvents.toLocaleString());set('architecture',brain.layers.join(' · '));
 set('state',trained?'Aprendizaje adaptativo activo':'Lista para entrenar');
 if($('progressBar'))$('progressBar').style.width=`${Math.max(0,Math.min(100,progress))}%`;
 renderMemoryTable();if(typeof networkDirty!=='undefined')networkDirty=true;if(typeof networkCache!=='undefined')networkCache=null;invalidateNetwork()
}

async function timeout(p,ms=8000){return Promise.race([p,new Promise((_,r)=>setTimeout(()=>r(Error('Tiempo de espera agotado')),ms))])}

async function initDb(){
 setStatus('conectando…');if(!window.supabase?.createClient){setStatus('modo local');return false}
 try{db=window.supabase.createClient(SUPABASE_URL,SUPABASE_KEY,{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false}});return true}
 catch(e){console.error(e);setStatus('modo local');return false}
}

async function load(){
 memory=loadLocal();update();if(!await initDb())return;
 try{
  const r=await timeout(db.from('neural_memory').select('question,answer,created_at').order('created_at',{ascending:false}).limit(500));
  if(r.error)throw r.error;
  const rows=r.data||[],stateRow=rows.find(x=>x.question===STATE_KEY),remote=rows.filter(x=>x.question!==STATE_KEY);
  if(remote.length)memory=remote.map(x=>({...x,learned:true,hits:Number(x.hits)||1}));
  if(stateRow){try{const s=JSON.parse(stateRow.answer);brain.restore(s.network);epoch=Number(s.epoch)||0;loss=Number(s.loss)||1;trained=Boolean(s.trained);learningEvents=Number(s.learningEvents)||0}catch(e){console.warn(e)}}
  saveLocal();setStatus(`conectado · ${memory.length} recuerdos`);update()
 }catch(e){console.warn('Supabase:',e);setStatus('modo local');update()}
}

async function saveState(){
 if(!db)return;
 const payload=JSON.stringify({network:brain.serialize(),epoch,loss,trained,learningEvents,updatedAt:new Date().toISOString()});
 try{
  const old=await timeout(db.from('neural_memory').select('question').eq('question',STATE_KEY).limit(1));if(old.error)throw old.error;
  const op=old.data?.length?db.from('neural_memory').update({answer:payload}).eq('question',STATE_KEY):db.from('neural_memory').insert({question:STATE_KEY,answer:payload});
  const r=await timeout(op);if(r.error)throw r.error
 }catch(e){console.warn('estado:',e)}
}

function queueSave(){clearTimeout(saveTimer);saveTimer=setTimeout(saveState,700)}

async function saveMemory(question,answer){
 const n=normalize(question),old=memory.find(x=>normalize(x.question)===n);
 if(old){old.answer=answer;old.learned=true;old.hits=(Number(old.hits)||0)+1;old.created_at=new Date().toISOString()}
 else memory.unshift({question,answer,learned:true,hits:1,created_at:new Date().toISOString()});
 memory=memory.slice(0,2000);saveLocal();renderMemoryTable();
 if(!db)return;
 try{const r=await timeout(db.from('neural_memory').insert({question:question.slice(0,300),answer:answer.slice(0,600)}));if(r.error)throw r.error}catch(e){console.warn('memoria:',e)}
}

function trainEpoch(){let total=0;const batch=XOR.slice().sort(()=>Math.random()-.5);for(const s of batch)total+=brain.train(s.input,s.target);return total/XOR.length}

async function train(){
 if(isTraining)return;isTraining=true;const b=$('train');if(b)b.disabled=true;if($('reset'))$('reset').disabled=true;if($('state'))$('state').textContent='Entrenando…';
 try{
  const total=3200;
  for(let i=0;i<total;i++){loss=trainEpoch();epoch++;if(i%80===0){update(i/total*100);await new Promise(requestAnimationFrame)}}
  loss=XOR.reduce((s,x)=>s+(x.target[0]-brain.predict(x.input))**2,0)/XOR.length;trained=true;learningEvents++;update(100);queueSave();
  if($('state'))$('state').textContent=`Entrenada · error ${loss.toFixed(4)}`
 }catch(e){console.error(e);if($('state'))$('state').textContent='Error: '+e.message}
 finally{isTraining=false;if(b)b.disabled=false;if($('reset'))$('reset').disabled=false}
}

function learnInput(text){
 const n=normalize(text),count=words(n).length;if(!n)return;
 const target=Math.min(.92,Math.max(.08,(count%12+1)/13)),input=[Math.min(.99,n.length/320),Math.min(.99,count/32)];
 let e=0;for(let i=0;i<4;i++)e+=brain.train(input,[target]);
 loss=e/4;epoch+=4;trained=true
}

function teachPair(question,answerText){
 question=question.trim();answerText=answerText.trim();if(!question||!answerText)return false;
 memory=memory.filter(x=>normalize(x.question)!==normalize(question));memory.unshift({question,answer:answerText,learned:true,hits:1,created_at:new Date().toISOString()});
 memory=memory.slice(0,2000);learningEvents++;saveLocal();renderMemoryTable();learnInput(question);saveMemory(question,answerText);queueSave();
 if($('learnResult'))$('learnResult').textContent='✓ RDio aprendió este par y lo guardó en su memoria.';return true
}

function autoLearnFromInteraction(question,answerText,source='interaction'){
 question=String(question||'').trim();answerText=String(answerText||'').trim();
 if(!question||!answerText||answerText.length>600)return false;
 const normalized=normalize(question);
 if(normalized.length<3||FALLBACKS.includes(answerText))return false;
 const existing=memory.find(x=>normalize(x.question)===normalized);
 if(existing){existing.answer=answerText;existing.learned=true;existing.hits=(Number(existing.hits)||0)+1;existing.source=source;saveLocal();renderMemoryTable();return false}
 memory.unshift({question,answer:answerText,learned:true,hits:1,source,created_at:new Date().toISOString()});
 memory=memory.slice(0,2000);learningEvents++;learnInput(question);saveLocal();renderMemoryTable();queueSave();return true
}

function chooseVariant(list){
 const available=list.filter(x=>!recentReplies.includes(x));
 const pool=available.length?available:list;
 const value=pool[Math.floor(Math.random()*pool.length)];
 recentReplies=[...recentReplies.slice(-5),value];
 return value
}

function answer(text){
 const n=normalize(text);
 const calculation=calculate(text);
 if(calculation!==null)return calculation;

 const remembered=findMemory(text);
 if(remembered)return remembered.answer;

 let best=null,bestScore=0;
 for(const [q,variants] of KNOWLEDGE){
  const s=similarity(n,q);
  if(s>bestScore){bestScore=s;best={q,variants}}
 }
 if(best&&bestScore>=.72)return chooseVariant(best.variants);

 if(/^(hola|holi|hey|buenas|hello)$/.test(n))return chooseVariant(['Hola. 🧠','Buenas. Soy RDio.','Hola, ¿qué tal? 🧠']);
 if(n.includes('que aprendiste')||n.includes('que has aprendido')||n.includes('que sabes'))return `Tengo ${memory.length.toLocaleString()} recuerdos y ${learningEvents.toLocaleString()} eventos de aprendizaje registrados.`;
 if(n.includes('quien eres')||n.includes('que eres'))return chooseVariant(KNOWLEDGE[1][1]);
 if(n.includes('que es una red neuronal'))return chooseVariant(KNOWLEDGE[3][1]);
 if(n.includes('como funciona'))return chooseVariant(KNOWLEDGE[4][1]);
 if(n.includes('como estas'))return chooseVariant(KNOWLEDGE[11][1]);
 if(n.includes('gracias'))return chooseVariant(KNOWLEDGE[12][1]);

 return chooseVariant(FALLBACKS)
}

function calculate(text){
 let expr=normalize(text).replace(/^(cuanto es|cuanto da|calcula|resuelve|resultado de|cuanto seria)\s+/,'').replace(/\s+/g,'');
 if(!/^[0-9+*/().\-]+$/.test(expr)||!/[+*/\-]/.test(expr))return null;
 try{
  const value=Function('"use strict";return ('+expr+')')();
  if(typeof value!=='number'||!Number.isFinite(value))return null;
  return Number.isInteger(value)?String(value):String(Number(value.toFixed(10)))
 }catch{return null}
}

function renderMemoryTable(){
 const body=$('memoryTableBody'),wrap=document.querySelector('.memoryTableWrap'),empty=$('memoryEmpty'),badge=$('memoryGrowth');if(!body)return;
 body.innerHTML='';const rows=memory.slice(0,2000);
 rows.forEach((item,i)=>{const tr=document.createElement('tr');tr.className='memoryRow';const learned=item.learned===true;
  tr.innerHTML='<td>'+String(i+1)+'</td><td class="memoryQuestion"></td><td class="memoryAnswer"></td><td><span class="memoryStatus">'+(learned?'APRENDIDO':'RECUERDO')+'</span></td>';
  tr.children[1].textContent=item.question;tr.children[2].textContent=item.answer;body.appendChild(tr)
 });
 if(wrap)wrap.classList.toggle('hasRows',rows.length>0);if(empty)empty.hidden=rows.length>0;if(badge)badge.textContent=rows.length.toLocaleString()+' '+(rows.length===1?'fila':'filas')
}

function add(text,type){
 const box=$('messages');if(!box)return;
 const el=document.createElement('div');el.className=type==='user'?'message userMessage':'message neuronMessage';el.textContent=text;box.appendChild(el);box.scrollTop=box.scrollHeight
}

function toggleMemory(force){
 const drawer=$('memoryDrawer'),backdrop=$('memoryBackdrop'),button=$('memoryToggle');if(!drawer||!button)return;
 const open=typeof force==='boolean'?force:!drawer.classList.contains('open');
 drawer.classList.toggle('open',open);drawer.setAttribute('aria-hidden',String(!open));button.setAttribute('aria-expanded',String(open));
 if(backdrop)backdrop.hidden=!open;document.body.style.overflow=open?'hidden':'';
}

function pos(layer,index,w,h){const count=brain.layers[layer],x=38+(w-76)*layer/(brain.layers.length-1),gap=Math.min(45,(h-90)/Math.max(1,count-1));return{x,y:h/2+(index-(count-1)/2)*gap}}
let networkCache=null,networkDirty=true;

function renderNetwork(){
 const c=$('networkCanvas');if(!c)return;const r=c.getBoundingClientRect();if(!r.width||!r.height)return;
 const w=Math.max(320,Math.round(r.width||640)),h=Math.max(320,Math.round(r.height||360)),d=Math.min(devicePixelRatio||1,1.5);
 if(c.width===Math.round(w*d)&&c.height===Math.round(h*d)&&networkCache&&!networkDirty)return;
 c.width=Math.round(w*d);c.height=Math.round(h*d);const ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);
 const p=brain.layers.map((n,l)=>Array.from({length:n},(_,i)=>pos(l,i,w,h)));
 for(let l=0;l<p.length-1;l++)for(let a=0;a<p[l].length;a++)for(let b=0;b<p[l+1].length;b++){const wt=brain.weights[l][b][a];ctx.beginPath();ctx.moveTo(p[l][a].x,p[l][a].y);ctx.lineTo(p[l+1][b].x,p[l+1][b].y);ctx.lineWidth=Math.min(2.2,.4+Math.abs(wt));ctx.strokeStyle=wt>=0?'rgba(40,40,44,.13)':'rgba(130,130,136,.11)';ctx.stroke()}
 p.forEach((layer,l)=>layer.forEach((v,i)=>{const active=selectedNode?.l===l&&selectedNode?.i===i,rad=l===0||l===p.length-1?17:11;ctx.beginPath();ctx.arc(v.x,v.y,rad+(active?3:0),0,Math.PI*2);ctx.fillStyle=active?'#17181b':'rgba(255,255,255,.72)';ctx.fill();ctx.lineWidth=1;ctx.strokeStyle='rgba(80,80,86,.28)';ctx.stroke();ctx.fillStyle=active?'#fff':'#555';ctx.font='600 8px system-ui';ctx.textAlign='center';ctx.fillText(l===0?'x'+(i+1):l===p.length-1?'y':'h'+l+'.'+(i+1),v.x,v.y+3)}));
 ctx.fillStyle='#8d9097';ctx.font='9px system-ui';ctx.textAlign='center';ctx.fillText(trained?'RDio · '+epoch.toLocaleString()+' épocas · '+memory.length.toLocaleString()+' recuerdos':'RDio · esperando entrenamiento',w/2,h-10);networkCache=true;networkDirty=false
}

function invalidateNetwork(){networkCache=null;networkDirty=true;requestAnimationFrame(()=>requestAnimationFrame(renderNetwork))}

function reset(){brain.reset();epoch=0;loss=1;trained=false;learningEvents=0;memory=[];recentReplies=[];saveLocal();update();renderMemoryTable();if($('state'))$('state').textContent='Modelo reiniciado';queueSave();invalidateNetwork()}

$('learnForm')?.addEventListener('submit',e=>{e.preventDefault();const q=$('learnQuestion').value,a=$('learnAnswer').value;if(teachPair(q,a)){$('learnQuestion').value='';$('learnAnswer').value=''}});

$('train')?.addEventListener('click',train);
$('reset')?.addEventListener('click',reset);
$('memoryToggle')?.addEventListener('click',()=>toggleMemory());
$('memoryClose')?.addEventListener('click',()=>toggleMemory(false));
$('memoryBackdrop')?.addEventListener('click',()=>toggleMemory(false));
$('openMemoryHint')?.addEventListener('click',()=>toggleMemory(true));
$('statusMemory')?.addEventListener('click',()=>toggleMemory(true));
document.addEventListener('keydown',e=>{if(e.key==='Escape')toggleMemory(false)});

$('chatForm')?.addEventListener('submit',e=>{
 e.preventDefault();const input=$('chatInput'),q=input.value.trim();if(!q)return;add(q,'user');input.value='';
 if(!trained){add('Entrena el núcleo desde Home primero. 🧠','neuron');return}
 const response=answer(q);add(response,'neuron');
 // Solo consolida respuestas conocidas. Las respuestas de fallback nunca se convierten en "verdades".
 const knownBefore=memory.length;
 if(response&&!FALLBACKS.includes(response))autoLearnFromInteraction(q,response,'chat');
 if(memory.length!==knownBefore)update();
});

window.addEventListener('resize',()=>{networkCache=null;networkDirty=true;renderNetwork()});
window.addEventListener('load',()=>{networkCache=null;networkDirty=true;requestAnimationFrame(()=>requestAnimationFrame(renderNetwork))});

setStatus('cargando RDio…');update();renderMemoryTable();load();invalidateNetwork();
