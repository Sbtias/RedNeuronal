/* IAYO Local AI Engine
 * WebLLM + Llama 3.2 3B Instruct
 * The model runs locally in the browser through WebGPU.
 */
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const MODEL = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
let engine = null;
let loading = null;
let ready = false;
let chatHistory = [];

function setStatus(text) {
  const a = document.getElementById("dbState");
  const b = document.getElementById("chatState");
  if (a) a.textContent = text;
  if (b) b.textContent = text;
}

function readMemory() {
  try {
    const rows = JSON.parse(localStorage.getItem("iayo_memory_v7") || "[]");
    return Array.isArray(rows) ? rows.filter(x => x?.learned === true).slice(0, 40) : [];
  } catch {
    return [];
  }
}

function memoryContext() {
  const rows = readMemory();
  if (!rows.length) return "No hay recuerdos permanentes todavía.";
  return rows.map((x, i) => `[${i + 1}] ${x.question} => ${x.answer}`).join("\n");
}

async function load() {
  if (ready) return engine;
  if (loading) return loading;
  if (!("gpu" in navigator)) throw new Error("Este navegador no tiene WebGPU disponible.");

  loading = (async () => {
    setStatus("descargando cerebro…");
    engine = await webllm.CreateMLCEngine(MODEL, {
      initProgressCallback: (p) => {
        const value = Math.max(0, Math.min(100, Math.round((p.progress || 0) * 100)));
        setStatus(`cerebro ${value}%`);
        const bar = document.getElementById("progressBar");
        if (bar) bar.style.width = value + "%";
      }
    });
    ready = true;
    chatHistory = [];
    setStatus("IA local activa");
    return engine;
  })().catch((error) => {
    loading = null;
    ready = false;
    setStatus("IA local no disponible");
    throw error;
  });

  return loading;
}

async function chat(question) {
  const ai = await load();
  const memories = memoryContext();

  const system = [
    "Eres IAYO, un asistente local experimental.",
    "Responde en español salvo que el usuario use otro idioma.",
    "Sé natural, claro y útil. No inventes recuerdos.",
    "Los recuerdos siguientes son información guardada por el usuario. Úsalos como contexto, pero no los conviertas automáticamente en hechos universales.",
    "MEMORIA DE IAYO:",
    memories
  ].join("\n");

  const messages = [
    { role: "system", content: system },
    ...chatHistory.slice(-10),
    { role: "user", content: question }
  ];

  const result = await ai.chat.completions.create({
    messages,
    temperature: 0.75,
    top_p: 0.9,
    max_tokens: 500
  });

  const answer = result.choices?.[0]?.message?.content?.trim();
  if (!answer) throw new Error("El modelo no devolvió una respuesta.");

  chatHistory.push({ role: "user", content: question });
  chatHistory.push({ role: "assistant", content: answer });
  chatHistory = chatHistory.slice(-12);

  return answer;
}

window.IAYOAI = {
  model: MODEL,
  load,
  chat,
  isReady: () => ready
};

setStatus("IA local lista · cargar al enviar");
