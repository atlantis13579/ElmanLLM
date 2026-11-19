"use strict";

const byId = id => document.getElementById(id);

let capabilities;
let history = [];
let chatRequest;
let imageRequest;
let imageUrl;
let finalBitmap;

function showError(element, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

async function apiError(response) {
  try {
    const data = await response.json();
    if (data.error?.message) return new Error(data.error.message);
    if (typeof data.detail === "string") return new Error(data.detail);
  } catch {
    // Fall back to the HTTP status when the server supplies no error JSON.
  }
  return new Error(`Request failed (${response.status}).`);
}

async function connect() {
  const status = byId("server-status");
  try {
    const response = await fetch("/health");
    if (!response.ok) throw await apiError(response);
    capabilities = await response.json();
    if (capabilities.model_type === "llm") {
      byId("image").remove();
      byId("chat").hidden = false;
      byId("chat-model").textContent = capabilities.model;
      byId("message").disabled = !capabilities.chat_text;
      byId("send").disabled = !capabilities.chat_text;
      if (!capabilities.chat_text) byId("chat-empty").textContent = "Text chat requires a model tokenizer.";
    } else if (capabilities.model_type === "diffusion") {
      byId("chat").remove();
      byId("image").hidden = false;
      byId("image-model").textContent = capabilities.model;
      if (capabilities.image_sampler === "FlowMatchEuler") {
        document.title = capabilities.model + " | Elman";
        byId("image-title").textContent = "Create with FLUX.2";
        byId("steps").value = "4";
        byId("guidance").value = "1";
        byId("guidance-label").hidden = true;
        byId("negative-label").hidden = true;
        byId("negative-prompt").hidden = true;
      }
      randomizeSeed();
      byId("image-fields").disabled = false;
    } else throw new Error("Unknown model type");
    status.textContent = "Connected";
    status.classList.add("online");
  } catch {
    status.textContent = "Connection failed";
    status.classList.add("problem");
  }
}

function scrollToLatest() {
  const conversation = byId("conversation");
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(role, text = "") {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  const mark = document.createElement("span");
  mark.className = "message-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = role === "user" ? "Y" : "E";
  const body = document.createElement("div");
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Elman";
  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;
  body.append(label, content);
  row.append(mark, body);
  byId("chat-empty").hidden = true;
  byId("messages").append(row);
  scrollToLatest();
  return content;
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, {stream: true});
      for (let end = buffer.indexOf("\n\n"); end >= 0; end = buffer.indexOf("\n\n")) {
        const event = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        if (event.startsWith("data: ")) yield event.slice(6);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function streamChat(messages, signal, append) {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({model: capabilities.model, messages, stream: true}),
    signal
  });
  if (!response.ok) throw await apiError(response);
  for await (const data of readEvents(response)) {
    if (data === "[DONE]") return;
    const chunk = JSON.parse(data);
    if (chunk.error) throw new Error(chunk.error.message);
    const text = chunk.choices?.[0]?.delta?.content;
    if (text) append(text);
  }
  throw new Error("The response ended before generation completed.");
}

byId("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const field = byId("message");
  const text = field.value.trim();
  if (!capabilities?.chat_text || !text || chatRequest) return;
  const user = {role: "user", content: text};
  const incoming = new AbortController();
  chatRequest = incoming;
  field.value = "";
  byId("send").disabled = true;
  byId("new-chat").disabled = true;
  byId("stop-chat").hidden = false;
  byId("chat-status").textContent = "Waiting for response";
  showError(byId("chat-error"));
  const sent = addMessage("user", text);
  let reply;
  let answer = "";
  try {
    await streamChat([...history, user], incoming.signal, piece => {
      if (!reply) {
        reply = addMessage("assistant");
        byId("chat-status").textContent = "Generating";
      }
      answer += piece;
      reply.append(piece);
      scrollToLatest();
    });
    if (!reply) addMessage("assistant", "No text returned.");
    history.push(user, {role: "assistant", content: answer});
    byId("chat-status").textContent = "Response complete";
  } catch (error) {
    const cancelled = incoming.signal.aborted;
    if (!reply) {
      sent.closest("article").remove();
      byId("chat-empty").hidden = byId("messages").childElementCount !== 0;
      if (!field.value) field.value = text;
    }
    byId("chat-status").textContent = cancelled ? "Generation stopped" : "";
    if (!cancelled) showError(byId("chat-error"), error.message);
  } finally {
    chatRequest = undefined;
    byId("send").disabled = false;
    byId("new-chat").disabled = byId("messages").childElementCount === 0;
    byId("stop-chat").hidden = true;
  }
});

byId("stop-chat").addEventListener("click", () => chatRequest?.abort());
byId("new-chat").addEventListener("click", () => {
  history = [];
  byId("messages").replaceChildren();
  byId("chat-empty").hidden = false;
  byId("chat-status").textContent = "";
  byId("new-chat").disabled = true;
  showError(byId("chat-error"));
  byId("message").focus();
});

function decodeBitmap(encoded) {
  const decoded = atob(encoded);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; ++i) bytes[i] = decoded.charCodeAt(i);
  return new Blob([bytes], {type: "image/bmp"});
}

function displayImage(bitmap, description) {
  const url = URL.createObjectURL(bitmap);
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = url;
  const image = byId("generated-image");
  image.src = imageUrl;
  image.alt = description;
  image.hidden = false;
  byId("image-placeholder").hidden = true;
}

async function imageForDownload(source, format) {
  if (format === "bmp") return source;
  const type = format === "jpg" ? "image/jpeg" : "image/png";
  const bitmap = await createImageBitmap(source);
  const canvas = document.createElement("canvas");
  try {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Image conversion is unavailable.");
    context.drawImage(bitmap, 0, 0);
  } finally {
    bitmap.close();
  }
  const image = await new Promise(resolve => canvas.toBlob(resolve, type, 0.95));
  if (!image || image.type !== type) throw new Error(`Could not encode ${format.toUpperCase()}.`);
  return image;
}

byId("download").addEventListener("click", async () => {
  const source = finalBitmap;
  if (!source) return;
  const format = byId("output-format").value;
  const button = byId("download");
  const errorBox = byId("download-error");
  button.disabled = true;
  showError(errorBox);
  try {
    const file = await imageForDownload(source, format);
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `elman-image.${format}`;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    if (source === finalBitmap) showError(errorBox, error.message);
  } finally {
    button.disabled = false;
  }
});

function randomizeSeed() {
  const seed = byId("seed");
  const random = new Uint32Array(1);
  do { crypto.getRandomValues(random); } while (random[0] === seed.valueAsNumber);
  seed.value = String(random[0]);
}

async function streamImage(body, signal, status, bar, image, caption) {
  const response = await fetch("/sdapi/v1/txt2img", {
    method: "POST",
    headers: {"Content-Type": "application/json", "Accept": "text/event-stream"},
    body: JSON.stringify(body),
    signal
  });
  if (!response.ok) throw await apiError(response);
  for await (const data of readEvents(response)) {
    signal.throwIfAborted();
    const event = JSON.parse(data);
    if (event.error) throw new Error(event.detail ?? event.error);
    if (event.type === "result") return event.result;
    if (event.type === "progress") {
      status.textContent = event.status;
      bar.max = event.steps;
      bar.value = event.step;
    } else if (event.type === "preview") {
      displayImage(decodeBitmap(event.image), `Preview at denoising step ${event.step}`);
      caption.textContent = `Preview at step ${event.step}/${event.steps}`;
      caption.hidden = false;
      await image.decode();
    }
  }
  throw new Error("The response ended before generation completed.");
}

byId("image-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (imageRequest || !capabilities?.image_generation) return;
  const incoming = new AbortController();
  imageRequest = incoming;
  const taskId = Array.from(crypto.getRandomValues(new Uint8Array(12)), byte => byte.toString(16).padStart(2, "0")).join("");
  const body = {
    prompt: byId("prompt").value.trim(),
    negative_prompt: byId("negative-prompt").value,
    width: Number(byId("width").value),
    height: Number(byId("height").value),
    steps: byId("steps").valueAsNumber,
    cfg_scale: byId("guidance").valueAsNumber,
    seed: byId("seed").valueAsNumber,
    force_task_id: taskId,
    live_preview: true
  };
  const fields = byId("image-fields");
  const taskStatus = byId("image-task-status");
  const status = byId("image-status");
  const bar = byId("image-progress");
  const generate = byId("generate");
  const image = byId("generated-image");
  const caption = byId("image-preview-label");
  const downloads = byId("image-download");
  if (imageUrl) {
    URL.revokeObjectURL(imageUrl);
    imageUrl = undefined;
    image.removeAttribute("src");
  }
  image.hidden = true;
  caption.hidden = true;
  downloads.hidden = true;
  finalBitmap = undefined;
  showError(byId("download-error"));
  byId("image-placeholder").hidden = false;
  fields.disabled = true;
  generate.textContent = "Generating image";
  status.textContent = "Waiting for server";
  taskStatus.hidden = false;
  bar.max = body.steps;
  bar.value = 0;
  bar.hidden = false;
  byId("stop-image").hidden = false;
  showError(byId("image-error"));
  const statusBounds = taskStatus.getBoundingClientRect();
  if (statusBounds.top < 0 || statusBounds.bottom > window.innerHeight - 24) {
    taskStatus.scrollIntoView({block: "center"});
  }
  try {
    const {images} = await streamImage(body, incoming.signal, status, bar, image, caption);
    if (!images?.[0]) throw new Error("The server did not return an image.");
    const bitmap = decodeBitmap(images[0]);
    displayImage(bitmap, `Generated image for ${body.prompt}`);
    await image.decode();
    incoming.signal.throwIfAborted();
    finalBitmap = bitmap;
    caption.hidden = true;
    downloads.hidden = false;
    status.textContent = "Image ready";
    randomizeSeed();
  } catch (error) {
    status.textContent = incoming.signal.aborted ? "Generation cancelled" : "Image generation failed";
    if (!incoming.signal.aborted) showError(byId("image-error"), error.message);
  } finally {
    incoming.abort();
    bar.hidden = true;
    imageRequest = undefined;
    fields.disabled = false;
    generate.textContent = "Generate image";
    byId("stop-image").hidden = true;
  }
});
byId("stop-image").addEventListener("click", () => imageRequest?.abort());

connect();
