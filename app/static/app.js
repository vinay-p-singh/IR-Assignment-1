"use strict";

// Every block the server sends has a `type`; this file owns one renderer per
// type. All text goes in through textContent, never innerHTML, so an article
// containing angle brackets cannot inject markup.

const transcript = document.getElementById("transcript");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const indexSelect = document.getElementById("indexSelect");
const corpusLabel = document.getElementById("corpusLabel");
const quick = document.getElementById("quick");
const modal = document.getElementById("docModal");

const QUICK_COMMANDS = [
  ":help", ":stats", ":index", ":term oil", ":stem nationalization",
  ":wild econom*", ":spell recieve", ":compare oil AND price",
];

let busy = false;

// ------------------------------------------------------------------ helpers

function el(tag, className, textContent) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent !== undefined) node.textContent = textContent;
  return node;
}

function card(title) {
  const box = el("div", "card");
  if (title) box.appendChild(el("h3", null, title));
  return box;
}

/** Wrap each occurrence of any highlight prefix in <mark>, without innerHTML. */
function highlighted(text, terms) {
  const span = el("span");
  const cleaned = (terms || []).filter((t) => t && t.length > 2)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!cleaned.length) {
    span.textContent = text;
    return span;
  }
  const re = new RegExp("\\b(" + cleaned.join("|") + ")\\w*", "gi");
  let last = 0;
  for (const match of text.matchAll(re)) {
    if (match.index > last) span.appendChild(document.createTextNode(text.slice(last, match.index)));
    span.appendChild(el("mark", null, match[0]));
    last = match.index + match[0].length;
  }
  span.appendChild(document.createTextNode(text.slice(last)));
  return span;
}

function animateFill(node, ratio) {
  requestAnimationFrame(() => {
    node.style.width = Math.max(0, Math.min(1, ratio)) * 100 + "%";
  });
}

// ---------------------------------------------------------------- renderers

const renderers = {
  text(block) {
    const box = el("div", "card");
    box.appendChild(el("p", "body-text " + (block.tone || "plain"), block.text));
    return box;
  },

  kv(block) {
    const box = card(block.title);
    const dl = el("dl", "kv");
    for (const [key, value] of block.items) {
      dl.appendChild(el("dt", null, key));
      dl.appendChild(el("dd", null, value));
    }
    box.appendChild(dl);
    return box;
  },

  table(block) {
    const box = card(block.title);
    const scroll = el("div", "scroll");
    const table = el("table");
    const thead = el("thead");
    const hrow = el("tr");
    block.columns.forEach((c) => hrow.appendChild(el("th", null, c)));
    thead.appendChild(hrow);
    const tbody = el("tbody");
    block.rows.forEach((row) => {
      const tr = el("tr");
      row.forEach((cell) => tr.appendChild(el("td", null, cell)));
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    scroll.appendChild(table);
    box.appendChild(scroll);
    return box;
  },

  bar(block) {
    const box = card(block.title);
    const wrap = el("div", "bars");
    const max = block.unit === "ratio"
      ? 1
      : Math.max(1, ...block.items.map((i) => Number(i.value) || 0));
    block.items.forEach((item) => {
      const row = el("div", "barrow");
      row.appendChild(el("div", "lbl", item.label));
      const track = el("div", "track");
      const fill = el("div", "fill");
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("div", "val", item.note));
      wrap.appendChild(row);
      animateFill(fill, (Number(item.value) || 0) / max);
    });
    box.appendChild(wrap);
    return box;
  },

  metrics(block) {
    const box = card(block.title);
    const grid = el("div", "metrics");
    [["precision", block.precision], ["recall", block.recall], ["F1", block.f1]]
      .forEach(([name, value]) => {
        const cell = el("div", "metric");
        cell.appendChild(el("div", "num", value.toFixed(3)));
        cell.appendChild(el("div", "cap", name));
        const track = el("div", "track");
        const fill = el("div", "fill");
        track.appendChild(fill);
        cell.appendChild(track);
        grid.appendChild(cell);
        animateFill(fill, value);
      });
    box.appendChild(grid);
    box.appendChild(el("div", "counts",
      `retrieved ${block.retrieved} · reference ${block.relevant} · overlap ${block.true_positives}`));
    return box;
  },

  chips(block) {
    const box = card(block.title);
    const wrap = el("div", "quick");
    block.items.forEach((item) => {
      const looksRunnable = /^[:/]/.test(item) || /^[\w()][^→]*$/.test(item);
      const chip = el("button", "chip" + (looksRunnable ? "" : " static"), item);
      chip.type = "button";
      if (looksRunnable) chip.dataset.send = item.replace(/\s*\(d=\d+\)$/, "");
      wrap.appendChild(chip);
    });
    box.appendChild(wrap);
    if (block.note) box.appendChild(el("div", "more", block.note));
    return box;
  },

  steps(block) {
    const box = card(block.title);
    const wrap = el("div", "steps");
    let previous = null;
    block.items.forEach((item, i) => {
      if (i > 0) wrap.appendChild(el("span", "arrow", "→"));
      const step = el("div", "step" + (previous !== null && item.value !== previous ? " changed" : ""));
      step.appendChild(el("span", "n", item.label));
      step.appendChild(el("span", "v", item.value));
      wrap.appendChild(step);
      previous = item.value;
    });
    box.appendChild(wrap);
    return box;
  },

  matrix(block) {
    const box = card(block.title);
    const scroll = el("div", "scroll");
    const table = el("table", "matrix");
    const head = el("tr");
    head.appendChild(el("th", null, ""));
    block.colLabels.forEach((c) => head.appendChild(el("th", null, c || "ε")));
    table.appendChild(head);
    const max = Math.max(1, ...block.rows.flat());
    block.rows.forEach((row, i) => {
      const tr = el("tr");
      tr.appendChild(el("th", null, block.rowLabels[i] || "ε"));
      row.forEach((value, j) => {
        const td = el("td", null, String(value));
        const heat = 1 - value / max;
        td.style.background = `color-mix(in srgb, var(--accent-soft) ${Math.round(heat * 100)}%, transparent)`;
        if (i === block.rows.length - 1 && j === row.length - 1) {
          td.style.fontWeight = "700";
          td.style.color = "var(--accent)";
        }
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    scroll.appendChild(table);
    box.appendChild(scroll);
    return box;
  },

  docs(block) {
    if (!block.items.length) return null;
    const box = card(`${block.title} — ${block.total} document${block.total === 1 ? "" : "s"}`);
    const list = el("div", "doclist");
    block.items.forEach((doc) => {
      const item = el("div", "docitem");
      item.tabIndex = 0;
      const head = el("div", "head");
      head.appendChild(el("span", "id", "#" + doc.doc_id));
      head.appendChild(el("span", "cat", doc.category));
      head.appendChild(el("span", "title", doc.title));
      item.appendChild(head);
      const snip = el("div", "snip");
      snip.appendChild(highlighted(doc.snippet, block.highlights));
      item.appendChild(snip);
      item.addEventListener("click", () => openDoc(doc.doc_id));
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") openDoc(doc.doc_id); });
      list.appendChild(item);
    });
    box.appendChild(list);
    if (block.total > block.items.length) {
      box.appendChild(el("div", "more", `showing the first ${block.items.length}`));
    }
    return box;
  },

  article(block) {
    const box = card(block.title);
    const pre = el("pre");
    pre.style.whiteSpace = "pre-wrap";
    pre.style.margin = "0";
    pre.style.font = "13.5px/1.6 inherit";
    pre.textContent = block.text;
    box.appendChild(pre);
    return box;
  },
};

// ------------------------------------------------------------------ chat io

function addUserTurn(message) {
  const turn = el("div", "turn user", message);
  transcript.appendChild(turn);
  scroll();
}

function addBotTurn(blocks) {
  const turn = el("div", "turn bot");
  blocks.forEach((block) => {
    const render = renderers[block.type];
    const node = render ? render(block) : renderers.text({ text: JSON.stringify(block) });
    if (node) turn.appendChild(node);
  });
  transcript.appendChild(turn);
  scroll();
  return turn;
}

function scroll() {
  transcript.scrollTop = transcript.scrollHeight;
}

function applyState(state) {
  if (!state) return;
  corpusLabel.textContent = `${state.num_docs} BBC articles`;
  indexSelect.textContent = "";
  state.configs.forEach((config) => {
    const option = el("option", null, config.name + (config.built ? "" : " (builds on use)"));
    option.value = config.name;
    option.selected = config.name === state.active;
    indexSelect.appendChild(option);
  });
}

async function send(message) {
  if (busy || !message.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  addUserTurn(message);
  const pending = el("div", "turn bot");
  pending.appendChild(el("div", "thinking", "working"));
  transcript.appendChild(pending);
  scroll();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    pending.remove();
    addBotTurn(data.blocks || [{ type: "text", text: data.error || "empty reply", tone: "error" }]);
    applyState(data.state);
  } catch (err) {
    pending.remove();
    addBotTurn([{ type: "text", text: "Server unreachable: " + err.message, tone: "error" }]);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

async function openDoc(docId) {
  const response = await fetch("/api/doc?id=" + encodeURIComponent(docId));
  if (!response.ok) return;
  const doc = await response.json();
  document.getElementById("docTitle").textContent =
    `#${doc.doc_id} · ${doc.category} · ${doc.filename}`;
  document.getElementById("docBody").textContent = doc.text;
  modal.showModal();
}

// ------------------------------------------------------------------ wiring

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value;
  input.value = "";
  send(message);
});

document.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-send]");
  if (chip) send(chip.dataset.send);
});

indexSelect.addEventListener("change", () => send(":index " + indexSelect.value));

document.getElementById("docClose").addEventListener("click", () => modal.close());

document.getElementById("themeToggle").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("irChatTheme", next);
});

QUICK_COMMANDS.forEach((command) => {
  const chip = el("button", "chip", command);
  chip.type = "button";
  chip.dataset.send = command;
  quick.appendChild(chip);
});

(async function boot() {
  const response = await fetch("/api/boot");
  const data = await response.json();
  applyState(data.state);
  addBotTurn(data.blocks);
  input.focus();
})();
