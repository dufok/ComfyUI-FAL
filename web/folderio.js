// ComfyUI-FolderIO — browser side.
//
//  📁 Load Images (upload folder)
//     Adds "Upload folder…" / "Upload files…" buttons and accepts a folder dropped onto the node.
//     Every file goes to input/<folder>/ through the stock /upload/image endpoint — one request per
//     file, so proxies with small body caps (Cloudflare free: 100 MB) never bite — and the folder
//     combo is then pointed at the new folder. Nested folders are flattened into one.
//
//  💾 Save Images + ZIP
//     After a run, the node's "⬇ Download ZIP" button links to the ZIP of that run.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD_NODE = "FolderIOLoadImages";
const SAVE_NODE = "FolderIOSaveZip";
const UPLOAD_BTN = "📁 Upload folder…";
const FILES_BTN = "🖼 Upload files…";
const ZIP_BTN = "⬇ Download ZIP (after run)";
const IMAGE_RE = /\.(jpe?g|png|webp|bmp|tiff?|gif|heic|heif|hif|avif)$/i;
const PARALLEL = 3; // concurrent uploads

const log = (...a) => console.log("[FolderIO]", ...a);
const fmtMB = (b) => `${(b / 1048576).toFixed(1)} MB`;
const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);
const isImageFile = (f) => !!f && f.size > 0 && !f.name.startsWith(".") && IMAGE_RE.test(f.name);

function toast(severity, summary, detail, life = 6000) {
  try {
    app.extensionManager?.toast?.add({ severity, summary, detail, life });
  } catch (e) {
    /* very old frontend without the toast API */
  }
  log(summary, detail ?? "");
}

function safeFolderName(name) {
  const clean = String(name ?? "")
    .normalize("NFC")
    .replace(/[\\/:*?"<>|\x00-\x1f]/g, "_")
    .replace(/^[\s.]+|[\s.]+$/g, "");
  return clean || "photos";
}

function currentFolder(node) {
  const v = findWidget(node, "folder")?.value;
  return typeof v === "string" && v.trim() ? v.trim() : "";
}

function askFolderName(node) {
  const name = window.prompt("Folder name inside input/ for these files:", currentFolder(node) || "photos");
  return name == null ? "" : safeFolderName(name);
}

function pickFiles({ directory }) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    if (directory) input.webkitdirectory = true;
    else input.accept = "image/*,.heic,.heif,.hif,.avif";
    input.style.display = "none";
    document.body.appendChild(input);
    const finish = (files) => {
      input.remove();
      resolve(files);
    };
    input.onchange = () => finish(Array.from(input.files ?? []));
    input.oncancel = () => finish([]);
    input.click();
  });
}

function selectFolder(node, folder) {
  const w = findWidget(node, "folder");
  if (!w) return;
  const values = w.options?.values;
  if (Array.isArray(values) && !values.includes(folder)) {
    values.push(folder);
    values.sort();
  }
  const prev = w.value;
  w.value = folder;
  w.callback?.(folder);
  node.onWidgetChanged?.("folder", folder, prev, w);
  node.graph?.setDirtyCanvas(true, true);
  refreshFolderCombos();
}

const graphNodes = () => app.graph?.nodes ?? app.graph?._nodes ?? [];

// Re-read just this node's definition (cheap, no toasts — unlike app.refreshComboInNodes) and give
// every folder combo in the graph the fresh list of input/ subfolders.
async function refreshFolderCombos() {
  try {
    const res = await api.fetchApi(`/object_info/${LOAD_NODE}`);
    if (res.status !== 200) return;
    const values = (await res.json())?.[LOAD_NODE]?.input?.required?.folder?.[0];
    if (!Array.isArray(values)) return;
    for (const n of graphNodes()) {
      if (n.type !== LOAD_NODE) continue;
      const w = findWidget(n, "folder");
      if (!w?.options) continue;
      const cur = w.value;
      w.options.values = values.slice();
      if (typeof cur === "string" && cur && !w.options.values.includes(cur)) w.options.values.push(cur);
    }
    app.graph?.setDirtyCanvas(true, true);
  } catch (e) {
    log("folder list refresh failed", e);
  }
}

async function uploadFiles(node, allFiles, folder) {
  const files = allFiles.filter(isImageFile);
  const skipped = allFiles.length - files.length;
  if (!files.length) {
    toast("warn", "FolderIO: nothing to upload", "no image files in the selection");
    return;
  }
  if (node._folderioBusy) {
    toast("warn", "FolderIO: an upload is already running");
    return;
  }
  const total = files.reduce((s, f) => s + f.size, 0);
  const ok = window.confirm(
    `Upload ${files.length} image(s), ${fmtMB(total)} → input/${folder}/ ?` +
      (skipped ? `\n(${skipped} non-image file(s) skipped)` : "") +
      "\nFiles with the same name in that folder are overwritten."
  );
  if (!ok) return;

  const btn = findWidget(node, UPLOAD_BTN);
  const setProgress = (text) => {
    if (btn) btn.label = text;
    node.graph?.setDirtyCanvas(true, false);
  };
  node._folderioBusy = true;
  const failed = [];
  const queue = files.slice();
  let done = 0;
  const worker = async () => {
    while (queue.length) {
      const f = queue.shift();
      try {
        const fd = new FormData();
        fd.append("image", f, f.name);
        fd.append("subfolder", folder);
        fd.append("type", "input");
        fd.append("overwrite", "true");
        const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (res.status !== 200) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        await res.json();
      } catch (e) {
        failed.push(`${f.name}: ${e?.message ?? e}`);
      }
      done += 1;
      setProgress(`⏳ ${done}/${files.length} → input/${folder}`);
    }
  };
  setProgress(`⏳ 0/${files.length} → input/${folder}`);
  try {
    await Promise.all(Array.from({ length: Math.min(PARALLEL, files.length) }, worker));
  } finally {
    node._folderioBusy = false;
    setProgress(UPLOAD_BTN);
  }
  const uploaded = files.length - failed.length;
  if (uploaded) selectFolder(node, folder);
  if (failed.length) {
    toast("error", `FolderIO: ${failed.length} of ${files.length} upload(s) failed`, failed.slice(0, 5).join("\n"), 12000);
  } else {
    toast("success", `FolderIO: ${uploaded} file(s) → input/${folder}/`, "Folder selected on the node — press Run.");
  }
}

// --- folder drag & drop -----------------------------------------------------------------------

const entryToFile = (entry) => new Promise((res, rej) => entry.file(res, rej));

async function walkDirectory(dir, out) {
  const reader = dir.createReader();
  for (;;) {
    // readEntries hands out batches (Chrome: 100) — loop until it returns an empty one.
    const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    for (const en of batch) {
      if (en.isDirectory) await walkDirectory(en, out);
      else if (en.isFile) out.push(await entryToFile(en));
    }
  }
}

function handleDrop(node, e) {
  const dt = e?.dataTransfer;
  if (!dt) return false;
  // Entries must be read synchronously — DataTransferItems are gone once the event returns.
  const entries = dt.items ? Array.from(dt.items).map((it) => it.webkitGetAsEntry?.()).filter(Boolean) : [];
  const plainFiles = Array.from(dt.files ?? []);
  if (!entries.length && !plainFiles.length) return false;
  (async () => {
    let files = [];
    let folder = "";
    if (entries.length) {
      for (const en of entries) {
        if (en.isDirectory) {
          folder ||= en.name;
          await walkDirectory(en, files);
        } else if (en.isFile) {
          files.push(await entryToFile(en));
        }
      }
    } else {
      files = plainFiles;
    }
    folder = folder ? safeFolderName(folder) : currentFolder(node) || askFolderName(node);
    if (!folder) return;
    await uploadFiles(node, files, folder);
  })().catch((err) => toast("error", "FolderIO: drop failed", String(err)));
  return true;
}

// --- ZIP download ------------------------------------------------------------------------------

const storedZip = (node) => app.nodeOutputs?.[String(node.id)]?.zip?.[0];

function downloadZip(node) {
  const z = node._folderioZip ?? storedZip(node);
  if (!z) {
    toast("info", "FolderIO: run the graph first", "the ZIP appears here after Save Images + ZIP executes");
    return;
  }
  const a = document.createElement("a");
  a.href = api.apiURL(z.url);
  a.download = z.filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function setZip(node, z) {
  if (!z) return;
  node._folderioZip = z;
  const btn = findWidget(node, ZIP_BTN);
  if (btn) btn.label = `⬇ Download ZIP (${z.count} files, ${fmtMB(z.bytes)})`;
  node.graph?.setDirtyCanvas(true, true);
}

function onLoadExecuted(node, message) {
  const warning = message?.folderio_warning?.[0];
  if (warning) toast("warn", "FolderIO", String(warning), 15000);
}

function addButton(node, name, callback) {
  const w = node.addWidget("button", name, null, callback, { serialize: false });
  if (w) w.serialize = false; // keep it out of widgets_values in both litegraph generations
  return w;
}

app.registerExtension({
  name: "Comfy.FolderIO",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === LOAD_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const node = this;
        addButton(node, UPLOAD_BTN, async () => {
          const files = await pickFiles({ directory: true });
          if (!files.length) return;
          const top = files[0].webkitRelativePath?.split("/")[0];
          await uploadFiles(node, files, safeFolderName(top || "photos"));
        });
        addButton(node, FILES_BTN, async () => {
          const files = await pickFiles({ directory: false });
          if (!files.length) return;
          const folder = askFolderName(node);
          if (folder) await uploadFiles(node, files, folder);
        });
        return r;
      };
      nodeType.prototype.onDragOver = function (e) {
        const types = e?.dataTransfer?.types;
        return !!types && Array.from(types).includes("Files");
      };
      nodeType.prototype.onDragDrop = function (e) {
        return handleDrop(this, e);
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        onLoadExecuted(this, message);
      };
    }
    if (nodeData.name === SAVE_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        addButton(this, ZIP_BTN, () => downloadZip(this));
        return r;
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        setZip(this, message?.zip?.[0]);
      };
    }
  },
  // Fired when outputs are restored from the queue history (F5, or opening a past run): the live
  // onExecuted never runs on that path, so relabel the ZIP buttons from the stored outputs here.
  onNodeOutputsUpdated(outputs) {
    for (const n of graphNodes()) {
      if (n.type === SAVE_NODE) setZip(n, outputs?.[String(n.id)]?.zip?.[0]);
    }
  },
});
