/* Reusable image import control: drag-and-drop or click-to-browse, shows an
 * inline preview once a file is picked, with replace/remove affordances -
 * used everywhere this app needs to bring in an image, instead of each
 * screen rolling its own file input. Local object-URL preview for formats
 * the browser can render natively (PNG/JPEG/WebP); TIFF gets a placeholder
 * icon until the caller supplies a server-rendered preview via setPreviewUrl().
 */

const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"];
const NATIVELY_PREVIEWABLE = /\.(png|jpe?g|webp)$/i;

function isAcceptedFile(file) {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const ICONS = {
  upload: `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
    <path d="M12 16V4M12 4l-4 4M12 4l4 4" stroke-linecap="round" stroke-linejoin="round" />
    <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" stroke-linecap="round" stroke-linejoin="round" />
  </svg>`,
  file: `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
    <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z" stroke-linejoin="round" />
    <path d="M14 2v6h6" stroke-linejoin="round" />
  </svg>`,
  replace: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5" stroke-linecap="round" stroke-linejoin="round" />
    <path d="M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5" stroke-linecap="round" stroke-linejoin="round" />
  </svg>`,
  remove: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M18 6 6 18M6 6l12 12" stroke-linecap="round" />
  </svg>`,
};

function createImageImport({
  label = "Drop an image here",
  hint = "or click to browse — TIFF, PNG, JPEG, WebP",
  onFile,
} = {}) {
  const root = document.createElement("div");
  root.className = "image-import";
  root.dataset.state = "empty";
  root.tabIndex = 0;

  root.innerHTML = `
    <input type="file" class="ii-input" accept="image/png,image/jpeg,image/webp,.tif,.tiff" hidden />

    <div class="ii-empty">
      <div class="ii-icon">${ICONS.upload}</div>
      <p class="t-h5 ii-label">${label}</p>
      <p class="t-caption ii-hint">${hint}</p>
    </div>

    <div class="ii-filled">
      <img class="ii-preview" alt="" hidden />
      <div class="ii-placeholder-icon">${ICONS.file}</div>
      <div class="ii-chip">
        <span class="ii-chip-name t-cap-bold"></span>
        <span class="ii-chip-size t-caption"></span>
      </div>
      <div class="ii-controls">
        <button type="button" class="btn btn-icon ii-replace" title="Replace image">${ICONS.replace}</button>
        <button type="button" class="btn btn-icon ii-remove" title="Remove image">${ICONS.remove}</button>
      </div>
      <div class="ii-status t-caption" hidden></div>
    </div>

    <div class="ii-error t-caption" hidden></div>
  `;

  const input = root.querySelector(".ii-input");
  const previewImg = root.querySelector(".ii-preview");
  const placeholderIcon = root.querySelector(".ii-placeholder-icon");
  const chipName = root.querySelector(".ii-chip-name");
  const chipSize = root.querySelector(".ii-chip-size");
  const statusEl = root.querySelector(".ii-status");
  const errorEl = root.querySelector(".ii-error");
  const replaceButton = root.querySelector(".ii-replace");
  const removeButton = root.querySelector(".ii-remove");

  let currentFile = null;
  let objectUrl = null;
  let errorTimer = null;

  function showError(message) {
    clearTimeout(errorTimer);
    errorEl.textContent = message;
    errorEl.hidden = false;
    root.classList.add("has-error");
    errorTimer = setTimeout(() => {
      errorEl.hidden = true;
      root.classList.remove("has-error");
    }, 2600);
  }

  function selectFile(file) {
    if (!file) return;
    if (!isAcceptedFile(file)) {
      showError("Unsupported file — use TIFF, PNG, or JPEG");
      return;
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    currentFile = file;

    if (NATIVELY_PREVIEWABLE.test(file.name)) {
      objectUrl = URL.createObjectURL(file);
      previewImg.src = objectUrl;
      previewImg.hidden = false;
      placeholderIcon.hidden = true;
    } else {
      previewImg.hidden = true;
      placeholderIcon.hidden = false;
    }

    chipName.textContent = file.name;
    chipSize.textContent = formatBytes(file.size);
    statusEl.hidden = true;
    root.dataset.state = "filled";

    if (onFile) onFile(file);
  }

  function reset(notify = true) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    currentFile = null;
    input.value = "";
    previewImg.src = "";
    root.dataset.state = "empty";
    if (notify && onFile) onFile(null);
  }

  root.addEventListener("click", (event) => {
    if (root.dataset.state === "empty") input.click();
  });
  root.addEventListener("keydown", (event) => {
    if (root.dataset.state === "empty" && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      input.click();
    }
  });

  replaceButton.addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  removeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    reset();
  });

  input.addEventListener("click", (event) => event.stopPropagation());
  input.addEventListener("change", () => selectFile(input.files[0]));

  ["dragenter", "dragover"].forEach((name) =>
    root.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      root.dataset.drag = "true";
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    root.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      root.dataset.drag = "false";
    })
  );
  root.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

  return {
    el: root,
    getFile: () => currentFile,
    setPreviewUrl(url) {
      previewImg.src = url;
      previewImg.hidden = false;
      placeholderIcon.hidden = true;
    },
    setStatus(text) {
      if (text) {
        statusEl.textContent = text;
        statusEl.hidden = false;
      } else {
        statusEl.hidden = true;
      }
    },
    reset,
  };
}
