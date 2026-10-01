const TONE_CLASS = { good: "tag-mint", bad: "tag-rose", warn: "tag-yellow" };

function renderReport(dl, entries) {
  dl.innerHTML = entries
    .map(([label, value, tone]) => {
      const dd = tone ? `<span class="tag ${TONE_CLASS[tone]}">${value}</span>` : value;
      return `<dt>${label}</dt><dd>${dd}</dd>`;
    })
    .join("");
}

const ACTIVE_TAB_STORAGE_KEY = "ai-prepress-active-tab";

function setupViewTabs() {
  const buttons = document.querySelectorAll("#view-tabs .pill-tab");

  function activateTab(view) {
    const button = Array.from(buttons).find((b) => b.dataset.view === view);
    if (!button) return false;
    buttons.forEach((b) => (b.dataset.active = "false"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    button.dataset.active = "true";
    document.getElementById(`view-${view}`).classList.add("active");
    return true;
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      activateTab(button.dataset.view);
      // per-viewer convenience only (which tab was open) - never anything the app needs back,
      // so a blocked/cleared store just means it reopens on Inspect, not a broken feature
      try {
        localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, button.dataset.view);
      } catch {
        /* private browsing or blocked storage - fine, just won't remember the tab */
      }
    });
  });

  try {
    const savedView = localStorage.getItem(ACTIVE_TAB_STORAGE_KEY);
    if (savedView) activateTab(savedView);
  } catch {
    /* same as above - start on the default (Inspect) tab */
  }
}

function labelize(key) {
  return key.charAt(0).toUpperCase() + key.slice(1).replaceAll("_", " ");
}

function setupInspect() {
  const placeholder = document.getElementById("inspect-placeholder");
  const report = document.getElementById("inspect-report");
  const exifBlock = document.getElementById("exif-block");
  const rawTagsBlock = document.getElementById("raw-tags-block");

  const importer = createImageImport({
    label: "Drop an image here",
    hint: "or click to browse — TIFF, PNG, JPEG, WebP",
    onFile: (file) => {
      if (!file) {
        placeholder.hidden = false;
        report.hidden = true;
        exifBlock.hidden = true;
        rawTagsBlock.hidden = true;
        return;
      }
      runInspect(file);
    },
  });
  document.getElementById("inspect-import-mount").appendChild(importer.el);

  async function runInspect(file) {
    placeholder.hidden = true;
    report.hidden = true;
    exifBlock.hidden = true;
    rawTagsBlock.hidden = true;
    importer.setStatus("reading...");

    const body = new FormData();
    body.append("image", file);

    let response;
    try {
      response = await fetch("/api/inspect", { method: "POST", body });
    } catch (err) {
      importer.setStatus(`request failed: ${err}`);
      return;
    }

    if (!response.ok) {
      importer.setStatus(`server error: ${response.status}`);
      return;
    }

    const info = await response.json();
    importer.setStatus(null);
    importer.setPreviewUrl(`/api/file/${info.file_id}/preview.jpg?t=${Date.now()}`);

    // every field the profile itself declares - real data, not inferred.
    // empty when there's genuinely no profile to read.
    const iccRows = Object.entries(info.icc_profile || {}).map(([key, value]) => [labelize(key), value]);

    renderReport(report, [
      ["Dimensions", `${info.width} x ${info.height}`],
      ["Channels", info.channels],
      ["Dtype", info.dtype],
      ["Bit depth", `${info.bit_depth}-bit`],
      ["File size", `${(info.file_size_bytes / 1024).toFixed(1)} KB`],
      ["Unique values / ch", info.unique_values_per_channel.join(", ")],
      ["Upsampled from 8-bit?", info.looks_upsampled_from_8bit ? "yes" : "no", info.looks_upsampled_from_8bit ? "warn" : "good"],
      ["ICC profile", info.icc_profile_present ? "present" : "missing", info.icc_profile_present ? "good" : "bad"],
      ...iccRows,
      // heuristic bucket used internally for color-transfer math - shown alongside the
      // real profile fields above, not instead of them, since it isn't always the same thing
      ["Color space (guess)", info.colourspace_guess],
      ["DPI", info.dpi ? info.dpi.map((v) => v.toFixed(0)).join(" x ") : "not set"],
      ["Compression", info.compression || "n/a"],
    ]);
    report.hidden = false;

    const exifEntries = Object.entries(info.exif || {});
    if (exifEntries.length) {
      renderReport(document.getElementById("exif-report"), exifEntries);
      exifBlock.hidden = false;
    }

    const rawTagEntries = Object.entries(info.raw_tags || {});
    if (rawTagEntries.length) {
      renderReport(document.getElementById("raw-tags-report"), rawTagEntries);
      rawTagsBlock.hidden = false;
    }
  }
}

function setupMatchLook() {
  const submitButton = document.getElementById("match-submit");
  const changePhotosButton = document.getElementById("match-change-photos");
  const status = document.getElementById("match-status");
  const placeholder = document.getElementById("match-placeholder");
  const report = document.getElementById("match-report");
  const downloadLink = document.getElementById("match-download");
  const importPair = document.getElementById("match-import-pair");
  const resultPair = document.getElementById("match-result-pair");

  function refreshSubmitState() {
    submitButton.disabled = !(targetImporter.getFile() && referenceImporter.getFile());
  }

  const targetImporter = createImageImport({
    label: "Drop target image",
    hint: "or click to browse",
    onFile: refreshSubmitState,
  });
  document.getElementById("target-import-mount").appendChild(targetImporter.el);

  const referenceImporter = createImageImport({
    label: "Drop reference image",
    hint: "or click to browse",
    onFile: refreshSubmitState,
  });
  document.getElementById("reference-import-mount").appendChild(referenceImporter.el);

  changePhotosButton.addEventListener("click", () => {
    importPair.hidden = false;
    resultPair.hidden = true;
    changePhotosButton.hidden = true;
  });

  submitButton.addEventListener("click", async () => {
    status.textContent = "running...";
    placeholder.hidden = true;
    report.hidden = true;
    downloadLink.hidden = true;

    const body = new FormData();
    body.append("target", targetImporter.getFile());
    body.append("reference", referenceImporter.getFile());
    body.append("method", document.getElementById("method").value);

    let response;
    try {
      response = await fetch("/api/match-look", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();
    status.textContent = "done";

    document.getElementById("match-target-preview").src = `/api/file/${result.target_id}/preview.jpg?t=${Date.now()}`;
    document.getElementById("match-result-preview").src = `/api/file/${result.result_id}/preview.jpg?t=${Date.now()}`;
    downloadLink.href = `/api/file/${result.result_id}/download`;

    renderReport(report, [
      ["Delta-E mean", result.delta_e_mean.toFixed(2)],
      ["Delta-E max", result.delta_e_max.toFixed(2)],
      ["Bit depth collapsed", result.bit_depth_collapsed ? "yes" : "no", result.bit_depth_collapsed ? "bad" : "good"],
      ["ICC profile", result.icc_profile_present ? "present" : "missing", result.icc_profile_present ? "good" : "bad"],
    ]);
    report.hidden = false;
    downloadLink.hidden = false;
    importPair.hidden = true;
    resultPair.hidden = false;
    changePhotosButton.hidden = false;
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function hexToRgba(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// one polygon per retouching region ai_prepress.face_landmarks derives from the 478-point
// mesh - see that module's docstring for why these replaced the old 19-class semantic labels
// (there's no under-eye/cheek/forehead class in any face-parsing dataset's taxonomy, so no
// model swap was ever going to produce these regions; geometry from dense landmarks can).
// Colors reuse matplotlib's tab20 hue families, same as the old palette, for bilateral pairs.
const REGION_DEFS = [
  { key: "skin_oval", label: "Skin", color: "#bcbd22", cutoutsKey: "skin_cutouts" },
  { key: "forehead", label: "Forehead", color: "#ff7f0e" },
  { key: "cheek_right", label: "R cheek", color: "#17becf" },
  { key: "cheek_left", label: "L cheek", color: "#17becf" },
  { key: "under_eye_right", label: "R under-eye", color: "#e377c2" },
  { key: "under_eye_left", label: "L under-eye", color: "#e377c2" },
];

// skin is the one "bulk" region here (the rest are already small and targeted), so it's the
// one left unchecked by default - same reasoning as the old label defaults, just one region
// this time instead of a whole class of them.
const DEFAULT_ACTIVE_REGIONS = new Set([
  "forehead", "cheek_right", "cheek_left", "under_eye_right", "under_eye_left",
]);

function setupFaceRegions() {
  const submitButton = document.getElementById("face-submit");
  const changePhotoButton = document.getElementById("face-change-photo");
  const mount = document.getElementById("face-import-mount");
  const placeholder = document.getElementById("face-placeholder");
  const status = document.getElementById("face-status");
  const canvasWrap = document.getElementById("face-canvas-wrap");
  const canvas = document.getElementById("face-canvas");
  const opacityRow = document.getElementById("face-opacity-row");
  const opacitySlider = document.getElementById("face-opacity");
  const legend = document.getElementById("face-legend");

  function resetResult() {
    mount.hidden = false;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    opacityRow.hidden = true;
    legend.hidden = true;
    placeholder.hidden = false;
    status.textContent = "";
    sourceImage = null;
    regions = null;
  }

  const importer = createImageImport({
    label: "Drop a portrait here",
    hint: "or click to browse",
    onFile: (file) => {
      submitButton.disabled = !file;
      if (!file) resetResult();
    },
  });
  mount.appendChild(importer.el);

  changePhotoButton.addEventListener("click", () => importer.reset());

  let sourceImage = null; // the uploaded photo, redrawn under the overlay on every change
  let regions = null; // { region_key: [[x,y], ...], ... } from /api/face-regions
  let hoveredKey = null; // set while a legend row is hovered, dims every other active region
  const activeKeys = new Set();

  function fillAlpha() {
    return Number(opacitySlider.value) / 100;
  }

  function tracePath(ctx, points) {
    points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
  }

  // fill + a same-hue outline, matching the fill+outline convention professional segmentation
  // viewers (detectron2, CVAT) use over flat fill alone. Hovering a legend row pops that one
  // region and dims the rest instead of hiding them outright, so context isn't lost.
  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
    if (!regions) return;

    const base = fillAlpha();
    for (const def of REGION_DEFS) {
      if (!activeKeys.has(def.key) || !regions[def.key]) continue;
      let alpha = base;
      if (hoveredKey !== null) {
        alpha = def.key === hoveredKey ? Math.min(1, alpha + 0.25) : alpha * 0.2;
      }

      ctx.beginPath();
      tracePath(ctx, regions[def.key]);
      if (def.cutoutsKey && regions[def.cutoutsKey]) {
        for (const cutout of regions[def.cutoutsKey]) tracePath(ctx, cutout);
        ctx.fillStyle = hexToRgba(def.color, alpha);
        ctx.fill("evenodd");
      } else {
        ctx.fillStyle = hexToRgba(def.color, alpha);
        ctx.fill();
      }
      ctx.strokeStyle = hexToRgba(def.color, Math.min(1, alpha + 0.3));
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  let redrawQueued = false;
  function requestRedraw() {
    if (redrawQueued) return;
    redrawQueued = true;
    requestAnimationFrame(() => {
      redrawQueued = false;
      redraw();
    });
  }

  opacitySlider.addEventListener("input", requestRedraw);

  submitButton.addEventListener("click", async () => {
    const file = importer.getFile();
    if (!file) return;

    status.textContent = "finding regions...";
    placeholder.hidden = true;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    opacityRow.hidden = true;
    legend.hidden = true;

    const body = new FormData();
    body.append("image", file);

    let response;
    try {
      response = await fetch("/api/face-regions", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();
    if (!result.face_detected) {
      status.textContent = "no face detected in this image";
      mount.hidden = false;
      return;
    }

    let source;
    try {
      source = await loadImage(`/api/file/${result.file_id}/preview.jpg?t=${Date.now()}`);
    } catch (err) {
      status.textContent = `couldn't load the preview: ${err}`;
      return;
    }

    sourceImage = source;
    canvas.width = source.naturalWidth;
    canvas.height = source.naturalHeight;
    regions = result.regions;
    hoveredKey = null;
    activeKeys.clear();
    REGION_DEFS.filter((def) => DEFAULT_ACTIVE_REGIONS.has(def.key)).forEach((def) => activeKeys.add(def.key));

    legend.innerHTML = REGION_DEFS.map((def) => {
      const checked = activeKeys.has(def.key) ? "checked" : "";
      return `
        <li class="legend-item" data-region-key="${def.key}">
          <label>
            <input type="checkbox" ${checked} data-region-key="${def.key}" />
            <span class="legend-swatch" style="background: ${def.color}"></span>
            <span class="legend-name">${def.label}</span>
          </label>
        </li>
      `;
    }).join("");

    legend.querySelectorAll("input[type=checkbox]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const key = checkbox.dataset.regionKey;
        if (checkbox.checked) activeKeys.add(key);
        else activeKeys.delete(key);
        redraw();
      });
    });

    legend.querySelectorAll(".legend-item").forEach((item) => {
      const key = item.dataset.regionKey;
      item.addEventListener("mouseenter", () => {
        hoveredKey = key;
        redraw();
      });
      item.addEventListener("mouseleave", () => {
        hoveredKey = null;
        redraw();
      });
    });

    status.textContent = "done";
    mount.hidden = true;
    canvasWrap.hidden = false;
    changePhotoButton.hidden = false;
    opacityRow.hidden = false;
    legend.hidden = false;
    redraw();
  });
}

// one strength value per slider key, all sliders default to their existing behavior's neutral
// point (0 for every effect strength, 1x/0 for feather/edge - matching RetouchStrengths'
// own Python-side dataclass defaults exactly, see retouch_faces.py)
function defaultRetouchStrengths() {
  return {
    dark_circles: 0, even_skin: 0, even_skin_texture: 0, contouring: 0,
    feather_amount: 1, edge_amount: 0,
    eye_whiten: 0, teeth_whiten: 0, lip_enhance: 0,
  };
}

function setupRetouchFaces() {
  const submitButton = document.getElementById("retouch-submit");
  const changePhotoButton = document.getElementById("retouch-change-photo");
  const mount = document.getElementById("retouch-import-mount");
  const placeholder = document.getElementById("retouch-placeholder");
  const status = document.getElementById("retouch-status");
  const resultPair = document.getElementById("retouch-result-pair");
  const report = document.getElementById("retouch-report");
  const downloadLink = document.getElementById("retouch-download");
  const faceList = document.getElementById("retouch-face-list");
  const controls = document.getElementById("retouch-controls");

  // slider UI ranges aren't all 0-100 the same way (even_skin_texture and edge_amount are
  // signed, feather_amount is a 0-200 percentage of the existing default) - one scale table
  // instead of repeating the conversion at every call site
  const sliders = {
    dark_circles: document.getElementById("retouch-dark-circles"),
    even_skin: document.getElementById("retouch-even-skin"),
    even_skin_texture: document.getElementById("retouch-even-skin-texture"),
    contouring: document.getElementById("retouch-contouring"),
    eye_whiten: document.getElementById("retouch-eye-whiten"),
    teeth_whiten: document.getElementById("retouch-teeth-whiten"),
    lip_enhance: document.getElementById("retouch-lip-enhance"),
    feather_amount: document.getElementById("retouch-feather"),
    edge_amount: document.getElementById("retouch-edge"),
  };
  const SLIDER_SCALE = {
    dark_circles: 100, even_skin: 100, even_skin_texture: 100, contouring: 100,
    eye_whiten: 100, teeth_whiten: 100, lip_enhance: 100,
    feather_amount: 100, edge_amount: 100,
  };

  let fileId = null;
  let faceCount = 0;
  let facesData = []; // raw /api/face-regions "faces" array (face_index, bbox, regions)
  let sourceImage = null; // preview image, cropped client-side for face-list thumbnails
  let selectedTarget = "all"; // "all" | a face_index number
  let usedPerFace = false; // did the user ever touch an individual face's sliders
  let strengthsByTarget = { all: defaultRetouchStrengths() };

  function currentStrengths() {
    return strengthsByTarget[selectedTarget];
  }

  function loadSlidersFromCurrentTarget() {
    const s = currentStrengths();
    for (const [key, slider] of Object.entries(sliders)) {
      slider.value = Math.round(s[key] * SLIDER_SCALE[key]);
    }
  }

  function refreshSubmitState() {
    submitButton.disabled = !(fileId && faceCount > 0);
  }

  function resetDetectionState() {
    fileId = null;
    faceCount = 0;
    facesData = [];
    sourceImage = null;
    selectedTarget = "all";
    usedPerFace = false;
    strengthsByTarget = { all: defaultRetouchStrengths() };

    mount.hidden = false;
    faceList.hidden = true;
    faceList.innerHTML = "";
    controls.hidden = true;
    resultPair.hidden = true;
    changePhotoButton.hidden = true;
    report.hidden = true;
    downloadLink.hidden = true;
    placeholder.hidden = false;
    status.textContent = "";
    refreshSubmitState();
  }

  function cropThumbnail(bbox) {
    const canvas = document.createElement("canvas");
    canvas.width = 40;
    canvas.height = 40;
    const ctx = canvas.getContext("2d");
    const [[x0, y0], [x1, y1]] = bbox;
    const w = Math.max(1, x1 - x0);
    const h = Math.max(1, y1 - y0);
    const pad = 0.2; // a little room around the face oval, not a razor-tight crop
    const sx = Math.max(0, x0 - w * pad);
    const sy = Math.max(0, y0 - h * pad);
    const sw = Math.min(sourceImage.naturalWidth - sx, w * (1 + 2 * pad));
    const sh = Math.min(sourceImage.naturalHeight - sy, h * (1 + 2 * pad));
    ctx.drawImage(sourceImage, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL();
  }

  function highlightSelectedRow() {
    faceList.querySelectorAll(".legend-item").forEach((item) => {
      item.style.background = item.dataset.target === String(selectedTarget) ? "var(--surface-raised)" : "";
      item.style.opacity = item.dataset.target === "all" && usedPerFace ? "0.4" : "1";
    });
  }

  function renderFaceList() {
    const allRow = `
      <li class="legend-item" data-target="all">
        <label><span class="legend-swatch" style="background: var(--primary)"></span><span class="legend-name">All faces</span></label>
      </li>`;
    const faceRows = facesData.map(
      (face) => `
      <li class="legend-item" data-target="${face.face_index}">
        <label>
          <img class="legend-thumb" src="${cropThumbnail(face.bbox)}" alt="" />
          <span class="legend-name">Face ${face.face_index + 1}</span>
        </label>
      </li>`
    );
    faceList.innerHTML = [allRow, ...faceRows].join("");
    faceList.hidden = false;

    faceList.querySelectorAll(".legend-item").forEach((item) => {
      item.addEventListener("click", () => {
        const target = item.dataset.target;
        if (target === "all" && usedPerFace) {
          // once any individual face has been customized, "All faces" edits would silently be
          // dropped at submit time (the API sends either an all_faces broadcast or a per_face
          // map, never both) - refusing the click here is clearer than a silent no-op later
          status.textContent = "per-face settings are active for this photo - edit individual faces, or Change photo to start over";
          return;
        }
        selectedTarget = target === "all" ? "all" : Number(target);
        loadSlidersFromCurrentTarget();
        highlightSelectedRow();
      });
    });
    highlightSelectedRow();
  }

  async function detectFaces(file) {
    status.textContent = "detecting faces...";

    const body = new FormData();
    body.append("image", file);
    let response;
    try {
      response = await fetch("/api/face-regions", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }
    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();
    fileId = result.file_id;
    faceCount = result.face_count;
    facesData = result.faces;
    refreshSubmitState();

    if (faceCount === 0) {
      status.textContent = "no face detected in this image";
      return;
    }

    strengthsByTarget = { all: defaultRetouchStrengths() };
    for (let i = 0; i < faceCount; i++) strengthsByTarget[i] = defaultRetouchStrengths();
    selectedTarget = "all";
    usedPerFace = false;

    try {
      sourceImage = await loadImage(`/api/file/${fileId}/preview.jpg?t=${Date.now()}`);
    } catch (err) {
      status.textContent = `couldn't load the preview: ${err}`;
      return;
    }

    placeholder.hidden = true;
    renderFaceList();
    controls.hidden = false;
    loadSlidersFromCurrentTarget();
    status.textContent = faceCount === 1 ? "1 face detected" : `${faceCount} faces detected`;
  }

  const importer = createImageImport({
    label: "Drop a portrait here",
    hint: "or click to browse",
    onFile: (file) => {
      resetDetectionState();
      if (file) detectFaces(file);
    },
  });
  mount.appendChild(importer.el);

  Object.entries(sliders).forEach(([key, slider]) => {
    slider.addEventListener("input", () => {
      currentStrengths()[key] = Number(slider.value) / SLIDER_SCALE[key];
      if (selectedTarget !== "all" && !usedPerFace) {
        usedPerFace = true;
        highlightSelectedRow(); // grey out "All faces" now that it's a dead end for this photo
      }
    });
  });

  changePhotoButton.addEventListener("click", () => importer.reset());

  submitButton.addEventListener("click", async () => {
    if (!fileId || faceCount === 0) return;

    status.textContent = "retouching...";
    report.hidden = true;
    downloadLink.hidden = true;

    const payload = usedPerFace
      ? { file_id: fileId, mode: "per_face", per_face: Object.fromEntries(Array.from({ length: faceCount }, (_, i) => [i, strengthsByTarget[i]])) }
      : { file_id: fileId, mode: "all_faces", strengths: strengthsByTarget.all };

    let response;
    try {
      response = await fetch("/api/retouch-faces/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();
    if (!result.face_detected) {
      status.textContent = "no face detected in this image";
      return;
    }

    status.textContent = "done";
    document.getElementById("retouch-original-preview").src = `/api/file/${result.file_id}/preview.jpg?t=${Date.now()}`;
    document.getElementById("retouch-result-preview").src = `/api/file/${result.result_id}/preview.jpg?t=${Date.now()}`;
    downloadLink.href = `/api/file/${result.result_id}/download`;

    renderReport(report, [
      ["Delta-E mean", result.delta_e_mean.toFixed(2)],
      ["Delta-E max", result.delta_e_max.toFixed(2)],
      ["Bit depth collapsed", result.bit_depth_collapsed ? "yes" : "no", result.bit_depth_collapsed ? "bad" : "good"],
      ["ICC profile", result.icc_profile_present ? "present" : "missing", result.icc_profile_present ? "good" : "bad"],
    ]);
    report.hidden = false;
    downloadLink.hidden = false;
    mount.hidden = true;
    resultPair.hidden = false;
    changePhotoButton.hidden = false;
  });
}

function setupLayerSeparation() {
  const submitButton = document.getElementById("layer-sep-submit");
  const changePhotoButton = document.getElementById("layer-sep-change-photo");
  const mount = document.getElementById("layer-sep-import-mount");
  const placeholder = document.getElementById("layer-sep-placeholder");
  const status = document.getElementById("layer-sep-status");
  const canvasWrap = document.getElementById("layer-sep-canvas-wrap");
  const canvas = document.getElementById("layer-sep-canvas");
  const ctx = canvas.getContext("2d");
  const layerList = document.getElementById("layer-sep-list");
  const showAllButton = document.getElementById("layer-sep-show-all");
  const downloadAllButton = document.getElementById("layer-sep-download-all");

  let file = null;
  // one row per stacking element, in DRAW order (bottom to top): original, background,
  // layer 1..N ascending - matches the model's own bottom-to-top layer_index convention
  // (see deploy/layer_separation.py's module docstring), so the sidebar list (which shows
  // topmost-first, Photoshop convention) is a reversal of this array, not a second source of
  // truth for ordering.
  let rows = [];

  function resetState() {
    file = null;
    rows = [];
    mount.hidden = false;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    showAllButton.hidden = true;
    layerList.hidden = true;
    layerList.innerHTML = "";
    downloadAllButton.hidden = true;
    placeholder.hidden = false;
    status.textContent = "";
    submitButton.disabled = true;
  }

  function buildRows(result) {
    const built = [
      { kind: "original", label: "Original", resultId: result.file_id, visible: false, image: null },
      { kind: "background", label: "Background", resultId: result.background_id, visible: true, image: null },
    ];
    for (const layer of result.layers) {
      built.push({
        kind: "layer",
        // pre-filled from layer_naming (InternVL3.5-2B) when it succeeded - falls back to the
        // generic placeholder if naming failed or wasn't run, same manual rename field either way
        label: layer.suggested_name || `Layer ${layer.layer_index + 1}`,
        resultId: layer.result_id,
        visible: true,
        image: null,
        coverage: layer.alpha_coverage,
      });
    }
    return built;
  }

  async function loadRowImages() {
    await Promise.all(
      rows.map(async (row) => {
        row.image = await loadImage(`/api/file/${row.resultId}/preview.png?t=${Date.now()}`);
      })
    );
  }

  function composite() {
    const loaded = rows.find((row) => row.image);
    if (!loaded) return;
    canvas.width = loaded.image.naturalWidth;
    canvas.height = loaded.image.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // draw order is `rows`' own order (bottom to top) - checkerboard shows through wherever no
    // visible row covers a pixel, same as any real layered file
    for (const row of rows) {
      if (row.visible && row.image) ctx.drawImage(row.image, 0, 0);
    }
  }

  function rowMarkup(row) {
    const index = rows.indexOf(row);
    const thumbSrc = `/api/file/${row.resultId}/preview.png?t=${Date.now()}`;
    const nameHtml =
      row.kind === "layer"
        ? `<input type="text" class="legend-name-input" data-row-index="${index}" value="${row.label}" />`
        : `<span class="legend-name">${row.label}</span>`;
    const coverageHtml =
      row.kind === "layer" ? `<span class="tag tag-indigo layer-sep-coverage">${Math.round(row.coverage * 100)}%</span>` : "";
    return `
      <li class="legend-item">
        <div class="legend-item-main">
          <input type="checkbox" class="layer-sep-eye" data-row-index="${index}" ${row.visible ? "checked" : ""} />
          <img class="legend-thumb checkerboard" data-row-index="${index}" src="${thumbSrc}" alt="" title="Click to view this alone" />
          ${nameHtml}
        </div>
        <div class="legend-item-side">
          ${coverageHtml}
          <a class="btn-link" data-download-for="${index}" download="${row.label}.tiff" href="/api/file/${row.resultId}/download">Download</a>
        </div>
      </li>`;
  }

  function renderList() {
    // sidebar shows topmost-first (Photoshop convention) - a reversal of `rows`' own bottom-to-
    // top draw order, not a second ordering to keep in sync
    const original = rows.find((row) => row.kind === "original");
    const layers = rows
      .filter((row) => row.kind === "layer")
      .slice()
      .reverse();
    const background = rows.find((row) => row.kind === "background");

    const sections = [rowMarkup(original)];
    if (layers.length) {
      sections.push('<hr class="layer-sep-divider" />', ...layers.map(rowMarkup));
    }
    sections.push('<hr class="layer-sep-divider" />', rowMarkup(background));

    layerList.innerHTML = sections.join("");
    layerList.hidden = false;
    showAllButton.hidden = false;
    downloadAllButton.hidden = false;

    layerList.querySelectorAll(".layer-sep-eye").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        rows[Number(checkbox.dataset.rowIndex)].visible = checkbox.checked;
        composite();
      });
    });

    // click-to-solo (a real Photoshop convention: alt/option-clicking an eye icon) - "Show all"
    // is the deliberately simple way back, not a remembered-previous-state toggle
    layerList.querySelectorAll(".legend-thumb").forEach((thumb) => {
      thumb.addEventListener("click", () => {
        const soloIndex = Number(thumb.dataset.rowIndex);
        rows.forEach((row, i) => (row.visible = i === soloIndex));
        renderList();
        composite();
      });
    });

    // the manual-rename field standing in for the naming VLM (see the plan's roadmap) - purely
    // client-side, no backend change, both the download link and "Download all"'s payload read
    // the row's own current label
    layerList.querySelectorAll(".legend-name-input").forEach((input) => {
      input.addEventListener("input", () => {
        const index = Number(input.dataset.rowIndex);
        rows[index].label = input.value || rows[index].label;
        const link = layerList.querySelector(`[data-download-for="${index}"]`);
        if (link) link.download = `${rows[index].label}.tiff`;
      });
    });
  }

  const importer = createImageImport({
    label: "Drop a photo here",
    hint: "or click to browse",
    onFile: (chosenFile) => {
      resetState();
      if (chosenFile) {
        file = chosenFile;
        submitButton.disabled = false;
      }
    },
  });
  mount.appendChild(importer.el);

  changePhotoButton.addEventListener("click", () => {
    importer.reset();
    resetState();
  });

  showAllButton.addEventListener("click", () => {
    rows.forEach((row) => (row.visible = true));
    renderList();
    composite();
  });

  downloadAllButton.addEventListener("click", async () => {
    const files = rows
      .filter((row) => row.kind !== "original")
      .map((row) => ({ result_id: row.resultId, filename: `${row.label}.tiff` }));

    downloadAllButton.disabled = true;
    downloadAllButton.textContent = "Zipping...";

    let response;
    try {
      response = await fetch("/api/layer-separation/download-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files }),
      });
    } catch (err) {
      status.textContent = `download failed: ${err}`;
      downloadAllButton.disabled = false;
      downloadAllButton.textContent = "Download all (.zip)";
      return;
    }
    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      downloadAllButton.disabled = false;
      downloadAllButton.textContent = "Download all (.zip)";
      return;
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "layers.zip";
    link.click();
    URL.revokeObjectURL(url);

    downloadAllButton.disabled = false;
    downloadAllButton.textContent = "Download all (.zip)";
  });

  submitButton.addEventListener("click", async () => {
    if (!file) return;

    submitButton.disabled = true;
    status.textContent = "separating layers - this can take a few minutes on a cold start...";
    canvasWrap.hidden = true;
    layerList.hidden = true;

    const body = new FormData();
    body.append("image", file);

    let response;
    try {
      response = await fetch("/api/layer-separation", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      submitButton.disabled = false;
      return;
    }
    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      submitButton.disabled = false;
      return;
    }

    const result = await response.json();
    status.textContent = "loading previews...";

    rows = buildRows(result);
    try {
      await loadRowImages();
    } catch (err) {
      status.textContent = `couldn't load previews: ${err}`;
      submitButton.disabled = false;
      return;
    }

    status.textContent =
      result.layer_count === 0
        ? "no separable layers found - showing the reconstructed background only"
        : result.layer_count === 1
          ? "1 layer separated"
          : `${result.layer_count} layers separated`;

    renderList();
    composite();

    placeholder.hidden = true;
    mount.hidden = true;
    canvasWrap.hidden = false;
    changePhotoButton.hidden = false;
    submitButton.disabled = false;
  });
}

// ---- AI Crop geometry: exact JS port of ai_prepress.features.ai_crop.compute_crop -------------
// Kept in sync by hand with the Python version. This is what lets aspect ratio, margin, and
// manual drag/resize all update instantly with zero server round-trips - only Detect and Apply
// touch the server. See ai_prepress/features/ai_crop.py's own module docstring for why that
// split exists.
function computeCropRect(imgW, imgH, bbox, aspectRatio, margin, center) {
  const [x0, y0, x1, y1] = bbox;
  const bboxW = x1 - x0;
  const bboxH = y1 - y0;
  const [cx, cy] = center || [(x0 + x1) / 2, (y0 + y1) / 2];

  const pad = margin * Math.max(bboxW, bboxH);
  const paddedW = bboxW + 2 * pad;
  const paddedH = bboxH + 2 * pad;

  let cropW, cropH;
  if (paddedW / paddedH > aspectRatio) {
    cropW = paddedW;
    cropH = cropW / aspectRatio;
  } else {
    cropH = paddedH;
    cropW = cropH * aspectRatio;
  }

  // can't ask for a crop bigger than the source image itself along either axis
  const scale = Math.min(1, imgW / cropW, imgH / cropH);
  cropW *= scale;
  cropH *= scale;

  const left = Math.min(Math.max(cx - cropW / 2, 0), imgW - cropW);
  const top = Math.min(Math.max(cy - cropH / 2, 0), imgH - cropH);

  const rx0 = Math.round(left);
  const ry0 = Math.round(top);
  const rx1 = Math.min(imgW, rx0 + Math.round(cropW));
  const ry1 = Math.min(imgH, ry0 + Math.round(cropH));
  return [rx0, ry0, rx1, ry1];
}

function clampRectToImage(rect, imgW, imgH) {
  let [x0, y0, x1, y1] = rect;
  const w = x1 - x0;
  const h = y1 - y0;
  x0 = Math.min(Math.max(x0, 0), imgW - w);
  y0 = Math.min(Math.max(y0, 0), imgH - h);
  return [x0, y0, x0 + w, y0 + h];
}

function rectCenter(rect) {
  return [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2];
}

// desired on-SCREEN size in CSS px, not canvas-internal px - the canvas is sized to the full
// preview image (often 1000+ internal px) then shrunk by object-fit:contain to fit the viewport,
// so a fixed canvas-space constant here ends up a few real screen pixels, nearly impossible to
// grab with a mouse. Both the drawn handle and its hit-test radius get multiplied by the actual
// display scale (see displayScale() below) so they stay a constant, grabbable size on screen
// regardless of how far the canvas is shrunk for display.
const CROP_HANDLE_SCREEN_SIZE = 12; // drawn square, CSS px
const CROP_HANDLE_HIT_SCREEN_RADIUS = 16; // click target, CSS px - deliberately larger than the
// drawn handle itself, same "generous invisible hit area around a smaller visible control"
// convention most drag-handle UIs use (Figma, native OS resize corners)

function setupAiCrop() {
  const detectButton = document.getElementById("crop-detect");
  const applyButton = document.getElementById("crop-apply");
  const resetButton = document.getElementById("crop-reset");
  const changePhotoButton = document.getElementById("crop-change-photo");
  const mount = document.getElementById("crop-import-mount");
  const placeholder = document.getElementById("crop-placeholder");
  const status = document.getElementById("crop-status");
  const canvasWrap = document.getElementById("crop-canvas-wrap");
  const canvas = document.getElementById("crop-canvas");
  const modeSelect = document.getElementById("crop-mode");
  const candidateHint = document.getElementById("crop-candidate-hint");
  const aspectSelect = document.getElementById("crop-aspect");
  const customRow = document.getElementById("crop-custom-row");
  const customW = document.getElementById("crop-custom-w");
  const customH = document.getElementById("crop-custom-h");
  const swapButton = document.getElementById("crop-swap-orientation");
  const lockAspectCheckbox = document.getElementById("crop-lock-aspect");
  const marginSlider = document.getElementById("crop-margin");
  const gridSelect = document.getElementById("crop-grid");
  const report = document.getElementById("crop-report");
  const download = document.getElementById("crop-download");

  let sourceImage = null;
  let imgW = 0;
  let imgH = 0; // preview-space dimensions - what /detect returned, already stride-scaled
  let stride = 1;
  let fileId = null;
  let candidates = []; // [{bbox:[x0,y0,x1,y1], center:[cx,cy], label}]
  let selectedIndex = 0;
  let cropRect = null; // [x0,y0,x1,y1], preview space
  let currentCenter = null; // [cx,cy], preview space - follows manual drags/resizes
  let orientationSwapped = false;
  let dragMode = null; // null | "move" | "resize"
  let dragHandleAnchor = null; // the FIXED opposite corner while resizing
  let dragStart = null;
  let rectAtDragStart = null;

  function resetAll() {
    mount.hidden = false;
    canvasWrap.hidden = true;
    applyButton.hidden = true;
    resetButton.hidden = true;
    changePhotoButton.hidden = true;
    report.hidden = true;
    download.hidden = true;
    candidateHint.hidden = true;
    placeholder.hidden = false;
    status.textContent = "";
    sourceImage = null;
    candidates = [];
    cropRect = null;
    currentCenter = null;
    fileId = null;
  }

  const importer = createImageImport({
    label: "Drop an image here",
    hint: "or click to browse",
    onFile: (file) => {
      detectButton.disabled = !file;
      if (!file) resetAll();
    },
  });
  mount.appendChild(importer.el);

  changePhotoButton.addEventListener("click", () => importer.reset());
  // mode/aspect/margin/grid/lock settings deliberately survive "Change photo" - the same rule
  // reapplies to the next upload with no need to reconfigure every control, a lightweight version
  // of Capture One's own "set a reference crop, apply it to the next shot" idea (see README)
  // that doesn't need a multi-file batch queue to be useful.

  function aspectRatio() {
    let base;
    if (aspectSelect.value === "original") {
      base = imgW && imgH ? imgW / imgH : 1;
    } else if (aspectSelect.value === "custom") {
      base = (Number(customW.value) || 1) / (Number(customH.value) || 1);
    } else {
      base = Number(aspectSelect.value) || 1;
    }
    return orientationSwapped ? 1 / base : base;
  }

  function marginFraction() {
    return Number(marginSlider.value) / 100;
  }

  function recomputeFromCenter() {
    if (!candidates.length || !currentCenter) return;
    cropRect = computeCropRect(
      imgW, imgH, candidates[selectedIndex].bbox, aspectRatio(), marginFraction(), currentCenter
    );
    redraw();
  }

  function selectCandidate(index) {
    selectedIndex = index;
    currentCenter = candidates[index].center.slice();
    recomputeFromCenter();
  }

  aspectSelect.addEventListener("change", () => {
    customRow.hidden = aspectSelect.value !== "custom";
    recomputeFromCenter();
  });
  customW.addEventListener("input", recomputeFromCenter);
  customH.addEventListener("input", recomputeFromCenter);
  marginSlider.addEventListener("input", recomputeFromCenter);
  gridSelect.addEventListener("change", redraw);
  swapButton.addEventListener("click", () => {
    orientationSwapped = !orientationSwapped;
    recomputeFromCenter();
  });
  resetButton.addEventListener("click", () => selectCandidate(selectedIndex));

  // --- drawing --------------------------------------------------------------------------------

  function drawGrid(rect) {
    const kind = gridSelect.value;
    if (kind === "none") return;
    const [x0, y0, x1, y1] = rect;
    const w = x1 - x0;
    const h = y1 - y0;
    // rule of thirds splits at 1/3 and 2/3; a golden-ratio (phi) grid sits closer to center,
    // the standard approximation used across Lightroom/Photoshop/Affinity's own overlay sets
    const fracs = kind === "golden" ? [0.382, 0.618] : [1 / 3, 2 / 3];
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.65)";
    ctx.lineWidth = 1;
    for (const f of fracs) {
      ctx.beginPath();
      ctx.moveTo(x0 + w * f, y0);
      ctx.lineTo(x0 + w * f, y1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x0, y0 + h * f);
      ctx.lineTo(x1, y0 + h * f);
      ctx.stroke();
    }
    ctx.restore();
  }

  function cropHandlePoints(rect) {
    const [x0, y0, x1, y1] = rect;
    return [
      [x0, y0],
      [x1, y0],
      [x0, y1],
      [x1, y1],
    ];
  }

  // canvas-internal px per CSS px - multiply a desired on-screen size by this to get the
  // equivalent size in the coordinate space redraw()/hitHandle() both actually work in
  function displayScale() {
    const bounds = canvas.getBoundingClientRect();
    return bounds.width ? canvas.width / bounds.width : 1;
  }

  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

    // candidate markers only matter when there's more than one to choose between (several
    // faces, or a face set plus its "All faces" union) - a single candidate needs no picker
    if (candidates.length > 1) {
      candidates.forEach((c, i) => {
        const [cx, cy] = c.center;
        ctx.beginPath();
        ctx.arc(cx, cy, 7, 0, Math.PI * 2);
        ctx.fillStyle = i === selectedIndex ? "#8a1224" : "rgba(28,28,38,0.85)";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }

    if (!cropRect) return;
    drawGrid(cropRect);

    const [x0, y0, x1, y1] = cropRect;
    ctx.strokeStyle = "#8a1224";
    ctx.lineWidth = 3;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    const handleSize = CROP_HANDLE_SCREEN_SIZE * displayScale();
    const half = handleSize / 2;
    for (const [hx, hy] of cropHandlePoints(cropRect)) {
      ctx.fillStyle = "#8a1224";
      ctx.fillRect(hx - half, hy - half, handleSize, handleSize);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hx - half, hy - half, handleSize, handleSize);
    }
  }

  // --- interaction: drag to move, drag a corner to resize, click a marker to re-center --------

  function canvasPoint(event) {
    const bounds = canvas.getBoundingClientRect();
    const scaleX = canvas.width / bounds.width;
    const scaleY = canvas.height / bounds.height;
    return [(event.clientX - bounds.left) * scaleX, (event.clientY - bounds.top) * scaleY];
  }

  function hitHandle(point) {
    if (!cropRect) return null;
    const radius = CROP_HANDLE_HIT_SCREEN_RADIUS * displayScale();
    for (const corner of cropHandlePoints(cropRect)) {
      if (Math.abs(point[0] - corner[0]) <= radius && Math.abs(point[1] - corner[1]) <= radius) {
        return corner;
      }
    }
    return null;
  }

  function pointInRect(point, rect) {
    return point[0] >= rect[0] && point[0] <= rect[2] && point[1] >= rect[1] && point[1] <= rect[3];
  }

  canvas.addEventListener("mousedown", (event) => {
    if (!cropRect) return;
    const point = canvasPoint(event);
    const handle = hitHandle(point);

    if (handle) {
      dragMode = "resize";
      const [x0, y0, x1, y1] = cropRect;
      dragHandleAnchor = [
        Math.abs(handle[0] - x0) < Math.abs(handle[0] - x1) ? x1 : x0,
        Math.abs(handle[1] - y0) < Math.abs(handle[1] - y1) ? y1 : y0,
      ];
    } else if (pointInRect(point, cropRect)) {
      dragMode = "move";
      dragStart = point;
      rectAtDragStart = cropRect.slice();
    } else if (candidates.length > 1) {
      const index = candidates.findIndex((c) => pointInRect(point, c.bbox));
      if (index >= 0) selectCandidate(index);
      return;
    } else {
      return;
    }
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragMode) return;
    const point = canvasPoint(event);

    if (dragMode === "move") {
      const dx = point[0] - dragStart[0];
      const dy = point[1] - dragStart[1];
      const [sx0, sy0, sx1, sy1] = rectAtDragStart;
      cropRect = clampRectToImage([sx0 + dx, sy0 + dy, sx1 + dx, sy1 + dy], imgW, imgH);
    } else {
      // resize from a fixed opposite corner - aspect-locked resizing picks whichever axis
      // implies the larger rect, same "farthest drag wins" convention Photoshop/Figma use
      const [ax, ay] = dragHandleAnchor;
      let x0, y0, x1, y1;
      if (lockAspectCheckbox.checked) {
        const ratio = aspectRatio();
        const dx = point[0] - ax;
        const dy = point[1] - ay;
        const wFromX = Math.abs(dx);
        const hFromX = wFromX / ratio;
        const hFromY = Math.abs(dy);
        const wFromY = hFromY * ratio;
        const useX = wFromX * hFromX >= wFromY * hFromY;
        const w = useX ? wFromX : wFromY;
        const h = useX ? hFromX : hFromY;
        const signX = dx >= 0 ? 1 : -1;
        const signY = dy >= 0 ? 1 : -1;
        x0 = ax; y0 = ay; x1 = ax + signX * w; y1 = ay + signY * h;
      } else {
        x0 = ax; y0 = ay; x1 = point[0]; y1 = point[1];
      }
      const rect = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
      cropRect = clampRectToImage(rect, imgW, imgH);
    }
    currentCenter = rectCenter(cropRect);
    redraw();
  });

  window.addEventListener("mouseup", () => {
    dragMode = null;
  });

  canvasWrap.addEventListener("keydown", (event) => {
    if (!cropRect) return;
    const step = event.shiftKey ? 10 : 1;
    let dx = 0, dy = 0;
    if (event.key === "ArrowLeft") dx = -step;
    else if (event.key === "ArrowRight") dx = step;
    else if (event.key === "ArrowUp") dy = -step;
    else if (event.key === "ArrowDown") dy = step;
    else return;
    event.preventDefault();

    const [x0, y0, x1, y1] = cropRect;
    cropRect = clampRectToImage([x0 + dx, y0 + dy, x1 + dx, y1 + dy], imgW, imgH);
    currentCenter = rectCenter(cropRect);
    redraw();
  });

  // --- detect / apply ---------------------------------------------------------------------------

  detectButton.addEventListener("click", async () => {
    const file = importer.getFile();
    if (!file) return;

    status.textContent = "detecting...";
    placeholder.hidden = true;
    canvasWrap.hidden = true;
    applyButton.hidden = true;
    resetButton.hidden = true;
    changePhotoButton.hidden = true;
    report.hidden = true;
    download.hidden = true;
    candidateHint.hidden = true;

    const body = new FormData();
    body.append("image", file);
    body.append("mode", modeSelect.value);

    let response;
    try {
      response = await fetch("/api/ai-crop/detect", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      status.textContent = detail.detail || `server error: ${response.status}`;
      mount.hidden = false;
      return;
    }

    const result = await response.json();

    let source;
    try {
      source = await loadImage(`/api/file/${result.file_id}/preview.jpg?t=${Date.now()}`);
    } catch (err) {
      status.textContent = `couldn't load the preview: ${err}`;
      return;
    }

    sourceImage = source;
    fileId = result.file_id;
    stride = result.stride;
    imgW = source.naturalWidth;
    imgH = source.naturalHeight;
    canvas.width = imgW;
    canvas.height = imgH;
    candidates = result.candidates;
    selectedIndex = result.primary_index;
    currentCenter = candidates[selectedIndex].center.slice();

    if (candidates.length > 1) {
      candidateHint.textContent =
        result.mode_used === "face"
          ? `${candidates.length - 1} face(s) found - click one on the preview to crop around it (or "All faces", selected by default).`
          : `${candidates.length} candidates found - click one on the preview to crop around it.`;
      candidateHint.hidden = false;
    }

    recomputeFromCenter();

    status.textContent = `done (${result.mode_used} mode)`;
    mount.hidden = true;
    canvasWrap.hidden = false;
    applyButton.hidden = false;
    resetButton.hidden = false;
    changePhotoButton.hidden = false;
    canvasWrap.focus();
  });

  applyButton.addEventListener("click", async () => {
    if (!cropRect || !fileId) return;
    status.textContent = "applying...";

    const [px0, py0, px1, py1] = cropRect;
    const payload = {
      file_id: fileId,
      x0: Math.round(px0 * stride),
      y0: Math.round(py0 * stride),
      x1: Math.round(px1 * stride),
      y1: Math.round(py1 * stride),
    };

    let response;
    try {
      response = await fetch("/api/ai-crop/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();
    status.textContent = "done";

    renderReport(report, [
      ["Crop size (full-res)", `${payload.x1 - payload.x0} x ${payload.y1 - payload.y0}`],
      ["ICC profile", result.icc_profile_present ? "present" : "missing", result.icc_profile_present ? "good" : "bad"],
    ]);
    report.hidden = false;
    download.href = `/api/file/${result.result_id}/download`;
    download.hidden = false;
  });
}

setupViewTabs();
setupInspect();
setupMatchLook();
setupFaceRegions();
setupRetouchFaces();
setupLayerSeparation();
setupAiCrop();
