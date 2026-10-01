const TONE_CLASS = { good: "tag-mint", bad: "tag-rose", warn: "tag-yellow" };

function renderReport(dl, entries) {
  dl.innerHTML = entries
    .map(([label, value, tone]) => {
      const dd = tone ? `<span class="tag ${TONE_CLASS[tone]}">${value}</span>` : value;
      return `<dt>${label}</dt><dd>${dd}</dd>`;
    })
    .join("");
}

function setupViewTabs() {
  const buttons = document.querySelectorAll("#view-tabs .pill-tab");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((b) => (b.dataset.active = "false"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      button.dataset.active = "true";
      document.getElementById(`view-${button.dataset.view}`).classList.add("active");
    });
  });
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

function setupAiCrop() {
  const submitButton = document.getElementById("crop-submit");
  const changePhotoButton = document.getElementById("crop-change-photo");
  const mount = document.getElementById("crop-import-mount");
  const placeholder = document.getElementById("crop-placeholder");
  const status = document.getElementById("crop-status");
  const canvasWrap = document.getElementById("crop-canvas-wrap");
  const canvas = document.getElementById("crop-canvas");
  const modeSelect = document.getElementById("crop-mode");
  const faceIndexRow = document.getElementById("crop-face-index-row");
  const faceIndexInput = document.getElementById("crop-face-index");
  const aspectSelect = document.getElementById("crop-aspect");
  const marginSlider = document.getElementById("crop-margin");
  const report = document.getElementById("crop-report");
  const download = document.getElementById("crop-download");

  let sourceImage = null;
  let bbox = null; // [x0, y0, x1, y1], already scaled to the preview's own stride
  let cropRect = null; // same scaling

  function resetResult() {
    mount.hidden = false;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    report.hidden = true;
    download.hidden = true;
    placeholder.hidden = false;
    status.textContent = "";
    sourceImage = null;
    bbox = null;
    cropRect = null;
  }

  const importer = createImageImport({
    label: "Drop an image here",
    hint: "or click to browse",
    onFile: (file) => {
      submitButton.disabled = !file;
      if (!file) resetResult();
    },
  });
  mount.appendChild(importer.el);

  changePhotoButton.addEventListener("click", () => importer.reset());

  modeSelect.addEventListener("change", () => {
    faceIndexRow.hidden = modeSelect.value !== "face";
  });

  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
    if (!bbox || !cropRect) return;

    // detected bbox: thin dashed outline - what the model found
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#f5dc5e";
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

    // computed crop rect: bold solid outline - what actually gets cropped
    ctx.setLineDash([]);
    ctx.strokeStyle = "#8a1224";
    ctx.lineWidth = 3;
    ctx.strokeRect(cropRect[0], cropRect[1], cropRect[2] - cropRect[0], cropRect[3] - cropRect[1]);
  }

  submitButton.addEventListener("click", async () => {
    const file = importer.getFile();
    if (!file) return;

    status.textContent = "computing crop...";
    placeholder.hidden = true;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    report.hidden = true;
    download.hidden = true;

    const body = new FormData();
    body.append("image", file);
    body.append("mode", modeSelect.value);
    body.append("aspect_ratio", aspectSelect.value);
    body.append("margin", String(Number(marginSlider.value) / 100));
    body.append("face_index", faceIndexInput.value);

    let response;
    try {
      response = await fetch("/api/ai-crop", { method: "POST", body });
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
    canvas.width = source.naturalWidth;
    canvas.height = source.naturalHeight;
    bbox = result.bbox;
    cropRect = result.crop_rect;

    renderReport(report, [
      ["Detected bbox", bbox.map((v) => v.toFixed(0)).join(", ")],
      ["Crop rect", cropRect.map((v) => v.toFixed(0)).join(", ")],
      ["ICC profile", result.icc_profile_present ? "present" : "missing", result.icc_profile_present ? "good" : "bad"],
    ]);
    download.href = `/api/file/${result.result_id}/download`;

    status.textContent = "done";
    mount.hidden = true;
    canvasWrap.hidden = false;
    changePhotoButton.hidden = false;
    report.hidden = false;
    download.hidden = false;
    redraw();
  });
}

setupViewTabs();
setupInspect();
setupMatchLook();
setupFaceRegions();
setupRetouchFaces();
setupLayerSeparation();
setupAiCrop();
