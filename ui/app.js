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
  const form = document.getElementById("match-form");
  const submitButton = document.getElementById("match-submit");
  const status = document.getElementById("match-status");
  const output = document.getElementById("match-output");

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

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.textContent = "running...";
    output.hidden = true;

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

    document.getElementById("result-preview").src = `/api/file/${result.result_id}/preview.jpg?t=${Date.now()}`;
    document.getElementById("match-download").href = `/api/file/${result.result_id}/download`;

    renderReport(document.getElementById("match-report"), [
      ["Delta-E mean", result.delta_e_mean.toFixed(2)],
      ["Delta-E max", result.delta_e_max.toFixed(2)],
      ["Bit depth collapsed", result.bit_depth_collapsed ? "yes" : "no", result.bit_depth_collapsed ? "bad" : "good"],
      ["ICC profile", result.icc_profile_present ? "present" : "missing", result.icc_profile_present ? "good" : "bad"],
    ]);
    output.hidden = false;
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

// matplotlib's tab20, https://matplotlib.org/stable/gallery/color/colormap_reference.html -
// 10 hue families, each a [saturated, light] pair. Replaces an earlier hue-rotation formula
// that turned out to be the same "garish, unrelated colors" mistake baked into the reference
// face-parsing repo's own visualization code (zllrunning/face-parsing.PyTorch's vis_parsing_maps
// uses the same kind of raw hue-cycle) - not a one-off bug, a known failure mode in this space.
const TAB20 = [
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
  "#17becf", "#9edae5",
];

// hand-assigned rather than palette[index] - pairs bilateral and otherwise-related regions
// onto the same hue family's saturated/light slots (l_eye+r_eye, l_ear+r_ear, l_brow+r_brow,
// u_lip+l_lip, hair+hat, ear_r+neck_l as "accessories") so related parts read as connected
// instead of random. "background" is deliberately absent - it's never drawn.
const LABEL_COLOR_SLOT = {
  l_eye: 0, r_eye: 1,
  skin: 2, neck: 3,
  l_ear: 4, r_ear: 5,
  u_lip: 6, l_lip: 7,
  l_brow: 8, r_brow: 9,
  hair: 10, hat: 11,
  mouth: 12, nose: 13,
  cloth: 14,
  ear_r: 16, neck_l: 17,
  eye_g: 18,
};

// fixed anatomical order for the legend - sorting by pixel count instead (what an earlier
// version did) reshuffles the list on every photo and breaks the bilateral pairing above,
// since whichever side happens to have marginally more pixels jumps around independently
const LEGEND_ORDER = [
  "skin", "hair", "hat",
  "l_eye", "r_eye", "l_brow", "r_brow",
  "nose", "l_ear", "r_ear", "ear_r", "eye_g",
  "mouth", "u_lip", "l_lip",
  "neck", "neck_l", "cloth",
];

// the "bulk" regions (skin, hair, headwear, neck, clothing) wash the whole photo in color
// when active and add little - this is a region-inspection tool for retouching work, and the
// point is isolating small anatomical regions (see the README's dark-circle-correction example),
// so only those start checked. Bulk regions are still one click away via their checkbox.
const DEFAULT_ACTIVE_LABELS = new Set([
  "l_eye", "r_eye", "l_brow", "r_brow", "nose",
  "u_lip", "l_lip", "mouth", "l_ear", "r_ear", "ear_r", "eye_g",
]);

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildPalette(labelNames) {
  return labelNames.map((name) => {
    const slot = LABEL_COLOR_SLOT[name];
    return slot === undefined ? [122, 122, 122] : hexToRgb(TAB20[slot]);
  });
}

// marks pixels whose right or down neighbor has a different label - cheap single pass,
// checking two of four neighbors is enough to catch every boundary edge somewhere in the scan
function computeBoundaryMask(labelPixels, width, height) {
  const data = labelPixels.data;
  const at = (x, y) => data[(y * width + x) * 4];
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const here = at(x, y);
      const right = x + 1 < width ? at(x + 1, y) : here;
      const down = y + 1 < height ? at(x, y + 1) : here;
      mask[y * width + x] = here !== right || here !== down ? 1 : 0;
    }
  }
  return mask;
}

function setupFaceParsing() {
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
    labelPixels = null;
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
  let labelPixels = null; // ImageData of the (downsampled, lossless) label-index map
  let boundaryMask = null; // precomputed once per result, not per redraw - see computeBoundaryMask
  let palette = [];
  let hoveredIndex = null; // set while a legend row is hovered, dims every other active region
  const activeLabels = new Set();

  function fillAlpha() {
    return Number(opacitySlider.value) / 100;
  }

  // fill + a darker same-hue outline on boundary pixels, matching how detectron2's visualizer
  // pairs alpha fill with a full-opacity edge rather than flat fill alone (which is the "garish
  // paint bucket" look the first version had). Hovering a legend row pops that one region and
  // dims the rest instead of hiding them outright, so context isn't lost while isolating one.
  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
    if (!labelPixels) return;

    const base = fillAlpha();
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let p = 0; p < labelPixels.data.length; p += 4) {
      const labelIndex = labelPixels.data[p]; // grayscale source: R channel is the class index
      if (labelIndex === 0 || !activeLabels.has(labelIndex)) continue; // 0 = background

      const isBoundary = boundaryMask[p / 4] === 1;
      let alpha = isBoundary ? Math.min(1, base + 0.4) : base;
      if (hoveredIndex !== null) {
        alpha = labelIndex === hoveredIndex ? Math.min(1, alpha + 0.25) : alpha * 0.2;
      }

      const [r, g, b] = palette[labelIndex];
      const shade = isBoundary ? 0.7 : 1; // darker outline, same hue as the fill
      frame.data[p] = frame.data[p] * (1 - alpha) + r * shade * alpha;
      frame.data[p + 1] = frame.data[p + 1] * (1 - alpha) + g * shade * alpha;
      frame.data[p + 2] = frame.data[p + 2] * (1 - alpha) + b * shade * alpha;
    }
    ctx.putImageData(frame, 0, 0);
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

    status.textContent = "parsing... (the remote model can take a while on a cold start)";
    placeholder.hidden = true;
    canvasWrap.hidden = true;
    changePhotoButton.hidden = true;
    opacityRow.hidden = true;
    legend.hidden = true;

    const body = new FormData();
    body.append("image", file);

    let response;
    try {
      response = await fetch("/api/face-parse", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const result = await response.json();

    let source, labelsImg;
    try {
      [source, labelsImg] = await Promise.all([
        loadImage(`/api/file/${result.file_id}/preview.jpg?t=${Date.now()}`),
        loadImage(`/api/file/${result.labels_id}/download?t=${Date.now()}`),
      ]);
    } catch (err) {
      status.textContent = `couldn't load the result images: ${err}`;
      return;
    }

    sourceImage = source;
    canvas.width = labelsImg.naturalWidth;
    canvas.height = labelsImg.naturalHeight;

    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = labelsImg.naturalWidth;
    labelCanvas.height = labelsImg.naturalHeight;
    const labelCtx = labelCanvas.getContext("2d");
    labelCtx.drawImage(labelsImg, 0, 0);
    labelPixels = labelCtx.getImageData(0, 0, labelCanvas.width, labelCanvas.height);
    boundaryMask = computeBoundaryMask(labelPixels, canvas.width, canvas.height);

    palette = buildPalette(result.label_names);
    activeLabels.clear();
    hoveredIndex = null;

    const counts = result.label_counts;
    const present = LEGEND_ORDER.filter((name) => name in counts).map((name) => [name, counts[name]]);
    present
      .filter(([name]) => DEFAULT_ACTIVE_LABELS.has(name))
      .forEach(([name]) => activeLabels.add(result.label_names.indexOf(name)));

    legend.innerHTML = present
      .map(([name, count]) => {
        const index = result.label_names.indexOf(name);
        const [r, g, b] = palette[index];
        const checked = activeLabels.has(index) ? "checked" : "";
        return `
          <li class="legend-item" data-label-index="${index}">
            <label>
              <input type="checkbox" ${checked} data-label-index="${index}" />
              <span class="legend-swatch" style="background: rgb(${r},${g},${b})"></span>
              <span class="legend-name">${labelize(name)}</span>
            </label>
            <span class="legend-count">${count.toLocaleString()} px</span>
          </li>
        `;
      })
      .join("");

    legend.querySelectorAll("input[type=checkbox]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const index = Number(checkbox.dataset.labelIndex);
        if (checkbox.checked) activeLabels.add(index);
        else activeLabels.delete(index);
        redraw();
      });
    });

    legend.querySelectorAll(".legend-item").forEach((item) => {
      const index = Number(item.dataset.labelIndex);
      item.addEventListener("mouseenter", () => {
        hoveredIndex = index;
        redraw();
      });
      item.addEventListener("mouseleave", () => {
        hoveredIndex = null;
        redraw();
      });
    });

    status.textContent = present.length ? "done" : "done - no face regions detected in this image";
    mount.hidden = true;
    canvasWrap.hidden = false;
    changePhotoButton.hidden = false;
    opacityRow.hidden = false;
    legend.hidden = false;
    redraw();
  });
}

setupViewTabs();
setupInspect();
setupMatchLook();
setupFaceParsing();
