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

// evenly-spaced hues rather than a hand-picked palette - guarantees every label gets a
// visually distinct color regardless of how many of the 19 classes actually show up
function hslToRgb(hue, saturationPct, lightnessPct) {
  const s = saturationPct / 100;
  const l = lightnessPct / 100;
  const k = (n) => (n + hue / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}

function labelPalette(count) {
  return Array.from({ length: count }, (_, i) => hslToRgb(Math.round((360 * i) / count), 70, 55));
}

function setupFaceParsing() {
  const form = document.getElementById("face-form");
  const submitButton = document.getElementById("face-submit");
  const status = document.getElementById("face-status");
  const output = document.getElementById("face-output");
  const canvas = document.getElementById("face-canvas");
  const legend = document.getElementById("face-legend");

  const importer = createImageImport({
    label: "Drop a portrait here",
    hint: "or click to browse",
    onFile: (file) => {
      submitButton.disabled = !file;
    },
  });
  document.getElementById("face-import-mount").appendChild(importer.el);

  let sourceImage = null; // the uploaded photo, redrawn under the overlay on every toggle
  let labelPixels = null; // ImageData of the (downsampled, lossless) label-index map
  let palette = [];
  const activeLabels = new Set();

  // draws from scratch every time rather than patching pixels incrementally - simpler to get
  // right, and a few hundred thousand pixels is well under a frame even on modest hardware
  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
    if (!labelPixels) return;

    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < labelPixels.data.length; i += 4) {
      const labelIndex = labelPixels.data[i]; // grayscale source: R channel is the class index
      if (labelIndex === 0 || !activeLabels.has(labelIndex)) continue; // 0 = background
      const [r, g, b] = palette[labelIndex];
      frame.data[i] = frame.data[i] * 0.35 + r * 0.65;
      frame.data[i + 1] = frame.data[i + 1] * 0.35 + g * 0.65;
      frame.data[i + 2] = frame.data[i + 2] * 0.35 + b * 0.65;
    }
    ctx.putImageData(frame, 0, 0);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.textContent = "parsing... (the remote model can take a while on a cold start)";
    output.hidden = true;

    const body = new FormData();
    body.append("image", importer.getFile());

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

    palette = labelPalette(result.label_names.length);
    activeLabels.clear();

    const present = Object.entries(result.label_counts)
      .filter(([name]) => name !== "background")
      .sort((a, b) => b[1] - a[1]);
    present.forEach(([name]) => activeLabels.add(result.label_names.indexOf(name)));

    legend.innerHTML = present
      .map(([name, count]) => {
        const index = result.label_names.indexOf(name);
        const [r, g, b] = palette[index];
        return `
          <li class="legend-item">
            <label>
              <input type="checkbox" checked data-label-index="${index}" />
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

    status.textContent = present.length ? "done" : "done - no face regions detected in this image";
    output.hidden = false;
    redraw();
  });
}

setupViewTabs();
setupInspect();
setupMatchLook();
setupFaceParsing();
