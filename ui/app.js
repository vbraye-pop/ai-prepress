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
      buttons.forEach((b) => b.dataset.active = "false");
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      button.dataset.active = "true";
      document.getElementById(`view-${button.dataset.view}`).classList.add("active");
    });
  });
}

function setupInspect() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("inspect-file-input");
  const workspace = document.getElementById("inspect-workspace");
  const resetButton = document.getElementById("inspect-reset");

  const runInspect = async (file) => {
    if (!file) return;

    const body = new FormData();
    body.append("image", file);

    dropzone.hidden = true;
    workspace.hidden = false;
    document.getElementById("inspect-filename").textContent = `reading ${file.name}...`;

    let response;
    try {
      response = await fetch("/api/inspect", { method: "POST", body });
    } catch (err) {
      document.getElementById("inspect-filename").textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      document.getElementById("inspect-filename").textContent = `server error: ${response.status}`;
      return;
    }

    const info = await response.json();
    document.getElementById("inspect-filename").textContent = file.name;
    document.getElementById("inspect-preview").src = `/api/file/${info.file_id}/preview.png?t=${Date.now()}`;

    renderReport(document.getElementById("inspect-report"), [
      ["Dimensions", `${info.width} x ${info.height}`],
      ["Channels", info.channels],
      ["Dtype", info.dtype],
      ["Bit depth", `${info.bit_depth}-bit`],
      ["File size", `${(info.file_size_bytes / 1024).toFixed(1)} KB`],
      ["Unique values / ch", info.unique_values_per_channel.join(", ")],
      ["Upsampled from 8-bit?", info.looks_upsampled_from_8bit ? "yes" : "no", info.looks_upsampled_from_8bit ? "warn" : "good"],
      ["ICC profile", info.icc_profile_present ? "present" : "missing", info.icc_profile_present ? "good" : "bad"],
      ["Color space guess", info.colourspace_guess],
      ["DPI", info.dpi ? info.dpi.map((v) => v.toFixed(0)).join(" x ") : "not set"],
      ["Compression", info.compression || "n/a"],
    ]);

    const exifBlock = document.getElementById("exif-block");
    const exifEntries = Object.entries(info.exif || {});
    if (exifEntries.length) {
      renderReport(document.getElementById("exif-report"), exifEntries);
      exifBlock.hidden = false;
    } else {
      exifBlock.hidden = true;
    }
  };

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", () => runInspect(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.dataset.drag = "true";
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.dataset.drag = "false";
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    runInspect(file);
  });

  resetButton.addEventListener("click", () => {
    workspace.hidden = true;
    dropzone.hidden = false;
    fileInput.value = "";
  });
}

function setupMatchLook() {
  const form = document.getElementById("match-form");
  const status = document.getElementById("match-status");
  const output = document.getElementById("match-output");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.textContent = "running...";
    output.hidden = true;

    const body = new FormData();
    body.append("target", document.getElementById("target").files[0]);
    body.append("reference", document.getElementById("reference").files[0]);
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

    const bust = `?t=${Date.now()}`;
    document.getElementById("target-preview").src = `/api/file/${result.target_id}/preview.png${bust}`;
    document.getElementById("reference-preview").src = `/api/file/${result.reference_id}/preview.png${bust}`;
    document.getElementById("result-preview").src = `/api/file/${result.result_id}/preview.png${bust}`;
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

setupViewTabs();
setupInspect();
setupMatchLook();
