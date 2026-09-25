function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function renderReport(dl, entries) {
  dl.innerHTML = entries
    .map(([label, value, warn]) => `<dt>${label}</dt><dd${warn ? ' class="warn"' : ""}>${value}</dd>`)
    .join("");
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
      ["delta-E mean", result.delta_e_mean.toFixed(2)],
      ["delta-E max", result.delta_e_max.toFixed(2)],
      ["bit depth collapsed", result.bit_depth_collapsed, result.bit_depth_collapsed],
      ["ICC profile present", result.icc_profile_present, !result.icc_profile_present],
    ]);
    output.hidden = false;
  });
}

function setupInspect() {
  const form = document.getElementById("inspect-form");
  const status = document.getElementById("inspect-status");
  const output = document.getElementById("inspect-output");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.textContent = "reading...";
    output.hidden = true;

    const body = new FormData();
    body.append("image", document.getElementById("inspect-image").files[0]);

    let response;
    try {
      response = await fetch("/api/inspect", { method: "POST", body });
    } catch (err) {
      status.textContent = `request failed: ${err}`;
      return;
    }

    if (!response.ok) {
      status.textContent = `server error: ${response.status}`;
      return;
    }

    const info = await response.json();
    status.textContent = "done";

    document.getElementById("inspect-preview").src = `/api/file/${info.file_id}/preview.png?t=${Date.now()}`;

    renderReport(document.getElementById("inspect-report"), [
      ["dimensions", `${info.width} x ${info.height}`],
      ["channels", info.channels],
      ["dtype", info.dtype],
      ["bit depth", `${info.bit_depth}-bit`],
      ["file size", `${(info.file_size_bytes / 1024).toFixed(1)} KB`],
      ["unique values / channel", info.unique_values_per_channel.join(", ")],
      ["looks upsampled from 8-bit", info.looks_upsampled_from_8bit, info.looks_upsampled_from_8bit],
      ["ICC profile present", info.icc_profile_present, !info.icc_profile_present],
      ["color space guess", info.colourspace_guess],
      ["DPI", info.dpi ? info.dpi.map((v) => v.toFixed(0)).join(" x ") : "not set"],
      ["compression", info.compression || "n/a"],
    ]);

    const exifBlock = document.getElementById("exif-block");
    const exifEntries = Object.entries(info.exif || {});
    if (exifEntries.length) {
      renderReport(document.getElementById("exif-report"), exifEntries);
      exifBlock.hidden = false;
    } else {
      exifBlock.hidden = true;
    }

    output.hidden = false;
  });
}

setupTabs();
setupMatchLook();
setupInspect();
