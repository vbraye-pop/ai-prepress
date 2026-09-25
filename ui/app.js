const form = document.getElementById("match-form");
const status = document.getElementById("status");
const output = document.getElementById("output");
const preview = document.getElementById("preview");
const report = document.getElementById("report");
const download = document.getElementById("download");

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

  preview.src = `/api/result/${result.result_id}/preview.png?t=${Date.now()}`;
  download.href = `/api/result/${result.result_id}/download`;
  report.innerHTML = `
    <dt>delta-E mean</dt><dd>${result.delta_e_mean.toFixed(2)}</dd>
    <dt>delta-E max</dt><dd>${result.delta_e_max.toFixed(2)}</dd>
    <dt>bit depth collapsed</dt><dd>${result.bit_depth_collapsed}</dd>
    <dt>ICC profile present</dt><dd>${result.icc_profile_present}</dd>
  `;
  output.hidden = false;
});
