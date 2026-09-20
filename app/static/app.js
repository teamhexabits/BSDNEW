// BorderGuard AI - Intelligent Travel Document Screening Workstation Logic

const dropzone = document.getElementById("dropzone");
const multiDocInput = document.getElementById("multi-doc-input");
const browseBtn = document.getElementById("browse-btn");
const stagingList = document.getElementById("staging-list");
const stagingEmpty = document.getElementById("staging-empty");
const stagedCountBadge = document.getElementById("staged-count-badge");
const clearAllBtn = document.getElementById("clear-all-btn");
const screenBatchBtn = document.getElementById("screen-batch-btn");
const screenBtnText = document.getElementById("screen-btn-text");
const demoSpecimensBtn = document.getElementById("demo-specimens-btn");
const loadBatchDemoTop = document.getElementById("load-batch-demo-top");

// Results Elements
const loadingState = document.getElementById("loading-state");
const emptyState = document.getElementById("empty-state");
const resultsContainer = document.getElementById("results");

// Clearance Banner Elements
const scoreVal = document.getElementById("score");
const scoreCircle = document.getElementById("score-circle");
const riskBadge = document.getElementById("risk-badge");
const batchTag = document.getElementById("batch-tag");
const recommendationText = document.getElementById("recommendation-text");
const screeningSummary = document.getElementById("screening-summary");

// Meta Indicators
const metaDocCount = document.getElementById("meta-doc-count");
const metaConsistency = document.getElementById("meta-consistency");
const metaChecksStatus = document.getElementById("meta-checks-status");
const metaHighFindings = document.getElementById("meta-high-findings");

// Dossier Elements
const dossierName = document.getElementById("dossier-name");
const dossierDob = document.getElementById("dossier-dob");
const dossierGender = document.getElementById("dossier-gender");
const dossierNat = document.getElementById("dossier-nat");
const dossierPassport = document.getElementById("dossier-passport");
const dossierVisa = document.getElementById("dossier-visa");
const dossierAadhaar = document.getElementById("dossier-aadhaar");
const dossierDl = document.getElementById("dossier-dl");
const dossierAddress = document.getElementById("dossier-address");
const dossierConsistencyPill = document.getElementById("dossier-consistency-pill");

// Cross Checks & Tabs
const crossChecksList = document.getElementById("cross-checks-list");
const faceMatrixPanel = document.getElementById("face-matrix-panel");
const faceMatchesGrid = document.getElementById("face-matches-grid");
const docTabs = document.getElementById("doc-tabs");
const tabContent = document.getElementById("tab-content");
const toggleJsonBtn = document.getElementById("toggle-json-btn");
const rawJsonViewer = document.getElementById("raw-json-viewer");
const allFindingsList = document.getElementById("all-findings-list");
const findingsCount = document.getElementById("findings-count");

// Webcam Elements
const video = document.getElementById("webcam");
const canvas = document.getElementById("webcam-canvas");
const preview = document.getElementById("capture-preview");
const cameraStatus = document.getElementById("camera-status");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const startBtn = document.getElementById("start-camera");
const captureBtn = document.getElementById("capture-photo");
const retakeBtn = document.getElementById("retake-photo");
const faceVerifyResult = document.getElementById("face-verify-result");

// Staged files list
let stagedFiles = [];
let cameraStream = null;
let livePhotoBlob = null;
let currentBatchData = null;
let activeDocIndex = 0;

// Document Type Guessing Helper
function guessDocTypeFromName(name) {
  const lower = name.toLowerCase();
  if (lower.includes("pass") || lower.includes("passport")) return "indian_passport";
  if (lower.includes("visa")) return "visa";
  if (lower.includes("aadhaar") || lower.includes("adhaar") || lower.includes("uid")) return "national_id";
  if (lower.includes("dl") || lower.includes("licen") || lower.includes("drive")) return "driving_license";
  return "generic_document";
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

// Staging List UI Management
function updateStagingUI() {
  const count = stagedFiles.length;
  stagedCountBadge.textContent = `${count} Document${count === 1 ? "" : "s"}`;
  screenBtnText.textContent = `Screen All Documents (${count})`;
  screenBatchBtn.disabled = count === 0;
  clearAllBtn.hidden = count === 0;
  stagingEmpty.hidden = count > 0;

  stagingList.innerHTML = "";
  stagedFiles.forEach((item, index) => {
    const el = document.createElement("div");
    el.className = "staged-item";
    el.innerHTML = `
      <img src="${item.previewUrl}" class="staged-thumb" alt="thumb" />
      <div class="staged-meta">
        <div class="staged-name" title="${item.file.name}">${item.file.name}</div>
        <div class="staged-controls">
          <select class="staged-type-select" data-index="${index}">
            <option value="indian_passport" ${item.docType === "indian_passport" ? "selected" : ""}>Passport</option>
            <option value="visa" ${item.docType === "visa" ? "selected" : ""}>Visa</option>
            <option value="national_id" ${item.docType === "national_id" ? "selected" : ""}>Aadhaar</option>
            <option value="driving_license" ${item.docType === "driving_license" ? "selected" : ""}>Driving Licence</option>
            <option value="generic_document" ${item.docType === "generic_document" ? "selected" : ""}>Auto-Detect / Generic</option>
          </select>
          <span class="staged-size">${formatBytes(item.file.size)}</span>
        </div>
      </div>
      <button type="button" class="staged-remove-btn" data-index="${index}" title="Remove">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;
    stagingList.appendChild(el);
  });

  // Attach select & remove listeners
  stagingList.querySelectorAll(".staged-type-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      if (stagedFiles[idx]) {
        stagedFiles[idx].docType = e.target.value;
      }
    });
  });

  stagingList.querySelectorAll(".staged-remove-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.currentTarget.dataset.index, 10);
      removeStagedFile(idx);
    });
  });
}

function addFilesToStaging(files) {
  Array.from(files).forEach((file) => {
    if (!file.type.startsWith("image/") && !file.name.match(/\.(png|jpe?g|heic)$/i)) {
      return;
    }
    const guessed = guessDocTypeFromName(file.name);
    const previewUrl = URL.createObjectURL(file);
    stagedFiles.push({
      file,
      docType: guessed,
      previewUrl,
    });
  });
  updateStagingUI();
}

function removeStagedFile(index) {
  if (stagedFiles[index]) {
    URL.revokeObjectURL(stagedFiles[index].previewUrl);
    stagedFiles.splice(index, 1);
    updateStagingUI();
  }
}

function clearAllStaged() {
  stagedFiles.forEach((item) => URL.revokeObjectURL(item.previewUrl));
  stagedFiles = [];
  updateStagingUI();
}

// Drag and drop listeners
browseBtn.addEventListener("click", () => multiDocInput.click());
dropzone.addEventListener("click", () => multiDocInput.click());

multiDocInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    addFilesToStaging(e.target.files);
    multiDocInput.value = "";
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files.length > 0) {
    addFilesToStaging(dt.files);
  }
});

clearAllBtn.addEventListener("click", clearAllStaged);

// Webcam Functions
function setCameraStatus(text) {
  cameraStatus.textContent = text;
}

function stopCamera() {
  if (!cameraStream) return;
  for (const track of cameraStream.getTracks()) {
    track.stop();
  }
  cameraStream = null;
  video.srcObject = null;
}

startBtn.addEventListener("click", async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCameraStatus("Camera API unavailable in this browser.");
    return;
  }
  try {
    stopCamera();
    livePhotoBlob = null;
    preview.hidden = true;
    cameraPlaceholder.hidden = true;
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = cameraStream;
    video.hidden = false;
    captureBtn.disabled = false;
    retakeBtn.disabled = true;
    setCameraStatus("Camera is active. Frame traveller face and click Capture Frame.");
  } catch (err) {
    stopCamera();
    video.hidden = true;
    cameraPlaceholder.hidden = false;
    captureBtn.disabled = true;
    setCameraStatus(`Camera error: ${err.message || err.name}`);
  }
});

captureBtn.addEventListener("click", async () => {
  if (!cameraStream) return;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    setCameraStatus("Camera feed not ready.");
    return;
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);
  livePhotoBlob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  if (!livePhotoBlob) return;
  preview.src = URL.createObjectURL(livePhotoBlob);
  preview.hidden = false;
  video.hidden = true;
  stopCamera();
  captureBtn.disabled = true;
  retakeBtn.disabled = false;
  setCameraStatus("Live photo captured. Will be cross-matched against all documents.");
});

retakeBtn.addEventListener("click", () => {
  livePhotoBlob = null;
  preview.hidden = true;
  retakeBtn.disabled = true;
  captureBtn.disabled = true;
  startBtn.click();
});

// Run Screening Batch
async function executeBatchScreening(formData) {
  emptyState.hidden = true;
  resultsContainer.hidden = true;
  loadingState.hidden = false;

  try {
    const res = await fetch("/api/screen/batch", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errText = await res.text();
      alert(`Screening error (${res.status}): ${errText}`);
      loadingState.hidden = true;
      emptyState.hidden = false;
      return;
    }
    const data = await res.json();
    currentBatchData = data;
    renderBatchResults(data);
  } catch (err) {
    alert(`Network or server error: ${err.message}`);
    loadingState.hidden = true;
    emptyState.hidden = false;
  } finally {
    loadingState.hidden = true;
  }
}

screenBatchBtn.addEventListener("click", () => {
  if (stagedFiles.length === 0) return;
  const formData = new FormData();
  stagedFiles.forEach((item) => {
    formData.append("documents", item.file, item.file.name);
    formData.append("document_types", item.docType);
  });
  if (livePhotoBlob) {
    formData.append("live_photo", livePhotoBlob, "webcam_capture.jpg");
  }
  executeBatchScreening(formData);
});

// 1-Click Specimen Demo
async function executeDemoBatch() {
  emptyState.hidden = true;
  resultsContainer.hidden = true;
  loadingState.hidden = false;

  try {
    const res = await fetch("/api/demo/batch", { method: "POST" });
    if (!res.ok) {
      alert(await res.text());
      loadingState.hidden = true;
      emptyState.hidden = false;
      return;
    }
    const data = await res.json();
    currentBatchData = data;
    renderBatchResults(data);
  } catch (err) {
    alert(`Demo failed: ${err.message}`);
    loadingState.hidden = true;
    emptyState.hidden = false;
  } finally {
    loadingState.hidden = true;
  }
}

demoSpecimensBtn.addEventListener("click", executeDemoBatch);
loadBatchDemoTop.addEventListener("click", executeDemoBatch);

// Format Field Label Helper
function formatFieldLabel(key) {
  const map = {
    passport_number: "Passport Number",
    mrz_number: "MRZ Document Number",
    visa_number: "Visa / Sticker Number",
    travel_document_number: "Travel Document (Passport) Ref",
    aadhaar_number: "Aadhaar Number",
    aadhaar_display: "Aadhaar Number (UIDAI)",
    dl_number: "Driving Licence Number",
    full_name: "Full Legal Name",
    surname: "Surname",
    given_names: "Given Name(s)",
    date_of_birth: "Date of Birth",
    date_of_issue: "Date of Issue",
    date_of_expiry: "Date of Expiry",
    valid_from: "Visa Valid From",
    valid_until: "Visa Valid Until",
    nationality: "Nationality",
    issuing_country: "Issuing Country",
    issuing_post: "Issuing Authority / Post",
    issuing_state: "Issuing State",
    sex: "Gender",
    gender: "Gender",
    blood_group: "Blood Group",
    place_of_birth: "Place of Birth",
    place_of_issue: "Place of Issue",
    parent_or_spouse: "Guardian / Spouse",
    address: "Declared Address",
    visa_type: "Visa Type / Category",
    period_of_stay: "Period of Stay",
    entries: "Number of Entries",
    remarks: "Remarks / Conditions",
    mrz_line1: "MRZ Line 1",
    mrz_line2: "MRZ Line 2",
    vehicle_classes: "Authorized Vehicle Classes",
  };
  return map[key] || key.replace(/_/g, " ").toUpperCase();
}

// Render Results Dashboard
function renderBatchResults(data) {
  resultsContainer.hidden = false;
  loadingState.hidden = true;
  emptyState.hidden = true;

  // 1. Clearance Banner
  scoreVal.textContent = Math.round(data.overall_authenticity_score);
  batchTag.textContent = data.batch_id || "BATCH-SCREEN";
  riskBadge.textContent = `RISK: ${data.risk_level}`;
  riskBadge.className = `risk-badge ${data.risk_level}`;

  recommendationText.textContent = (data.recommendation || "").replace(/_/g, " ");
  screeningSummary.textContent = data.summary;

  // Score circle color
  if (data.overall_authenticity_score >= 80) {
    scoreCircle.style.borderColor = "var(--low)";
    scoreCircle.style.boxShadow = "0 0 25px rgba(61, 220, 151, 0.3)";
  } else if (data.overall_authenticity_score >= 55) {
    scoreCircle.style.borderColor = "var(--mid)";
    scoreCircle.style.boxShadow = "0 0 25px rgba(229, 185, 59, 0.3)";
  } else {
    scoreCircle.style.borderColor = "var(--high)";
    scoreCircle.style.boxShadow = "0 0 25px rgba(240, 93, 94, 0.3)";
  }

  // Meta grid
  metaDocCount.textContent = `${data.documents.length} Credential(s)`;
  metaConsistency.textContent = (data.dossier.identity_consistency || "CONSISTENT").replace(/_/g, " ");
  
  const passedChecks = data.cross_checks.filter(c => c.status === "MATCH").length;
  metaChecksStatus.textContent = `${passedChecks}/${data.cross_checks.length} Passed`;

  const highFindings = data.total_findings.filter(f => f.severity === "high").length;
  metaHighFindings.textContent = `${highFindings} Alert${highFindings === 1 ? "" : "s"}`;
  metaHighFindings.style.color = highFindings > 0 ? "var(--high)" : "var(--low)";

  // 2. Dossier Panel
  const d = data.dossier || {};
  dossierName.textContent = d.primary_name || "—";
  dossierDob.textContent = d.primary_dob || "—";
  dossierGender.textContent = d.primary_gender ? (d.primary_gender === "M" ? "Male (M)" : d.primary_gender === "F" ? "Female (F)" : d.primary_gender) : "—";
  dossierNat.textContent = d.primary_nationality || "—";
  dossierPassport.textContent = d.passport_number || "—";
  dossierVisa.textContent = d.visa_number || "—";
  dossierAadhaar.textContent = d.aadhaar_number || "—";
  dossierDl.textContent = d.dl_number || "—";
  dossierAddress.textContent = (d.addresses && d.addresses.length > 0) ? d.addresses.join(" · ") : "None declared";

  dossierConsistencyPill.textContent = (d.identity_consistency || "CONSISTENT").replace(/_/g, " ");
  dossierConsistencyPill.className = `dossier-pill ${d.identity_consistency}`;

  // 3. Cross Checks Matrix
  crossChecksList.innerHTML = "";
  if (data.cross_checks && data.cross_checks.length > 0) {
    data.cross_checks.forEach((chk) => {
      const card = document.createElement("div");
      card.className = "matrix-card";
      card.innerHTML = `
        <div class="matrix-left">
          <span class="matrix-name">${chk.name}</span>
          <span class="matrix-summary">${chk.summary}</span>
        </div>
        <span class="matrix-status-pill ${chk.status}">${chk.status}</span>
      `;
      crossChecksList.appendChild(card);
    });
  } else {
    crossChecksList.innerHTML = `<p class="hint">No cross-document checks available.</p>`;
  }

  // 4. Biometric Cross Face Matrix
  const faceMatches = data.cross_face_matches || [];
  const liveMatches = data.live_face_matches || [];
  if (faceMatches.length > 0 || liveMatches.length > 0) {
    faceMatrixPanel.hidden = false;
    faceMatchesGrid.innerHTML = "";

    faceMatches.forEach((m) => {
      const el = document.createElement("div");
      el.className = "face-match-card";
      el.innerHTML = `
        <div class="face-match-header">
          <span class="matrix-name">${m.doc1_type} ↔ ${m.doc2_type}</span>
          <span class="matrix-status-pill ${m.matched ? "MATCH" : "MISMATCH"}">${m.matched ? "MATCH" : "MISMATCH"}</span>
        </div>
        <div class="face-match-sim" style="color: ${m.matched ? "var(--low)" : "var(--high)"}">
          Similarity: ${m.similarity !== null ? m.similarity.toFixed(3) : "N/A"}
        </div>
        <p class="hint" style="margin:0.2rem 0 0">Doc photo cross-match (Threshold: ${m.threshold})</p>
      `;
      faceMatchesGrid.appendChild(el);
    });

    liveMatches.forEach((m) => {
      const el = document.createElement("div");
      el.className = "face-match-card";
      el.innerHTML = `
        <div class="face-match-header">
          <span class="matrix-name">WebCam ↔ ${m.doc_type}</span>
          <span class="matrix-status-pill ${m.matched ? "MATCH" : "MISMATCH"}">${m.matched ? "MATCH" : "MISMATCH"}</span>
        </div>
        <div class="face-match-sim" style="color: ${m.matched ? "var(--low)" : "var(--high)"}">
          Similarity: ${m.similarity !== null ? m.similarity.toFixed(3) : "N/A"}
        </div>
        <p class="hint" style="margin:0.2rem 0 0">Live face vs ${m.filename}</p>
      `;
      faceMatchesGrid.appendChild(el);
    });
  } else {
    faceMatrixPanel.hidden = true;
  }

  // 5. Document Tabs
  docTabs.innerHTML = "";
  data.documents.forEach((doc, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `doc-tab-btn ${idx === 0 ? "active" : ""}`;
    const cleanType = (doc.document_type || "Doc").replace(/_/g, " ").toUpperCase();
    btn.textContent = `[${idx + 1}] ${cleanType}`;
    btn.addEventListener("click", () => {
      docTabs.querySelectorAll(".doc-tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeDocIndex = idx;
      renderActiveDocTab(doc);
    });
    docTabs.appendChild(btn);
  });

  if (data.documents.length > 0) {
    activeDocIndex = 0;
    renderActiveDocTab(data.documents[0]);
  }

  // 6. Adverse Findings
  allFindingsList.innerHTML = "";
  findingsCount.textContent = `${data.total_findings.length} findings`;
  if (data.total_findings.length === 0) {
    allFindingsList.innerHTML = `<li class="low">✓ No adverse forensic or validation findings detected across submitted credentials.</li>`;
  } else {
    data.total_findings.forEach((f) => {
      const li = document.createElement("li");
      li.className = f.severity;
      li.innerHTML = `<strong>[${f.module.toUpperCase()} / ${f.severity.toUpperCase()}] ${f.code}</strong> — ${f.message}`;
      allFindingsList.appendChild(li);
    });
  }

  // Raw JSON
  rawJsonViewer.textContent = JSON.stringify(data, null, 2);
}

// Render Deep Dive Tab Content
function renderActiveDocTab(doc) {
  const fields = doc.extracted_fields || {};
  const modules = [doc.ocr, doc.validation, doc.tampering, doc.face].filter(Boolean);

  // Module scores row
  let modulesHtml = `<div class="doc-modules-row">`;
  modules.forEach((m) => {
    modulesHtml += `
      <div class="module-mini-card">
        <div class="mod-title">${m.name}</div>
        <div class="mod-score">${Math.round(m.score)} <span style="font-size:0.75rem;color:var(--text-sub)">/ 100</span></div>
        <div class="progress-track">
          <div class="progress-fill" style="width:${m.score}%"></div>
        </div>
        <p class="hint" style="margin:0.35rem 0 0;font-size:0.75rem">${m.summary || ""}</p>
      </div>
    `;
  });
  modulesHtml += `</div>`;

  // Extracted fields grid
  let fieldsHtml = `<div class="extracted-grid">`;
  const skipKeys = new Set(["raw_text", "image_size", "checks"]);

  Object.entries(fields).forEach(([k, v]) => {
    if (skipKeys.has(k) || v === null || v === "" || v === undefined) return;
    const isMrz = k.includes("mrz");
    const isFull = isMrz || k === "address" || Array.isArray(v);
    const displayVal = Array.isArray(v) ? v.join(", ") : String(v);

    fieldsHtml += `
      <div class="field-cell ${isFull ? "full-span" : ""}">
        <span class="field-label">${formatFieldLabel(k)}</span>
        <span class="field-val ${isMrz ? "mrz" : ""}">${displayVal}</span>
      </div>
    `;
  });
  fieldsHtml += `</div>`;

  // Specific doc findings
  let docFindingsHtml = "";
  if (doc.findings && doc.findings.length > 0) {
    docFindingsHtml = `<div style="margin-top:0.8rem"><h4 style="margin:0 0 0.4rem;font-size:0.85rem;color:var(--text-muted)">DOCUMENT FINDINGS</h4><ul class="findings-list">`;
    doc.findings.forEach(f => {
      docFindingsHtml += `<li class="${f.severity}">[${f.code}] ${f.message}</li>`;
    });
    docFindingsHtml += `</ul></div>`;
  }

  tabContent.innerHTML = modulesHtml + fieldsHtml + docFindingsHtml;
}

// Toggle JSON viewer
toggleJsonBtn.addEventListener("click", () => {
  const isHidden = rawJsonViewer.hidden;
  rawJsonViewer.hidden = !isHidden;
  toggleJsonBtn.textContent = isHidden ? "Hide JSON" : "View JSON";
});

window.addEventListener("beforeunload", stopCamera);
