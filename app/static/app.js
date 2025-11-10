const state = {
  config: {
    departments: [],
    ui: {},
    security: {},
  },
  topicText: "",
  normalizedPreview: "",
  themes: [],
  selectedDepartment: "",
  results: [],
  resultLookup: new Map(),
  shortlist: [],
  shortlistOpen: false,
  isAnalyzing: false,
  isMatching: false,
};

const elements = {
  views: {
    home: document.getElementById("homeView"),
    themes: document.getElementById("themesView"),
    results: document.getElementById("resultsView"),
  },
  topicForm: document.getElementById("topicForm"),
  topicInput: document.getElementById("topicInput"),
  charCounter: document.getElementById("charCounter"),
  fileUpload: document.getElementById("fileUpload"),
  departmentFilter: document.getElementById("departmentFilter"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  themesChips: document.getElementById("themesChips"),
  addChipInput: document.getElementById("addChipInput"),
  addChipBtn: document.getElementById("addChipBtn"),
  normalizedPreview: document.getElementById("normalizedPreview"),
  runMatchBtn: document.getElementById("runMatchBtn"),
  backToInputBtn: document.getElementById("backToInputBtn"),
  resultsDepartmentFilter: document.getElementById("resultsDepartmentFilter"),
  resultsSort: document.getElementById("resultsSort"),
  resultsCount: document.getElementById("resultsCount"),
  backToThemesBtn: document.getElementById("backToThemesBtn"),
  resultsList: document.getElementById("resultsList"),
  shortlistButton: document.getElementById("shortlistButton"),
  shortlistDrawer: document.getElementById("shortlistDrawer"),
  closeShortlistBtn: document.getElementById("closeShortlistBtn"),
  shortlistItems: document.getElementById("shortlistItems"),
  exportPDFBtn: document.getElementById("exportPDFBtn"),
  exportJSONBtn: document.getElementById("exportJSONBtn"),
  modalBackdrop: document.getElementById("modalBackdrop"),
  modalBody: document.getElementById("modalBody"),
  modalTitle: document.getElementById("modalTitle"),
  modalCloseBtn: document.getElementById("modalCloseBtn"),
  toastContainer: document.getElementById("toastContainer"),
};

let shortlistSaveTimeout = null;

window.addEventListener("DOMContentLoaded", () => {
  setupListeners();
  initialize();
});

async function initialize() {
  try {
    await loadConfig();
    await loadShortlist();
    renderDepartmentOptions();
    updateShortlistBadge();
  } catch (error) {
    showToast(error.message || "Klarte ikke å laste konfigurasjon.", "error");
  }
}

function setupListeners() {
  elements.topicInput.addEventListener("input", handleTopicInput);
  elements.fileUpload.addEventListener("change", handleFileChange);
  elements.departmentFilter.addEventListener("change", (event) => {
    state.selectedDepartment = event.target.value;
  });

  elements.topicForm.addEventListener("submit", handleAnalyze);
  elements.addChipBtn.addEventListener("click", addChipFromInput);
  elements.addChipInput.addEventListener("keyup", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addChipFromInput();
    }
  });

  elements.runMatchBtn.addEventListener("click", fetchMatches);
  elements.resultsDepartmentFilter.addEventListener("change", (event) => {
    state.selectedDepartment = event.target.value;
    fetchMatches();
  });
  elements.resultsSort.addEventListener("change", () => {
    sortResults();
    renderResults();
  });
  elements.backToThemesBtn.addEventListener("click", () => setView("themes"));
  elements.backToInputBtn.addEventListener("click", () => setView("home"));

  elements.shortlistButton.addEventListener("click", () => toggleShortlist(true));
  elements.closeShortlistBtn.addEventListener("click", () => toggleShortlist(false));
  elements.exportJSONBtn.addEventListener("click", () => exportShortlist("json"));
  elements.exportPDFBtn.addEventListener("click", () => exportShortlist("pdf"));

  elements.modalCloseBtn.addEventListener("click", closeModal);
  elements.modalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.modalBackdrop) {
      closeModal();
    }
  });

  elements.themesChips.addEventListener("click", handleThemesChipClick);
  elements.resultsList.addEventListener("click", handleResultsClick);
  elements.shortlistItems.addEventListener("input", handleShortlistInput);
  elements.shortlistItems.addEventListener("click", handleShortlistClick);
}

async function loadConfig() {
  const response = await fetch("/config");
  if (!response.ok) {
    throw new Error("Klarte ikke å hente konfigurasjon.");
  }
  state.config = await response.json();
  updateFileHelpText();
}

function updateFileHelpText() {
  const help = document.getElementById("fileHelp");
  if (!help || !state.config.security) {
    return;
  }
  const formats = state.config.security.allowFileTypes || [];
  help.textContent = `Maks ${state.config.security.maxUploadMb} MB per fil. Støttede formater: ${formats
    .map((item) => item.toUpperCase())
    .join(", ")}.`;
  elements.fileUpload.setAttribute(
    "accept",
    formats.map((item) => `.${item}`).join(",")
  );
}

function renderDepartmentOptions() {
  const { departments } = state.config;
  const selects = [elements.departmentFilter, elements.resultsDepartmentFilter];
  selects.forEach((select) => {
    if (!select) return;
    select.innerHTML = '<option value="">Alle</option>';
    departments.forEach((dept) => {
      const option = document.createElement("option");
      option.value = dept;
      option.textContent = dept;
      select.appendChild(option);
    });
  });
}

async function loadShortlist() {
  try {
    const response = await fetch("/shortlist");
    if (!response.ok) {
      throw new Error("Feil ved lasting av kortliste.");
    }
    const data = await response.json();
    state.shortlist = data.items || [];
    renderShortlist();
  } catch (error) {
    showToast(error.message, "warning");
  }
}

function handleTopicInput(event) {
  state.topicText = event.target.value;
  const length = state.topicText.length;
  elements.charCounter.textContent = `${length} / 3200`;
  updateAnalyzeButtonState();
}

function handleFileChange() {
  updateAnalyzeButtonState();
}

function updateAnalyzeButtonState() {
  const textValue = elements.topicInput.value || "";
  state.topicText = textValue;
  const hasText = textValue.trim().length > 0;
  const hasFiles = elements.fileUpload.files && elements.fileUpload.files.length > 0;
  elements.analyzeBtn.disabled = !(hasText || hasFiles) || state.isAnalyzing;
}

function setView(target) {
  Object.entries(elements.views).forEach(([key, section]) => {
    section.classList.toggle("active", key === target);
  });
  if (target === "results") {
    elements.resultsDepartmentFilter.value = state.selectedDepartment || "";
  }
}

async function handleAnalyze(event) {
  event.preventDefault();
  if (state.isAnalyzing) return;

  const topicText = elements.topicInput.value || "";
  const trimmedText = topicText.trim();
  const files = elements.fileUpload.files || [];
  const hasFiles = files.length > 0;
  if (!trimmedText && !hasFiles) {
    showToast("Legg inn en kort beskrivelse eller last opp filer før du analyserer.", "warning");
    return;
  }

  state.topicText = topicText;
  state.isAnalyzing = true;
  updateAnalyzeButtonState();
  elements.analyzeBtn.textContent = "Analyserer…";

  try {
    const formData = new FormData();
    formData.append("text", state.topicText);
    if (state.selectedDepartment) {
      formData.append("department", state.selectedDepartment);
    }
    for (const file of files) {
      formData.append("files", file, file.name);
    }

    const response = await fetch("/analyze-topic", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      let errorMessage = "Analyse mislyktes.";
      try {
        const data = await response.json();
        const detail = data?.detail;
        const formatted = extractErrorMessage(detail);
        if (formatted) {
          errorMessage = formatted;
        }
      } catch (parseError) {
        console.error("Kunne ikke tolke feilrespons fra /analyze-topic:", parseError);
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    state.themes = data.themes || [];
    state.normalizedPreview = data.normalizedPreview || "";

    renderThemes();
    elements.normalizedPreview.textContent = state.normalizedPreview || "Ingen data ennå.";
    setView("themes");
    showToast("Temaer analysert. Gjennomgå før matching.");
  } catch (error) {
    console.error("Analyse mislyktes:", error);
    showToast(error.message || "Noe gikk galt under analysen.", "error");
  } finally {
    state.isAnalyzing = false;
    elements.analyzeBtn.textContent = "Analyser tema";
    updateAnalyzeButtonState();
  }
}

function extractErrorMessage(detail) {
  if (!detail) return null;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item?.msg) return item.msg;
        if (item?.detail) return extractErrorMessage(item.detail);
        return JSON.stringify(item);
      })
      .join(" ");
  }
  if (typeof detail === "object") {
    if (detail.message) return detail.message;
    if (detail.error) return detail.error;
    if (detail.detail) return extractErrorMessage(detail.detail);
    return JSON.stringify(detail);
  }
  return String(detail);
}

function renderThemes() {
  elements.themesChips.innerHTML = "";
  if (!state.themes.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Ingen temaer foreslått ennå. Legg til manuelt.";
    elements.themesChips.appendChild(empty);
    return;
  }

  state.themes.forEach((theme, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.setAttribute("role", "listitem");
    chip.innerHTML = `${theme}<button type="button" aria-label="Fjern tema" data-index="${index}">×</button>`;
    elements.themesChips.appendChild(chip);
  });
}

function handleThemesChipClick(event) {
  const target = event.target;
  if (target.matches("button[data-index]")) {
    const idx = Number.parseInt(target.dataset.index, 10);
    state.themes.splice(idx, 1);
    renderThemes();
  }
}

function addChipFromInput() {
  const value = elements.addChipInput.value.trim();
  if (!value) return;
  if (state.themes.includes(value.toLowerCase())) {
    showToast("Temaet ligger allerede i listen.", "warning");
    return;
  }
  state.themes.push(value.toLowerCase());
  elements.addChipInput.value = "";
  renderThemes();
}

async function fetchMatches() {
  if (!state.themes.length) {
    showToast("Legg til minst ett tema før du søker.", "warning");
    return;
  }
  if (state.isMatching) return;

  state.isMatching = true;
  elements.runMatchBtn.textContent = "Søker…";
  setView("results");
  elements.resultsList.innerHTML = '<div class="empty-state">Søker etter relevante ansatte…</div>';

  try {
    const response = await fetch("/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        themes: state.themes,
        department: state.selectedDepartment || null,
      }),
    });
    if (!response.ok) {
      let errorMessage = "Kunne ikke hente treff.";
      try {
        const data = await response.json();
        const detail = data?.detail;
        const formatted = extractErrorMessage(detail);
        if (formatted) {
          errorMessage = formatted;
        }
      } catch (parseError) {
        console.error("Kunne ikke tolke feilrespons fra /match:", parseError);
      }
      throw new Error(errorMessage);
    }
    const data = await response.json();
    state.results = data.results || [];
    state.resultLookup = new Map(state.results.map((result) => [result.id, result]));
    sortResults();
    renderResults();
    showToast(`Fant ${state.results.length} kandidater.`);
  } catch (error) {
    showToast(error.message || "Noe gikk galt under søket.", "error");
  } finally {
    state.isMatching = false;
    elements.runMatchBtn.textContent = "Finn relevante ansatte";
  }
}

function sortResults() {
  const sort = elements.resultsSort.value;
  if (sort === "alpha") {
    state.results.sort((a, b) => a.name.localeCompare(b.name, "nb"));
  } else {
    state.results.sort((a, b) => b.score - a.score);
  }
}

function renderResults() {
  elements.resultsList.innerHTML = "";
  if (!state.results.length) {
    elements.resultsList.innerHTML =
      '<div class="empty-state">Ingen treff. Juster temaene eller fjern avdelingsfilteret.</div>';
    elements.resultsCount.textContent = "Ingen treff.";
    return;
  }

  elements.resultsCount.textContent = `Viser ${state.results.length} treff`;

  state.results.forEach((result) => {
    const inShortlist = state.shortlist.some((item) => item.id === result.id);
    const keywordHtml = renderKeywordSection(result.keywords);
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-header">
        <div>
          <h3>${result.name}</h3>
          <span class="department-pill">${result.department}</span>
        </div>
        <div class="score-indicator">Score: ${result.score.toFixed(1)}</div>
      </div>
      <p class="result-why">${result.why}</p>
      ${keywordHtml}
      <div class="result-actions">
        <button class="btn-secondary" data-action="toggle-shortlist" data-id="${result.id}">
          ${inShortlist ? "Fjern fra kortliste" : "Legg til kortliste"}
        </button>
        <button class="btn-text" data-action="view-sources" data-id="${result.id}">Vis kilder</button>
        <a class="btn-text" href="${result.profile_url}" target="_blank" rel="noopener">Åpne profil</a>
      </div>
    `;
    elements.resultsList.appendChild(card);
  });

}

function handleResultsClick(event) {
  const target = event.target;
  const id = target.dataset.id;
  if (!id) return;
  const action = target.dataset.action;
  if (action === "toggle-shortlist") {
    toggleShortlistItem(id);
  } else if (action === "view-sources") {
    openSourcesModal(id);
  }
}

function toggleShortlistItem(resultId) {
  const existingIndex = state.shortlist.findIndex((item) => item.id === resultId);
  if (existingIndex >= 0) {
    state.shortlist.splice(existingIndex, 1);
    showToast("Fjernet fra kortlisten.");
  } else {
    const result = state.resultLookup.get(resultId);
    if (!result) {
      showToast("Fant ikke kandidaten i resultatlisten.", "error");
      return;
    }
    state.shortlist.push({
      id: result.id,
      name: result.name,
      department: result.department,
      why: result.why,
      citations: result.citations || [],
      score: result.score,
      keywords: result.keywords || [],
      notes: "",
    });
    showToast("Lagt til i kortlisten.");
  }
  renderResults();
  renderShortlist();
  scheduleShortlistSave();
  updateShortlistBadge();
}

function renderShortlist() {
  elements.shortlistItems.innerHTML = "";
  if (!state.shortlist.length) {
    elements.shortlistItems.innerHTML =
      '<div class="empty-state">Kortlisten er tom. Legg til kandidater fra resultatsiden.</div>';
    return;
  }

  state.shortlist.forEach((item) => {
    const card = document.createElement("div");
    card.className = "shortlist-card";
    const keywordHtml = renderKeywordSection(item.keywords);
    card.innerHTML = `
      <div>
        <strong>${item.name}</strong>
        <div class="department-pill">${item.department}</div>
      </div>
      <p>${item.why}</p>
      ${keywordHtml}
      <div>
        ${(item.citations || [])
          .map(
            (citation) =>
              `<a href="${citation.url}" target="_blank" rel="noopener">${citation.id} ${citation.title}</a>`
          )
          .join("<br />")}
      </div>
      <textarea data-id="${item.id}" placeholder="Egennotater">${item.notes || ""}</textarea>
      <button class="btn-text" data-remove="${item.id}">Fjern</button>
    `;
    elements.shortlistItems.appendChild(card);
  });

}

function renderKeywordSection(keywords) {
  if (!Array.isArray(keywords) || keywords.length === 0) {
    return "";
  }
  const chips = keywords
    .slice(0, 8)
    .map((keyword) => `<span class="keyword-chip">${keyword}</span>`)
    .join("");
  return `
    <div class="keyword-list">
      <span class="keyword-label">Nøkkelord:</span>
      <div class="keyword-chips">${chips}</div>
    </div>
  `;
}

function handleShortlistInput(event) {
  if (event.target.matches("textarea[data-id]")) {
    const id = event.target.dataset.id;
    const entry = state.shortlist.find((item) => item.id === id);
    if (entry) {
      entry.notes = event.target.value;
      scheduleShortlistSave();
    }
  }
}

function handleShortlistClick(event) {
  if (event.target.matches("button[data-remove]")) {
    const id = event.target.dataset.remove;
    state.shortlist = state.shortlist.filter((item) => item.id !== id);
    renderShortlist();
    renderResults();
    scheduleShortlistSave();
    updateShortlistBadge();
  }
}

function updateShortlistBadge() {
  elements.shortlistButton.textContent = `Kortliste (${state.shortlist.length})`;
}

function toggleShortlist(open) {
  state.shortlistOpen = open ?? !state.shortlistOpen;
  elements.shortlistDrawer.classList.toggle("open", state.shortlistOpen);
  elements.shortlistDrawer.setAttribute("aria-hidden", (!state.shortlistOpen).toString());
}

function openSourcesModal(resultId) {
  const result = state.resultLookup.get(resultId);
  if (!result || !result.citations || !result.citations.length) {
    showToast("Ingen kilder tilgjengelig for denne kandidaten.", "warning");
    return;
  }
  elements.modalTitle.textContent = `Kilder brukt for: ${result.name}`;
  elements.modalBody.innerHTML = result.citations
    .map(
      (citation) => `
      <div class="modal-citation">
        <h4>${citation.id} ${citation.title}</h4>
        <p>${citation.snippet}</p>
        <a href="${citation.url}" target="_blank" rel="noopener">Åpne lenke</a>
      </div>
    `
    )
    .join("");
  elements.modalBackdrop.classList.add("active");
  elements.modalBackdrop.setAttribute("aria-hidden", "false");
}

function closeModal() {
  elements.modalBackdrop.classList.remove("active");
  elements.modalBackdrop.setAttribute("aria-hidden", "true");
}

function scheduleShortlistSave() {
  if (shortlistSaveTimeout) {
    clearTimeout(shortlistSaveTimeout);
  }
  shortlistSaveTimeout = setTimeout(saveShortlist, 600);
}

async function saveShortlist() {
  try {
    await fetch("/shortlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: state.shortlist }),
    });
  } catch (error) {
    showToast("Kunne ikke lagre kortlisten.", "error");
  }
}

async function exportShortlist(format) {
  if (!state.shortlist.length) {
    showToast("Kortlisten er tom.", "warning");
    return;
  }

  if (format === "json") {
    const blob = new Blob([JSON.stringify({ items: state.shortlist }, null, 2)], {
      type: "application/json",
    });
    downloadBlob(blob, `kortliste-${Date.now()}.json`);
    showToast("Kortliste eksportert til JSON.");
    return;
  }

  try {
    if (shortlistSaveTimeout) {
      clearTimeout(shortlistSaveTimeout);
      shortlistSaveTimeout = null;
    }
    await saveShortlist();
    const response = await fetch("/shortlist/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        format,
        metadata: {
          topic: state.topicText,
          department: state.selectedDepartment,
          themes: state.themes,
        },
      }),
    });
    if (!response.ok) {
      let errorMessage = "Eksport mislyktes.";
      try {
        const data = await response.json();
        const detail = data?.detail;
        const formatted = extractErrorMessage(detail);
        if (formatted) {
          errorMessage = formatted;
        }
      } catch (parseError) {
        console.error("Kunne ikke tolke feilrespons fra /shortlist/export:", parseError);
      }
      throw new Error(errorMessage);
    }
    const data = await response.json();
    if (data?.path) {
      showToast(`Eksport lagret: ${data.path}`);
    } else {
      showToast("Eksport fullført.");
    }
  } catch (error) {
    showToast(error.message || "Eksport mislyktes.", "error");
  }
}

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = typeof message === "string" ? message : String(message);
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("fade");
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}
