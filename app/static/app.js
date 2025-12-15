const PAGE_SIZE = 5;

const state = {
  uiLanguage: "no",
  query: "",
  results: [],
  visibleCount: 0,
  loading: false,
  config: {},
  activeModalId: null,
};

const copy = {
  no: {
    noscript: "Vennligst aktiver JavaScript for å bruke ONH Expert Connector.",
    kicker: "Finn interne eksperter",
    title: "ONH Expert Connector",
    searchBtn: "Søk",
    idleHelper: "Beskriv behovet ditt og trykk Søk.",
    helperText: "Start et søk for å se ansatte med relevante treff.",
    loadMore: "Last inn flere",
    statusSearching: "Søker …",
    statusResults: (q, n) => `Viser ${n} treff for «${q}»`,
    errorMissingQuery: "Skriv inn et søk før du fortsetter.",
    errorMatchFail: "Kunne ikke hente treff.",
    emptyResults: "Ingen treff. Prøv mer spesifikt språk eller færre ord.",
    snippetLabel: "Beskrivelse",
    tagsLabel: "Nøkkelord",
    viewDetails: "Vis detaljer",
    hideDetails: "Skjul detaljer",
    openProfile: "Åpne profil",
    openLink: "Åpne lenke",
    sources: "Kilder",
    toastCopied: "Kopiert til utklippstavlen.",
  },
  en: {
    noscript: "Please enable JavaScript to use ONH Expert Connector.",
    kicker: "Find internal experts",
    title: "ONH Expert Connector",
    searchBtn: "Search",
    idleHelper: "Describe what you need and press Search.",
    helperText: "Start a search to see staff with relevant matches.",
    loadMore: "Load more",
    statusSearching: "Searching…",
    statusResults: (q, n) => `Showing ${n} results for “${q}”`,
    errorMissingQuery: "Enter a search before continuing.",
    errorMatchFail: "Could not fetch matches.",
    emptyResults: "No matches found. Try more specific language or fewer words.",
    snippetLabel: "Description",
    tagsLabel: "Keywords",
    viewDetails: "View details",
    hideDetails: "Hide details",
    openProfile: "Open profile",
    openLink: "Open link",
    sources: "Sources",
    toastCopied: "Copied to clipboard.",
  },
};

const elements = {
  hero: document.getElementById("hero"),
  searchForm: document.getElementById("searchForm"),
  queryInput: document.getElementById("queryInput"),
  searchBtn: document.getElementById("searchBtn"),
  statusLine: document.getElementById("statusLine"),
  errorBanner: document.getElementById("errorBanner"),
  resultsList: document.getElementById("resultsList"),
  emptyState: document.getElementById("emptyState"),
  loadMoreBtn: document.getElementById("loadMoreBtn"),
  langNoBtn: document.getElementById("langNoBtn"),
  langEnBtn: document.getElementById("langEnBtn"),
  toastContainer: document.getElementById("toastContainer"),
  modalBackdrop: document.getElementById("modalBackdrop"),
  modalBody: document.getElementById("modalBody"),
  modalTitle: document.getElementById("modalTitle"),
  modalCloseBtn: document.getElementById("modalCloseBtn"),
};

window.addEventListener("DOMContentLoaded", () => {
  setupListeners();
  initialize();
});

async function initialize() {
  await loadConfig();
  state.uiLanguage = (state.config?.ui?.language || "no").toLowerCase();
  await applyLanguage(state.uiLanguage);
}

function setupListeners() {
  elements.searchForm.addEventListener("submit", handleSearch);
  elements.queryInput.addEventListener("input", () => {
    elements.errorBanner.hidden = true;
  });
  elements.loadMoreBtn.addEventListener("click", () => {
    state.visibleCount = Math.min(state.results.length, state.visibleCount + PAGE_SIZE);
    renderResults();
    updateLoadMore();
  });

  elements.langNoBtn.addEventListener("click", () => applyLanguage("no"));
  elements.langEnBtn.addEventListener("click", () => applyLanguage("en"));
  elements.resultsList.addEventListener("click", handleResultsClick);
  elements.modalCloseBtn.addEventListener("click", closeModal);
  elements.modalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.modalBackdrop) {
      closeModal();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.modalBackdrop.hidden) {
      closeModal();
    }
  });
}

async function loadConfig() {
  try {
    const response = await fetch("/config");
    if (!response.ok) return;
    state.config = await response.json();
    document.title = copy[state.uiLanguage]?.title ?? "ONH Expert Connector";
  } catch (error) {
    console.warn("Kunne ikke laste konfigurasjon:", error);
  }
}

async function applyLanguage(lang) {
  state.uiLanguage = lang === "en" ? "en" : "no";
  document.documentElement.lang = state.uiLanguage;
  elements.langNoBtn.classList.toggle("active", state.uiLanguage === "no");
  elements.langEnBtn.classList.toggle("active", state.uiLanguage === "en");

  const dict = copy[state.uiLanguage];
  document.querySelectorAll("[data-l10n]").forEach((node) => {
    const key = node.dataset.l10n;
    const value = dict[key];
    if (typeof value === "string") {
      node.textContent = value;
    }
  });

  document.title = dict.title;
  elements.queryInput.placeholder =
    state.uiLanguage === "en"
      ? "Type what you need — course, topic, question, or goal"
      : "Skriv hva du trenger — kurs, tema, spørsmål eller mål";
  elements.searchBtn.textContent = dict.searchBtn;
  updateStatusForLanguage();
  renderResults();
  if (needsLanguageRefresh()) {
    await refreshResultsForLanguage();
  }
  if (state.activeModalId) {
    const result = state.results.find((r) => r.id === state.activeModalId);
    if (result) openModal(result);
  }
}

async function handleSearch(event) {
  event.preventDefault();
  const query = (elements.queryInput.value || "").trim();
  if (!query) {
    showError(copy[state.uiLanguage].errorMissingQuery);
    return;
  }
  await runSearch(query, { resetVisibleCount: true });
}

async function runSearch(query, { resetVisibleCount = true, preserveVisibleCount } = {}) {
  state.query = query;
  const dict = copy[state.uiLanguage];
  const initialVisible = resetVisibleCount
    ? PAGE_SIZE
    : preserveVisibleCount ?? state.visibleCount || PAGE_SIZE;

  if (resetVisibleCount) {
    state.visibleCount = 0;
    state.results = [];
    elements.resultsList.innerHTML = "";
  }

  setStatus(dict.statusSearching, { loading: true });
  elements.hero.classList.add("hero--collapsed");
  elements.errorBanner.hidden = true;
  elements.emptyState.hidden = true;

  try {
    state.loading = true;
    const data = await fetchMatches(query);
    state.results = data.results || [];
    const available = state.results.length;
    state.visibleCount = available ? Math.min(initialVisible, available) : 0;
    renderResults();
    updateLoadMore();

    if (!state.results.length) {
      elements.emptyState.hidden = false;
      elements.emptyState.textContent = dict.emptyResults;
    }

    setStatus(dict.statusResults(query, state.results.length));
  } catch (error) {
    console.error("Match error:", error);
    showError(error.message || dict.errorMatchFail);
    setStatus(dict.idleHelper);
    elements.emptyState.hidden = false;
    elements.emptyState.textContent = dict.helperText;
  } finally {
    state.loading = false;
    elements.statusLine.classList.remove("is-searching");
  }
}

async function fetchMatches(query) {
  const response = await fetch("/match", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-UI-Language": state.uiLanguage,
    },
    body: JSON.stringify({
      themes: [query],
      department: null,
    }),
  });

  if (!response.ok) {
    const errText = await safeExtractError(response);
    throw new Error(errText || copy[state.uiLanguage].errorMatchFail);
  }
  return response.json();
}

async function refreshResultsForLanguage() {
  if (!state.query || state.loading) return;
  const preserveVisibleCount = state.visibleCount || PAGE_SIZE;
  await runSearch(state.query, { resetVisibleCount: false, preserveVisibleCount });
}

function updateStatusForLanguage() {
  const dict = copy[state.uiLanguage];
  if (state.loading) {
    setStatus(dict.statusSearching, { loading: true });
    return;
  }
  if (state.results.length) {
    setStatus(dict.statusResults(state.query, state.results.length));
    return;
  }
  setStatus(dict.idleHelper);
}

function needsLanguageRefresh() {
  if (!state.results.length) return false;
  return state.results.some((result) => {
    const whyByLang = result?.whyByLang;
    if (!whyByLang || typeof whyByLang !== "object") return true;
    return !(whyByLang[state.uiLanguage] || "").trim();
  });
}

function renderResults() {
  elements.emptyState.hidden = true;
  elements.resultsList.innerHTML = "";
  if (!state.results.length) return;

  const visible = state.results.slice(0, state.visibleCount || PAGE_SIZE);
  visible.forEach((result) => {
    const card = document.createElement("article");
    card.className = "result-card";

    const snippet = pickSnippet(result);
    const highlighted = highlightSnippet(snippet, state.query);

    card.innerHTML = `
      <div class="result-top">
        <div>
          <h3 class="result-name">${escapeHtml(result.name)}</h3>
          <div class="result-meta">${escapeHtml(result.department || "—")}</div>
        </div>
      </div>
      <div class="result-snippet">
        <strong>${copy[state.uiLanguage].snippetLabel}:</strong>
        <p>${highlighted}</p>
      </div>
      ${renderTags(result.keywords)}
      <div class="card-actions">
        <button class="btn-primary btn-sm" data-action="open-details" data-id="${result.id}">
          ${copy[state.uiLanguage].viewDetails}
        </button>
        <a class="btn-outline btn-sm" href="${result.profile_url}" target="_blank" rel="noopener">
          ${copy[state.uiLanguage].openProfile}
        </a>
      </div>
    `;

    elements.resultsList.appendChild(card);
  });
}

function handleResultsClick(event) {
  const btn = event.target.closest("[data-action='open-details']");
  if (!btn) return;
  const id = btn.dataset.id;
  const result = state.results.find((r) => r.id === id);
  if (result) {
    openModal(result);
  }
}

function renderTags(keywords) {
  if (!Array.isArray(keywords) || keywords.length === 0) return "";
  const chips = keywords
    .slice(0, 8)
    .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
    .join("");
  return `
    <div class="tags" aria-label="${copy[state.uiLanguage].tagsLabel}">
      ${chips}
    </div>
  `;
}

function renderCitations(result) {
  if (!result?.citations?.length) {
    return `<p class="empty-state">${copy[state.uiLanguage].sources}: –</p>`;
  }
  const items = result.citations
    .slice(0, 4)
    .map((c) => {
      const safeSnippet = escapeHtml(c.snippet || "");
      return `
        <div class="modal-citation">
          <div><strong>${escapeHtml(c.title || c.id || "")}</strong></div>
          <p>${safeSnippet}</p>
          <a href="${c.url}" target="_blank" rel="noopener">${copy[state.uiLanguage].openLink}</a>
        </div>
      `;
    })
    .join("");
  return items;
}

function updateLoadMore() {
  const hasMore = state.results.length > state.visibleCount;
  elements.loadMoreBtn.hidden = !hasMore;
}

function pickSnippet(result) {
  const whyByLang = result?.whyByLang;
  if (whyByLang && typeof whyByLang === "object") {
    const preferred = (whyByLang[state.uiLanguage] || "").trim();
    if (preferred) return preferred;
    const fallbackLang = state.uiLanguage === "en" ? "no" : "en";
    const fallback = (whyByLang[fallbackLang] || "").trim();
    if (fallback) return fallback;
  }
  const why = (result?.why || "").trim();
  if (why) return why;
  if (result?.citations?.length) {
    return truncateText(result.citations[0].snippet || "");
  }
  return "";
}

function highlightSnippet(text, query) {
  if (!text) return "";
  const safe = escapeHtml(text);
  const tokens = (query || "")
    .split(/\s+/)
    .filter((w) => w.length > 3)
    .slice(0, 6)
    .map(escapeRegExp);
  if (!tokens.length) return safe;
  const pattern = new RegExp(`(${tokens.join("|")})`, "gi");
  return safe.replace(pattern, '<mark class="highlight">$1</mark>');
}

function setStatus(message, { loading = false } = {}) {
  elements.statusLine.textContent = message;
  elements.statusLine.classList.toggle("is-searching", loading);
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
  showToast(message, "error");
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

function truncateText(text, maxLength = 240) {
  const normalized = (text || "").trim();
  if (!normalized) return "";
  if (normalized.length <= maxLength) return normalized;
  const cut = normalized.slice(0, maxLength);
  const lastSpace = cut.lastIndexOf(" ");
  const trimmed = lastSpace > maxLength * 0.6 ? cut.slice(0, lastSpace) : cut;
  return `${trimmed.trim().replace(/[,:;]$/, "")}...`;
}

async function safeExtractError(response) {
  try {
    const data = await response.json();
    return data?.detail || data?.message || null;
  } catch (error) {
    return null;
  }
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function openModal(result) {
  state.activeModalId = result.id;
  elements.modalTitle.textContent = result.name;
  const tags = renderTags(result.keywords);
  const meta = `
    <div class="result-meta">${escapeHtml(result.department || "—")}</div>
  `;
  const snippet = `
    <div class="result-snippet">
      <strong>${copy[state.uiLanguage].snippetLabel}:</strong>
      <p>${highlightSnippet(pickSnippet(result), state.query)}</p>
    </div>
  `;
  elements.modalBody.innerHTML = `
    ${meta}
    ${tags}
    ${snippet}
    <div>
      <strong>${copy[state.uiLanguage].sources}</strong>
      <div class="modal-citation-list">${renderCitations(result)}</div>
    </div>
  `;
  elements.modalBackdrop.hidden = false;
  elements.modalBackdrop.setAttribute("aria-hidden", "false");
  elements.modalCloseBtn.focus();
}

function closeModal() {
  state.activeModalId = null;
  elements.modalBackdrop.hidden = true;
  elements.modalBackdrop.setAttribute("aria-hidden", "true");
}
