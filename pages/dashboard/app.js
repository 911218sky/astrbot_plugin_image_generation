(() => {
  "use strict";

  const AUTO_REFRESH_MS = 30000;

  const state = {
    data: null,
    page: "overview",
    taskQuery: "",
    taskStatus: "all",
    historyQuery: "",
    historyStatus: "all",
    cacheImages: [],
    cachePage: 1,
    cachePageSize: 12,
    cacheTotal: 0,
    cacheTotalPages: 1,
    cacheLoading: false,
    previewOpen: false,
    previewImage: null,
    timer: null,
  };

  const STATUS_CLASS = {
    queued: "queued",
    running: "running",
    success: "success",
    failed: "failed",
    blocked: "blocked",
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const element = byId(id);
    if (element) element.textContent = value;
  }

  function endpoint(path) {
    return "page/" + String(path).replace(/^\/+/, "").replace(/\/+/g, "/");
  }

  async function apiGet(path, params) {
    const bridge = window.AstrBotPluginPage;
    if (!bridge || typeof bridge.apiGet !== "function") {
      throw new Error("請從 AstrBot 官方插件 Pages 開啟此頁面");
    }
    const response = await bridge.apiGet(endpoint(path), params || {});
    if (response && response.status === "error") {
      throw new Error(response.message || "API 請求失敗");
    }
    return response && response.status === "ok" ? response.data || {} : response || {};
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatSeconds(value) {
    const seconds = Math.max(0, Math.floor(Number(value) || 0));
    const minutes = Math.floor(seconds / 60);
    const remain = seconds % 60;
    if (minutes <= 0) return `${remain}s`;
    return `${minutes}m ${String(remain).padStart(2, "0")}s`;
  }

  function includesText(task, query) {
    if (!query) return true;
    const text = [
      task.task_id,
      task.user,
      task.user_label,
      task.prompt_preview,
      task.error,
      task.mode,
      (task.files || []).join(" "),
    ]
      .join(" ")
      .toLowerCase();
    return text.includes(query.toLowerCase());
  }

  function showToast(message) {
    const toast = byId("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove("visible");
    void toast.offsetWidth;
    toast.classList.add("visible");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => toast.classList.remove("visible"), 2600);
  }

  function setElementText(id, value) {
    const element = byId(id);
    if (element) element.textContent = value || "--";
  }

  function readTheme() {
    try {
      const bridge = window.AstrBotPluginPage;
      if (bridge && typeof bridge.getContext === "function") {
        const ctx = bridge.getContext();
        if (ctx && typeof ctx.isDark === "boolean") return ctx.isDark ? "dark" : "light";
      }
    } catch (_) {}
    try {
      const stored = localStorage.getItem("imagegen_theme");
      if (stored) return stored;
    } catch (_) {}
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const darkIcon = byId("theme-icon-dark");
    const lightIcon = byId("theme-icon-light");
    if (darkIcon && lightIcon) {
      darkIcon.classList.toggle("hidden", theme === "light");
      lightIcon.classList.toggle("hidden", theme === "dark");
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const next = current === "light" ? "dark" : "light";
    applyTheme(next);
    try {
      localStorage.setItem("imagegen_theme", next);
    } catch (_) {}
  }

  function listenBridgeTheme() {
    try {
      const bridge = window.AstrBotPluginPage;
      if (!bridge || typeof bridge.onContext !== "function") return;
      bridge.onContext((ctx) => {
        if (!ctx || typeof ctx.isDark !== "boolean") return;
        applyTheme(ctx.isDark ? "dark" : "light");
      });
    } catch (_) {}
  }

  function setPage(page) {
    state.page = page;
    document.querySelectorAll(".nav-item[data-page]").forEach((button) => {
      button.classList.toggle("active", button.dataset.page === page);
    });
    document.querySelectorAll(".page").forEach((section) => {
      section.classList.toggle("active", section.id === `page-${page}`);
    });
  }

  async function refreshData(manual) {
    if (!manual && state.previewOpen) return;
    const sync = byId("sync-state");
    if (sync) sync.textContent = "更新中...";
    try {
      state.data = await apiGet("overview");
      render();
      if (manual) await refreshCache(false);
      if (sync) {
        const at = state.data.summary && state.data.summary.generated_at;
        sync.textContent = at ? `已更新 ${at}` : "已更新";
      }
      if (manual) showToast("資料已重新整理");
    } catch (error) {
      if (sync) sync.textContent = "更新失敗";
      showToast(error.message || "資料讀取失敗");
    }
  }

  function render() {
    const data = state.data || {};
    const summary = data.summary || {};
    setText("stat-active", summary.active_count ?? "--");
    setText("stat-active-detail", `排隊 ${summary.queued_count ?? 0} / 生成中 ${summary.running_count ?? 0}`);
    setText("stat-concurrent", summary.max_concurrent ?? "--");
    setText("stat-running", `正在使用 ${summary.running_count ?? 0} 個生成槽`);
    setText("stat-model", summary.model || "未初始化");
    setText("stat-provider", summary.provider ? `供應商 ${summary.provider}` : "供應商 --");
    setText("stat-cache", summary.cache_file_count ?? "--");
    setText("queue-pill", `${summary.active_count ?? 0} 個任務`);
    setText("cache-pill", `${summary.cache_file_count ?? 0} 個檔案`);

    renderOverviewTasks(data.tasks || []);
    renderSettings(data);
    renderTasksTable(data.tasks || []);
    renderHistory(data.recent_tasks || []);
    renderCache(state.cacheImages);
    renderCachePager();
    renderModels(data.providers || []);
  }

  async function refreshCache(manual) {
    if (!manual && state.previewOpen) return;
    state.cacheLoading = true;
    renderCachePager();
    try {
      const pageData = await apiGet("images", {
        page: state.cachePage,
        page_size: state.cachePageSize,
      });
      state.cacheImages = pageData.items || [];
      state.cachePage = Number(pageData.page) || 1;
      state.cachePageSize = Number(pageData.page_size) || state.cachePageSize;
      state.cacheTotal = Number(pageData.total) || 0;
      state.cacheTotalPages = Number(pageData.total_pages) || 1;
      const select = byId("cache-page-size");
      if (select) select.value = String(state.cachePageSize);
      renderCache(state.cacheImages);
      renderCachePager();
      if (manual) showToast("照片快取已重新整理");
    } catch (error) {
      showToast(error.message || "照片快取讀取失敗");
    } finally {
      state.cacheLoading = false;
      renderCachePager();
    }
  }

  function taskBadge(task) {
    const cls = STATUS_CLASS[task.status] || "";
    return `<span class="badge ${cls}">${escapeHtml(task.status_label || task.status || "未知")}</span>`;
  }

  function renderOverviewTasks(tasks) {
    const box = byId("overview-tasks");
    if (!box) return;
    if (!tasks.length) {
      box.innerHTML = '<div class="empty-state">目前沒有排隊或生成中的圖片任務。</div>';
      return;
    }
    box.innerHTML = tasks
      .slice(0, 6)
      .map(
        (task) => `
          <article class="task-row">
            ${taskBadge(task)}
            <div class="task-main">
              <div class="task-title">
                <span>${escapeHtml(task.mode)}</span>
                <span class="mono">${escapeHtml(task.task_id)}</span>
              </div>
              <div class="task-meta">${escapeHtml(task.user_label)} · ${escapeHtml(task.aspect_ratio)} · ${escapeHtml(task.resolution)} · 參考圖 ${task.reference_count || 0}</div>
              <div class="task-prompt">${escapeHtml(task.prompt_preview || "沒有提示詞摘要")}</div>
            </div>
            <div class="task-time">${formatSeconds(task.elapsed)}</div>
          </article>
        `
      )
      .join("");
  }

  function renderSettings(data) {
    const box = byId("settings-list");
    if (!box) return;
    const summary = data.summary || {};
    const settings = data.settings || {};
    const items = [
      ["初始化", summary.initialized ? "已就緒" : "未就緒"],
      ["每日限制", summary.daily_limit_enabled ? `${summary.daily_limit} 次/日` : "未啟用"],
      ["頻率限制", summary.rate_limit_seconds ? `${summary.rate_limit_seconds}s` : "未啟用"],
      ["預設比例", settings.default_aspect_ratio || "自動"],
      ["預設解析度", settings.default_resolution || "1K"],
      ["背景任務", `${summary.background_tasks || 0} 個`],
      ["循環/每日任務", `${summary.loop_tasks || 0} / ${summary.daily_tasks || 0}`],
      ["快取目錄", settings.cache_dir || "-"],
      ["黑名單/稽核白名單", `${settings.blocked_sessions || 0} / ${settings.audit_whitelist || 0}`],
    ];
    box.innerHTML = items
      .map(
        ([label, value]) => `
          <div class="info-item">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
          </div>
        `
      )
      .join("");
  }

  function renderTasksTable(tasks) {
    const body = byId("task-table-body");
    if (!body) return;
    const filtered = tasks.filter((task) => {
      const statusOk = state.taskStatus === "all" || task.status === state.taskStatus;
      return statusOk && includesText(task, state.taskQuery);
    });
    if (!filtered.length) {
      body.innerHTML = '<tr><td class="empty-state" colspan="6">沒有符合條件的進行中任務。</td></tr>';
      return;
    }
    body.innerHTML = filtered
      .map(
        (task) => `
          <tr>
            <td>${taskBadge(task)}</td>
            <td><strong>${escapeHtml(task.mode)}</strong><br><span class="mono">${escapeHtml(task.task_id)}</span></td>
            <td>${escapeHtml(task.user_label)}<br><span class="muted">${escapeHtml(task.started_at || "")}</span></td>
            <td class="cell-prompt">${escapeHtml(task.prompt_preview || "")}</td>
            <td>${escapeHtml(task.aspect_ratio)} / ${escapeHtml(task.resolution)}<br><span class="muted">參考圖 ${task.reference_count || 0}</span></td>
            <td class="mono">${formatSeconds(task.elapsed)}</td>
          </tr>
        `
      )
      .join("");
  }

  function renderHistory(tasks) {
    const box = byId("history-grid");
    if (!box) return;
    const filtered = tasks.filter((task) => {
      const statusOk = state.historyStatus === "all" || task.status === state.historyStatus;
      return statusOk && includesText(task, state.historyQuery);
    });
    if (!filtered.length) {
      box.innerHTML = '<div class="empty-state">尚無符合條件的最近紀錄。新的生成任務完成後會出現在這裡。</div>';
      return;
    }
    box.innerHTML = filtered
      .map((task) => {
        const files = (task.files || []).slice(0, 3).join(", ");
        const fileText = files ? `檔案：${files}` : task.error || "沒有附加訊息";
        return `
          <article class="history-card">
            <div class="history-head">
              <div class="history-title">
                ${escapeHtml(task.mode)}
                <span class="mono">${escapeHtml(task.task_id)}</span>
              </div>
              ${taskBadge(task)}
            </div>
            <p>${escapeHtml(task.prompt_preview || "沒有提示詞摘要")}</p>
            <div class="task-meta">${escapeHtml(task.user_label)} · ${escapeHtml(task.finished_at || "")} · ${formatSeconds(task.duration)}</div>
            <div class="task-meta">圖片 ${task.image_count || 0} · ${escapeHtml(fileText)}</div>
          </article>
        `;
      })
      .join("");
  }

  function renderCache(images) {
    const box = byId("cache-grid");
    if (!box) return;
    state.cacheImages = images;
    if (!images.length) {
      box.innerHTML = '<div class="empty-state">目前快取目錄沒有圖片檔案。</div>';
      return;
    }
    box.innerHTML = images
      .map((image, index) => {
        const metadata = image.metadata || {};
        const prompt = metadata.prompt_preview || (metadata.known ? metadata.prompt || "" : "尚無生成紀錄");
        const model = metadata.model_full || (metadata.known ? metadata.model || "" : "未記錄模型");
        const taskInfo = metadata.task_id
          ? `${metadata.status_label || metadata.status || "已記錄"} · ${metadata.task_id}`
          : image.kind === "generated"
            ? "舊快取或未記錄任務"
            : "參考圖片";
        return `
          <article class="file-card">
            <button class="file-preview" type="button" data-action="preview" data-index="${index}" aria-label="預覽 ${escapeHtml(image.name)}">
              ${
                image.preview_data_url
                  ? `<img src="${escapeHtml(image.preview_data_url)}" alt="${escapeHtml(image.name)}" loading="lazy" />`
                  : `<div class="file-icon">
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <path d="m21 15-5-5L5 21"></path>
                      </svg>
                    </div>`
              }
            </button>
            <div class="file-name">${escapeHtml(image.name)}</div>
            <div class="file-model">${escapeHtml(model || "--")}</div>
            <div class="file-prompt">${escapeHtml(prompt || "--")}</div>
            <div class="task-meta">${escapeHtml(taskInfo)} · ${escapeHtml(image.size_label)}</div>
            <div class="task-meta">${escapeHtml(image.modified_at || "")}</div>
            <div class="file-actions">
              <button class="mini-btn" type="button" data-action="preview" data-index="${index}">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <span>預覽</span>
              </button>
              <button class="mini-btn" type="button" data-action="download" data-index="${index}">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <path d="M7 10l5 5 5-5"></path>
                  <path d="M12 15V3"></path>
                </svg>
                <span>下載</span>
              </button>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderCachePager() {
    const info = byId("cache-page-info");
    const prev = byId("cache-prev");
    const next = byId("cache-next");
    if (info) {
      if (state.cacheLoading && !state.cacheImages.length) {
        info.textContent = "圖片載入中...";
      } else if (!state.cacheTotal) {
        info.textContent = "沒有圖片";
      } else {
        info.textContent = `第 ${state.cachePage} / ${state.cacheTotalPages} 頁，共 ${state.cacheTotal} 張`;
      }
    }
    if (prev) prev.disabled = state.cacheLoading || state.cachePage <= 1;
    if (next) next.disabled = state.cacheLoading || state.cachePage >= state.cacheTotalPages;
  }

  function setCachePage(page) {
    const nextPage = Math.min(Math.max(1, page), state.cacheTotalPages || 1);
    if (nextPage === state.cachePage && state.cacheImages.length) return;
    state.cachePage = nextPage;
    refreshCache(false);
  }

  function imageByIndex(index) {
    const number = Number(index);
    if (!Number.isInteger(number) || number < 0) return null;
    return state.cacheImages[number] || null;
  }

  async function fetchImageData(name) {
    if (!name) throw new Error("圖片檔名無效");
    const data = await apiGet("image", { name });
    if (!data.data_url) throw new Error("圖片資料不存在");
    if (!String(data.data_url).startsWith("data:image/")) {
      throw new Error("圖片資料格式無效");
    }
    return data;
  }

  function fillPreview(data) {
    const metadata = data.metadata || {};
    const image = byId("preview-image");
    if (image) {
      image.src = data.data_url;
      image.alt = data.name || "cached image";
    }
    setElementText("preview-title", data.name || "圖片預覽");
    setElementText("preview-name", data.name || "");
    setElementText("preview-model", metadata.model_full || metadata.model || "未記錄模型");
    setElementText("preview-prompt", metadata.prompt || metadata.prompt_preview || "尚無生成紀錄");
    setElementText(
      "preview-task",
      metadata.task_id
        ? `${metadata.status_label || metadata.status || "已記錄"} · ${metadata.task_id}`
        : "未記錄任務"
    );
    setElementText("preview-user", metadata.user || "未記錄會話");
    setElementText("preview-file", `${data.size_label || "--"} · ${data.modified_at || "--"}`);
  }

  async function openPreview(image) {
    const modal = byId("image-modal");
    if (!modal || !image) return;
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
    state.previewOpen = true;
    state.previewImage = null;
    setElementText("preview-title", "載入中...");
    setElementText("preview-name", image.name || "");
    setElementText("preview-model", "載入中...");
    setElementText("preview-prompt", "載入中...");
    try {
      const data = await fetchImageData(image.name);
      state.previewImage = data;
      fillPreview(data);
    } catch (error) {
      closePreview();
      showToast(error.message || "圖片載入失敗");
    }
  }

  function closePreview() {
    const modal = byId("image-modal");
    const image = byId("preview-image");
    if (modal) {
      modal.classList.add("hidden");
      modal.setAttribute("aria-hidden", "true");
    }
    if (image) image.removeAttribute("src");
    state.previewOpen = false;
    state.previewImage = null;
  }

  function triggerDownload(data) {
    const link = document.createElement("a");
    link.href = data.data_url;
    link.download = data.name || "image.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  async function downloadImage(image) {
    if (!image) return;
    try {
      const data =
        state.previewImage && state.previewImage.name === image.name
          ? state.previewImage
          : await fetchImageData(image.name);
      triggerDownload(data);
      showToast("已開始下載圖片");
    } catch (error) {
      showToast(error.message || "圖片下載失敗");
    }
  }

  function renderModels(providers) {
    const box = byId("model-list");
    if (!box) return;
    if (!providers.length) {
      box.innerHTML = '<div class="empty-state">目前設定中沒有可展示的模型。</div>';
      return;
    }
    box.innerHTML = providers
      .map(
        (provider) => `
          <div class="model-item">
            <span>${escapeHtml(provider.name)}</span>
            <strong>${escapeHtml(provider.model)} ${provider.current ? '<span class="badge success">目前</span>' : ""}</strong>
          </div>
        `
      )
      .join("");
  }

  function bindEvents() {
    document.querySelectorAll(".nav-item[data-page]").forEach((button) => {
      button.addEventListener("click", () => setPage(button.dataset.page));
    });
    byId("refresh-btn").addEventListener("click", () => refreshData(true));
    byId("theme-toggle").addEventListener("click", toggleTheme);
    byId("task-search").addEventListener("input", (event) => {
      state.taskQuery = event.target.value.trim();
      renderTasksTable((state.data && state.data.tasks) || []);
    });
    byId("task-status").addEventListener("change", (event) => {
      state.taskStatus = event.target.value;
      renderTasksTable((state.data && state.data.tasks) || []);
    });
    byId("history-search").addEventListener("input", (event) => {
      state.historyQuery = event.target.value.trim();
      renderHistory((state.data && state.data.recent_tasks) || []);
    });
    byId("history-status").addEventListener("change", (event) => {
      state.historyStatus = event.target.value;
      renderHistory((state.data && state.data.recent_tasks) || []);
    });
    byId("cache-grid").addEventListener("click", (event) => {
      const trigger = event.target.closest("[data-action]");
      if (!trigger) return;
      const image = imageByIndex(trigger.dataset.index);
      if (trigger.dataset.action === "preview") openPreview(image);
      if (trigger.dataset.action === "download") downloadImage(image);
    });
    byId("cache-prev").addEventListener("click", () => setCachePage(state.cachePage - 1));
    byId("cache-next").addEventListener("click", () => setCachePage(state.cachePage + 1));
    byId("cache-page-size").addEventListener("change", (event) => {
      state.cachePageSize = Number(event.target.value) || 12;
      state.cachePage = 1;
      refreshCache(false);
    });
    byId("image-modal").addEventListener("click", (event) => {
      const trigger = event.target.closest('[data-action="close-preview"]');
      if (trigger) closePreview();
    });
    byId("preview-download").addEventListener("click", () => {
      if (state.previewImage) downloadImage(state.previewImage);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closePreview();
    });
  }

  function init() {
    applyTheme(readTheme());
    listenBridgeTheme();
    bindEvents();
    refreshData(false);
    refreshCache(false);
    state.timer = setInterval(() => refreshData(false), AUTO_REFRESH_MS);
    window.addEventListener("beforeunload", () => {
      if (state.timer) clearInterval(state.timer);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
