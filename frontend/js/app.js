(() => {
  "use strict";
  const $ = s => document.querySelector(s);
  const uploadZone = $("#uploadZone");
  const fileInput = $("#fileInput");
  const progressSection = $("#progressSection");
  const fileInfo = $("#fileInfo");
  const progressBar = $("#progressBar");
  const progressText = $("#progressText");
  const results = $("#results");
  const btnReset = $("#btnReset");

  // Drag & drop
  uploadZone.addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach(e =>
    uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(e =>
    uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.remove("dragover"); })
  );
  uploadZone.addEventListener("drop", ev => {
    const f = ev.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  btnReset.addEventListener("click", () => {
    results.classList.add("hidden");
    uploadZone.classList.remove("hidden");
    progressSection.classList.add("hidden");
    fileInput.value = "";
  });

  async function handleFile(file) {
    uploadZone.classList.add("hidden");
    results.classList.add("hidden");
    progressSection.classList.remove("hidden");
    fileInfo.textContent = `${file.name} — ${humanSize(file.size)}`;
    progressBar.style.width = "0%";
    progressText.textContent = "Загрузка файла...";

    try {
      // Upload
      const formData = new FormData();
      formData.append("file", file);
      const uploadRes = await uploadWithProgress("/api/upload", formData, pct => {
        progressBar.style.width = pct + "%";
        progressText.textContent = `Загрузка: ${pct}%`;
      });

      if (!uploadRes.fileId) throw new Error(uploadRes.detail || "Upload failed");

      // Analyze
      progressBar.style.width = "100%";
      progressText.textContent = "🧠 AI анализирует аудио... Это может занять 10-30 секунд";

      const res = await fetch(`/api/analyze/${uploadRes.fileId}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      renderResults(data);

      // Cleanup
      fetch(`/api/file/${uploadRes.fileId}`, { method: "DELETE" }).catch(() => {});

    } catch (err) {
      progressText.textContent = `❌ Ошибка: ${err.message}`;
      progressBar.style.width = "100%";
      progressBar.style.background = "var(--red)";
      setTimeout(() => {
        progressSection.classList.add("hidden");
        uploadZone.classList.remove("hidden");
        progressBar.style.background = "";
      }, 3000);
    }
  }

  function uploadWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.upload.addEventListener("progress", e => {
        if (e.lengthComputable) onProgress(Math.round(e.loaded / e.total * 100));
      });
      xhr.addEventListener("load", () => {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { reject(new Error("Invalid response")); }
      });
      xhr.addEventListener("error", () => reject(new Error("Network error")));
      xhr.send(formData);
    });
  }

  function renderResults(data) {
    progressSection.classList.add("hidden");
    results.classList.remove("hidden");

    renderGenres(data.genres, data.sub_genres);
    renderMoods(data.moods);
    renderRhythm(data.rhythm, data.duration);
    renderCharacteristics(data);
    renderSunoPrompt(data.suno_prompt);
  }

  function renderGenres(genres, subGenres) {
    const main = $("#genreMain");
    const bars = $("#genreBars");
    const subs = $("#subGenres");

    if (!genres || !genres.length) {
      main.innerHTML = '<span class="genre-name">Не определён</span>';
      bars.innerHTML = "";
      subs.innerHTML = "";
      return;
    }

    const top = genres[0];
    main.innerHTML = `
      <span class="genre-name">${top.genre}</span>
      <span class="genre-score">${top.score}%</span>
    `;

    const colors = ["bar-c1","bar-c2","bar-c3","bar-c4","bar-c5","bar-c6","bar-c7","bar-c8"];
    const maxScore = genres[0].score;
    bars.innerHTML = genres.slice(1).map((g, i) => `
      <div class="genre-row">
        <span class="label">${g.genre}</span>
        <div class="bar-wrap">
          <div class="bar ${colors[(i+1) % colors.length]}" style="width:${(g.score / maxScore * 100).toFixed(0)}%"></div>
        </div>
        <span class="score">${g.score}%</span>
      </div>
    `).join("");

    if (subGenres && subGenres.length) {
      subs.innerHTML = `
        <h3>Поджанры</h3>
        <div>${subGenres.map(s =>
          `<span class="sub-tag">${s.genre} › ${s.subgenre}<span class="pct">${s.score}%</span></span>`
        ).join("")}</div>
      `;
    } else {
      subs.innerHTML = "";
    }
  }

  const MOOD_META = {
    happy: { icon: "😊", label: "Весёлое", color: "var(--yellow)" },
    sad: { icon: "😢", label: "Грустное", color: "var(--blue)" },
    aggressive: { icon: "🔥", label: "Агрессивное", color: "var(--red)" },
    relaxed: { icon: "😌", label: "Расслабленное", color: "var(--green)" },
    electronic: { icon: "🤖", label: "Электронное", color: "var(--accent)" },
    acoustic: { icon: "🎸", label: "Акустическое", color: "var(--orange)" },
  };

  function renderMoods(moods) {
    const grid = $("#moodGrid");
    grid.innerHTML = Object.entries(moods).map(([key, val]) => {
      const m = MOOD_META[key] || { icon: "🎵", label: key, color: "var(--accent)" };
      return `
        <div class="mood-item">
          <div class="mood-label">${m.icon} ${m.label}</div>
          <div class="mood-bar-wrap">
            <div class="mood-bar" style="width:${val}%;background:${m.color}"></div>
          </div>
          <div class="mood-val">${val}%</div>
        </div>
      `;
    }).join("");
  }

  function renderRhythm(rhythm, duration) {
    const grid = $("#rhythmGrid");
    const keyDisplay = rhythm.key ? `${rhythm.key} ${rhythm.scale === "minor" ? "min" : "maj"}` : "—";
    const durDisplay = duration ? formatDuration(duration) : "—";
    grid.innerHTML = `
      <div class="rhythm-item">
        <div class="r-value" style="color:var(--accent)">${rhythm.bpm}</div>
        <div class="r-label">BPM</div>
      </div>
      <div class="rhythm-item">
        <div class="r-value" style="color:var(--green)">${keyDisplay}</div>
        <div class="r-label">Тональность</div>
      </div>
      <div class="rhythm-item">
        <div class="r-value" style="color:var(--yellow)">${durDisplay}</div>
        <div class="r-label">Длительность</div>
      </div>
    `;
  }

  function renderCharacteristics(data) {
    const grid = $("#charGrid");
    const vocal = data.vocal;
    const items = [
      {
        icon: "💃",
        label: "Танцевальность",
        value: data.danceability + "%",
      },
      {
        icon: vocal.has_vocals ? "🎤" : "🎹",
        label: "Тип",
        value: vocal.has_vocals
          ? `Вокал (${vocal.gender === "female" ? "жен" : "муж"}, ${vocal.confidence}%)`
          : `Инструментал (${vocal.confidence}%)`,
      },
      {
        icon: "🎼",
        label: "Тональная",
        value: data.tonal ? "Да" : "Атональная",
      },
      {
        icon: "⚡",
        label: "Энергия",
        value: data.rhythm.energy.toFixed(1),
      },
      {
        icon: "🔊",
        label: "Громкость",
        value: data.rhythm.loudness.toFixed(1) + " dB",
      },
      {
        icon: "📈",
        label: "Динамика",
        value: data.rhythm.dynamic_complexity.toFixed(1),
      },
    ];
    grid.innerHTML = items.map(it => `
      <div class="char-item">
        <span class="c-icon">${it.icon}</span>
        <div class="c-info">
          <span class="c-label">${it.label}</span>
          <span class="c-value">${it.value}</span>
        </div>
      </div>
    `).join("");
  }

  function renderSunoPrompt(prompt) {
    const el = $("#sunoPrompt");
    const btn = $("#sunoCopy");
    const msg = $("#sunoCopied");
    el.textContent = prompt || "";
    btn.onclick = () => {
      navigator.clipboard.writeText(prompt || "").then(() => {
        msg.classList.remove("hidden");
        setTimeout(() => msg.classList.add("hidden"), 2000);
      });
    };
  }

  function humanSize(bytes) {
    const units = ["B","KB","MB","GB"];
    let v = bytes;
    for (const u of units) {
      if (v < 1024 || u === "GB") return (u === "B" ? v : v.toFixed(1)) + " " + u;
      v /= 1024;
    }
    return bytes + " B";
  }

  function formatDuration(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
})();
