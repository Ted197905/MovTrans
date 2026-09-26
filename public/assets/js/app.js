window.MT = (() => {
  const csrf = () => {
    const m = document.querySelector('meta[name="csrf"]');
    return m ? { name: m.dataset.name, hash: m.content } : null;
  };
  const base = document.querySelector('.brand').getAttribute('href').replace(/videos$/, '');
  const fmtSize = b => b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' GB' : (b / 1048576).toFixed(1) + ' MB';
  const httpError = s => s === 401 ? '로그인이 필요합니다' : s === 413 ? '파일이 너무 큽니다' : 'HTTP ' + s;

  async function post(url, body) {
    const t = csrf();
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest', ...(t ? { 'X-CSRF-TOKEN': t.hash } : {}) }, body: JSON.stringify(body) });
    let j = {}; try { j = await res.json(); } catch (e) {}
    if (!res.ok || j.error) throw new Error(j.error || httpError(res.status));
    return j;
  }

  function putChunk(fd, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', base + 'api/upload/chunk');
      xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
      xhr.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded); };
      xhr.onload = () => {
        let j = {}; try { j = JSON.parse(xhr.responseText); } catch (e) {}
        (xhr.status === 200 && j.ok) ? resolve(j) : reject(new Error(j.error || httpError(xhr.status)));
      };
      xhr.onerror = () => reject(new Error('네트워크 오류'));
      xhr.send(fd);
    });
  }

  // keep the last chosen option per browser; storage may be unavailable (private mode)
  let uploading = 0;

  function remember(sel, key) {
    try { const v = localStorage.getItem(key); if (v && [...sel.options].some(o => o.value === v)) sel.value = v; } catch (e) {}
    sel.addEventListener('change', () => { try { localStorage.setItem(key, sel.value); } catch (e) {} });
  }

  function uploader(drop, input, list, langSel, ratingSel) {
    remember(langSel, 'mt.lang');
    remember(ratingSel, 'mt.rating');
    const row = (name) => {
      const el = document.createElement('div');
      el.className = 'up';
      el.innerHTML = '<div class="name"><span></span><span class="pct">0%</span></div><div class="bar"><i></i></div>';
      el.querySelector('.name span').textContent = name;
      list.prepend(el);
      return {
        set: (p, t) => { el.querySelector('.bar i').style.width = p + '%'; el.querySelector('.pct').textContent = t; },
        done: (t, href) => { el.classList.add('done'); el.querySelector('.bar i').style.width = '100%'; el.querySelector('.pct').innerHTML = '<a href="' + href + '">' + t + '</a>'; },
        fail: (t) => { el.classList.add('fail'); el.querySelector('.pct').textContent = t; },
      };
    };
    async function upload(file) {
      const r = row(file.name + ' (' + fmtSize(file.size) + ')');
      uploading++;
      let uploadId = null;
      try {
        const init = await post(base + 'api/upload/init', { name: file.name, size: file.size, lang: langSel.value, rating: ratingSel.value });
        uploadId = init.uploadId;
        const size = init.chunkSize || 8 * 1024 * 1024;
        const total = Math.max(1, Math.ceil(file.size / size));
        let sent = 0;
        for (let i = 0; i < total; i++) {
          const blob = file.slice(i * size, Math.min(file.size, (i + 1) * size));
          const t = csrf();
          const fd = new FormData();
          fd.append('uploadId', uploadId); fd.append('index', i); fd.append('chunk', blob);
          if (t) fd.append(t.name, t.hash);
          for (let tries = 0; ; tries++) {
            try {
              await putChunk(fd, loaded => { const p = Math.round((sent + loaded) / file.size * 100); r.set(p, p + '%'); });
              break;
            } catch (e) {
              if (tries >= 2) throw e;
              r.set(Math.round(sent / file.size * 100), '재시도 ' + (tries + 1) + '...');
              await new Promise(res => setTimeout(res, 1000 * (tries + 1)));
            }
          }
          sent += blob.size;
        }
        r.set(100, '등록 중...');
        const fin = await post(base + 'api/upload/finish', { uploadId, total });
        r.done('업로드 완료', fin.url);
        if (uploading === 1) setTimeout(() => location.reload(), 1500);
      } catch (e) {
        r.fail(e.message);
        if (uploadId) post(base + 'api/upload/abort', { uploadId }).catch(() => {});
      } finally {
        uploading--;
      }
    }
    const handle = files => { for (const f of files) upload(f); };
    input.addEventListener('change', () => { handle(input.files); input.value = ''; });
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => handle(e.dataTransfer.files));
  }

  /* list page: refresh status labels while any job is waiting/running; reload when one finishes (not mid-upload) */
  function list(table) {
    if (!table) return;
    const active = s => s === 'queued' || s === 'processing';
    const tick = async () => {
      try {
        const res = await fetch(base + 'api/videos/status', { headers: { 'X-Requested-With': 'XMLHttpRequest' } });
        if (!res.ok) throw new Error(httpError(res.status));
        let finished = false;
        for (const v of await res.json()) {
          const el = table.querySelector('tr[data-id="' + v.id + '"] .status');
          if (!el) continue;
          if (active(el.dataset.state) && !active(v.status)) finished = true;
          el.className = 'status ' + v.status; el.dataset.state = v.status; el.textContent = v.label;
        }
        if (finished && !uploading) { location.reload(); return; }
      } catch (e) { /* transient; keep polling */ }
      if ([...table.querySelectorAll('.status')].some(el => active(el.dataset.state))) setTimeout(tick, 3000);
    };
    if ([...table.querySelectorAll('.status')].some(el => active(el.dataset.state))) setTimeout(tick, 3000);
  }

  /* progress page: poll /api/jobs/{video} until done or failed */
  function progress(root, job) {
    const stages = [...root.querySelectorAll('.stage')];
    const order = stages.map(s => s.dataset.stage);
    const errEl = root.querySelector('#error');
    const logEl = document.getElementById('log');
    const render = (j) => {
      if (!j) return;
      const cur = j.stage === 'done' ? order.length : order.indexOf(j.stage);
      stages.forEach((s, i) => {
        s.classList.remove('active', 'done', 'fail');
        let p = 0, t = '';
        if (i < cur) { p = 100; t = '완료'; s.classList.add('done'); }
        else if (i === cur) { p = j.progress; t = j.status === 'failed' ? '실패' : j.status === 'queued' ? '대기' : j.progress + '%'; s.classList.add(j.status === 'failed' ? 'fail' : 'active'); }
        s.querySelector('.bar i').style.width = p + '%';
        s.querySelector('.pct').textContent = t;
      });
      if (j.status === 'failed' && j.error) { errEl.hidden = false; errEl.textContent = j.error; } else errEl.hidden = true;
      if (logEl && j.log != null) logEl.textContent = j.log;
    };
    render(job);
    if (!job || job.status === 'done' || job.status === 'failed') return;
    const tick = async () => {
      try {
        const res = await fetch(base + 'api/jobs/' + root.dataset.video, { headers: { 'X-Requested-With': 'XMLHttpRequest' } });
        if (!res.ok) throw new Error(httpError(res.status));
        const j = await res.json();
        render(j.job);
        if (j.job.status === 'done') { location.reload(); return; }
        if (j.job.status === 'failed') return;
      } catch (e) { /* transient; keep polling */ }
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 2000);
  }

  function player(video, sel) {
    if (!video || !sel) return;
    const apply = () => {
      for (const t of video.textTracks) {
        const isOrig = t.label.startsWith('원어');
        t.mode = (sel.value === 'ko' && !isOrig) || (sel.value === 'orig' && isOrig) ? 'showing' : 'hidden';
      }
    };
    sel.addEventListener('change', apply);
    video.addEventListener('loadedmetadata', apply);
  }

  return { uploader, list, progress, player };
})();
