<?= $this->extend('layouts/main') ?>
<?= $this->section('content') ?>
<div class="card">
  <h1>영상 업로드</h1>
  <div class="row">
    <label>원어
      <select id="lang">
        <?php foreach ($langs as $code => $name): ?>
          <option value="<?= $code ?>" <?= $code === 'en' ? 'selected' : '' ?>><?= esc($name) ?> (<?= $code ?>)</option>
        <?php endforeach ?>
      </select>
    </label>
    <label>번역 수위
      <select id="rating">
        <?php foreach ($ratings as $code => $name): ?>
          <option value="<?= $code ?>" <?= $code === 'rated' ? 'selected' : '' ?>><?= esc($name) ?></option>
        <?php endforeach ?>
      </select>
    </label>
  </div>
  <div id="drop" class="drop">
    <p>여기에 영상을 끌어다 놓거나 <label class="linkbtn" for="file">파일 선택</label></p>
    <input type="file" id="file" accept="video/*,.mkv,.ts" multiple hidden>
  </div>
  <div id="uploads"></div>
</div>

<div class="card">
  <h1>목록</h1>
  <?php if (! $videos): ?>
    <p class="muted">아직 업로드한 영상이 없습니다.</p>
  <?php else: ?>
  <table class="list">
    <thead><tr><th>제목</th><th>원어</th><th>수위</th><th>길이</th><th>상태</th><th></th></tr></thead>
    <tbody>
    <?php foreach ($videos as $v): $j = $v['job']; ?>
      <tr>
        <td><a href="<?= site_url('videos/' . $v['id']) ?>"><?= esc($v['title']) ?></a></td>
        <td><?= esc($v['lang']) ?></td>
        <td><?= esc(strtoupper($v['rating'] ?? 'rated')) ?></td>
        <td><?= $v['duration'] ? gmdate((float) $v['duration'] >= 3600 ? 'G:i:s' : 'i:s', (int) $v['duration']) : '-' ?></td>
        <td><span class="status <?= esc($v['status']) ?>"><?= esc($v['status']) ?><?= $j && $j['status'] === 'running' ? ' / ' . esc($j['stage']) . ' ' . (int) $j['progress'] . '%' : '' ?></span></td>
        <td class="actions">
          <form method="post" action="<?= site_url('videos/' . $v['id'] . '/delete') ?>" onsubmit="return confirm('삭제할까요?')"><?= csrf_field() ?><button class="linkbtn danger" type="submit">삭제</button></form>
        </td>
      </tr>
    <?php endforeach ?>
    </tbody>
  </table>
  <?php endif ?>
</div>
<?= $this->endSection() ?>

<?= $this->section('scripts') ?>
<script>MT.uploader(document.getElementById('drop'), document.getElementById('file'), document.getElementById('uploads'), document.getElementById('lang'), document.getElementById('rating'));</script>
<?= $this->endSection() ?>
