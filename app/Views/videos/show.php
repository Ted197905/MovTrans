<?= $this->extend('layouts/main') ?>
<?= $this->section('content') ?>
<?php
$stages = ['prepare' => '준비 (오디오 추출 / 프록시)', 'transcribe' => '원어 자막 (WhisperX)', 'scene' => '장면 분석 (Qwen3-VL)', 'translate' => '한국어 자막 (Qwen3-VL)'];
if ($video['lang'] === 'ko') unset($stages['scene'], $stages['translate']);
$hasKo   = in_array(['ko', 'vtt'], $subtitles, true);
$hasOrig = in_array(['orig', 'vtt'], $subtitles, true);
?>
<div class="card">
  <div class="head">
    <h1><?= esc($video['title']) ?></h1>
    <a class="linkbtn" href="<?= site_url('videos') ?>">목록</a>
  </div>
  <p class="muted">
    원어 <?= esc($langs[$video['lang']] ?? $video['lang']) ?> /
    <?= $video['width'] && $video['height'] ? $video['width'] . 'x' . $video['height'] . ' ' : '' ?><?= esc($video['vcodec'] ?? '') ?> /
    <?= $video['duration'] ? gmdate((float) $video['duration'] >= 3600 ? 'G:i:s' : 'i:s', (int) $video['duration']) : '' ?> /
    <?= number_format($video['size'] / 1048576, 1) ?> MB
  </p>

  <div id="progress" class="stages" data-video="<?= $video['id'] ?>" data-status="<?= esc($video['status']) ?>">
    <?php foreach ($stages as $key => $label): ?>
      <div class="stage" data-stage="<?= $key ?>">
        <div class="label"><span><?= esc($label) ?></span><span class="pct"></span></div>
        <div class="bar"><i></i></div>
      </div>
    <?php endforeach ?>
    <div id="error" class="alert" hidden><?= $job && $job['error'] ? esc($job['error']) : '' ?></div>
  </div>
  <details class="log"><summary>로그</summary><pre id="log"><?= $job ? esc($job['log'] ?? '') : '' ?></pre></details>
</div>

<div class="card" id="player" <?= $video['status'] === 'done' ? '' : 'hidden' ?>>
  <h1>재생</h1>
  <video id="video" controls preload="metadata" crossorigin="use-credentials" src="<?= site_url('media/' . $video['id'] . '/video') ?>">
    <?php if ($hasKo): ?><track kind="subtitles" label="한국어" srclang="ko" src="<?= site_url('media/' . $video['id'] . '/sub/ko.vtt') ?>" default><?php endif ?>
    <?php if ($hasOrig): ?><track kind="subtitles" label="원어 (<?= esc($video['lang']) ?>)" srclang="<?= esc($video['lang']) ?>" src="<?= site_url('media/' . $video['id'] . '/sub/orig.vtt') ?>"><?php endif ?>
  </video>
  <div class="row">
    <label>자막
      <select id="track">
        <option value="ko">한국어</option>
        <option value="orig">원어</option>
        <option value="off">끄기</option>
      </select>
    </label>
    <span class="spacer"></span>
    <?php foreach ($subtitles as [$t, $e]): ?>
      <a class="btn" href="<?= site_url("media/{$video['id']}/sub/{$t}.{$e}?dl") ?>"><?= $t === 'ko' ? '한국어' : '원어' ?> .<?= $e ?></a>
    <?php endforeach ?>
  </div>
</div>

<div class="card">
  <h1>다시 실행</h1>
  <form method="post" action="<?= site_url('videos/' . $video['id'] . '/rerun') ?>" class="row">
    <?= csrf_field() ?>
    <label>원어
      <select name="lang">
        <?php foreach ($langs as $code => $name): ?>
          <option value="<?= $code ?>" <?= $code === $video['lang'] ? 'selected' : '' ?>><?= esc($name) ?> (<?= $code ?>)</option>
        <?php endforeach ?>
      </select>
    </label>
    <button class="btn" type="submit" <?= $video['status'] === 'processing' ? 'disabled' : '' ?>>파이프라인 다시 실행</button>
  </form>
</div>
<?= $this->endSection() ?>

<?= $this->section('scripts') ?>
<script>
MT.progress(document.getElementById('progress'), <?= json_encode($job ?: null) ?>);
MT.player(document.getElementById('video'), document.getElementById('track'));
</script>
<?= $this->endSection() ?>
