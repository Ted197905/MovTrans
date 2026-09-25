<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="csrf" data-name="<?= csrf_token() ?>" content="<?= csrf_hash() ?>">
<title><?= isset($title) ? esc($title) . ' - ' : '' ?>MovTrans</title>
<link rel="stylesheet" href="<?= asset_url('assets/css/app.css') ?>">
</head>
<body>
<nav class="nav"><div class="in">
  <a class="brand" href="<?= site_url('videos') ?>">MovTrans</a>
  <span class="spacer"></span>
  <?php if (session()->get('user_id')): ?>
    <span class="nav-user"><?= esc(session()->get('email')) ?></span>
    <form method="post" action="<?= site_url('logout') ?>"><?= csrf_field() ?><button class="linkbtn" type="submit">로그아웃</button></form>
  <?php endif ?>
</div></nav>
<main class="main">
<?= $this->renderSection('content') ?>
</main>
<script src="<?= asset_url('assets/js/app.js') ?>"></script>
<?= $this->renderSection('scripts') ?>
</body>
</html>
