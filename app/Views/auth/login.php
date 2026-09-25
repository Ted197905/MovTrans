<?= $this->extend('layouts/main') ?>
<?= $this->section('content') ?>
<div class="card narrow">
  <h1>로그인</h1>
  <?php if ($errors = session()->getFlashdata('errors')): ?>
    <div class="alert"><?= esc(implode(' ', (array) $errors)) ?></div>
  <?php endif ?>
  <form method="post" action="<?= site_url('login') ?>">
    <?= csrf_field() ?>
    <label>이메일<input type="email" name="email" value="<?= old('email') ?>" required autofocus></label>
    <label>비밀번호<input type="password" name="password" required></label>
    <button class="btn primary" type="submit">로그인</button>
  </form>
</div>
<?= $this->endSection() ?>
