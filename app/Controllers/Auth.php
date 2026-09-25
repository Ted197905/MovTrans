<?php

namespace App\Controllers;

use App\Models\UserModel;

class Auth extends BaseController
{
    public function login()
    {
        if (session()->get('user_id')) {
            return redirect()->to('/videos');
        }
        return view('auth/login', ['title' => '로그인']);
    }

    public function attemptLogin()
    {
        if (! $this->validate(['email' => 'required|valid_email', 'password' => 'required'])) {
            return redirect()->back()->withInput()->with('errors', $this->validator->getErrors());
        }
        $user = (new UserModel())->where('email', $this->request->getPost('email'))->first();
        if (! $user || ! password_verify((string) $this->request->getPost('password'), $user['password_hash'])) {
            return redirect()->back()->withInput()->with('errors', ['auth' => '이메일 또는 비밀번호가 올바르지 않습니다.']);
        }
        session()->regenerate(true);
        session()->set(['user_id' => (int) $user['id'], 'email' => $user['email']]);
        return redirect()->to('/videos');
    }

    public function logout()
    {
        session()->destroy();
        return redirect()->to('/login');
    }
}
