<?php

namespace App\Commands;

use App\Models\UserModel;
use CodeIgniter\CLI\BaseCommand;
use CodeIgniter\CLI\CLI;

class AdminSet extends BaseCommand
{
    protected $group       = 'movtrans';
    protected $name        = 'admin:set';
    protected $description = 'Create or update the admin login: php spark admin:set <email> <password>';

    public function run(array $params)
    {
        $email = $params[0] ?? null;
        $pass  = $params[1] ?? null;
        if (! $email || ! $pass || ! filter_var($email, FILTER_VALIDATE_EMAIL)) {
            CLI::error('usage: php spark admin:set <email> <password>');
            return;
        }
        $users = new UserModel();
        $hash  = password_hash($pass, PASSWORD_DEFAULT);
        $row   = $users->where('email', $email)->first();
        $row ? $users->update($row['id'], ['password_hash' => $hash]) : $users->insert(['email' => $email, 'password_hash' => $hash]);
        CLI::write('admin ' . $email . ($row ? ' updated' : ' created'), 'green');
    }
}
