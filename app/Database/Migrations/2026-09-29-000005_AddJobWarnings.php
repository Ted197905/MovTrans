<?php

namespace App\Database\Migrations;

use CodeIgniter\Database\Migration;

class AddJobWarnings extends Migration
{
    public function up()
    {
        $this->forge->addColumn('jobs', [
            'warnings' => ['type' => 'TEXT', 'null' => true, 'after' => 'error'],
        ]);
    }

    public function down()
    {
        $this->forge->dropColumn('jobs', 'warnings');
    }
}
