<?php

namespace App\Database\Migrations;

use CodeIgniter\Database\Migration;

class CreateJobs extends Migration
{
    public function up()
    {
        $this->forge->addField([
            'id'          => ['type' => 'INT', 'unsigned' => true, 'auto_increment' => true],
            'video_id'    => ['type' => 'INT', 'unsigned' => true],
            'status'      => ['type' => 'VARCHAR', 'constraint' => 20, 'default' => 'queued'],
            'stage'       => ['type' => 'VARCHAR', 'constraint' => 20, 'default' => 'queued'],
            'progress'    => ['type' => 'TINYINT', 'unsigned' => true, 'default' => 0],
            'log'         => ['type' => 'TEXT', 'null' => true],
            'error'       => ['type' => 'TEXT', 'null' => true],
            'started_at'  => ['type' => 'DATETIME', 'null' => true],
            'finished_at' => ['type' => 'DATETIME', 'null' => true],
            'created_at'  => ['type' => 'DATETIME', 'null' => true],
            'updated_at'  => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->addKey(['status', 'id']);
        $this->forge->addKey('video_id');
        $this->forge->createTable('jobs');
    }

    public function down()
    {
        $this->forge->dropTable('jobs');
    }
}
