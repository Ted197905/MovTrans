<?php

namespace App\Database\Migrations;

use CodeIgniter\Database\Migration;

class CreateVideos extends Migration
{
    public function up()
    {
        $this->forge->addField([
            'id'         => ['type' => 'INT', 'unsigned' => true, 'auto_increment' => true],
            'title'      => ['type' => 'VARCHAR', 'constraint' => 255],
            'filename'   => ['type' => 'VARCHAR', 'constraint' => 64],
            'size'       => ['type' => 'BIGINT', 'unsigned' => true, 'default' => 0],
            'duration'   => ['type' => 'DECIMAL', 'constraint' => '10,3', 'null' => true],
            'width'      => ['type' => 'INT', 'unsigned' => true, 'null' => true],
            'height'     => ['type' => 'INT', 'unsigned' => true, 'null' => true],
            'vcodec'     => ['type' => 'VARCHAR', 'constraint' => 32, 'null' => true],
            'has_proxy'  => ['type' => 'TINYINT', 'unsigned' => true, 'default' => 0],
            'lang'       => ['type' => 'VARCHAR', 'constraint' => 8],
            'status'     => ['type' => 'VARCHAR', 'constraint' => 20, 'default' => 'queued'],
            'created_at' => ['type' => 'DATETIME', 'null' => true],
            'updated_at' => ['type' => 'DATETIME', 'null' => true],
        ]);
        $this->forge->addKey('id', true);
        $this->forge->createTable('videos');
    }

    public function down()
    {
        $this->forge->dropTable('videos');
    }
}
