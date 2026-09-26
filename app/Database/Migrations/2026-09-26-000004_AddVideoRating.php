<?php

namespace App\Database\Migrations;

use CodeIgniter\Database\Migration;

class AddVideoRating extends Migration
{
    public function up()
    {
        $this->forge->addColumn('videos', [
            'rating' => ['type' => 'VARCHAR', 'constraint' => 10, 'default' => 'rated', 'after' => 'lang'],
        ]);
    }

    public function down()
    {
        $this->forge->dropColumn('videos', 'rating');
    }
}
