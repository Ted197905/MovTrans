<?php

namespace App\Libraries;

class Storage
{
    /** Group-writable so the web user (php-fpm) and the worker (systemd, www-data) can both touch files. */
    public static function relax(string $path): void
    {
        @chmod($path, is_dir($path) ? 0775 : 0664);
    }

    public static function removeDir(string $dir): void
    {
        if (! is_dir($dir)) return;
        foreach (scandir($dir) ?: [] as $f) {
            if ($f === '.' || $f === '..') continue;
            $p = $dir . '/' . $f;
            is_dir($p) ? self::removeDir($p) : @unlink($p);
        }
        @rmdir($dir);
    }
}
