<?php

use CodeIgniter\Router\RouteCollection;

/**
 * @var RouteCollection $routes
 */
$routes->get('/', static fn () => redirect()->to(session()->get('user_id') ? '/videos' : '/login'));

$routes->get('login', 'Auth::login');
$routes->post('login', 'Auth::attemptLogin');
$routes->post('logout', 'Auth::logout');

$routes->group('', ['filter' => 'auth'], static function (RouteCollection $routes) {
    $routes->get('videos', 'Videos::index');
    $routes->get('videos/(:num)', 'Videos::show/$1');
    $routes->post('videos/(:num)/delete', 'Videos::delete/$1');
    $routes->post('videos/(:num)/rerun', 'Videos::rerun/$1');

    $routes->post('api/upload/init', 'Upload::init');
    $routes->post('api/upload/chunk', 'Upload::chunk');
    $routes->post('api/upload/finish', 'Upload::finish');
    $routes->post('api/upload/abort', 'Upload::abort');
    $routes->get('api/jobs/(:num)', 'Jobs::show/$1');

    $routes->get('media/(:num)/video', 'Media::video/$1');
    $routes->get('media/(:num)/sub/([a-z]+)\.([a-z]+)', 'Media::subtitle/$1/$2/$3');
});
