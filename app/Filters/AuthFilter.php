<?php

namespace App\Filters;

use CodeIgniter\Filters\FilterInterface;
use CodeIgniter\HTTP\RequestInterface;
use CodeIgniter\HTTP\ResponseInterface;

class AuthFilter implements FilterInterface
{
    public function before(RequestInterface $request, $arguments = null)
    {
        if ((int) session()->get('user_id') > 0) {
            return;
        }
        if ($request->isAJAX() || str_starts_with($request->getUri()->getPath(), 'api/')) {
            return service('response')->setStatusCode(401)->setJSON(['error' => 'unauthorized']);
        }
        return redirect()->to('/login');
    }

    public function after(RequestInterface $request, ResponseInterface $response, $arguments = null)
    {
    }
}
