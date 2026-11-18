import unittest
import requests
import execute_403
from kaggle_backend import KaggleBackend


class Slug403(unittest.TestCase):
    def test_403_404_free_others_and_existing_block(self):
        def failing(code):
            def get(slug):
                r=requests.Response();r.status_code=code;raise requests.HTTPError(response=r)
            return get
        b=KaggleBackend.__new__(KaggleBackend)
        for code in (403,404):
            b.get_kernel=failing(code);execute_403.require_new_slug(b,'o/s')
        for code in (401,500):
            b.get_kernel=failing(code)
            with self.assertRaises(requests.HTTPError):execute_403.require_new_slug(b,'o/s')
        b.get_kernel=lambda s:object()
        with self.assertRaisesRegex(ValueError,'already exists'):execute_403.require_new_slug(b,'o/s')


if __name__=='__main__':unittest.main()
