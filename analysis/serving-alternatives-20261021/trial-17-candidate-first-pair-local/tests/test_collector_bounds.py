import base64, hashlib, io, json, shutil, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collection_tools'))
import collector
IDENTITY = dict(kernel_id=140493086, version=1, source_exact=True, status='COMPLETE', inner_sha256='fdab81dc2113e336eaf38d46ccff1ed9e8b4b65efbe7811d219d88933c3559f4')

def collect(*args, **kwargs):
    return collector.collect(*args, identity=IDENTITY, **kwargs)

class Stream:

    def __init__(self, data, status=200, length=None):
        self.data = data
        self.status_code = status
        self.headers = {'content-length': str(len(data) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('HTTP failure')

    def iter_content(self, chunk_size):
        yield self.data

class API:

    def __init__(self, files, pages=None):
        self.files = files
        self.pages = pages or [[(n, 'https://fixture.invalid/' + n + '?secret=DO_NOT_SAVE') for n in files]]
        self.downloads = []
        self.lists = []
        self.identities = []

    def readback(self, destination, remaining):
        destination.mkdir()
        self.identities.append(str(destination))
        return dict(IDENTITY)

    def list_page(self, token, page_size):
        index = int(token or 0)
        self.lists.append((token, page_size))
        return (self.pages[index], str(index + 1) if index + 1 < len(self.pages) else '')

    def download(self, url, remaining):
        name = url.split('fixture.invalid/', 1)[1].split('?', 1)[0]
        self.downloads.append(name)
        value = self.files[name]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Stream) else Stream(value)

def row(name, conditional=False, **kwargs):
    return dict(name=name, conditional=conditional, expected_sha256=None, expected_bytes=None, **kwargs)

class CollectorTests(unittest.TestCase):

    def test_only_exact_missing_names_are_downloaded(self):
        api = API({'needed/raw.bin': b'abc', 'needed/raw.bin.extra': b'ignore', 'already/recovered.bin': b'ignore'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('needed/raw.bin')], Path(d) / 'new', api)
            self.assertTrue(result['complete'])
            self.assertEqual(api.downloads, ['needed/raw.bin'])
            self.assertEqual((Path(d) / 'new/members/needed/raw.bin').read_bytes(), b'abc')
            self.assertNotIn('DO_NOT_SAVE', json.dumps(result))
            self.assertNotIn('https://', json.dumps(result))

    def test_pending_body_waits_for_http200_evidence(self):
        name = 'candidate/rid/body.bin'
        h = 'candidate/rid/http.json'
        http = json.dumps(dict(rid='rid', status=200, received_bytes=3, persisted_bytes=3, headers=[['content-length', '5']], complete=False)).encode()
        api = API({name: b'abc', h: http})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row(name, True), row(h)], Path(d) / 'new', api)
            self.assertTrue(result['complete'])
            self.assertEqual(api.downloads, [h, name])

    def test_no_headers_and_zero_bytes_skips_conditional_body(self):
        name = 'candidate/rid/body.bin'
        h = 'candidate/rid/http.json'
        http = json.dumps(dict(rid='rid', status=None, received_bytes=0, persisted_bytes=0, complete=False)).encode()
        api = API({h: http})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row(name, True), row(h)], Path(d) / 'new', api)
            self.assertTrue(result['complete'])
            self.assertEqual(api.downloads, [h])
            self.assertEqual(result['skipped_conditionals'][0]['name'], name)

    def test_malformed_pending_receipt_stops_before_body(self):
        name = 'candidate/rid/body.bin'
        h = 'candidate/rid/http.json'
        api = API({h: b'{"rid":"other","status":200}', name: b'x'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row(name, True), row(h)], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(api.downloads, [h])
            self.assertEqual(len(api.identities), 1)

    def test_unsafe_manifest_has_no_api_or_download(self):
        for name in ('../escape', '/absolute', 'a/../b', 'a\\b', 'a//b'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                api = API({})
                result = collect([row(name)], Path(d) / 'new', api)
                self.assertFalse(result['complete'])
                self.assertEqual(api.lists, [])
                self.assertEqual(api.identities, [])

    def test_repeated_pagination_token_stops(self):

        class Repeated(API):

            def list_page(self, token, page_size):
                self.lists.append((token, page_size))
                return ([], 'same')
        api = Repeated({})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('missing')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(len(api.lists), 2)
            self.assertEqual(api.downloads, [])

    def test_page_limit_stops_before_next_request(self):
        api = API({'a': b'a'}, pages=[[], [('a', 'https://fixture.invalid/a')]])
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api, limits={'list_pages': 1})
            self.assertFalse(result['complete'])
            self.assertEqual(len(api.lists), 1)
            self.assertEqual(api.downloads, [])

    def test_one_failed_file_stops_without_retry_or_post_readback(self):
        api = API({'a': RuntimeError('https://fixture.invalid/a?secret=DO_NOT_SAVE'), 'b': b'b'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a'), row('b')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(api.downloads, ['a'])
            self.assertEqual(len(api.identities), 1)
            self.assertNotIn('DO_NOT_SAVE', json.dumps(result))

    def test_byte_cap_refuses_body_before_stream_consumption(self):
        api = API({'a': b'abcd'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api, limits={'bytes': 3})
            self.assertFalse(result['complete'])
            self.assertEqual(result['downloaded_bytes'], 0)
            self.assertFalse((Path(d) / 'new/members/a').exists())

    def test_hash_mismatch_retains_actual_bytes_then_stops(self):
        api = API({'a': b'actual', 'b': b'b'})
        with tempfile.TemporaryDirectory() as d:
            item = row('a')
            item['expected_sha256'] = '0' * 64
            result = collect([item, row('b')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual((Path(d) / 'new/members/a').read_bytes(), b'actual')
            self.assertEqual(api.downloads, ['a'])

    def test_after_identity_drift_invalidates_completed_downloads(self):

        class Drift(API):

            def readback(self, destination, remaining):
                value = super().readback(destination, remaining)
                if len(self.identities) == 2:
                    value['version'] = 2
                return value
        api = Drift({'a': b'a'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(api.downloads, ['a'])
            self.assertIn('after identity', result['error']['reason'])

    def test_wall_lateness_stops_after_inflight_list_and_no_download(self):

        class Clock:
            now = 0

            def __call__(self):
                return self.now
        clock = Clock()

        class Late(API):

            def list_page(self, *args):
                result = super().list_page(*args)
                clock.now = 11
                return result
        api = Late({'a': b'a'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api, limits={'wall_seconds': 10}, clock=clock)
            self.assertFalse(result['complete'])
            self.assertEqual(api.downloads, [])

    def test_exclusive_destination_refuses_overwrite(self):
        api = API({})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                collect([], Path(d), api)
            self.assertEqual(api.identities, [])

    def test_duplicate_requested_name_has_no_download(self):
        api = API({'a': b'a'}, pages=[[('a', 'https://fixture.invalid/a')], [('a', 'https://fixture.invalid/a')]])
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(api.downloads, [])

    def test_expanded_manifest_stops_before_any_request(self):
        api = API({'a': b'a', 'b': b'b'})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a'), row('b')], Path(d) / 'new', api, limits={'downloads': 1})
            self.assertFalse(result['complete'])
            self.assertEqual(api.identities, [])

    def test_missing_content_length_is_invalid_without_writing(self):
        stream = Stream(b'abc')
        stream.headers = {}
        api = API({'a': stream})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual(result['downloaded_bytes'], 0)

    def test_error_capture_never_reads_unbuffered_body_or_persists_url(self):
        import remote

        class Response:
            status_code = 403
            headers = {'content-type': 'text/plain', 'Authorization': 'DO_NOT_SAVE'}

            @property
            def content(self):
                raise AssertionError('must not read error stream')
        error = RuntimeError('https://signed.invalid/file?token=DO_NOT_SAVE')
        error.response = Response()
        value = remote.Remote.error_record(error)
        self.assertEqual(value['status'], 403)
        self.assertNotIn('DO_NOT_SAVE', json.dumps(value))
        self.assertNotIn('signed.invalid', json.dumps(value))

    def test_oversized_chunk_fails_closed_and_retains_only_bounded_prefix(self):
        api = API({'a': Stream(b'oversize', length=1)})
        with tempfile.TemporaryDirectory() as d:
            result = collect([row('a')], Path(d) / 'new', api)
            self.assertFalse(result['complete'])
            self.assertEqual((Path(d) / 'new/members/a').stat().st_size, 0)
if __name__ == '__main__':
    unittest.main()
