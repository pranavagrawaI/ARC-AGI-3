"""A single local health request; parent owns its absolute deadline/process group."""
import errno
import json
import sys
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def capture(response):
    """First line is metadata; remaining bytes are the exact bounded raw body."""
    metadata = dict(status=response.status, headers=list(response.headers.items()), body_format='raw bytes after first newline', body_limit=65536)
    print(json.dumps(metadata), flush=True)
    size = 0
    while size <= 65536:
        chunk = response.read1(min(4096, 65537 - size))
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        size += len(chunk)
    return 0 if response.status == 200 and size <= 65536 else 1


def main(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(url, timeout=60) as response:
            return capture(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return capture(exc)
    except urllib.error.URLError as exc:
        print(repr(exc), flush=True)
        return 75 if getattr(exc.reason, 'errno', None) == errno.ECONNREFUSED else 1
    except Exception as exc:
        print(repr(exc), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
