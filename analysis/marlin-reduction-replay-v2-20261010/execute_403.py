"""Unsealed wrapper: live GetKernel answers 403 (not 404) for an unknown slug in the owner's namespace.
Control probes 2026-10-10: existing own kernel -> OK; v2 slug and a random slug -> identical 403.
Sealed bytes stay untouched; this only widens the 'slug is free' test, still failing on 401/5xx or an existing kernel."""
import requests
import execute
import kaggle_backend


def require_new_slug(self,slug):
    try:self.get_kernel(slug)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code in {403,404}:return
        raise
    raise ValueError('Remote slug already exists; do not reuse a consumed or ambiguous run identity')


if __name__=='__main__':
    kaggle_backend.KaggleBackend.require_new_slug=require_new_slug
    execute.main()
