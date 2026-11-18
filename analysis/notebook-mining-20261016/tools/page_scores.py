"""Render Kaggle notebook pages with headless Chrome; extract public/best score, runtime, version, inputs.
usage: python3 -B page_scores.py OUTDIR REF [REF ...]   -> OUTDIR/<owner>__<slug>.txt and prints a JSON line per ref
"""
import sys, re, html, json, subprocess, os, concurrent.futures as cf
out = sys.argv[1]; refs = sys.argv[2:]
def one(ref):
    fn = os.path.join(out, ref.replace('/', '__') + '.txt')
    if not os.path.exists(fn) or os.path.getsize(fn) < 500:
        prof = '/tmp/nbm-chrome-' + ref.replace('/', '_')
        try:
            dom = subprocess.run(['google-chrome-stable', '--headless=new', '--disable-gpu', '--no-sandbox',
                f'--user-data-dir={prof}', '--virtual-time-budget=20000', '--dump-dom',
                f'https://www.kaggle.com/code/{ref}'], capture_output=True, text=True, timeout=120).stdout
        except subprocess.TimeoutExpired:
            dom = ''
        t = re.sub(r'<script.*?</script>|<style.*?</style>', '', dom, flags=re.S)
        t = html.unescape(re.sub(r'<[^>]+>', '\n', t)); t = re.sub(r'\n\s*\n+', '\n', t)
        open(fn, 'w').write(t)
    t = open(fn).read()
    def after(lbl):
        m = re.search(re.escape(lbl) + r'\n([^\n]+)', t); return m.group(1).strip() if m else None
    ver = re.search(r'Version (\d+) of (\d+)', t)
    rt = re.search(r'([\dhms ]+) · GPU ([^\n]+)', t)
    return dict(ref=ref, public=after('Public Score'), best=after('Best Score'),
                version=ver.group(0) if ver else None, runtime=rt.group(1).strip() if rt else None, chars=len(t))
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(one, refs): print(json.dumps(r), flush=True)
