"""Render a URL with headless Chrome and save text. usage: python3 -B render.py URL OUTFILE [budget_ms]"""
import sys, re, html, subprocess
url, out = sys.argv[1], sys.argv[2]; budget = sys.argv[3] if len(sys.argv) > 3 else '25000'
prof = '/tmp/nbm-chrome-render-' + re.sub(r'\W', '_', url)[-60:]
dom = subprocess.run(['google-chrome-stable', '--headless=new', '--disable-gpu', '--no-sandbox', f'--user-data-dir={prof}',
    f'--virtual-time-budget={budget}', '--dump-dom', url], capture_output=True, text=True, timeout=180).stdout
links = sorted(set((m.group(1), re.sub(r'<[^>]+>',' ',m.group(2)).strip()[:120]) for m in re.finditer(r'href="(/competitions/arc-prize-2026-arc-agi-3/discussion/\d+)"[^>]*>(.*?)</a>', dom, flags=re.S)))
links = [a + '  ' + html.unescape(b) for a, b in links]
t = re.sub(r'<script.*?</script>|<style.*?</style>', '', dom, flags=re.S)
t = html.unescape(re.sub(r'<[^>]+>', '\n', t)); t = re.sub(r'\n\s*\n+', '\n', t)
open(out, 'w').write(t + '\n\nLINKS:\n' + '\n'.join(links))
print(len(t), 'chars', len(links), 'discussion links')
