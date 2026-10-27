"""Restore one draft-release artifact without forwarding credentials to storage URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request

BASE = 'https://api.github.com/repos/NathanTong0921/ViT-token-compression'


def credential():
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            capture_output=True, text=True, check=True)
    return dict(row.split('=', 1) for row in result.stdout.splitlines() if '=' in row)['password']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def restore(path):
    path = Path(path)
    asset = json.loads(path.with_suffix('.github.json').read_text())
    metadata = json.loads(path.with_suffix('.json').read_text())
    destination = Path('cache/transfers/flowers-github-recovery') / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    request = urllib.request.Request(BASE + f"/releases/assets/{asset['id']}", headers={
        'Authorization': 'Bearer ' + credential(), 'Accept': 'application/octet-stream'})
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=90)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        response = urllib.request.urlopen(error.headers['Location'], timeout=90)
    with response, destination.open('wb') as target:
        shutil.copyfileobj(response, target, 8 * 1024 * 1024)
    digest = hashlib.sha256()
    with destination.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert destination.stat().st_size == metadata['bytes']
    assert digest.hexdigest() == metadata['sha256']
    shutil.copy2(path.with_suffix('.json'), destination.with_suffix('.json'))
    print(json.dumps({'asset_id': asset['id'], 'restored_to': str(destination),
                      'bytes': metadata['bytes'], 'matches_retained_local_archive': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    restore(parser.parse_args().path)
