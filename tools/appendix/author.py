"""Write explicitly authored answer blocks to editable manifests; no generated fallback answers."""
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'app/content/solutions'
SOURCES = {}
for line in (Path(__file__).with_name('sources.tsv')).read_text().splitlines():
    if not line.strip(): continue
    key, title, url, why = line.split('\t')
    SOURCES[key] = dict(title=title, url=url, publisher=url.split('/')[2], tags='', why=why, access='Publisher page; full text may require access' if key == 'pinn' else 'Free online paper, book, or documentation')

def write(blocks):
    for block in blocks.strip().split('\n@@ '):
        header, body = block.removeprefix('@@ ').split('\n',1)
        number, keys = header.split('|')
        number=int(number); keys=keys.split()
        assert len(keys)==5 and len(set(keys))==5, header
        references=[SOURCES[k] for k in keys]
        def cite(m):
            rank=int(m[1]); assert 1 <= rank <= 5
            return f'[{rank}]({references[rank-1]["url"]})'
        body=re.sub(r'\[(\d)\](?!\()',cite,body.strip())
        assert len(body.split())>=60, (number, 'answer too short')
        path=DIRECTORY/f'q{number:03}.json'
        item=json.loads(path.read_text()) if path.exists() else {'id':number}
        item.update(answer=body, readings=references)
        path.write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
        print(number, end=' ')
    print()
