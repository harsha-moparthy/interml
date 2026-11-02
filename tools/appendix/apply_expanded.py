"""Apply carefully authored beginner explanations after the original chapter scripts."""
import json,re
from pathlib import Path
from author import DIRECTORY
HERE=Path(__file__).parent/'expanded'
for path in sorted(HERE.glob('q*.md')):
    target=DIRECTORY/(path.stem+'.json');item=json.loads(target.read_text())
    resources=path.with_suffix('.resources.json')
    if resources.exists():item['readings']=json.loads(resources.read_text())
    refs=item['readings'];assert len(refs)==5
    def cite(match):
        rank=int(match[1]);return f'[{rank}]({refs[rank-1]["url"]})'
    item['answer']=re.sub(r'\[(\d)\](?!\()',cite,path.read_text().strip())
    item['explanation_revision']='beginner-worked-v2'
    target.write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
print(f'Applied {len(list(HERE.glob("q*.md")))} worked explanations')
