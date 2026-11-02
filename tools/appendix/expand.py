"""Prepend explicitly authored teaching passages to the existing technical answer."""
import json
from pathlib import Path
from author import DIRECTORY
HERE=Path(__file__).parent/'expanded';HERE.mkdir(exist_ok=True)
MARKER='\n\n**Technical details and references.**\n\n'
def write(blocks):
    for block in blocks.strip().split('\n@@ '):
        number,body=block.removeprefix('@@ ').split('\n',1)
        path=HERE/f'q{number}.md'
        old=path.read_text().split(MARKER,1)[-1] if path.exists() else json.loads((DIRECTORY/f'q{number}.json').read_text())['answer']
        path.write_text(body.strip()+MARKER+old.strip()+'\n')
    print('Expanded',len(blocks.strip().split('\n@@ ')),'answers')
