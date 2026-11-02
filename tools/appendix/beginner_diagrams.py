"""Original worked-example diagrams; no externally reproduced artwork."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'app/frontend/public/answer-diagrams'
def diagram(name,title,description,body):
    (OUT/f'{name}.svg').write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 400" role="img" aria-labelledby="title desc"><title id="title">{title}</title><desc id="desc">{description}</desc><rect width="760" height="400" rx="16" fill="#f4f8fc"/><g font-family="sans-serif" fill="#172b45" font-size="19"><text x="28" y="40" font-size="25" font-weight="bold">{title}</text>{body}</g></svg>\n''')
diagram('entropy-kl','Cross-entropy = source uncertainty + mismatch','Original numerical example in bits. P is 0.75, 0.25. A matching model has cross-entropy 0.8113 and KL zero. A uniform model has cross-entropy 1, consisting of entropy 0.8113 plus KL 0.1887.', '''<text x="28" y="80">True weather: 75% sunshine, 25% rain</text>
<text x="28" y="126">Model Q = P</text><rect x="28" y="142" width="486.78" height="48" rx="5" fill="#167d67"/><text x="47" y="173" fill="white">Entropy: 0.8113 bits</text><text x="535" y="173">KL = 0</text>
<text x="28" y="230">Model Q = (0.5, 0.5)</text><rect x="28" y="246" width="486.78" height="48" fill="#167d67"/><rect x="514.78" y="246" width="113.22" height="48" fill="#d1812a"/><text x="47" y="277" fill="white">Same entropy: 0.8113</text><text x="571" y="277" text-anchor="middle" fill="#172b45">+ 0.1887</text>
<text x="28" y="335">Total cross-entropy: 1 bit. Only the mismatch changes.</text><text x="28" y="373">For fixed P, minimizing H(P,Q) minimizes KL(P || Q).</text>''')
diagram('in-batch-negatives','In-batch negatives: reuse the other positive items','Original schematic. Rows are Alice, Ben, Cara; columns are Guitar, Football, Pasta. Diagonal pairs are observed positives, all six off-diagonal pairs are sampled negatives, not confirmed dislikes. Batch selection follows logged item popularity.', '''<text x="28" y="78">One batch → nine scores from one matrix multiplication</text>
<text x="218" y="122">Guitar</text><text x="398" y="122">Football</text><text x="583" y="122">Pasta</text>
<text x="28" y="168">Alice</text><text x="28" y="228">Ben</text><text x="28" y="288">Cara</text>
<rect x="185" y="138" width="165" height="44" rx="5" fill="#167d67"/><text x="267" y="166" text-anchor="middle" fill="white">Positive</text>
<rect x="370" y="138" width="165" height="44" rx="5" fill="#e1edfa"/><text x="452" y="166" text-anchor="middle">Negative</text>
<rect x="555" y="138" width="165" height="44" rx="5" fill="#e1edfa"/><text x="637" y="166" text-anchor="middle">Negative</text>
<rect x="185" y="198" width="165" height="44" rx="5" fill="#e1edfa"/><text x="267" y="226" text-anchor="middle">Negative</text>
<rect x="370" y="198" width="165" height="44" rx="5" fill="#167d67"/><text x="452" y="226" text-anchor="middle" fill="white">Positive</text>
<rect x="555" y="198" width="165" height="44" rx="5" fill="#e1edfa"/><text x="637" y="226" text-anchor="middle">Negative</text>
<rect x="185" y="258" width="165" height="44" rx="5" fill="#e1edfa"/><text x="267" y="286" text-anchor="middle">Negative</text>
<rect x="370" y="258" width="165" height="44" rx="5" fill="#e1edfa"/><text x="452" y="286" text-anchor="middle">Negative</text>
<rect x="555" y="258" width="165" height="44" rx="5" fill="#167d67"/><text x="637" y="286" text-anchor="middle" fill="white">Positive</text>
<text x="28" y="345">Popular items recur in batches; rare items are seldom compared.</text><text x="28" y="378">“Negative” is a training comparison, not proof of dislike.</text>''')
for number,name,caption,anchor in [
(796,'entropy-kl','Original worked example: cross-entropy separates into fixed entropy and model mismatch.','**Entropy: the uncertainty in the source itself.**'),
(824,'in-batch-negatives','Original three-user batch: observed diagonal pairs and six reused negative comparisons.','This is computationally attractive.')]:
    path=Path(__file__).parent/'expanded'/f'q{number}.md'
    content=path.read_text()
    if f'/answer-diagrams/{name}.svg' not in content:
        content=content.replace(anchor,f'![{caption}](/answer-diagrams/{name}.svg)\n\n'+anchor,1)
        path.write_text(content)
print('Added two original worked-example diagrams')
