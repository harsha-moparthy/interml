"""T04: map workbook questions to PDF pages and compare titles and code. Usage: audit_source.py <ml-interview dir> <out.json>"""
import hashlib, json, re, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
pdf = root / "ml-coding-patterns-500-solutions.pdf"
files = {
    "pdf": pdf,
    "tex": root / "ml-coding-patterns-500-solutions.tex",
    "workbook": root / "ml-coding-workbook/workbook.json",
    "catalog": root / "ml-coding-catalog/catalog.json",
}
hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
workbook = json.loads(files["workbook"].read_text())
text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
pages = text.split("\f")
if pages and not pages[-1].strip():
    pages.pop()  # pdftotext ends with a form feed; drop the empty trailing split

def norm(s):
    """Fold PDF text substitutions: dash variants, curly quotes, backticks, dropped ligatures, headers."""
    s = re.sub(r"^ML Coding Patterns +600 questions and complete solutions$", "", s, flags=re.M)
    s = re.sub(r"^ +\d{1,3}$", "", s, flags=re.M)  # page-number footer
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"[\u2018\u2019`]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = s.replace("f", "")  # PDF text turns the f of fi/fl ligatures into a space; drop every f on both sides
    return re.sub(r"\s+", "", s)

heading = re.compile(r"^(\d{3})\. (.+)$", re.M)
found = {}
for pno, page in enumerate(pages, 1):
    for m in heading.finditer(page):
        n = int(m.group(1))
        found.setdefault(n, []).append({"page": pno, "title": m.group(2).strip(), "pos": m.start()})

records, problems = [], []
ids = [r["number"] for r in workbook]
if ids != list(range(1, 601)):
    problems.append("workbook IDs are not exactly 1..600 in order")
for r in workbook:
    n = r["number"]
    hits = found.get(n, [])
    rec = {"id": n, "title": r["title"], "pdf_pages": None, "title_match": None, "code_match": None}
    if len(hits) != 1:
        problems.append(f"{n:03d}: {len(hits)} PDF headings")
        records.append(rec); continue
    h = hits[0]
    # section text from this heading to the next heading of any question
    nxt = [x["page"] * 10**7 + x["pos"] for v in found.values() for x in v if (x["page"], x["pos"]) > (h["page"], h["pos"])]
    start = h["page"] * 10**7 + h["pos"]
    flat = "".join(f"\0{i}\0" + p for i, p in enumerate(pages, 1))
    seg_pages = []
    end = min(nxt) if nxt else None
    endp = (end // 10**7) if end else len(pages)
    rec["pdf_pages"] = [h["page"], endp] if endp != h["page"] else [h["page"], h["page"]]
    body = pages[h["page"] - 1][h["pos"]:]
    for p in range(h["page"] + 1, endp + 1):
        body += "\n" + pages[p - 1][: (end % 10**7) if p == endp and end else None]
    rec["title_match"] = norm(h["title"]) == norm(r["title"])
    if not rec["title_match"]:
        problems.append(f"{n:03d}: title differs: pdf={h['title']!r} workbook={r['title']!r}")
    rec["code_match"] = norm(r["code"]) in norm(body)
    if not rec["code_match"]:
        problems.append(f"{n:03d}: workbook code not found in PDF section")
    records.append(rec)

dups = sorted({n for n, v in found.items() if len(v) > 1})
extra = sorted(set(found) - set(range(1, 601)))
result = {
    "hashes": hashes, "pdf_pages": len(pages),
    "workbook_records": len(workbook), "pdf_headings": sum(len(v) for v in found.values()),
    "title_matches": sum(bool(r["title_match"]) for r in records),
    "code_matches": sum(bool(r["code_match"]) for r in records),
    "duplicate_pdf_ids": dups, "extra_pdf_ids": extra, "problems": problems, "records": records,
}
out.write_text(json.dumps(result, indent=1))
print({k: v for k, v in result.items() if k not in ("records", "problems")})
print(len(problems), "problems"); print("\n".join(problems[:40]))
