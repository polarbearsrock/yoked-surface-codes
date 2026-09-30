#!/usr/bin/env python3
"""Render the decoder explanation to PDF and a self-contained offline HTML file.

LaTeX compiles only inside DANTE_SCRATCH. Each PDF page becomes a vector SVG
embedded as an image in HTML, so the browser needs no math renderer, network,
or separate assets. Keeping each SVG in its own image also isolates font IDs.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORKSPACE / 'results/decoder_gap_explainer_20260924/decoder_gaps.tex'
ORIGINAL_PAGE_TITLES = (
    'The three configurations and the complementary gap',
    'Cluster geometry and the UF implementation',
    'The native MPP implementation and interpretation',
    'L2 decisions, comparison scope, and references',
)


def data_url(mime_type: str, content: bytes) -> str:
    """Inline an artifact without leaving an external file dependency."""
    encoded = base64.b64encode(content).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    output = source.parent
    # Expanded notes declare one title per deliberately designed page. Older
    # four-page notes remain reproducible without modifying their source.
    page_titles = re.findall(r'^% PAGE: (.+)$', source.read_text(), re.MULTILINE)
    if not page_titles:
        page_titles = ORIGINAL_PAGE_TITLES
    scratch = Path(os.environ['DANTE_SCRATCH']) / 'document-builds'
    scratch.mkdir(parents=True, exist_ok=True)
    for program in ('pdflatex', 'pdftocairo', 'pdftoppm', 'pdftotext', 'pdfinfo'):
        if shutil.which(program) is None:
            raise RuntimeError(f'Required rendering program is missing: {program}')

    commands: list[list[str]] = []

    with tempfile.TemporaryDirectory(prefix='decoder-gaps-', dir=scratch) as temporary:
        build = Path(temporary)
        environment = os.environ.copy()
        # TeX may create caches in addition to its normal auxiliary files.
        environment['TEXMFVAR'] = str(build / 'texmf-var')
        environment['TEXMFCONFIG'] = str(build / 'texmf-config')

        def run(arguments: list[str]) -> str:
            commands.append(arguments)
            result = subprocess.run(
                arguments, cwd=build, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode:
                raise RuntimeError(f'{arguments[0]} failed:\n{result.stdout}')
            return result.stdout

        latex_command = [
            'pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
            '-halt-on-error', '-output-directory', str(build), str(source),
        ]
        # The second pass resolves PDF cross-references and document metadata.
        run(latex_command)
        run(latex_command)
        log = (build / f'{source.stem}.log').read_text()
        warnings = re.findall(r'^.*(?:Overfull \\[hv]box|Missing character:).*$', log, re.MULTILINE)
        if warnings:
            raise RuntimeError('The document has layout or glyph problems:\n' + '\n'.join(warnings))

        pdf = build / f'{source.stem}.pdf'
        information = run(['pdfinfo', str(pdf)])
        pages = int(re.search(r'^Pages:\s+(\d+)$', information, re.MULTILINE).group(1))
        text_path = build / 'text.txt'
        run(['pdftotext', '-layout', str(pdf), str(text_path)])
        transcript = text_path.read_text()
        if pages != len(page_titles):
            first_lines = [
                f'{index}: {page.strip().splitlines()[0]}'
                for index, page in enumerate(transcript.split('\f'), start=1) if page.strip()
            ]
            raise RuntimeError(
                f'Expected {len(page_titles)} designed pages, got {pages}. Page starts:\n'
                + '\n'.join(first_lines))

        rendered_pages = []
        for page_number, title in enumerate(page_titles, start=1):
            svg = build / f'page-{page_number}.svg'
            run(['pdftocairo', '-svg', '-f', str(page_number), '-l', str(page_number),
                 str(pdf), str(svg)])
            rendered_pages.append(
                f'<figure id="page-{page_number}">'
                f'<figcaption>{page_number}. {html.escape(title)}</figcaption>'
                f'<img width="612" height="792" alt="{html.escape(title)}" '
                f'src="{data_url("image/svg+xml", svg.read_bytes())}"></figure>'
            )

        # A PNG gives chat clients and file browsers a ready-made preview.
        preview = build / 'preview'
        run(['pdftoppm', '-f', '1', '-singlefile', '-r', '140', '-png', str(pdf), str(preview)])
        pdf_name = f'{source.stem}.pdf'
        document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complementary gap and cluster gap</title>
<style>
  :root { color-scheme: light; font-family: system-ui, sans-serif; color: #203b60; }
  body { margin: 0; background: #eef2f6; }
  header, main, footer { max-width: 900px; margin: auto; padding: 24px; }
  h1 { font-size: clamp(24px, 4vw, 34px); line-height: 1.2; }
  p { line-height: 1.6; }
  nav { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }
  nav a { background: #203b60; color: white; padding: 10px 16px;
          border-radius: 6px; text-decoration: none; }
  main { padding-top: 0; }
  .contents { columns: 2; padding-left: 22px; line-height: 1.5; }
  .contents li { padding: 4px 8px 4px 0; break-inside: avoid; }
  .contents a { color: #203b60; }
  @media (max-width: 600px) { .contents { columns: 1; } }
  figure { margin: 0 0 30px; }
  figcaption { padding: 10px 0; font-weight: 600; }
  img { display: block; width: 100%; height: auto; background: white;
        box-shadow: 0 3px 16px #203b6018; }
  details { background: white; padding: 16px; border-radius: 6px; }
  summary { cursor: pointer; }
  pre { white-space: pre-wrap; color: #20252b; line-height: 1.5; }
  @media print {
    @page { size: letter; margin: 0; }
    body { background: white; }
    header, footer, figcaption, details { display: none; }
    main { max-width: none; padding: 0; }
    figure { margin: 0; break-after: page; }
    figure:last-of-type { break-after: auto; }
    img { box-shadow: none; }
  }
</style></head><body><header>
<h1>Complementary gap and cluster gap</h1>
<p>Research-meeting notes for the correlated MWPM and correlated UF experiments.
All equations and diagrams are already rendered. This file works offline.</p>
<nav>DOWNLOAD_LINKS</nav><ol class="contents">CONTENTS_LINKS</ol></header><main>RENDERED_PAGES
<details><summary>Text extracted from the PDF</summary><pre>TRANSCRIPT</pre></details>
</main><footer><p>Sources: <a href="https://arxiv.org/abs/2312.04522">Yoked surface codes</a>
and <a href="https://arxiv.org/html/2405.07433v2">Efficient soft-output decoders for the surface code</a>.
</p></footer></body></html>'''
        downloads = (
            f'<a download="{pdf_name}" href="{data_url("application/pdf", pdf.read_bytes())}">Download PDF</a>'
            f'<a download="{source.name}" href="{data_url("text/plain", source.read_bytes())}">Download LaTeX source</a>'
        )
        document = (document.replace('DOWNLOAD_LINKS', downloads)
                    .replace('CONTENTS_LINKS', ''.join(
                        f'<li><a href="#page-{index}">{html.escape(title)}</a></li>'
                        for index, title in enumerate(page_titles, start=1)))
                    .replace('RENDERED_PAGES', '\n'.join(rendered_pages))
                    .replace('TRANSCRIPT', html.escape(transcript)))
        (output / f'{source.stem}.html').write_text(document)
        shutil.copy2(pdf, output / pdf_name)
        shutil.copy2(preview.with_suffix('.png'), output / 'preview.png')
        (output / 'build.log').write_text(log)
        (output / 'text.txt').write_text(transcript)

        sources = [
            'src/yoked/hierarchical/_correlated_matching_gap.py',
            'src/yoked/hierarchical/_matching_gaps.py',
            'src/yoked/hierarchical/_patch_graphs.py',
            'src/yoked/hierarchical/_cluster_gap.py',
            'src/yoked/hierarchical/_correlated_uf.py',
            'src/yoked/hierarchical/_paper_decoder.py',
            'src/yoked/hierarchical/_mpp.py',
            'src/yoked/hierarchical/_mpp_experiment.py',
            'src/yoked/decoders/_union_find.py',
            'src/yoked/decoders/_correlated_union_find.py',
            'src/yoked/decoders/_correlations.py',
            'src/yoked/_yoked_memory_circuits.py',
            'native/mpp/mpp_fast.cc',
            'src/yoked/hierarchical/_outer_mwpm.py',
        ]
        repository = WORKSPACE / 'repos/yoked-surface-codes'
        commit = run(['git', '-C', str(repository), 'rev-parse', 'HEAD']).strip()
        artifacts = [pdf_name, f'{source.stem}.html', 'preview.png', source.name, 'text.txt']
        if (output / 'worked_example.json').exists():
            artifacts.append('worked_example.json')
        provenance = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'Rendered explanation; no simulations or decoder changes',
            'pages': pages,
            'decoder_commit': commit,
            'decoder_source_sha256': {name: sha256(repository / name) for name in sources},
            'recipe': str(Path(__file__).resolve()),
            'recipe_sha256': sha256(Path(__file__)),
            'latex_version': run(['pdflatex', '--version']).splitlines()[0],
            'commands': commands,
            'artifacts_sha256': {name: sha256(output / name) for name in artifacts},
            'checks': {'no_overfull_boxes': True, 'no_missing_glyphs': True,
                       'expected_page_count': True, 'offline_svg_pages': len(rendered_pages)},
        }
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    print(f'Rendered {pages} pages: {output / pdf_name}')
    print(f'Offline HTML: {output / (source.stem + ".html")}')


if __name__ == '__main__':
    main()
