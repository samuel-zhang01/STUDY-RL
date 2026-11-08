#!/usr/bin/env python3
"""Execute output-free source notebooks in isolated copied course trees.

Run from the repository root:
    python scripts/execute_reference_notebooks.py --jobs 3
    python scripts/execute_reference_notebooks.py --week 11 --week 26

The source cells run unchanged, in order, in fresh explicit-interpreter kernels.
Model downloads are not requested. Outputs go to ignored portal/generated state
and /tmp execution copies, never back into the source notebooks. A manifest
records notebook/source-code hashes, environment versions and cell indices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT/'portal/generated/notebook-outputs/manifest.json'
SIBLING_DEPENDENCIES = {
    'week-15-decision-transformer-diffusion': ['week-14-offline-rl-1/project/env.py'],
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class TableHTML(HTMLParser):
    """Retain inert table/text markup only; no style, links, SVG or JavaScript."""
    allowed = {'table','thead','tbody','tfoot','tr','th','td','caption','p','br',
               'strong','em','code','pre','span','div','b','i'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppress = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script','style','svg','iframe','object'}:
            self.suppress += 1
        elif not self.suppress and tag in self.allowed:
            self.parts.append('<'+tag+'>')

    def handle_endtag(self, tag):
        if tag in {'script','style','svg','iframe','object'}:
            self.suppress = max(0,self.suppress-1)
        elif not self.suppress and tag in self.allowed and tag != 'br':
            self.parts.append('</'+tag+'>')

    def handle_data(self, data):
        if not self.suppress:
            self.parts.append(escape(data))


def safe_output(output):
    kind = output.get('output_type')
    if kind == 'stream':
        return dict(output_type='stream', name=output.get('name','stdout'),
                    text=str(output.get('text',''))[:18000])
    if kind not in {'display_data','execute_result'}:
        return None
    data = {}
    for mime,value in output.get('data',{}).items():
        value = ''.join(value) if isinstance(value,list) else str(value)
        if mime == 'image/png' and len(value) <= 4_000_000:
            data[mime] = value
        elif mime == 'text/plain':
            data[mime] = value[:18000]
        elif mime == 'text/html':
            parser = TableHTML(); parser.feed(value[:100000]); parser.close()
            data[mime] = ''.join(parser.parts)
    if not data:
        return None
    result = dict(output_type=kind,data=data,metadata={})
    if kind == 'execute_result':
        result['execution_count'] = output.get('execution_count')
    return result


def code_sources(notebook):
    """Record only the trusted course roots actually available to the kernel."""
    paths = list((ROOT/'common').glob('*.py'))
    paths += list(notebook.parent.glob('*.py'))
    paths += list((notebook.parent/'project').rglob('*.py'))
    paths += list((notebook.parent/'llm_lab').rglob('*.py'))
    paths += [ROOT/p for p in SIBLING_DEPENDENCIES.get(notebook.parent.name,[])]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def execute_one(path, work_root, timeout):
    started = time.monotonic()
    relative = str(path.relative_to(ROOT))
    record = dict(source_sha256=digest(path),source_files_sha256=code_sources(path),
                  executed_at=datetime.now(timezone.utc).isoformat(),status='running',
                  execution_mode='reference',overrides=[],cells=[])
    execution = work_root/path.parent.name
    # A separate parent tree per notebook avoids cross-week `project` aliasing
    # and prevents outputs such as figs/*.pdf from changing tracked sources.
    tree = execution/'course'
    tree.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__','.pytest_cache','runs','checkpoints',
                                   'workbook.executed.ipynb','.ipynb_checkpoints')
    shutil.copytree(ROOT/'common',tree/'common',ignore=ignore)
    shutil.copytree(path.parent,tree/path.parent.name,ignore=ignore)
    for relative_dependency in SIBLING_DEPENDENCIES.get(path.parent.name,[]):
        destination = tree/relative_dependency
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative_dependency,destination)
    copied = tree/path.parent.name/path.name
    nb = nbformat.read(copied,as_version=4)
    original_sources = [c.source for c in nb.cells]
    kernel = KernelManager(kernel_name='python3')
    kernel.kernel_spec.argv = [sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client = NotebookClient(nb,timeout=timeout,kernel_name='python3',km=kernel,
                            resources={'metadata':{'path':str(copied.parent)}})
    try:
        nbformat.validate(nb)
        client.execute()
        if [c.source for c in nb.cells] != original_sources:
            raise RuntimeError('Execution changed a source cell')
        if digest(path) != record['source_sha256']:
            raise RuntimeError('Source notebook changed during execution; rerun for current provenance')
        if code_sources(path) != record['source_files_sha256']:
            raise RuntimeError('Course code changed during execution; rerun for current provenance')
        record['status'] = 'executed'
    except Exception as exc:
        record['status'] = 'failed'
        # Notebook inputs contain only public teaching code/synthetic fixtures.
        # Keep a bounded diagnostic; omit traceback and failed output objects.
        record['error'] = f'{type(exc).__name__}: {str(exc)[-2500:]}'
    finally:
        if kernel.has_kernel:
            kernel.shutdown_kernel(now=True)
        kernel.cleanup_resources()
    for index,cell in enumerate(nb.cells,1):
        outputs = [safe_output(o) for o in cell.get('outputs',[])]
        outputs = [o for o in outputs if o]
        if outputs:
            record['cells'].append(dict(index=index,outputs=outputs))
    record['code_cells'] = sum(c.cell_type == 'code' for c in nb.cells)
    record['executed_code_cells'] = sum(c.cell_type == 'code' and c.get('execution_count') is not None
                                        for c in nb.cells)
    record['duration_seconds'] = round(time.monotonic()-started,3)
    record['png_count'] = sum('image/png' in o.get('data',{}) for c in record['cells'] for o in c['outputs'])
    executed_path = execution/'workbook.executed.ipynb'
    nbformat.write(nb,executed_path)
    record['executed_notebook'] = str(executed_path)
    return relative,record


def write_manifest(path,manifest):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--week',type=int,action='append')
    parser.add_argument('--include-lab',action='store_true',help='Include optional lab with a selected week')
    parser.add_argument('--timeout',type=int,default=600,help='Per-cell seconds; unchanged source budgets')
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 8:
        parser.error('--jobs must be in [1,8]')
    # These limits affect CPU execution, not algorithm/sample budgets.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp(prefix='study-rl-matplotlib-')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    paths = sorted(ROOT.glob('week-*/workbook.ipynb'))
    if args.week:
        paths = [p for p in paths if int(p.parent.name.split('-')[1]) in args.week]
    if not args.week or args.include_lab:
        paths.append(ROOT/'llm-lab/workbook.ipynb')
    if not paths:
        parser.error('No selected notebooks')
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        manifest = dict(schema=1,notebooks={})
    manifest['environment'] = dict(python=sys.version.split()[0],
        packages={name:importlib.metadata.version(name)
                  for name in ['numpy','scipy','matplotlib','torch','nbclient','nbformat','ipykernel']},
        cpu_threads_per_kernel=1,model_downloads=False)
    manifest['generated_at'] = datetime.now(timezone.utc).isoformat()
    work_root = Path(tempfile.mkdtemp(prefix='study-rl-notebooks-'))
    failed = []
    print(json.dumps(dict(selected=len(paths),work_root=str(work_root),manifest=str(args.manifest))),flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute_one,p,work_root,args.timeout):p for p in paths}
        for future in as_completed(futures):
            relative,record = future.result()
            manifest['notebooks'][relative] = record
            write_manifest(args.manifest,manifest)
            print(json.dumps(dict(notebook=relative,status=record['status'],
                                  seconds=record['duration_seconds'],pngs=record['png_count'],
                                  executed_cells=record['executed_code_cells'],
                                  code_cells=record['code_cells'],error=record.get('error'))),flush=True)
            if record['status'] != 'executed':
                failed.append(relative)
    print(json.dumps(dict(selected=len(paths),passed=len(paths)-len(failed),failed=failed)),flush=True)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
