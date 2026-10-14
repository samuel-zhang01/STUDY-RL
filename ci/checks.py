"""Execute the existing Make targets; never replace their cases or browser scenarios."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def require(condition, message):
    if not condition: raise RuntimeError(message)


def run(args, *, cwd, timeout=1200, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=timeout)


def history_git(args, cwd, timeout=120):
    result = subprocess.run(['git']+args, cwd=cwd, timeout=timeout, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, 'History Git operation refused; raw diagnostics withheld')
    return result.stdout


def history_refs(text, allow_head=False):
    require(len(text.encode()) <= 4*1024**2, 'History ref metadata bound exceeded')
    result = {}
    for line in text.splitlines():
        oid, separator, name = line.partition(' ')
        require(separator and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', oid)
                and (name == 'HEAD' and allow_head or re.fullmatch(r'refs/[^\x00-\x20\x7f]+', name))
                and name not in result, 'History ref metadata refused')
        result[name] = oid
    require(result and len(result) <= 20000, 'History ref count refused')
    return result


def history_closure(text):
    objects = text.splitlines()
    require(objects and len(text.encode()) <= 32*1024**2
            and all(re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', value) for value in objects),
            'History reachable object metadata refused')
    objects = sorted(set(objects))
    return {'count': len(objects), 'sha256': hashlib.sha256(('\n'.join(objects)+'\n').encode('ascii')).hexdigest()}


def verify_history(work, record, revision):
    require(history_refs(history_git(['for-each-ref', '--format=%(objectname) %(refname)'], work)) == record['sourceRefs'],
            'Imported refs differ from exact source ref/OID map')
    require(history_git(['rev-parse', '--verify', 'HEAD^{commit}'], work).strip() == revision == record['revision'],
            'History detached HEAD differs from event revision')
    detached = subprocess.run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=work, timeout=30,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(detached.returncode == 1 and not detached.stdout and not detached.stderr,
            'History HEAD must be detached')
    require(history_closure(history_git(['rev-list', '--objects', '--all', 'HEAD', '--no-object-names'], work)) == record['reachableObjects'],
            'Imported reachable-object closure differs')


def prepare_history(source, work, revision):
    bundle = source/'ci/history.bundle'; manifest = source/'ci/history-manifest.json'
    require(bundle.is_file() and not bundle.is_symlink() and manifest.is_file() and not manifest.is_symlink()
            and manifest.stat().st_size <= 4*1024**2, 'Bound history inputs required')
    record = json.loads(manifest.read_text())
    require(set(record) == {'version', 'revision', 'sourceRefs', 'advertisedRefs', 'reachableObjects',
                           'bundleSha256', 'bundleBytes', 'scope'} and record['version'] == 1
            and record['revision'] == revision
            and record['scope'] == 'exact-ref-oid-and-object-closure-present-in-fetched-checkout',
            'Exact history input contract required')
    require(0 < record['bundleBytes'] <= 512*1024**2 and bundle.stat().st_size == record['bundleBytes']
            and hashlib.sha256(bundle.read_bytes()).hexdigest() == record['bundleSha256'],
            'History bundle content binding differs')
    advertised = history_refs(history_git(['bundle', 'list-heads', str(bundle)], source), allow_head=True)
    require(advertised == record['advertisedRefs'] == {**record['sourceRefs'], 'HEAD': revision},
            'Advertised bundle refs differ from prebundle source map')
    work.mkdir(mode=0o700)
    # Mirror keeps remote-tracking, notes and custom refs rather than remapping
    # or dropping them. Only this fresh temporary clone's local config changes.
    history_git(['-c', 'protocol.file.allow=always', 'clone', '--mirror', '--no-hardlinks', '--template=',
                 str(bundle), str(work/'.git')], source)
    require(history_refs(history_git(['--git-dir='+str(work/'.git'), 'for-each-ref', '--format=%(objectname) %(refname)'], work)) == record['sourceRefs'],
            'Mirror import omitted or changed a bundled ref')
    history_git(['--git-dir='+str(work/'.git'), 'config', '--local', 'core.bare', 'false'], work)
    history_git(['checkout', '--detach', revision], work)
    history_git(['fsck', '--full', '--strict', '--no-dangling'], work)
    verify_history(work, record, revision)
    return record


def main():
    mode, revision = sys.argv[1:]
    require(mode in {'cpu', 'browser', 'history'}, 'Unknown maintained mode')
    require(sys.version_info[:2] == (3, 12), 'Exact Python 3.12 required')
    require(os.getuid() == os.geteuid() == 10001 and os.getgid() == 10001, 'Non-root identity missing')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    require(int(status['CapEff'], 16) == 0 and status['NoNewPrivs'].strip() == '1'
            and status['Seccomp'].strip() == '2', 'Kernel privilege controls missing')
    limits = {p: (Path('/sys/fs/cgroup')/p).read_text().strip() for p in ('cpu.max', 'memory.max', 'pids.max')}
    quota, period = map(int, limits['cpu.max'].split())
    require(quota == period and quota > 0 and limits['memory.max'] == str(4*1024**3)
            and limits['pids.max'] == '512', 'Effective cgroup controls missing')
    require(not any(line.split()[1] == '00000000' for line in Path('/proc/net/route').read_text().splitlines()[1:]),
            'External network route forbidden')
    require(not any(Path('/dev').glob('nvidia*')) and shutil.which('docker') is None,
            'GPU and Docker authority forbidden')
    require(subprocess.check_output(['node', '--version'], text=True).strip() == 'v22.22.2', 'Exact Node required')
    source = Path('/input'); work = Path('/tmp/work'); Path('/tmp/home').mkdir(mode=0o700)
    manifest = json.loads((source/'ci/source-manifest.json').read_text())['files']
    require(not any((source/p).exists() for p in ('.git', '.env', 'runs', 'checkpoints')), 'Private state forbidden')
    if mode == 'history':
        history = prepare_history(source, work, revision)
    else:
        shutil.copytree(source, work)
        # Release checker needs only a credential-free temporary tracked index.
        run(['git', 'init', '--initial-branch=main'], cwd=work, timeout=30)
        run(['git', 'add', '--all'], cwd=work, timeout=30)
    def preserve():
        for relative, digest in manifest.items():
            require(hashlib.sha256((source/relative).read_bytes()).hexdigest() == digest
                    and hashlib.sha256((work/relative).read_bytes()).hexdigest() == digest,
                    'Original tracked source changed')
    preserve()
    if mode == 'cpu':
        import torch
        require(torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None and not torch.cuda.is_available(),
                'Exact CPU-only Torch required')
        require(not any(d.metadata['Name'].lower().startswith('nvidia-') for d in importlib.metadata.distributions()),
                'CUDA distributions forbidden')
        # All optional lab dependencies are installed. Missing imports may not
        # turn maintained importorskip cases into an apparent passing baseline.
        for package, version in [('transformers', '5.16.1'), ('peft', '0.20.0'), ('accelerate', '1.14.0')]:
            require(importlib.metadata.version(package) == version, 'Maintained lab dependency mismatch')
        run(['make', 'release-check', 'PY=python'], cwd=work)
        records = Path('/tmp/pytest-outcomes'); records.mkdir(mode=0o700)
        env = {**os.environ, 'PYTEST_PLUGINS': 'ci.pytest_gate', 'STUDY_CI_OUTCOMES': str(records)}
        result = subprocess.run(['make', 'tests', 'PY=python'], cwd=work, timeout=1500, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        require(len(result.stdout.encode()) <= 16*1024**2, 'Synthetic test output bound exceeded')
        print(result.stdout, end='', flush=True)
        require(result.returncode == 0, 'Original make tests failed')
        observed = [json.loads(p.read_text()) for p in records.glob('*.json')]
        expected_processes = len(list(work.glob('week-*/tests')))+3  # common, scripts and optional LLM lab
        require(len(observed) == expected_processes and all(x['accepted'] for x in observed),
                'Every maintained isolated pytest suite must execute without skips')
        unit_counts = re.findall(r'^Ran (\d+) tests? in ', result.stdout, re.MULTILINE)
        require(len(unit_counts) == 1 and int(unit_counts[0]) >= 21
                and len(re.findall(r'^OK\s*$', result.stdout, re.MULTILINE)) == 1,
                'Maintained Python unittest cases must execute without skips')
        node = {name: re.findall(r'^# '+name+r' (\d+)\s*$', result.stdout, re.MULTILINE)
                for name in ('tests', 'pass', 'fail', 'cancelled', 'skipped', 'todo')}
        require(all(len(value) == 1 for value in node.values()), 'Node accounting missing or ambiguous')
        node = {key: int(value[0]) for key, value in node.items()}
        require(node['tests'] >= 12 and node['tests'] == node['pass']
                and all(node[key] == 0 for key in ('fail', 'cancelled', 'skipped', 'todo')),
                'Maintained Node engine cases must execute without skips')
        print(json.dumps({'isolated_pytest_suites': len(observed), 'passed_cases': sum(x['passed'] for x in observed),
                          'unittest_cases': int(unit_counts[0]), 'node_cases': node['tests'], 'skipped': 0,
                          'make_tests_includes': ['weekly CPU', 'offline random-model LLM', 'Python unittest', 'Node engine']}))
    elif mode == 'browser':
        require(json.loads((work/'interactive/package.json').read_text())['devDependencies']['playwright'] == '1.64.0',
                'Exact browser package lock contract changed')
        (work/'interactive/node_modules').symlink_to('/opt/ci-node/node_modules', target_is_directory=True)
        env = {**os.environ, 'RL_LAB_BROWSERS': 'chromium,firefox,webkit', 'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright'}
        run(['make', 'browser-tests', 'PY=python'], cwd=work, timeout=360, env=env)
        print(json.dumps({'maintained_browser_engines': ['chromium', 'firefox', 'webkit'], 'original_scenarios_unchanged': True}))
    else:
        run(['make', 'secrets'], cwd=work, timeout=180)
        verify_history(work, history, revision)
        require(hashlib.sha256((source/'ci/history.bundle').read_bytes()).hexdigest() == history['bundleSha256'],
                'History bundle changed during scan')
        print(json.dumps({'history_scan': 'all reachable history in exactly bundled fetched refs, redacted',
                          'gitleaks_version': '8.30.1', 'source_ref_count': len(history['sourceRefs']),
                          'bundle_sha256': history['bundleSha256'], 'reachable_objects': history['reachableObjects'],
                          'unfetched_server_refs_or_deleted_history_claimed': False}))
    preserve()
    print(json.dumps({'mode': mode, 'accepted': True, 'uid': 10001, 'caps': 0, 'no_new_privs': 1,
                      'seccomp': 2, 'cgroup': limits, 'external_route': False, 'source_preserved': True,
                      'pretrained_models_providers_deployment': False}))


if __name__ == '__main__': main()
