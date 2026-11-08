#!/usr/bin/env python3
"""Create a bounded, reproducible course snapshot. No crawling outside course roots."""
from pathlib import Path
import re,json,subprocess,hashlib,html,shutil,tempfile,zipfile,sys
ROOT=Path(__file__).resolve().parents[2]
PREPARATION_FINGERPRINT=json.loads(subprocess.check_output(['node',str(ROOT/'portal/scripts/fingerprints.mjs')],cwd=ROOT,text=True))
OUT=Path(__file__).resolve().parents[1]/'generated'
ASSETS=OUT/'assets'; ASSETS.mkdir(parents=True,exist_ok=True)
STYLE=ROOT/'common/rlstudy.sty'
MACROS='\n'.join(x for x in STYLE.read_text().splitlines() if x.startswith('\\newcommand') or x.startswith('\\DeclareMathOperator'))
DOCS=[]; assets=set(); warnings=[]
def rel(p):return p.relative_to(ROOT).as_posix()
def asset(p):
 p=p.resolve();p.relative_to(ROOT)
 if not p.is_file():raise RuntimeError(f'Missing course asset: {p}')
 dest=ASSETS/rel(p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assets.add(rel(p));return rel(p)
def argument(s,pos):
 while pos<len(s) and s[pos].isspace():pos+=1
 if pos>=len(s) or s[pos]!='{':return None,pos
 depth=1;start=pos+1;pos+=1
 while pos<len(s) and depth:
  if s[pos]=='{' and (pos==0 or s[pos-1]!='\\'):depth+=1
  elif s[pos]=='}' and s[pos-1]!='\\':depth-=1
  pos+=1
 return s[start:pos-1],pos

def replace_command(s,name,fn):
 pattern=re.compile(r'\\'+name+r'(?![A-Za-z])(?:\*)?(?:\[[^\]]*\])?')
 pos=0;out=''
 while m:=pattern.search(s,pos):
  arg,end=argument(s,m.end())
  if arg is None:out+=s[pos:m.end()];pos=m.end();continue
  out+=s[pos:m.start()]+fn(arg);pos=end
 return out+s[pos:]

def algorithm(s):
 for command,count in [('ForEach',2),('ElseIf',2),('While',2),('Repeat',2),('For',2),('If',2)]:
  pat=re.compile(r'\\'+command+r'(?![A-Za-z])');pos=0;out=''
  while m:=pat.search(s,pos):
   cond,e=argument(s,m.end());body,end=argument(s,e)
   if cond is None or body is None:out+=s[pos:m.end()];pos=m.end();continue
   label={'ForEach':'For each','ElseIf':'Else if'}.get(command,command)
   if command=='Repeat':replacement=r'\par\textbf{Repeat:}\begin{quote}'+algorithm(body)+r'\end{quote}\textbf{Until }'+cond+r'\par '
   else:replacement=r'\par\textbf{'+label+r' }'+cond+r':\begin{quote}'+algorithm(body)+r'\end{quote}'
   out+=s[pos:m.start()]+replacement;pos=end
  s=out+s[pos:]
 for name,label in [('KwIn','Input'),('KwOut','Output'),('caption','Algorithm'),('tcp','Comment'),('tcc','Comment')]:
  s=replace_command(s,name,lambda a,label=label:r'\par\textbf{'+label+':} '+a+r'\par ')
 s=re.sub(r'\\(?:DontPrintSemicolon|SetAlgoLined|BlankLine)\b','',s)
 s=re.sub(r'\\Return\b',r'\\textbf{Return }',s)
 # Algorithm line terminators and math spacing share the same TeX command.
 # Preserve spacing inside maths; otherwise the conversion changes equations.
 return re.sub(r'\$[^$]*\$|\\;',lambda m:m.group() if m.group().startswith('$') else r'\par ',s)

def figure_svg(p):
 dest=(ASSETS/rel(p)).with_suffix('.svg');dest.parent.mkdir(parents=True,exist_ok=True)
 # Reconvert the exact current PDF. File modification times are not provenance:
 # restored or copied figures can change bytes while retaining an older timestamp.
 subprocess.run(['pdftocairo','-svg',str(p),str(dest)],check=True,capture_output=True)
 name=dest.relative_to(ASSETS).as_posix();assets.add(name);return name

def tikz_svg(tikz,p,n,preamble):
 digest=hashlib.sha256((tikz+MACROS+preamble).encode()).hexdigest()[:16]
 name=f'illustrations/{p.parent.name}-{p.stem}-{n}-{digest}.svg';dest=ASSETS/name
 if not dest.exists():
  dest.parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='rl-figure-') as temp:
   tex=Path(temp)/'figure.tex'
   tex.write_text(r'\documentclass[border=5pt]{standalone}'+'\n'+r'\usepackage{'+str(STYLE.with_suffix(''))+'}\n'+preamble+'\n'+r'\begin{document}'+'\n'+tikz+'\n'+r'\end{document}')
   run=subprocess.run(['pdflatex','-shell-escape','-interaction=batchmode','-halt-on-error','-output-directory',temp,str(tex)],cwd=p.parent,capture_output=True)
   if run.returncode:raise RuntimeError(f'TikZ compilation failed: {rel(p)} figure {n}; inspect isolated build, no source modified')
   subprocess.run(['pdftocairo','-svg',str(Path(temp)/'figure.pdf'),str(dest)],check=True,capture_output=True)
 assets.add(name);return name

def lecture(p):
 s=p.read_text(); preamble=s.split(r'\begin{document}')[0]
 extra='\n'.join(re.findall(r'^\\(?:newcommand|DeclareMathOperator|definecolor|tikzset|usetikzlibrary).*$',preamble,re.M))
 s=s.replace(r'\begin{document}',MACROS+'\n'+r'\begin{document}')
 i=0
 def tikz(m):
  nonlocal i;i+=1
  name=tikz_svg(m.group(),p,i,extra)
  return r'\includegraphics{'+ '/assets/'+name+'}'
 s=re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',tikz,s,flags=re.S)
 def image(m):
  name=m.group(1)
  if name.startswith('/assets/'):return m.group()
  src=(p.parent/name).resolve()
  if not src.suffix:src=src.with_suffix('.pdf')
  target=figure_svg(src) if src.suffix=='.pdf' else asset(src)
  return r'\includegraphics{/assets/'+target+'}'
 s=re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',image,s)
 s=re.sub(r'\\begin\{algorithm\}(?:\[[^\]]*\])?(.*?)\\end\{algorithm\}',lambda m:r'\begin{algorithm}'+algorithm(m.group(1))+r'\end{algorithm}',s,flags=re.S)
 s=replace_command(s,'marginnote',lambda a:r'\footnote{Aside: '+a+'}')
 s=re.sub(r'\\begin\{tcolorbox\}\[([^\n]+)\]',lambda m:r'\begin{tcolorbox}'+(m[1].split('title=',1)[1]+r'\par ' if 'title=' in m[1] else ''),s)
 # Preserve theorem/definition titles that Pandoc otherwise drops.
 s=re.sub(r'\\begin\{(theorem|definition|lemma|proposition|corollary|example|exercise)\}\[([^\]]+)\]',lambda m:r'\begin{'+m[1]+r'}\textbf{'+m[2]+r'}\par ',s)
 result=subprocess.run(['pandoc','-f','latex','-t','html5','--mathjax','--wrap=none','--no-highlight'],input=s,text=True,cwd=p.parent,capture_output=True)
 if result.returncode:raise RuntimeError(f'Pandoc failed: {rel(p)}: {result.stderr[:200]}')
 text=result.stdout
 # Native headings begin at h1 because chapters are top-level in the lecture.
 text=re.sub(r'<(/?)h([1-5])\b',lambda m:f'<{m[1]}h{int(m[2])+1}',text)
 for name,label in [('keyidea','Key idea'),('pitfall','Common pitfall'),('intuition','Intuition'),('definition','Definition'),('theorem','Theorem'),('lemma','Lemma'),('proposition','Proposition'),('corollary','Corollary'),('example','Worked example'),('exercise','Exercise'),('safetyangle','Safety angle'),('matching','Application'),('algorithm','Algorithm'),('remark','Interpretation'),('proof','Derivation'),('tcolorbox','Walkthrough')]:
  text=re.sub(r'<div([^>]*?)class="'+name+r'">',lambda m:f'<div{m[1]}class="callout {name}"><p class="callout-label">{label}</p>',text)
 text=re.sub(r'<p><strong>Study companion\.</strong>.*?</p>','',text,flags=re.S)
 DOCS.append(dict(path=rel(p),format='html',kind='lecture' if p.name=='notes.tex' else 'reading',body=text,title=f'{p.parent.name} {p.stem}',pdf=asset(p.with_suffix('.pdf'))))
 return rel(p)

TOP=['README.md','CONTRIBUTING.md','SECURITY.md','docs/ATTRIBUTION.md','llm-lab/README.md','llm-lab/NVIDIA.md','llm-lab/VALIDATION.md','study-lab/README.md','portal/README.md','interactive/README.md','learning_projects/README.md']
mds=[ROOT/x for x in TOP if (ROOT/x).is_file()]+sorted((ROOT/'docs').glob('*.md'))+sorted((ROOT/'docs/statistics').rglob('*.md'))+sorted(ROOT.glob('week-*/teaching_guide.md'))+sorted((ROOT/'research').glob('*.md'))+sorted((ROOT/'llm-lab/lessons').glob('*.md'))
for p in dict.fromkeys(mds):
 DOCS.append(dict(path=rel(p),format='markdown',kind='guide' if p.name=='teaching_guide.md' else 'resource',body=p.read_text(),title=p.read_text().splitlines()[0].lstrip('# ')))
for p in sorted(ROOT.glob('week-*/workbook.ipynb'))+[ROOT/'llm-lab/workbook.ipynb']:
 if not p.exists():continue
 nb=json.loads(p.read_text());cells=[]
 for i,c in enumerate(nb['cells']):
  source=''.join(c['source']);cells.append(dict(type=c['cell_type'],source=source,index=i+1))
 DOCS.append(dict(path=rel(p),format='notebook',kind='workbook',title=p.parent.name+' workbook',cells=cells,download=asset(p)))
# Only course code roots are eligible; never traverse credential/config trees.
code=[]
for week in sorted(ROOT.glob('week-*')):
 for directory in ['project','tests']:
  code+=list((week/directory).glob('**/*.py'))
for directory in ['common','learning_projects','llm-lab/llm_lab','llm-lab/tests']:
 code+=list((ROOT/directory).glob('**/*.py'))
code += [p for p in [ROOT/'scripts/new_week.py',ROOT/'scripts/execute_reference_notebooks.py'] if p.is_file()]
for p in sorted(set(code)):
 DOCS.append(dict(path=rel(p),format='code',kind='code',body=p.read_text(),title=rel(p),download=asset(p)))
for p in sorted(ROOT.glob('week-*/*.tex')):
 lecture(p)
for p in [ROOT/'syllabus/syllabus.pdf',ROOT/'requirements.txt',ROOT/'requirements-foundations.txt',ROOT/'requirements-extras.txt',ROOT/'portal/package-lock.json',ROOT/'LICENSE',ROOT/'llm-lab/requirements.txt',ROOT/'study-lab/ppo-signs.svg',ROOT/'week-11-contextual-bandits-ope/figs/fig_ope_bias.json',ROOT/'week-11-contextual-bandits-ope/figs/fig_bias_variance.json']:
 if p.exists():asset(p)
# Copy explicitly included lab runtime, excluding tests and repository metadata.
for filename in ['index.html','app.js','model.js','style.css']:
 asset(ROOT/'study-lab'/filename)
for filename in ['index.html','app.js','engine.js','style.css','package.json','package-lock.json']:
 asset(ROOT/'interactive'/filename)
for name in ['llm-lab/results/base_evaluation.json','llm-lab/results/sft_evaluation.json','llm-lab/results/sft_training.json','llm-lab/results/dpo_training.json']:
 if (ROOT/name).is_file():asset(ROOT/name)
for p in sorted((ROOT/'docs/archive').glob('*')):
 if p.suffix in {'.md','.tsv'}:asset(p)
# Small, self-contained per-week downloads. Only declared course materials enter ZIPs.
for week in sorted(ROOT.glob('week-*')):
 if not (week/'workbook.ipynb').is_file():continue
 files=[p for p in week.rglob('*') if p.is_file() and p.suffix in {'.py','.ipynb','.pdf','.tex','.md','.csv','.json','.sty','.bib'} and not any(x in p.parts for x in ['__pycache__','.ipynb_checkpoints']) and '.executed.' not in p.name]
 files += [p for p in (ROOT/'common').rglob('*') if p.is_file() and p.suffix in {'.py','.sty','.bib'} and '__pycache__' not in p.parts]
 files += sorted((ROOT/'docs/statistics').rglob('*.md'))
 files += [ROOT/name for name in ['requirements.txt','requirements-foundations.txt','requirements-extras.txt','conftest.py','LICENSE','docs/ATTRIBUTION.md']]
 # Week 15 deliberately reuses week 14's logger by file path.
 if week.name.startswith('week-15-'):files.append(ROOT/'week-14-offline-rl-1/project/env.py')
 # These two optional recipes delegate to the real pretrained-model lab.
 if week.name.startswith(('week-18-','week-19-')):
  for directory,extensions in [('llm-lab/llm_lab',{'.py'}),('llm-lab/data',{'.json','.jsonl'}),('llm-lab/lessons',{'.md'})]:
   files += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts]
  files += [ROOT/'llm-lab'/name for name in ['README.md','NVIDIA.md','requirements.txt','workbook.ipynb']]
 name=f'bundles/week-{week.name.split("-")[1]}.zip';dest=ASSETS/name;dest.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
  for p in sorted(set(files)):
   if p.exists():
    info=zipfile.ZipInfo(rel(p),date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
    bundle.writestr(info,p.read_bytes())
  info=zipfile.ZipInfo('GETTING_STARTED.txt',date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
  bundle.writestr(info,f'''STUDY-RL — {week.name}
Extract this whole ZIP. Keep the weekly directory next to common/.
Website and setup: https://study-rl.samuelzhang.co.uk/#/downloads?week={int(week.name.split("-")[1])}
Source: https://github.com/samuel-zhang01/STUDY-RL/tree/main/{week.name}
Python 3.12 recommended. Create and activate a virtual environment.
Weeks 1–4: python -m pip install -r requirements-foundations.txt
Later weeks: install swig==4.* first, then requirements.txt (see website for compiler requirements).
Register your kernel: python -m ipykernel install --user --name study-rl --display-name STUDY-RL
Start: python -m jupyter lab
Open {week.name}/workbook.ipynb and select the STUDY-RL kernel.
Some exercise cells call reference solutions. Write your own attempt before comparing.
Where setup defines REFERENCE_MODE, False routes scaffold checks to your implementation.
Complete the guarded scaffold, restart, then run all. Other demonstrations stay references.
Use the teaching guide's Notebook investigation to predict and test a changed input.
''')
 assets.add(name)
for stale in ASSETS.rglob('*'):
 if stale.is_file() and stale.relative_to(ASSETS).as_posix() not in assets:stale.unlink()
source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
content_fingerprint=json.loads(subprocess.check_output(['node',str(ROOT/'portal/scripts/fingerprints.mjs')],cwd=ROOT,text=True))
if content_fingerprint!=PREPARATION_FINGERPRINT and '--preview' not in sys.argv:raise RuntimeError('Canonical content changed during preparation. Finish source edits, then prepare and build again.')
source_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).strip())
(OUT/'documents.json').write_text(json.dumps({'documents':DOCS,'assets':sorted(assets),'source':{'repository':'https://github.com/samuel-zhang01/STUDY-RL','revision':source_revision,'workingTreeChanges':source_dirty,'contentFingerprint':content_fingerprint,'preparationStable':content_fingerprint==PREPARATION_FINGERPRINT}},ensure_ascii=False))
print(f'Prepared {len(DOCS)} documents, {len(assets)} assets, {sum(d["kind"]=="lecture" for d in DOCS)} full HTML lessons.')
