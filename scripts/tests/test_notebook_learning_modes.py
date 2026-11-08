"""Exercise the real notebook checks with intentionally wrong learner functions."""
import ast
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]


def notebook(week):
    return nbformat.read(next(ROOT.glob(f'week-{week:02d}-*/workbook.ipynb')),4)


def learner_namespace(week, reference_name):
    nb=notebook(week)
    namespace={'np':np,'torch':torch,'REFERENCE_MODE':False}
    def forbidden_reference(*args,**kwargs):
        raise AssertionError('A reference was executed in learner mode')
    namespace['solutions']=SimpleNamespace(**{reference_name:forbidden_reference})
    namespace['S']=namespace['solutions']
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        for node in ast.parse(cell.source).body:
            if isinstance(node,ast.FunctionDef) and node.name=='select_exercise':
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook mode>','exec'),namespace)
    return nb,namespace


def test_dqn_actual_check_rejects_wrong_and_accepts_correct_learner():
    nb,ns=learner_namespace(5,'exercise_1_gather_q')
    ns['gather_q']=lambda q,a:torch.zeros(len(a))
    with pytest.raises(AssertionError):
        exec(nb.cells[6].source,ns)
    ns['gather_q']=lambda q,a:q.gather(1,a[:,None]).squeeze(1)
    exec(nb.cells[6].source,ns)


def test_rlhf_actual_check_rejects_zero_loss_and_accepts_logistic_loss():
    nb,ns=learner_namespace(18,'exercise_1_bt_loss')
    for constant in [0.,1.]:
        ns['bt_loss']=lambda a,b:torch.tensor(constant)
        with pytest.raises(AssertionError):
            exec(nb.cells[7].source,ns)
    ns['bt_loss']=lambda a,b:torch.nn.functional.softplus(-(a-b)).mean()
    exec(nb.cells[7].source,ns)


def test_categorical_check_rejects_normalized_but_wrong_projection():
    nb,ns=learner_namespace(6,'exercise_6_categorical_project')
    ns['c51_project']=lambda *args:torch.full((1,3),1/3)
    with pytest.raises(AssertionError):
        exec(nb.cells[22].source,ns)
    ns['c51_project']=lambda *args:torch.tensor([[.16,.31,.53]])
    exec(nb.cells[22].source,ns)


def test_reference_wrapper_stops_without_learner_implementation():
    nb=notebook(1)
    ns={'REFERENCE_MODE':False}
    exec(nb.cells[5].source,ns)
    with pytest.raises(NotImplementedError,match='learner checks'):
        ns['exercise_1_incremental_mean']([1,2,3])


def test_existing_scaffolds_have_explicit_mode_and_dispatch_or_guard():
    for week in list(range(1,8))+list(range(14,25)):
        nb=notebook(week)
        assert nb.metadata['exercise_modes']['learner_fallback'] is False
        for cell in nb.cells:
            if cell.metadata.get('learner_scaffold'):
                name=cell.metadata['learner_scaffold']
                guarded='if not REFERENCE_MODE:' in cell.source
                dispatched=any(any(d['learner']==name for d in c.metadata.get('learner_checks',[]))
                               for c in nb.cells)
                assert guarded or dispatched,(week,name)
