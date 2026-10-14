"""Additive result accounting for each original isolated pytest invocation."""
import json
import os
from pathlib import Path

collected = passed = skipped = xfailed = deselected = 0


def pytest_collection_finish(session):
    global collected
    collected = len(session.items)


def pytest_deselected(items):
    global deselected
    deselected += len(items)


def pytest_collectreport(report):
    global skipped
    if report.skipped: skipped += 1


def pytest_runtest_logreport(report):
    global passed, skipped, xfailed
    if report.skipped: skipped += 1
    if hasattr(report, 'wasxfail'): xfailed += 1
    if report.when == 'call' and report.passed: passed += 1


def pytest_sessionfinish(session, exitstatus):
    accepted = int(exitstatus) == 0 and collected > 0 and collected == passed and skipped == xfailed == deselected == 0
    if not accepted: session.exitstatus = 1
    record = {'collected': collected, 'passed': passed, 'skipped': skipped, 'xfailed': xfailed,
              'deselected': deselected, 'accepted': accepted}
    destination = Path(os.environ['STUDY_CI_OUTCOMES'])/(str(os.getpid())+'.json')
    with destination.open('x') as output:
        json.dump(record, output, sort_keys=True)
