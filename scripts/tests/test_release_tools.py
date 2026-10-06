"""Release safeguards protect shipped materials and reject private artifacts."""
import subprocess

from scripts import check_release, clean_generated


def test_clean_keeps_pdfs_tracked_logs_and_environment(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.log\n*.aux\n.venv/\n__pycache__/\n")
    for name in ("notes.pdf", "tracked.log", "generated.log", "notes.aux"):
        (tmp_path / name).write_text("keep or clean")
    subprocess.run(["git", "add", "notes.pdf"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-f", "tracked.log"], cwd=tmp_path, check=True)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "internal.log").write_text("keep")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_text("clean")
    clean_generated.clean(tmp_path)
    assert (tmp_path / "notes.pdf").exists()
    assert (tmp_path / "tracked.log").exists()
    assert (tmp_path / ".venv" / "internal.log").exists()
    assert not (tmp_path / "generated.log").exists()
    assert not (tmp_path / "notes.aux").exists()
    assert not (tmp_path / "__pycache__").exists()


def test_release_rejects_local_settings_and_model_weights(tmp_path):
    for name in ("settings.local.json", "adapter.safetensors", ".env"):
        path = tmp_path / name
        path.write_text("{}")
        assert check_release.check_path(path, tmp_path)


def test_links_must_exist_inside_release_tree(tmp_path):
    (tmp_path / "valid.md").write_text("# Valid")
    source = tmp_path / "README.md"
    assert not check_release.check_links("[ok](valid.md#valid)", source, tmp_path)
    assert check_release.check_links("[missing](gone.md)", source, tmp_path)
    assert check_release.check_links("[outside](../outside.md)", source, tmp_path)
