"""Fail-closed tests for tools/publish_experiment.py.

The tool is not a package, so it is loaded from its path. Every test that
reaches the scanner sets or clears VYRA_PRIVATE_HANDLES itself. Probe strings
are built at runtime so this file does not contain a literal the scanner
would flag, and so no real personal handle is stored here.
"""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "publish_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("publish_experiment", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pe():
    return _load()


def _cat(*parts):
    return "".join(parts)


# The ten probes the gate must still catch. Split so this source does not
# itself match the patterns under test.
PROBES = [
    _cat("AKIA", "IOSFODNN7EXAMPLE"),
    _cat("ghp_", "abcdefghijklmnopqrstuvwxyz012345"),
    _cat("hf_", "abcdefghijklmnopqrstuvwxyz123456"),
    _cat("sk-", "abcdefghijklmnopqrstuvwx"),
    _cat("pass", "word = 'hunter2xyz'"),
    _cat("PASS", "WORD=hunter2xyz"),
    _cat("KAGGLE_", "USERNAME=ajuser"),
    _cat("kaggle_", "key: 1234567890abcdef"),
    _cat("API_", "TOKEN=9f8a7b6c5d4e3f2a1b0c"),
    _cat("AJ ", "t4", "21u"),
]


def _git(repo, *args, check=True):
    cp = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise AssertionError(f"git {args} failed: {cp.stderr or cp.stdout}")
    return cp


def _init_repo(tmp_path, branch="main"):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", branch)
    _git(repo, "config", "user.email", _cat("dev@", "example.test"))
    _git(repo, "config", "user.name", "dev")
    (repo / "docs").mkdir()
    (repo / "results").mkdir()
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("# Demo\n\n## Known limitations\n\nnone yet\n")
    (repo / "docs" / "EXPERIMENTS.md").write_text("# Experiments\n\n## Open questions\n\nnone\n")
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", "https://example.test/demo.git")
    return repo


def _results(tmp_path, raw):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "z_results.json").write_text(json.dumps(raw) + "\n")
    profile = {
        "step0": {
            "a": {"distance": 32, "token_acc": 0.1},
            "b": {"distance": 64, "token_acc": 0.2},
        },
        "step200": {
            "a": {"distance": 32, "token_acc": 0.3},
            "b": {"distance": 64, "token_acc": 0.4},
        },
    }
    (incoming / "profile.json").write_text(json.dumps(profile) + "\n")
    lines = []
    for name in ("z_results.json", "profile.json"):
        digest = hashlib.sha256((incoming / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (incoming / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return incoming


def _fingerprint(repo):
    found = {}
    for path in repo.rglob("*"):
        if ".git" in path.parts or ".pytest_cache" in path.parts or "__pycache__" in path.parts:
            continue
        if not path.is_file():
            continue
        found[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    status = _git(repo, "status", "--porcelain").stdout
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return found, status, head


def _run(pe, repo, incoming, monkeypatch, *extra):
    monkeypatch.chdir(repo)
    return pe.main([str(incoming), "--experiment", "Q", "--slug", "demo",
                    "--repo", str(repo), *extra])


def test_scan_text_catches_every_required_probe(pe, monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    for probe in PROBES:
        hits = pe.scan_text("probe", probe + "\n")
        assert hits, probe


def test_scan_text_ignores_legitimate_shapes(pe, monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    negatives = [
        '"authorized": True',
        "authorized",
        "```sh\nexport VYRA_LAB=./lab\n```",
        "export VYRA_LAB=./out",
        "kaggle.json\n",
        'bundle/vyra_X_step200_weights.pt',
        "FOO=changeme",
        "FOO=placeholder",
        "FOO=YOUR_TOKEN",
    ]
    for text in negatives:
        assert pe.scan_text("ok", text) == [], text


def test_narrowed_patterns_still_catch_realistic_bad_inputs(pe, monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    bad = [
        _cat("approved", "_by"),
        _cat("~/.kaggle/", "kaggle.json"),
        _cat("model_step", "200.pt"),
        _cat("/tmp/", "model.pt"),
    ]
    for text in bad:
        assert pe.scan_text("bad", text), text


def test_no_compiled_pattern_matches_empty_string(pe, monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    patterns = pe.build_scan_patterns()
    assert any(name == "personal_handle" for name, _, _ in patterns)
    for name, rx, _kind in patterns:
        assert rx.search("") is None, name
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", ",")
    patterns = pe.build_scan_patterns()
    assert all(name != "personal_handle" for name, _, _ in patterns)
    for name, rx, _kind in patterns:
        assert rx.search("") is None, name
    assert pe.personal_handle_regex(()) is None
    assert pe.personal_handle_regex(("", "")) is None
    # A handle that is only punctuation must stay a literal, not an empty match.
    star = pe.personal_handle_regex(("*",))
    compiled = pe.re.compile(star)
    assert compiled.search("") is None
    assert compiled.search("a * b")


def test_pattern_counts_follow_the_handle_check(pe, monkeypatch):
    monkeypatch.delenv("VYRA_PRIVATE_HANDLES", raising=False)
    assert pe.pattern_group_count() == 19
    assert pe.handle_check_active() is False
    skipped = pe.scan_report_line()
    assert skipped == "scan       : HANDLE CHECK SKIPPED (19 pattern groups, --allow-no-handle-check)"
    assert "clean" not in skipped
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    assert pe.pattern_group_count() == 20
    assert pe.handle_check_active() is True
    assert pe.scan_report_line() == "scan       : clean (20 pattern groups, 0 hits)"


def test_legitimate_repo_files_have_zero_hits(pe, monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "not-in-tree", "99"))
    hits = []
    # artifacts/ is the delivery bundle for this change. It quotes the probes on
    # purpose, so it is not part of the publication-source sweep.
    skip = {".git", "__pycache__", ".pytest_cache", "artifacts"}
    for path in ROOT.rglob("*"):
        if any(part in skip for part in path.parts) or not path.is_file():
            continue
        if path.suffix.lower() in pe.BINARY_EXT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(pe.scan_text(str(path.relative_to(ROOT)), text))
    assert hits == [], hits


def test_verify_manifest_failures(pe, tmp_path):
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(pe.PublishError, match="no SHA256SUMS"):
        pe.verify_manifest(missing)

    blank = tmp_path / "blank"
    blank.mkdir()
    (blank / "SHA256SUMS").write_text("\n# comment only\n")
    with pytest.raises(pe.PublishError, match="empty"):
        pe.verify_manifest(blank)

    gone = tmp_path / "gone"
    gone.mkdir()
    (gone / "SHA256SUMS").write_text("ab" * 32 + "  missing.dat\n")
    with pytest.raises(pe.PublishError, match="missing file"):
        pe.verify_manifest(gone)

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "note.txt").write_text("hello\n")
    (bad / "SHA256SUMS").write_text("0" * 64 + "  note.txt\n")
    with pytest.raises(pe.PublishError, match="checksum mismatch"):
        pe.verify_manifest(bad)


def test_unset_handles_refuse_before_any_write(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VYRA_PRIVATE_HANDLES", raising=False)
    repo = _init_repo(tmp_path)
    incoming = _results(tmp_path, {"comment": "ordinary"})
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch)
    err = capsys.readouterr().err
    after = _fingerprint(repo)
    assert rc == 2
    assert "VYRA_PRIVATE_HANDLES" in err
    assert "--allow-no-handle-check" in err
    assert "PUBLICATION ABORTED" in err
    assert before == after


def test_allow_no_handle_check_never_says_clean(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VYRA_PRIVATE_HANDLES", raising=False)
    repo = _init_repo(tmp_path, branch="publish-demo")
    incoming = _results(tmp_path, {"comment": "ordinary"})
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push", "--allow-no-handle-check")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc == 0, combined
    assert "clean" not in combined
    assert "scan       : HANDLE CHECK SKIPPED (19 pattern groups, --allow-no-handle-check)" in combined


def test_active_handle_check_reports_twenty_groups(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="publish-demo")
    incoming = _results(tmp_path, {"comment": "ordinary"})
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "scan       : clean (20 pattern groups, 0 hits)" in captured.out


def test_main_refuses_to_push_main(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="main")
    incoming = _results(tmp_path, {"comment": "ordinary"})
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch)
    err = capsys.readouterr().err
    after = _fingerprint(repo)
    assert rc == 2
    assert "refusing to push" in err
    assert "main" in err
    assert before == after


def test_no_push_still_commits_on_main(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="main")
    incoming = _results(tmp_path, {"comment": "ordinary"})
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    new = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert new != head
    assert "SKIPPED (--no-push)" in captured.out


def test_dry_run_restores_the_tree(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path)
    incoming = _results(tmp_path, {"comment": "ordinary"})
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch, "--dry-run")
    captured = capsys.readouterr()
    after = _fingerprint(repo)
    assert rc == 0, captured.out + captured.err
    assert before == after
    assert after[1] == ""


def test_planted_credential_refuses_and_restores(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="publish-demo")
    secret = _cat("pass", "word = 'hunter2xyz'")
    incoming = _results(tmp_path, {"comment": secret})
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    captured = capsys.readouterr()
    after = _fingerprint(repo)
    assert rc == 2, captured.out + captured.err
    assert "password_assignment" in captured.err
    assert "PUBLICATION ABORTED" in captured.err
    assert before == after


def test_planted_handle_refuses_and_restores(pe, tmp_path, monkeypatch, capsys):
    handle = _cat("zzqx", "fixture", "9")
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", handle)
    repo = _init_repo(tmp_path, branch="publish-demo")
    incoming = _results(tmp_path, {"comment": "owner " + handle})
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    captured = capsys.readouterr()
    after = _fingerprint(repo)
    assert rc == 2, captured.out + captured.err
    assert "personal_handle" in captured.err
    assert "PUBLICATION ABORTED" in captured.err
    assert before == after


def test_prestaged_secret_is_scanned_and_refused(pe, tmp_path, monkeypatch, capsys):
    """A file staged beside the publication outputs is part of the commit unless refused.

    The Y publication scanned four generated paths and never opened tools/.
    """
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="publish-demo")
    tool = repo / "tools"
    tool.mkdir()
    leaked = tool / "publish_experiment.py"
    leaked.write_text("token = " + _cat("AKIA", "IOSFODNN7EXAMPLE") + "\n")
    _git(repo, "add", "tools/publish_experiment.py")
    incoming = _results(tmp_path, {"comment": "ordinary"})
    before_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    err = capsys.readouterr().err
    assert rc == 2
    assert "aws_access_key" in err
    assert "tools/publish_experiment.py" in err
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before_head


def test_index_secret_is_not_hidden_by_a_clean_worktree(pe, tmp_path, monkeypatch, capsys):
    """The scanner must read the index. A clean worktree must not hide a staged secret."""
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="publish-demo")
    leak = repo / "leak.txt"
    leak.write_text("benign\n")
    _git(repo, "add", "leak.txt")
    _git(repo, "commit", "-m", "add leak placeholder")
    leak.write_text(_cat("AKIA", "IOSFODNN7EXAMPLE") + "\n")
    _git(repo, "add", "leak.txt")
    leak.write_text("benign\n")  # worktree no longer matches the index
    incoming = _results(tmp_path, {"comment": "ordinary"})
    before_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    err = capsys.readouterr().err
    assert rc == 2
    assert "aws_access_key" in err
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before_head


def test_commit_scanned_does_not_take_unlisted_index_paths(pe, tmp_path):
    repo = _init_repo(tmp_path, branch="publish-demo")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    pe.commit_scanned(repo, ["keep.txt"], "subject", "body")
    files = _git(repo, "ls-tree", "-r", "--name-only", "HEAD").stdout.split()
    assert "keep.txt" in files
    assert "drop.txt" not in files
    assert _git(repo, "show", "HEAD:keep.txt").stdout == "keep\n"


def test_unexpected_exception_restores_the_tree(pe, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", _cat("zzqx", "fixture", "9"))
    repo = _init_repo(tmp_path, branch="publish-demo")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "z_results.json").write_text("{}\n")
    # A list has no .values(); render_profile_svg raises AttributeError.
    (incoming / "profile.json").write_text(json.dumps({"step0": [1, 2, 3]}) + "\n")
    lines = []
    for name in ("z_results.json", "profile.json"):
        digest = hashlib.sha256((incoming / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (incoming / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    before = _fingerprint(repo)
    rc = _run(pe, repo, incoming, monkeypatch, "--no-push")
    err = capsys.readouterr().err
    after = _fingerprint(repo)
    assert rc == 2
    assert "PUBLICATION ABORTED" in err
    assert "AttributeError" in err
    assert before == after
