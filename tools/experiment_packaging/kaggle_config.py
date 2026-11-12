"""Private Kaggle identifiers, kept out of the repository.

The Kaggle username, the dataset and kernel slugs and the Kaggle filesystem
root are NOT stored in tracked files. They live in ``kaggle_config.json`` at the
repository root (gitignored; ``kaggle_config.example.json`` is the committed
template with placeholders), or in the file named by ``VYRA_KAGGLE_CONFIG``.

Tracked source refers to them with tokens such as ``@@KUSER@@``,
``@@KROOT@@`` and ``@@SLUG_X_BUNDLE@@``. Notebook asset modules keep the tokens
in their source; the builders expand every asset, the notebook program and the
dependency lock with ``expand`` *before* hashing, so the built notebook carries
the real strings and its SHA-256 is unchanged. Builder-side modules expand
their own module constants with ``expand_module(globals())``.

Fail closed: a missing or malformed config, an unknown token, or a token left
over after expansion raises ``KaggleConfigError``. The placeholder template is
refused by the builders unless ``VYRA_KAGGLE_ALLOW_EXAMPLE=1`` (the tests set it).
"""
import json
import os
import re
from pathlib import Path

ENV_PATH = "VYRA_KAGGLE_CONFIG"
ENV_ALLOW_EXAMPLE = "VYRA_KAGGLE_ALLOW_EXAMPLE"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "kaggle_config.json"
EXAMPLE_PATH = REPO_ROOT / "kaggle_config.example.json"
TOKEN_RE = re.compile(r"@@[A-Z0-9_]+@@")
REQUIRED_SLUGS = ("x_bundle", "y_bundle", "ckpt200", "ckpt225", "kernel_replay175", "kernel_replay175_approved",
                  "kernel_resume300", "kernel_y72fix", "kernel_y72fix2")
REQUIRED_KEYS = ("username", "root", "local_inputs", "slugs")
# Slugs added after the first scrub. Optional so that older private configs keep working for the older builders; a token
# of an absent optional slug is simply undefined, so any text that uses it fails closed ("unknown kaggle config token").
OPTIONAL_SLUGS = ("ckpt300", "kernel_y72fix2_cont400")


class KaggleConfigError(RuntimeError):
    pass


_CACHE = {}


def config_path():
    raw = os.environ.get(ENV_PATH)
    return Path(raw) if raw else DEFAULT_PATH


def _tokens(cfg):
    tokens = {"@@KUSER@@": cfg["username"], "@@KROOT@@": cfg["root"], "@@LOCAL_INPUTS@@": cfg["local_inputs"]}
    for key in REQUIRED_SLUGS:
        tokens["@@SLUG_%s@@" % key.upper()] = cfg["slugs"][key]
    for key in OPTIONAL_SLUGS:
        if key in cfg["slugs"]:
            tokens["@@SLUG_%s@@" % key.upper()] = cfg["slugs"][key]
    return tokens


def load():
    """Return (tokens, is_example) from the config file; fail closed if it is missing or malformed."""
    path = config_path()
    if path in _CACHE:
        return _CACHE[path]
    if not path.is_file():
        raise KaggleConfigError(
            "kaggle config missing: %s. Copy kaggle_config.example.json to kaggle_config.json (gitignored) and fill in the "
            "real Kaggle username, slugs and paths, or point %s at it. Refusing to build." % (path.name, ENV_PATH))
    try:
        cfg = json.loads(path.read_text())
    except ValueError as exc:
        raise KaggleConfigError("kaggle config is not valid JSON: %s (%s)" % (path.name, exc)) from None
    missing = [k for k in REQUIRED_KEYS if not isinstance(cfg.get(k), (str, dict)) or not cfg.get(k)]
    if not missing:
        missing = ["slugs." + k for k in REQUIRED_SLUGS if not isinstance(cfg["slugs"].get(k), str) or not cfg["slugs"][k]]
        missing += ["slugs." + k for k in OPTIONAL_SLUGS if k in cfg["slugs"] and (not isinstance(cfg["slugs"][k], str) or not cfg["slugs"][k])]
    if missing:
        raise KaggleConfigError("kaggle config %s is missing or has empty keys: %s" % (path.name, ", ".join(missing)))
    for value in _tokens(cfg).values():
        if TOKEN_RE.search(value):
            raise KaggleConfigError("kaggle config values must not contain @@TOKEN@@ text")
    result = (_tokens(cfg), bool(cfg.get("example")))
    _CACHE[path] = result
    return result


def is_example():
    return load()[1]


def require_real():
    """Builders call this: the placeholder template must not produce a package."""
    if is_example() and os.environ.get(ENV_ALLOW_EXAMPLE) != "1":
        raise KaggleConfigError("kaggle config is the placeholder template; refusing to build a package with it")


def expand(text):
    """Replace every token in ``text``; raise on an unknown token. Idempotent on already-expanded text."""
    if "@@" not in text:
        return text
    tokens, _ = load()
    unknown = sorted(set(TOKEN_RE.findall(text)) - set(tokens))
    if unknown:
        raise KaggleConfigError("unknown kaggle config token(s): " + ", ".join(unknown))
    return TOKEN_RE.sub(lambda m: tokens[m.group(0)], text)


def expand_value(value):
    if isinstance(value, str):
        return expand(value)
    if isinstance(value, tuple):
        return tuple(expand_value(v) for v in value)
    if isinstance(value, list):
        return [expand_value(v) for v in value]
    if isinstance(value, dict):
        return {expand_value(k): expand_value(v) for k, v in value.items()}
    return value


def expand_module(namespace):
    """Expand token-bearing string constants (also inside tuples, lists and dicts) of a builder-side module, in place."""
    load()
    for name, value in list(namespace.items()):
        if name.startswith("__"):
            continue
        if isinstance(value, (str, tuple, list, dict)):
            namespace[name] = expand_value(value)


def read_asset(path):
    """Read a tracked asset module/file and expand its tokens (what the notebook carries and the lock hashes)."""
    return expand(Path(path).read_text())


def assert_clean(text, what="text"):
    left = sorted(set(TOKEN_RE.findall(text)))
    if left:
        raise KaggleConfigError("unexpanded kaggle config token(s) in %s: %s" % (what, ", ".join(left)))


def token(name):
    """Real value of one token by short name, e.g. token('KUSER') or token('SLUG_X_BUNDLE')."""
    return load()[0]["@@%s@@" % name]


def dataset_id(slug_key):
    return token("KUSER") + "/" + token("SLUG_" + slug_key.upper())


def mount(slug_key):
    return token("KROOT") + "/input/datasets/" + dataset_id(slug_key)
