"""Store API credentials in the project's .env file without showing them on screen.

Run it yourself in a terminal (it asks for input, so it cannot be run for you):

    .venv\\Scripts\\python.exe src\\set_secret.py PREQIN_CLIENT_ID PREQIN_CLIENT_SECRET

Each value is typed at a hidden prompt; nothing is printed or logged. Existing entries with
the same name are replaced. .env is git-ignored, so it never reaches GitHub.
Code reads the values with load_env() below.
"""
import getpass
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def load_env():
    """{NAME: value} from .env (simple KEY=VALUE lines)."""
    if not ENV.exists():
        return {}
    out = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(names):
    if not names:
        sys.exit(__doc__)
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    for name in names:
        value = getpass.getpass(f"{name} (input hidden): ").strip()
        if not value:
            print(f"{name}: empty, skipped")
            continue
        lines = [l for l in lines if not l.split("=", 1)[0].strip() == name] + [f"{name}={value}"]
        print(f"{name}: saved ({len(value)} characters)")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
