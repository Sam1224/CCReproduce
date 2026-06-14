import subprocess

REPO = "https://github.com/xuxinzhang/oneretrieval"


def main() -> None:
    print(f"Verifying remote repo: {REPO}")
    out = subprocess.check_output(["git", "ls-remote", REPO], text=True)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("Empty repo or no refs returned")
    print(f"OK: {len(lines)} refs")
    print("Sample:")
    for ln in lines[:10]:
        print(ln)


if __name__ == "__main__":
    main()
