from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.path.join(ROOT, '.baseline-runtime'))
NEO4J_VERSION = "5.26.30"
NEO4J_URL = f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-unix.tar.gz"
JAVA_URL = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jre_x64_linux_hotspot_21.0.12.1_1.tar.gz"


def service_environment():
    path = Path(os.path.join(RUNTIME, 'credentials.json'))
    if not path.is_file():
        raise RuntimeError("run scripts/baseline_neo4j.py install first")
    credentials = json.loads(path.read_text())
    return {"HAMGF_BASELINE_NEO4J_URI": "bolt://127.0.0.1:17687",
            "HAMGF_BASELINE_NEO4J_USER": "neo4j", "HAMGF_BASELINE_NEO4J_PASSWORD": credentials["password"]}


def install():
    RUNTIME.mkdir(exist_ok=True)
    artifacts = []
    for name, url, checksum_url in (("neo4j", NEO4J_URL, NEO4J_URL + ".sha256"),
                                   ("java", JAVA_URL, JAVA_URL + ".sha256.txt")):
        archive = Path(os.path.join(RUNTIME, 'downloads', f'{name}.tar.gz'))
        archive.parent.mkdir(exist_ok=True)
        expected = urllib.request.urlopen(checksum_url, timeout=60).read().decode().split()[0]
        actual = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() if archive.is_file() else None
        if actual != expected:
            subprocess.run(["curl", "-fLsS", "-C", "-", "--retry", "2", "--max-time", "1800", url, "-o", str(archive)], check=True)
            with archive.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} checksum mismatch; refusing to execute")
        with tarfile.open(archive) as tar:
            names = {member.name.split('/')[0] for member in tar.getmembers()}
            if len(names) != 1:
                raise RuntimeError("unexpected runtime archive layout")
            folder = Path(os.path.join(RUNTIME, next(iter(names))))
            if not folder.exists():
                tar.extractall(RUNTIME, filter="data")
        artifacts.append({"name": name, "url": url, "sha256": actual, "path": str(folder)})
    (Path(os.path.join(RUNTIME, 'artifacts.json'))).write_text(json.dumps(artifacts, indent=2))
    credentials = Path(os.path.join(RUNTIME, 'credentials.json'))
    if not credentials.exists():
        fd = os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"password": secrets.token_urlsafe(24)}, stream)
        env = runtime_environment()
        result = subprocess.run([str(Path(os.path.join(neo4j_path(), 'bin', 'neo4j-admin'))), "dbms", "set-initial-password",
                                 service_environment()["HAMGF_BASELINE_NEO4J_PASSWORD"]],
                                env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Neo4j initial password setup failed (output suppressed)")
    output = Path(os.path.join(ROOT, 'tests', 'sol', 'baseline-environment'))
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'runtime-artifacts.json'))).write_text(json.dumps(artifacts, indent=2))
    print("Verified and installed local baseline runtime; credentials kept private.")


def neo4j_path():
    return Path(os.path.join(RUNTIME, f'neo4j-community-{NEO4J_VERSION}'))


def runtime_environment():
    artifacts = json.loads((Path(os.path.join(RUNTIME, 'artifacts.json'))).read_text())
    java = next(item["path"] for item in artifacts if item["name"] == "java")
    return {**os.environ, "JAVA_HOME": java, "NEO4J_CONF": str(Path(os.path.join(ROOT, 'docker', 'baseline-local-conf')))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "start", "console", "stop", "status"))
    args = parser.parse_args()
    if args.action == "install":
        install()
    else:
        subprocess.run([str(Path(os.path.join(neo4j_path(), 'bin', 'neo4j'))), args.action], env=runtime_environment(), check=True)


if __name__ == "__main__":
    main()
