"""
cpg_generator.py — Generate a Code Property Graph (CPG) from Python source using Joern.

Uses the Joern CLI binary (joern-parse / joern-export) to generate a CPG
and export it as a consolidated JSON file in GraphSON format.

Prerequisites:
    - Java 11+ runtime at JAVA_HOME or on PATH
    - Joern CLI installed at ~/bin/joern/ (joern-cli.zip extracted)

Usage:
    python cpg_generator.py [--input PATH] [--output PATH] [--joern-zip PATH]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT: Path = Path(__file__).parent / "test_samples" / "backdoor_sim.py"
DEFAULT_OUTPUT: Path = Path(__file__).parent / "output_cpg.json"
JOERN_INSTALL_DIR: Path = Path.home() / "bin" / "joern"
JAVA_HOME: Path = Path.home() / ".local" / "jdk" / "jdk-21.0.5+11"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("cpg_generator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _joern_env() -> dict[str, str]:
    """Build an environment dict with JAVA_HOME and PATH for Joern.

    Returns:
        A copy of the current environment with JAVA_HOME and the JDK
        bin directory prepended to PATH.
    """
    env: dict[str, str] = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + ":" + env.get("PATH", "")
    return env


def _find_joern_binary() -> Path:
    """Locate the Joern CLI binary.

    Searches in the following order:
        1. System PATH (``which joern``)
        2. ``~/bin/joern/joern`` (default non-root install location)
        3. ``/opt/joern/joern`` (default root install location)

    Returns:
        Absolute path to the ``joern`` executable.

    Raises:
        FileNotFoundError: If Joern CLI is not found in any search location.
    """
    system_joern: Optional[str] = shutil.which("joern")
    if system_joern is not None:
        logger.info("Found Joern on PATH: %s", system_joern)
        return Path(system_joern)

    for candidate in (JOERN_INSTALL_DIR, Path("/opt/joern")):
        joern_exe: Path = candidate / "joern"
        if joern_exe.is_file():
            logger.info("Found Joern at: %s", joern_exe)
            return joern_exe.resolve()

    raise FileNotFoundError(
        "Joern CLI binary not found. Install it via:\n"
        "  curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh | bash\n"
        "Or download joern-cli.zip and extract to ~/bin/joern/"
    )


def _check_java() -> str:
    """Verify that a Java runtime is available.

    Returns:
        The resolved path to the ``java`` executable.

    Raises:
        FileNotFoundError: If ``java`` is not on PATH and JAVA_HOME is not set.
        RuntimeError: If the found Java binary is not executable.
    """
    java_cmd: Optional[str] = shutil.which("java")
    if java_cmd is None:
        java_home: Optional[str] = os.environ.get("JAVA_HOME")
        if java_home:
            candidate = Path(java_home) / "bin" / "java"
            if candidate.is_file():
                java_cmd = str(candidate)

    if java_cmd is None:
        raise FileNotFoundError(
            "Java runtime not found. Install default-jdk via:\n"
            "  sudo apt-get install default-jdk\n"
            "Or set JAVA_HOME to a valid JDK installation."
        )

    try:
        result = subprocess.run(
            [java_cmd, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line: str = result.stderr.splitlines()[0] if result.stderr else "unknown"
        logger.info("Java detected: %s (exit_code=%d)", version_line, result.returncode)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Java version check timed out: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Java binary not executable at '{java_cmd}': {exc}") from exc

    return java_cmd


def _install_joern_from_zip(zip_path: Path, install_dir: Path) -> Path:
    """Install Joern CLI from a downloaded zip archive.

    Extracts the zip, flattens the joern-cli/ subdirectory, and makes
    all scripts executable.

    Args:
        zip_path: Path to the joern-cli.zip archive.
        install_dir: Target installation directory.

    Returns:
        Path to the installed ``joern`` binary.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        RuntimeError: If extraction fails.
    """
    import zipfile

    if not zip_path.is_file():
        raise FileNotFoundError(f"Joern zip not found: {zip_path}")

    logger.info("Installing Joern from %s to %s", zip_path, install_dir)

    try:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(install_dir)

        # Flatten joern-cli/ subdirectory
        cli_dir = install_dir / "joern-cli"
        if cli_dir.is_dir():
            for item in cli_dir.iterdir():
                target = install_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            cli_dir.rmdir()

        joern_exe = install_dir / "joern"
        if not joern_exe.is_file():
            raise RuntimeError(
                f"Joern binary not found after extraction. "
                f"Contents: {list(install_dir.iterdir())}"
            )

        # Make ALL scripts executable (bin/, frontends/, root scripts)
        for subdir in ("bin", "frontends"):
            subdir_path = install_dir / subdir
            if subdir_path.is_dir():
                for script in subdir_path.rglob("*"):
                    if script.is_file():
                        script.chmod(0o755)
        for f in install_dir.iterdir():
            if f.is_file():
                f.chmod(0o755)

        logger.info("Joern installed successfully at: %s", joern_exe)
        return joern_exe

    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Corrupt zip file {zip_path}: {exc}") from exc
    except PermissionError as exc:
        raise RuntimeError(f"Permission denied extracting to {install_dir}: {exc}") from exc


def _run_joern_parse(
    joern_bin: Path, input_file: Path, project_dir: Path
) -> Path:
    """Run joern-parse to generate a CPG binary.

    Uses the standalone ``joern-parse`` script from Joern's bin/ directory.
    The script must be run from its own working directory because it uses
    relative ``../lib`` paths.

    Args:
        joern_bin: Path to the main Joern executable (used to locate bin/).
        input_file: Path to the source file to analyze.
        project_dir: Temporary project directory; CPG is written here.

    Returns:
        Path to the generated CPG binary file (``cpg.bin``).

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If Joern parsing fails.
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"Input source file not found: {input_file}")

    project_dir.mkdir(parents=True, exist_ok=True)
    cpg_bin: Path = project_dir / "cpg.bin"

    # joern-parse is a standalone script in joern's bin/ dir
    joern_parse_script: Path = joern_bin.parent / "bin" / "joern-parse"
    if not joern_parse_script.is_file():
        raise FileNotFoundError(f"joern-parse script not found: {joern_parse_script}")

    parse_cmd: list[str] = [
        str(joern_parse_script),
        str(input_file),
    ]

    logger.info("Running Joern parse: %s", " ".join(parse_cmd))
    try:
        result = subprocess.run(
            parse_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(joern_parse_script.parent),
            env=_joern_env(),
        )
        if result.returncode != 0:
            logger.error("Joern parse stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"Joern parse failed (exit code {result.returncode}):\n{result.stderr}"
            )
        logger.info("Joern parse completed successfully.")
        if result.stdout:
            logger.debug("Joern parse stdout:\n%s", result.stdout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Joern parse timed out after 300s: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to execute Joern parse: {exc}") from exc

    # joern-parse writes cpg.bin to a temp dir; find it
    if cpg_bin.is_file():
        return cpg_bin

    # Search for cpg.bin in temp directories created by joern-parse
    tmp_root = Path(tempfile.gettempdir())
    cpg_candidates = sorted(
        tmp_root.glob("*/cpg.bin"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cpg_candidates:
        latest_cpg = cpg_candidates[0]
        logger.info("Found CPG at: %s", latest_cpg)
        # Copy to our project dir
        shutil.copy2(str(latest_cpg), str(cpg_bin))
        return cpg_bin

    raise RuntimeError(
        f"CPG binary not found after parsing. Searched: {cpg_bin}, "
        f"and {tmp_root}/*/cpg.bin"
    )


def _run_joern_export(
    joern_bin: Path, cpg_bin: Path, output_file: Path
) -> Path:
    """Run joern-export to export the CPG as JSON.

    Uses the standalone ``joern-export`` script with GraphSON format.
    The export creates a directory of per-node-type JSON files; this function
    consolidates them into a single JSON file.

    Args:
        joern_bin: Path to the main Joern executable.
        cpg_bin: Path to the CPG binary file.
        output_file: Destination path for the consolidated JSON.

    Returns:
        Path to the consolidated JSON file.

    Raises:
        FileNotFoundError: If the CPG binary does not exist.
        RuntimeError: If Joern export fails.
    """
    if not cpg_bin.is_file():
        raise FileNotFoundError(f"CPG binary not found: {cpg_bin}")

    joern_export_script: Path = joern_bin.parent / "bin" / "joern-export"
    if not joern_export_script.is_file():
        raise FileNotFoundError(f"joern-export script not found: {joern_export_script}")

    # Export to a temp directory, then consolidate
    export_dir: Path = output_file.parent / f"_cpg_export_{os.getpid()}"
    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_cmd: list[str] = [
        str(joern_export_script),
        str(cpg_bin),
        "--out", str(export_dir),
        "--repr", "cpg",
        "--format", "dot",
    ]

    logger.info("Running Joern export: %s", " ".join(export_cmd))
    try:
        result = subprocess.run(
            export_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(joern_export_script.parent),
            env=_joern_env(),
        )
        if result.returncode != 0:
            logger.error("Joern export stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"Joern export failed (exit code {result.returncode}):\n{result.stderr}"
            )
        logger.info("Joern export completed.")
        if result.stdout:
            logger.debug("Joern export stdout:\n%s", result.stdout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Joern export timed out after 300s: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to execute Joern export: {exc}") from exc

    # Consolidate DOT export directory into a single JSON
    _consolidate_dot(export_dir, output_file)
    return output_file


def _consolidate_dot(export_dir: Path, output_file: Path) -> None:
    """Merge per-node-type DOT files into a single JSON.

    Each export subdirectory contains an ``export.dot`` file with the
    DOT graph format. This function parses all of them, extracts nodes
    and edges, and writes a consolidated JSON with
    ``{"nodes": [...], "edges": [...]}``.

    Args:
        export_dir: Root directory of the DOT export.
        output_file: Destination path for the consolidated JSON.
    """
    import re

    all_nodes: dict[int, dict] = {}
    all_edges: list[dict] = []

    # Parse DOT format
    node_pattern = re.compile(r'"(\d+)"\s*\[([^\]]+)\]')
    edge_pattern = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[([^\]]+)\]')
    attr_pattern = re.compile(r'(\w+)="([^"]*)"')

    export_files = list(export_dir.rglob("export.dot"))
    logger.info("Consolidating %d DOT export files...", len(export_files))

    for ef in export_files:
        try:
            content = ef.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Skipping %s: %s", ef, exc)
            continue

        for match in node_pattern.finditer(content):
            node_id_str: str = match.group(1)
            node_id: int = int(node_id_str)
            attrs: dict[str, str] = dict(attr_pattern.findall(match.group(2)))
            if node_id not in all_nodes:
                all_nodes[node_id] = {"id": node_id, **attrs}

        for match in edge_pattern.finditer(content):
            src_id: int = int(match.group(1))
            dst_id: int = int(match.group(2))
            attrs = dict(attr_pattern.findall(match.group(3)))
            all_edges.append({"src": src_id, "dst": dst_id, **attrs})

    consolidated: dict[str, list] = {
        "nodes": list(all_nodes.values()),
        "edges": all_edges,
    }

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2)

    logger.info(
        "Consolidated %d nodes, %d edges -> %s",
        len(all_nodes),
        len(all_edges),
        output_file,
    )

    shutil.rmtree(export_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_cpg(
    input_file: Path = DEFAULT_INPUT,
    output_file: Path = DEFAULT_OUTPUT,
    joern_zip: Optional[Path] = None,
) -> Path:
    """End-to-end CPG generation pipeline.

    1. Verify Java is available.
    2. Locate (or install) Joern CLI.
    3. Parse the source file into a CPG binary.
    4. Export the CPG as consolidated JSON.

    Args:
        input_file: Path to the source file to analyze.
        output_file: Path for the JSON CPG output.
        joern_zip: Optional path to joern-cli.zip for auto-install.

    Returns:
        Path to the generated JSON CPG file.

    Raises:
        FileNotFoundError: If Java or Joern is not available.
        RuntimeError: If CPG generation fails at any step.
    """
    logger.info("=" * 60)
    logger.info("CPG Generation Pipeline")
    logger.info("=" * 60)
    logger.info("Input:  %s", input_file)
    logger.info("Output: %s", output_file)

    # Step 1: Check Java
    logger.info("Step 1/4: Checking Java runtime...")
    java_path: str = _check_java()
    logger.info("Java OK: %s", java_path)

    # Step 2: Locate Joern
    logger.info("Step 2/4: Locating Joern CLI...")
    try:
        joern_bin: Path = _find_joern_binary()
    except FileNotFoundError:
        if joern_zip is not None and joern_zip.is_file():
            logger.info("Joern not found, installing from %s...", joern_zip)
            joern_bin = _install_joern_from_zip(joern_zip, JOERN_INSTALL_DIR)
        else:
            raise
    logger.info("Joern OK: %s", joern_bin)

    # Step 3: Parse
    with tempfile.TemporaryDirectory(prefix="joern_project_") as tmp_dir:
        project_dir = Path(tmp_dir)
        logger.info("Step 3/4: Parsing source with Joern...")
        cpg_bin: Path = _run_joern_parse(joern_bin, input_file, project_dir)

        # Step 4: Export
        logger.info("Step 4/4: Exporting CPG to JSON...")
        _run_joern_export(joern_bin, cpg_bin, output_file)

    # Verify output
    if not output_file.is_file():
        raise RuntimeError(f"Expected output file was not created: {output_file}")

    file_size: int = output_file.stat().st_size
    logger.info("CPG generated successfully: %s (%d bytes)", output_file, file_size)

    # Quick sanity check
    try:
        with open(output_file, "r", encoding="utf-8") as fh:
            cpg_data = json.load(fh)
        node_count: int = len(cpg_data.get("nodes", []))
        edge_count: int = len(cpg_data.get("edges", []))
        logger.info("JSON validation OK. Nodes: %d, Edges: %d", node_count, edge_count)
    except json.JSONDecodeError as exc:
        logger.warning("Output is not valid JSON: %s", exc)

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for cpg_generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a Code Property Graph from Python source using Joern."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input source file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--joern-zip",
        type=Path,
        default=None,
        help="Path to joern-cli.zip for auto-install",
    )
    args = parser.parse_args()

    try:
        result_path: Path = generate_cpg(
            input_file=args.input,
            output_file=args.output,
            joern_zip=args.joern_zip,
        )
        print(f"\nSUCCESS: CPG written to {result_path}")
    except FileNotFoundError as exc:
        logger.error("Dependency not found: %s", exc)
        print(f"\nERROR (FileNotFound): {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("CPG generation failed: %s", exc)
        print(f"\nERROR (Runtime): {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        print(f"\nERROR (Unexpected): {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
