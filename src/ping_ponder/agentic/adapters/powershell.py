"""Bounded, argv-based PowerShell JSON transport for native adapters."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterError


class PowerShellTimeout(AdapterError):
    """The native helper exceeded its bounded execution time."""


class PowerShellProcessError(AdapterError):
    """The native helper exited unsuccessfully."""


class PowerShellOutputError(AdapterError):
    """The helper did not return a JSON object."""


class PowerShellRunner:
    """Run a packaged PowerShell helper without shell interpolation."""

    def __init__(self, *, executable: str = "powershell.exe",
                 prelude: Sequence[str] = ("-NoProfile", "-NonInteractive",
                                            "-ExecutionPolicy", "Bypass"),
                 script_flag: str | None = "-File",
                 timeout: float = 10.0) -> None:
        self.executable = executable
        self.prelude = tuple(prelude)
        self.script_flag = script_flag
        self.timeout = timeout

    async def run(self, script: str | Path, operation: str, *arguments: str) -> Mapping[str, Any]:
        script_path = str(script)
        # WSL callers launch Windows PowerShell with a Windows-visible path. Keep
        # this conversion local to the transport; capability code remains portable.
        if os.name != "nt" and self.executable.casefold().endswith(("powershell.exe", "pwsh.exe")) \
                and script_path.startswith("/"):
            converter = shutil.which("wslpath")
            if converter:
                converted = subprocess.run([converter, "-w", script_path], check=True,
                                           capture_output=True, text=True).stdout.strip()
                if converted:
                    script_path = converted
        command = [self.executable, *self.prelude]
        if self.script_flag:
            command.append(self.script_flag)
        command.extend((script_path, operation, *arguments))
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError as error:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.communicate()
            raise PowerShellTimeout(f"PowerShell operation '{operation}' timed out") from error
        except OSError as error:
            raise PowerShellProcessError(f"could not start PowerShell helper: {error}") from error

        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise PowerShellProcessError(
                f"PowerShell operation '{operation}' exited {process.returncode}"
                + (f": {detail}" if detail else ""))
        try:
            value = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PowerShellOutputError(
                f"PowerShell operation '{operation}' returned malformed JSON") from error
        if not isinstance(value, Mapping):
            raise PowerShellOutputError(
                f"PowerShell operation '{operation}' returned {type(value).__name__}, expected object")
        return dict(value)
