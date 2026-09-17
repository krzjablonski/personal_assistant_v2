"""Execute an approved argv in the fixed image; no application or credentials."""
import os
import sys

# Do not accept an inherited or workspace-controlled executable search path.
os.environ.update(PATH="/usr/local/bin:/usr/bin:/bin", HOME="/tmp", TZ="UTC")
try:
    os.chdir(sys.argv[1])
except OSError as error:
    print(f"Console cwd is unavailable: {sys.argv[1]} ({error.strerror})", file=sys.stderr)
    raise SystemExit(126)
try:
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
except FileNotFoundError:
    print("Program unavailable in the console image. Add dependencies through explicit image setup.", file=sys.stderr)
    raise SystemExit(127)
except (PermissionError, OSError) as error:
    print(f"Program could not start: {error}", file=sys.stderr)
    raise SystemExit(126)
