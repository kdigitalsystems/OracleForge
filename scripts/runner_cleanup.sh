#!/usr/bin/env bash
# Free what an OracleForge job left behind on a SHARED self-hosted runner.
#
# Run as the last step of a job (if: always()). It only removes OracleForge's
# own leftovers; the runner is shared with other projects, so it never touches
# their workspaces, the tool cache (_work/_tool), docker, or the Ollama models.
#
#   1. this job's workspace: untracked and ignored files (tracked files and
#      uncommitted edits to them are left alone), and __pycache__ directories
#   2. temp files our steps write: the Ollama start log and model list, and
#      any throwaway canary sandbox a killed job left behind
#   3. the pip cache the runner user shares across projects: only entries not
#      read for CLEANUP_PIP_DAYS (30) days, never the whole cache
#
# Paths can be overridden (CLEANUP_TMP, CLEANUP_PIP_CACHE) so the behavior is
# testable on decoy data; the defaults are what the workflows use.
set -u

WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"
TMPD="${CLEANUP_TMP:-/tmp}"
PIP_CACHE="${CLEANUP_PIP_CACHE:-$HOME/.cache/pip}"
PIP_DAYS="${CLEANUP_PIP_DAYS:-30}"

# Never run against anything that is not clearly a job workspace.
real=$(cd "$WORKSPACE" 2>/dev/null && pwd -P) || { echo "cleanup: workspace '$WORKSPACE' not found; nothing to do"; exit 0; }
case "$real" in
  /|/home|/root|/tmp|"$HOME"|"$(cd "$HOME" 2>/dev/null && pwd -P)")
    echo "cleanup: refusing to clean '$real' (not a job workspace)"; exit 0 ;;
esac

free_kb() { df -Pk "$real" | awk 'NR==2{print $4}'; }
before=$(free_kb)

# 1. workspace
if git -C "$real" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$real" clean -ffdxq || true
fi
find "$real" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 2. temp files this repo's steps create
rm -f "$TMPD/ollama.log" "$TMPD/ollama_tags.json" "$TMPD/ollama_canary.log"
rm -rf "$TMPD"/oracleforge-canary-* 2>/dev/null || true

# 3. shared pip cache: old entries only
if [ -d "$PIP_CACHE" ]; then
  find "$PIP_CACHE" -type f -atime +"$PIP_DAYS" -delete 2>/dev/null || true
  find "$PIP_CACHE" -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

after=$(free_kb)
echo "cleanup: $(( after / 1024 / 1024 )) GiB free on $(df -Pk "$real" | awk 'NR==2{print $6}'); this step freed $(( (after - before) / 1024 )) MiB"
exit 0
