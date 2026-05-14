#!/usr/bin/env bash
# cleanup-worktrees.sh — Remove stale .claude/worktrees/agent-* git worktrees.
#
# USAGE
#   cleanup-worktrees.sh [OPTIONS]
#
# OPTIONS
#   --all       Remove ALL agent-* worktrees regardless of merge status.
#   --dry-run   Print what would be done without making any changes.
#   --help      Show this help message and exit.
#
# DEFAULT BEHAVIOUR (no flags)
#   Only removes worktrees whose branch is fully merged into origin/main.
#
# EXIT CODES
#   0  success (including "nothing to remove")
#   1  unexpected error

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    # Print the leading comment block (everything before the first blank
    # comment line that signals the end of the header).
    awk '
        /^#!/ { next }
        /^#/ { print substr($0, 3); next }
        { exit }
    ' "$0"
    exit 0
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    echo "$*"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODE="merged"   # merged | all
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --all)      MODE="all"    ;;
        --dry-run)  DRY_RUN=true  ;;
        --help|-h)  usage         ;;
        *)          die "Unknown option: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate repo root (works when called from any directory or as a worktree)
# ---------------------------------------------------------------------------

REPO_ROOT="$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel 2>/dev/null \
             || git rev-parse --show-toplevel)"

# ---------------------------------------------------------------------------
# Fetch latest remote state (skip in dry-run to avoid network requirement)
# ---------------------------------------------------------------------------

if [[ "$MODE" == "merged" ]]; then
    log "Fetching origin …"
    if [[ "$DRY_RUN" == false ]]; then
        git -C "$REPO_ROOT" fetch origin --quiet
    else
        log "(dry-run: skipping fetch)"
    fi
fi

# ---------------------------------------------------------------------------
# Collect agent worktrees from porcelain listing
# ---------------------------------------------------------------------------

# git worktree list --porcelain emits blocks like:
#   worktree /abs/path
#   HEAD <sha>
#   branch refs/heads/<name>   (or "detached")
#   locked [reason]            (optional)
#   prunable gitdir file ...   (optional)
#   <blank line>

declare -a wt_paths=()
declare -a wt_branches=()

current_path=""
current_branch=""

while IFS= read -r line; do
    if [[ "$line" == worktree\ * ]]; then
        current_path="${line#worktree }"
        current_branch=""
    elif [[ "$line" == branch\ * ]]; then
        current_branch="${line#branch refs/heads/}"
    elif [[ -z "$line" ]]; then
        # End of a worktree block — check if it matches agent-* pattern
        if [[ "$current_path" == */.claude/worktrees/agent-* ]]; then
            wt_paths+=("$current_path")
            wt_branches+=("$current_branch")
        fi
        current_path=""
        current_branch=""
    fi
done < <(git -C "$REPO_ROOT" worktree list --porcelain; echo "")

total_found=${#wt_paths[@]}

if [[ $total_found -eq 0 ]]; then
    log "No .claude/worktrees/agent-* worktrees found. Nothing to do."
    exit 0
fi

log "Found $total_found agent worktree(s)."

# ---------------------------------------------------------------------------
# Detect the worktree we are currently running inside (path-format-agnostic)
# On Windows, git uses Drive:/... while $PWD uses /d/... — normalise both to
# a POSIX-style path so the self-protection guard works on all platforms.
# ---------------------------------------------------------------------------

# Converts a path to POSIX format:
#   - On Windows Git Bash / MSYS: delegates to cygpath if available.
#   - On Linux / macOS: returns the path as-is (already POSIX).
_to_posix_path() {
    local p="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$p" 2>/dev/null || echo "$p"
    else
        echo "$p"
    fi
}

CURRENT_WT_PATH="$(_to_posix_path "$PWD")"

# ---------------------------------------------------------------------------
# Process each candidate
# ---------------------------------------------------------------------------

removed_count=0
skipped_count=0
total_freed_bytes=0

for i in "${!wt_paths[@]}"; do
    path="${wt_paths[$i]}"
    branch="${wt_branches[$i]}"

    # Safety: never remove the worktree we are currently running inside.
    normalised_candidate="$(_to_posix_path "$path")"
    if [[ "$normalised_candidate" == "$CURRENT_WT_PATH" ]] \
       || [[ "$CURRENT_WT_PATH" == "$normalised_candidate"/* ]]; then
        log "  SKIP (current): $path"
        ((skipped_count++)) || true
        continue
    fi

    # Determine whether to remove based on mode
    should_remove=false
    reason=""

    if [[ "$MODE" == "all" ]]; then
        should_remove=true
        reason="--all flag"
    elif [[ "$MODE" == "merged" ]]; then
        # Check merge status; empty branch means detached HEAD — treat as merged.
        if [[ -z "$branch" ]]; then
            should_remove=true
            reason="detached HEAD (treating as merged)"
        else
            if git -C "$REPO_ROOT" branch -a --merged origin/main 2>/dev/null \
               | grep -qE "(^|[[:space:]])${branch}$"; then
                should_remove=true
                reason="branch '$branch' merged into origin/main"
            else
                log "  SKIP (unmerged): $path  [branch: $branch]"
                ((skipped_count++)) || true
                continue
            fi
        fi
    fi

    if [[ "$should_remove" == true ]]; then
        # Measure disk usage before removal
        dir_bytes=0
        if [[ -d "$path" ]]; then
            dir_bytes=$(du -sb "$path" 2>/dev/null | awk '{print $1}' || echo 0)
        fi

        log "  REMOVE ($reason): $path"

        if [[ "$DRY_RUN" == false ]]; then
            # Unlock (Claude harness locks worktrees to prevent accidental removal)
            git -C "$REPO_ROOT" worktree unlock "$path" 2>/dev/null || true

            # Remove the worktree directory and its .git bookkeeping
            git -C "$REPO_ROOT" worktree remove --force "$path" 2>/dev/null \
                || { log "    WARNING: git worktree remove failed for $path; attempting manual rm"; rm -rf "$path"; }

            # Delete the branch (ignore errors — might already be deleted or non-existent)
            if [[ -n "$branch" ]]; then
                git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
            fi

            # Also delete the auto-generated worktree-agent-<id> branch if present
            dir_name="$(basename "$path")"
            auto_branch="worktree-${dir_name}"
            if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$auto_branch" 2>/dev/null; then
                git -C "$REPO_ROOT" branch -D "$auto_branch" 2>/dev/null || true
            fi
        fi

        total_freed_bytes=$((total_freed_bytes + dir_bytes))
        ((removed_count++)) || true
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# Convert bytes to human-readable
freed_human="0 B"
if command -v du >/dev/null 2>&1 && [[ $total_freed_bytes -gt 0 ]]; then
    if   [[ $total_freed_bytes -ge $((1024 * 1024 * 1024)) ]]; then
        freed_human="$(awk "BEGIN {printf \"%.2f GB\", $total_freed_bytes / 1073741824}")"
    elif [[ $total_freed_bytes -ge $((1024 * 1024)) ]]; then
        freed_human="$(awk "BEGIN {printf \"%.1f MB\", $total_freed_bytes / 1048576}")"
    elif [[ $total_freed_bytes -ge 1024 ]]; then
        freed_human="$(awk "BEGIN {printf \"%.1f KB\", $total_freed_bytes / 1024}")"
    else
        freed_human="${total_freed_bytes} B"
    fi
fi

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN — no changes made ==="
fi
echo "Worktrees removed : $removed_count"
echo "Worktrees skipped : $skipped_count"
echo "Approx disk freed : $freed_human"
