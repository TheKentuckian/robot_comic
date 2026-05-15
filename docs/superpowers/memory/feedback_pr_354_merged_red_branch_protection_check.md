**Branch protection may not be blocking red merges. Investigate before relying on CI for safety.**

PR #354 (motion_safety, merged 2026-05-15 ~16:48 UTC) merged with BOTH ruff CI runs failing (job ids `76215944808` + `76215896915` on `conclusion: failure`). Possible causes: admin override, branch protection misconfigured to not require ruff, "required" check naming mismatch.

Check `Settings → Branches → main → branch protection rules`:

- Is "Require status checks to pass before merging" on?
- Is "ruff" listed in the required checks?
- Is "Do not allow bypassing the above settings" on for admins?

If admin bypass is the actual cause, that's fine for emergencies but: don't bypass on the 4b–4f stack. Use admin override only as a last resort; otherwise you propagate red main to every downstream branch and waste cycles diagnosing what looks like a PR-level failure but is actually a main-level one.
