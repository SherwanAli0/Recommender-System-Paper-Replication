# Branching and Pull Request workflow

This document describes how the four authors of this repository collaborate on changes. The model is **branch + Pull Request**: every change is made on a topic branch and merged into `main` only after at least one teammate has reviewed.

## Branch naming

```
<initials>-<type>/<short-description>
```

| Initials | Owner |
|---|---|
| `sa` | Sherwan Ali |
| `iu` | Ipek Ucar |
| `ysa` | Yaprak Sevinç Aldoğan |
| `yg` | Yasmine Guidoum |

| Type | Use for |
|---|---|
| `feature/` | new functionality |
| `fix/` | bug fix |
| `docs/` | documentation only |
| `test/` | test additions or changes |
| `experiment/` | new replication hypothesis in `past_tests/` |
| `refactor/` | non-functional code cleanup |
| `ci/` | GitHub Actions, build, tooling |

Examples:

```
sa-fix/p1-eval-path-bug
iu-experiment/per-user-fold-split
ysa-docs/p2-readme-update
yg-feature/gim-positive-only-flag
```

## Daily workflow

### One-time setup

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
git config user.name "Your Full Name"
git config user.email "you@example.com"
```

### Every change

```bash
# 1. Sync with main
git checkout main
git pull origin main

# 2. Create a branch
git checkout -b sa-feature/improve-cf-runtime

# 3. Edit files. Save.

# 4. Commit
git add <files-you-changed>
git commit -m "Speed up CF: vectorise Pearson correlation"

# 5. Push
git push -u origin sa-feature/improve-cf-runtime

# 6. Open the PR (one of these)
gh pr create
# or visit the URL printed by the push command, click "Compare & pull request"

# 7. After at least one approval, merge via the GitHub UI

# 8. Clean up
git checkout main
git pull origin main
git branch -d sa-feature/improve-cf-runtime
```

## Pull Request etiquette

1. **One PR = one logical change.** Don't bundle unrelated edits.
2. **Fill in the PR template.** It auto-loads when you open a PR.
3. **Run tests before opening the PR.** `python -m pytest tests/` must pass.
4. **No `__pycache__/` or local-path leaks.** `git status` should not show any after your commit.
5. **Reference an issue if one exists.** Type `Closes #12` in the PR body to auto-close it on merge.
6. **Request review from the right person.** CODEOWNERS auto-assigns based on which folder you touched.

## Reviewer responsibilities

When you're asked to review:

1. Open the PR's **Files changed** tab.
2. Read every changed file. Click any line to comment inline.
3. Verify:
   - PR template is filled in
   - Source citations are present in any new algorithmic code
   - Tests pass (CI badge is green)
   - Faithfulness check is intact: no novelty leaked into production tree
4. Click **Review changes** (top right):
   - **Approve** if you'd be happy with this in `main`
   - **Request changes** if something is wrong
   - **Comment** for non-blocking feedback

## Merging

- **Merge method:** Squash and merge. Keeps `main`'s history clean.
- **Required approvals:** at least 1 (configure under Settings → Branches → Branch protection).
- **Required status checks:** the `tests` workflow must be green.
- **Delete branch after merge:** GitHub does this automatically if you check the option.

## Handling merge conflicts

When `main` has changed since you branched:

```bash
git checkout main
git pull origin main
git checkout sa-feature/improve-cf-runtime
git merge main
```

If there are conflicts, git tells you which files. Open them, look for the markers:

```python
<<<<<<< HEAD
# your version
=======
# main's version
>>>>>>> main
```

Decide what to keep, delete the markers, then:

```bash
git add <conflicted-files>
git commit -m "Resolve merge conflict from main"
git push
```

The PR auto-updates.

## Useful git commands

```bash
git status                  # what changed?
git diff                    # show unstaged changes
git diff --staged           # show staged changes
git log --oneline -10       # last 10 commits
git branch                  # list local branches
git branch -a               # local + remote
git stash                   # temporarily shelve changes
git stash pop               # bring shelved changes back
git fetch --prune           # update remote refs, drop deleted branches
git checkout -              # switch to previous branch
```

## Useful gh CLI commands

```bash
gh repo create <name> --private
gh pr create
gh pr list
gh pr checkout 42           # check out PR #42 locally
gh pr view 42 --web         # open PR #42 in browser
gh pr review 42 --approve
gh issue create
gh issue list --label bug
```

## Branch protection (recommended setup)

Once the team is comfortable with the workflow, the repo owner should enable:

1. **Settings → Branches → Add rule** for `main`:
   - Require a pull request before merging
   - Require approvals: 1
   - Require status checks to pass before merging: select `tests`
   - Require branches to be up to date before merging
   - Do not allow bypassing the above settings

This prevents anyone from pushing directly to `main` and breaking it.
