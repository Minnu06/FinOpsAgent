# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Git Workflow — Follow these steps for EVERY change

1. Verify the remote repo: `git remote -v` → must show `https://github.com/Minnu06/FinOpsAgent`
2. Always pull latest code before starting: `git checkout main && git pull origin main`
3. Create a new branch: `git checkout -b claude/<short-description>`
   Example: `claude/add-calorie-tracker` or `claude/fix-login-bug`
4. Make the required code changes
5. Stage and commit: `git add . && git commit -m "clear description of what changed"`
6. Push the branch: `git push origin claude/<branch-name>`
7. Raise a PR: `gh pr create --repo Minnu06/FinOpsAgent --base main --fill`

## Hard Rules — Never break these

- NEVER push directly to `main`
- NEVER touch any repo other than https://github.com/Minnu06/FinOpsAgent
- NEVER commit secrets, API keys, or passwords
- Always create a new branch for every task
- while creating new branch the branch name should not present claude word
- If remote doesn't match the repo above, STOP and ask the user

## Permission Rules — Hard Limits, never break these

- NEVER read, write, or delete files outside of this repo folder
- NEVER install any package (pip, npm, apt) without explicitly asking the user first
- If a new package is needed, STOP and say: "I need to install `<package>` for `<reason>`. Do you approve?"
- Only proceed with installation after user says YES
