# Changelog

## 0.1.0 (unreleased)

- Compaction re-sends the captured provider request plus a summarize instruction, thinking disabled, so the server prefix cache covers the history.
- Post-compaction warm-up (1-token request with the new context).
- Fallback to Pi's default compaction on overflow, missing capture, low room, tool use, truncation, or errors.
- `/prefix-compaction` status command; global and project JSON config.
