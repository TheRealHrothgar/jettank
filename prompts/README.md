# Prompts

These files override the built-in prompts in the code. Edit one, then either:

* say **"Hank reload"**, or
* press **reload code** on the console, or
* just save - the watcher picks up changes within a few seconds.

| file | what it drives |
|---|---|
| `agent.md` | how Hank behaves in conversation and with tools |
| `planner.md` | the autonomous perception loop's JSON planner |
| `codegen.md` | the instructions used when Hank writes his own code |
| `self.md` | free text appended to his live system facts |

A missing or empty file means the built-in prompt is used. A bad edit cannot
crash him: the reload is wrapped, and on failure the previous prompt stays.
