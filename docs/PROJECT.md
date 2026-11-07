# Project record\n\n## Objective\n\nMinimal tool-using agent loop over a text-only LLM port: JSON actions, step, time and context budgets, loop detection, escaped tool results and content-free events.\n\n## Scope\n\n- **In-scope**: The public API (`Tool`, `Budget`, `run_agent`, `render_system_prompt`, `AgentResult`), typed and tested.\n- **Out-of-scope**: Native provider tool-calling APIs, parallel tool calls, persistent memory, planning strategies, streaming, async support.\n\n## Effort metrics\n\n| Metric | Value |\n|---|---|\n| Tokens (input / output) | pending measurement |\n| Hours (human) | pending measurement |\n| Cost (USD) | pending measurement |\n

## Factory run u120 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4488 | 10100 | 0 | 0 | 0.0206 |
| fix-1 | gemini-3.8-flash | no | 91892 | 3832 | 0 | 795970 | 0.1430 |
| fix-2 | gemini-3.8-flash | no | 26859 | 410 | 0 | 74425 | 0.0273 |
| fix-3 | gemini-3.8-flash | no | 97464 | 10011 | 0 | 897471 | 0.1779 |
| audit | gemini-3.1-flash-lite | no | 39989 | 238 | 0 | 93747 | 0.0127 |
| docs | gemini-3.1-flash-lite | no | 164857 | 1898 | 0 | 0 | 0.0441 |

Total: $0.4256. Prices: Anthropic list prices.
