# Minimal Agent Loop Architecture with Rigid Budgets

Status: accepted (generated, pending human review)

## Context

Agent frameworks often introduce complex graph abstractions, asynchronous engines, and hidden loop logic that make understanding failures difficult and allow runaway resource consumption or injection attacks.

## Decision

Implement a minimal, synchronous agent loop over a standard LLMPort protocol using standard library components only. Enforce strict sequential evaluations: deadline check, step count check, context character count check, followed by LLM completion, reply parsing, and argument type validation. Escape delimiter collisions in tool results to thwart prompt breakouts.

## Consequences

Predictable, auditable agent executions with clean terminal statuses ('final', 'max_steps', 'deadline', 'loop_detected', 'tool_errors', 'invalid_replies', 'context_exhausted') without external dependencies.
