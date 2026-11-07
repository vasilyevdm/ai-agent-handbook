<div align="center">

# AI Agent Engineering Handbook

**How modern AI agents are built — from 30+ open-source framework codebases.**

Should you use LangGraph or CrewAI? What makes OpenClaw tick? Why does Claude Code compact at 92%?
How do you stop context rot from killing your agent at 25% fill?

This guide answers these questions — and 200 more.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Read the Guide**](COMPREHENSIVE_AGENT_ENGINEERING_GUIDE_2026.md) · [**PDF**](COMPREHENSIVE_AGENT_ENGINEERING_GUIDE_2026.pdf)

</div>

---

## What is this

An organized collection of patterns, architectures, and implementation details from 30+ AI agent frameworks — extracted by reading their actual source code, system prompts, and compaction logic.

Not opinions. Not tutorials. Just documented patterns from codebases that are running in production.

## Why it exists

Most agent knowledge in 2026 is either surface-level blog posts or buried in source code nobody has time to read. We went through the codebases of OpenClaw, Claude Code, LangGraph, CrewAI, Hermes Agent, and 25+ others to document how they actually work — the agent loops, the prompt assembly, the context management, the memory systems, the tool architectures.

---

## Questions this answers

- LangGraph vs CrewAI vs PydanticAI — which one for what?
- How does OpenClaw's SOUL.md / AGENTS.md pattern work and why does everyone copy it?
- Why do agents get worse the longer they run? (context rot, and 12 ways to fight it)
- MCP servers are eating 72% of my context window — how do I fix tool sprawl?
- Should I use one agent or multiple? When does multi-agent actually help?
- How does Claude Code handle compaction? What survives, what gets dropped?
- What's the actual agent loop code? Is it really just a while loop?
- Skills as markdown vs. compiled tools — what do the top frameworks use?
- How does Hermes Agent improve itself over time? (episodic memory + self-evolution)
- What's the minimum viable agent architecture for production?

---

## What's covered

| Part | Topic | Contents |
|:-----|:------|:---------|
| I–II | **Agent Loops** | 8 loop variants with code: ReAct, Plan+Execute, Reflection, Compaction, Code-as-Action, Event-Driven, Graph State Machine, Heartbeat. Termination. Error recovery. |
| III | **System Prompts** | Assembly patterns. SOUL.md / AGENTS.md separation. Skill catalogs. Anti-patterns. |
| IV | **Context Management** | 6 compaction strategies with real prompts from Claude Code and OpenClaw. Trigger strategies. What survives compaction. |
| IV-B | **Context Rot** | 3 mechanisms. 12 defenses. The 40-60% rule. Agent Cognitive Compressor. Measuring degradation. |
| V | **Memory** | 5-tier hierarchy. File-based vs. vector vs. observational vs. episodic. 8 framework implementations compared. |
| VI | **Tools** | MCP. Code-as-action. Skills-as-markdown. JIT loading. Tool sprawl (72% context consumed). 7 solutions. Progressive disclosure. SDP. |
| VII | **Orchestration** | 6 multi-agent patterns. State passing. A2A protocol. Sizing and topology guidelines. |
| VIII | **Planning** | 5 strategies. Reflection loops. Cline's Plan/Act gold standard. |
| IX–XI | **Human-in-the-Loop, State, Security** | Permission models. Checkpointing. Durable execution. Sandboxing. Prompt injection defense. |
| XII–XIII | **Testing, Deployment** | Benchmarks. Eval strategies. Cost optimization. Observability. Gateway architecture. |
| XIV | **Synthesis** | Reference architecture. Decision framework for choosing your stack. |

---

## Selected findings

Things we found that weren't obvious:

- **A 100-line agent scores 74% on SWE-bench.** The loop is the easy part. Context assembly, tool design, and memory are what matter.
- **Context quality degrades starting at ~25% window fill**, not at 100%. Every frontier model tested shows this (Chroma Research, 18 models).
- **Three MCP servers consumed 143K of 200K tokens** with tool descriptions alone — before the agent read a single user message. Tool selection accuracy drops from 43% to 14% with bloated toolsets.
- **Personality and operational instructions should be separate files.** OpenClaw, Claude Code, and Hermes converged on this independently.
- **The primary reason to use sub-agents is context isolation**, not parallelism. Anthropic measured 90.2% improvement.
- **Skills defined as markdown files** (not code) is the dominant extensibility pattern across Claude Code, OpenClaw, and Cline. Progressive 3-tier loading cuts token cost by 94%.
- **Re-injecting instructions near the end of context** defeats "instruction centrifugation" — system prompt influence fading as context grows.

---

## 30+ Frameworks analyzed

| # | Framework | Stars | Category |
|:--|:----------|:------|:---------|
| 1 | [OpenClaw](https://github.com/openclaw/openclaw) | 210k+ | Personal AI Agent |
| 2 | [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 170k+ | Autonomous Agent |
| 3 | [n8n](https://github.com/n8n-io/n8n) | 150k+ | Workflow |
| 4 | [Dify](https://github.com/langgenius/dify) | 129k+ | Agent Platform |
| 5 | [OpenCode](https://github.com/opencode-ai/opencode) | 120k+ | Coding Agent |
| 6 | [MS Agent Framework](https://github.com/microsoft/agent-framework) | 75k+ | Enterprise |
| 7 | [Langflow](https://github.com/langflow-ai/langflow) | 55k+ | Visual Builder |
| 8 | [browser-use](https://github.com/browser-use/browser-use) | 50k+ | Browser Agent |
| 9 | [OpenHands](https://github.com/OpenHands/OpenHands) | 50k+ | Coding Agent |
| 10 | [MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 50k+ | Multi-Agent |
| 11 | [CrewAI](https://github.com/crewAIInc/crewAI) | 46k+ | Multi-Agent |
| 12 | [LangGraph](https://github.com/langchain-ai/langgraph) | 44.6k+ | Agent Framework |
| 13 | [AG2](https://github.com/ag2ai/ag2) | 40k+ | Multi-Agent |
| 14 | [Cline](https://github.com/cline/cline) | 35k+ | IDE Agent |
| 15 | [Aider](https://github.com/Aider-AI/aider) | 30k+ | Coding Agent |
| 16 | [Mastra](https://github.com/mastra-ai/mastra) | 25k+ | Agent Framework |
| 17 | [Goose](https://github.com/block/goose) | 25k+ | Coding Agent |
| 18 | [Roo Code](https://github.com/RooCodeInc/Roo-Code) | 22k+ | IDE Agent |
| 19 | [SWE-agent](https://github.com/SWE-agent/SWE-agent) | 20k+ | Research Agent |
| 20 | [Bolt.new](https://github.com/stackblitz/bolt.new) | 20k+ | Web Dev Agent |
| 21 | [Agno](https://github.com/agno-agi/agno) | 18.5k+ | Agent Runtime |
| 22 | [Google ADK](https://github.com/google/adk-python) | 17.8k+ | Agent Toolkit |
| 23 | [smolagents](https://github.com/huggingface/smolagents) | 15k+ | Agent Framework |
| 24 | [PydanticAI](https://github.com/pydantic/pydantic-ai) | 15.1k+ | Agent Framework |
| 25 | [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | 15k+ | Agent SDK |
| 26 | [Hermes Agent](https://github.com/NousResearch/hermes-agent) | 8.3k+ | Self-Improving |
| 27 | [Orchestrator](https://github.com/OrchestratorInc/agent-orchestrator) | 8k+ | Orchestrator |
| 28 | [Stagehand](https://github.com/browserbase/stagehand) | 8k+ | Browser Agent |
| 29 | [AWS Agent Squad](https://github.com/awslabs/agent-squad) | 5k+ | Orchestrator |
| 30 | [Devika](https://github.com/stitionai/devika) | — | AI Engineer |

<sub>Star counts as of March 2026.</sub>

---

## Quick decision guide

```
Coding agent             → Claude Agent SDK, Cline, Roo Code
Visual / no-code         → n8n, Dify, Langflow
Personal assistant       → OpenClaw, Hermes Agent
Multi-agent (graphs)     → LangGraph
Multi-agent (roles)      → CrewAI
Enterprise .NET/Java     → MS Agent Framework, Google ADK
Type safety              → PydanticAI
TypeScript native        → Mastra
Minimal footprint        → smolagents
Self-improving           → Hermes Agent
Browser automation       → browser-use, Stagehand
```

---

## Repo structure

```
├── README.md
├── COMPREHENSIVE_AGENT_ENGINEERING_GUIDE_2026.md   ← The guide (~5,000 lines)
├── COMPREHENSIVE_AGENT_ENGINEERING_GUIDE_2026.pdf  ← PDF version
├── CONTRIBUTING.md
└── LICENSE
```

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Corrections, new framework analyses, production patterns you've discovered — anything that makes this more accurate and useful.

---

## Author

[Dmitriy Vasilyev](https://github.com/vasilyevdm) · AI Enthusiast

---

## License

[MIT](LICENSE)
