Put downloaded benchmark files here (they are not committed to git).

| Benchmark | Where to get it | What to export |
| --- | --- | --- |
| BIPIA | github.com/microsoft/BIPIA | contexts as JSONL (`context`, `question`) + its attack strings as JSON |
| InjecAgent | github.com/uiuc-kang-lab/InjecAgent | tool outputs as JSONL with `task`, `chunk`, `label` |
| AgentDojo | github.com/ethz-spylab/agentdojo | tool outputs as JSONL with `task`, `chunk`, `label` |
| NotInject | Hugging Face dataset `leolee99/NotInject` | benign sentences as JSONL with `text` |

- BIPIA and NotInject: convert with `python -m tapid.build_dataset --source jsonl ... --test-only` (examples in that file's docstring).
- InjecAgent and AgentDojo exports that already have `task`, `chunk` and `label` go straight to `python -m tapid.evaluate --test <file>`.
