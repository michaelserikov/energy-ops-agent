# n8n run, 20 November 2026

n8n 2.35.7 (npm, Node 22, WSL Ubuntu), API in the same WSL (the Windows host is not reachable from WSL because of the firewall).

| Webhook call | Result |
|---|---|
| `energy-report?day=2024-06-01` | `{"day":"2024-06-01","status":"no alert","n_flags":0}` (branch: Log only) |
| `energy-report?day=2024-12-12` | `{"stored":true}`; `outbox.jsonl` holds the subject `Energy data alert 2024-12-12` and the report text with 14 unusual hours (branch: Message to operations) |

API log: 2 report requests, 1 outbox request. Schedule trigger not observed. The message store is a file; nothing was sent anywhere.
