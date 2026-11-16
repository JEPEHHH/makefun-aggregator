# makefun-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/JEPEHHH/makefun-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/JEPEHHH/makefun-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)


One long-running server that answers both the OpenAI wire (`POST /v1/chat/completions`)
and the Anthropic wire (`POST /v1/messages`), JSON and streaming, by dialling real free
models through the trial's gateway. `GET /v1/models` answers in whichever wire the
auth header names (`Authorization:` → OpenAI list, `x-api-key:` → Anthropic list).

## Run

```
MAKEFUN_PROMPT_FILE=... MAKEFUN_ENV_URL=... python3 main.py
```

- Single entrypoint, Python 3.13 standard library only, no third-party dependency.
- Picks a free port, listens on 0.0.0.0.
- `POST $MAKEFUN_ENV_URL/register` with `{"protocol_version":1,"port":N,"catalog":{...}}`,
  retried at most 3 times with a 0.5s backoff, then given up on cleanly.
- `--list` prints the catalog document byte for byte as registered, without touching the network.
- Polls `GET $MAKEFUN_ENV_URL/work`; exits 0 when `done` is true.
- Exits 0 on `SIGTERM` / `SIGINT`, releasing the listening port before it goes.

## Environment

| Variable | Meaning |
| --- | --- |
| `MAKEFUN_PROMPT_FILE` | Path to the brief JSON (models + prompts). Unset -> catalog is `{"providers": []}`. |
| `MAKEFUN_ENV_URL` | Trial environment base URL for `/register` and `/work`. |
| `OPENAI_BASE_URL` | Gateway base URL used for upstream Chat Completions calls. |
| `OPENAI_API_KEY` | Gateway bearer token. |
| `MAKEFUN_UPSTREAM_TIMEOUT` | Optional upstream deadline in seconds, clamped below 30s (default 25). |

## Behaviour worth knowing

- Every upstream call carries an explicit timeout; a deadline before the first byte counts as
  one failed candidate, so failover moves on instead of hanging.
- Candidates are ranked by fewest recent failures, then highest `rating_milli`; failover only
  happens before the first byte reaches the client.
- Streams always end with exactly one finish chunk, one usage chunk and one `[DONE]`.
- Requests that no free model satisfies are refused with a wire-shaped error, never a traceback.

## Layout

- `main.py` — HTTP server, both wires, catalog, routing/failover, registration, work loop.

## License

MIT — see `LICENSE`.
