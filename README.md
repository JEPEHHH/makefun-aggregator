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
MAKEFUN_PROMPT_FILE=... MAKEFUN_ENV_URL=... node main.ts
```

- Picks a free port, listens on 0.0.0.0.
- `POST $MAKEFUN_ENV_URL/register` with `{"protocol_version":1,"port":N,"catalog":{...}}`.
- `--list` prints the catalog document byte for byte as registered.
- Polls `GET $MAKEFUN_ENV_URL/work`; exits 0 when `done` is true.

## Layout

- `main.ts` — HTTP server, both wires, registration, catalog, work loop.
- `lib/providers.ts` — free-model catalog parsing and routing/failover over the gateway.
- `lib/wire.ts` — request/response shapes for each wire, including SSE event order.

## License

MIT — see `LICENSE`.
