# LLM Inference Gateway Implementation Plan

> **Executor:** Viraj writes all gateway, simserver, loadgen and test code himself (spec §1,
> agreed constraint). This plan therefore gives interfaces, behaviour contracts, test cases,
> concept notes and hints — **not implementation code**. Claude reviews each task's diff
> before the next task starts. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A measured Python gateway that routes, streams, fails over and sheds load in front of
simulated vLLM backends, calibrated on a real RTX 3060, plus one upstream PR to
vllm-project/production-stack.

**Architecture:** Four processes talking HTTP on localhost — `simserver` ×3 (fake vLLM),
`gateway` (policies, streaming proxy, health, failover, shedding), `loadgen` (seeded traffic +
timings), and `experiments` scripts that start them, inject faults and write results.

**Tech Stack:** Python 3.13, uv, FastAPI, uvicorn, httpx, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-27-llm-inference-gateway-design.md`

## Global Constraints

- 12 h/week, weeks 1–6 (Sep 28 → Nov 8, 2026). Interview prep wins every conflict.
- Viraj writes the code; every published number is labelled **simulated** or **real (RTX 3060)**.
- No Docker, Kubernetes, auth, billing or multi-replica gateway.
- Experiments: plan file (hypothesis, workload, pass bar, falsifier) committed **before** the
  run; results appended, never edited.
- Comparisons are paired per request on identical seeded traffic.
- Scope-cut order when a week slips: shared-GPU E1 run → PR #2 → formal E3 → metrics polling in
  LeastLoaded. Never cut E1, E2, tests, week-6 calibration, PR #1.

## Spec correction made while planning

Spec §3.2 keys `PrefixAware` on "system prompt + all messages except the last". That key
**changes every turn** (turn 2's prefix includes turn 1's assistant reply, turn 1's does not),
so consecutive turns of one conversation would hash to different backends — the opposite of
the goal. This plan keys on **system messages + the first user message**, which is stable for
the life of a conversation. Task 17 has a test that pins this.

## Review Focus

Inputs the spec implies but no feature test naturally hits; each has a test in its owning task:

1. **Client disconnects mid-stream** — upstream must be cancelled and the backend's in-flight
   count decremented, or the count leaks and LeastLoaded drifts. → Task 8.
2. **Every backend down or open-circuit** — respond `503` at once, never hang. → Tasks 8 and 19.
3. **Backend returns 200 then closes without `data: [DONE]`** — client gets a terminal SSE
   error event, not a silently truncated answer. → Task 19.
4. **Conversation with one message / no system prompt** — prefix key still defined and routing
   still works. → Task 17.
5. **Backend hangs before first byte** — first-byte timeout triggers failover instead of
   waiting forever. → Task 19.

## File structure

```
pyproject.toml
simserver/
  config.py         SimConfig dataclass, load_sim_config()
  timing.py         prefill_seconds(), decode_seconds_per_token()
  prefix_cache.py   PrefixCache
  slots.py          Slots (running/waiting accounting)
  faults.py         FaultState
  app.py            create_sim_app(cfg) — HTTP surface
  __main__.py       python -m simserver --port --config --name
gateway/
  config.py         GatewayConfig, load_gateway_config()
  backends.py       Backend, Registry
  policies.py       Router protocol, RoundRobin, LeastLoaded, PrefixAware
  hashring.py       HashRing
  prefix.py         conversation_key()
  metrics_parse.py  parse_prometheus()
  breaker.py        CircuitBreaker
  health.py         HealthChecker
  proxy.py          open_upstream(), relay()
  shedding.py       shed_decision()
  app.py            create_gateway_app(cfg, router)
  __main__.py       python -m gateway --config --policy
loadgen/
  workload.py       build_workload()
  runner.py         run_workload()
  summary.py        summarize()
  __main__.py       python -m loadgen --url --seed --rate --duration --out
experiments/
  harness.py        start/stop processes, inject faults, snapshot metrics
  E1-routing.md  E2-failover.md  E3-shedding.md   (plans, then results)
  results/
lab/
  calibration-session.md
  results/
notes/
  upstream.md       production-stack reading notes, PR links
configs/
  sim.default.toml  gateway.default.toml
tests/  (mirrors packages)
```

---

## Week 1 — simserver (Sep 28 → Oct 4)

### Task 1: Project scaffold + timing model

**Files:** Create `pyproject.toml`, `simserver/config.py`, `simserver/timing.py`,
`configs/sim.default.toml`, `tests/simserver/test_timing.py`

**Interfaces — produces:**
- `@dataclass(frozen=True) SimConfig`: `prefill_base_s: float`, `prefill_per_token_s: float`,
  `decode_base_s: float`, `decode_per_running_s: float`, `max_running: int`,
  `cache_capacity_blocks: int`, `block_size: int = 16`, `max_tokens_default: int`
- `load_sim_config(path: str) -> SimConfig` (use stdlib `tomllib`)
- `prefill_seconds(uncached_tokens: int, cfg: SimConfig) -> float`
  = `prefill_base_s + prefill_per_token_s * uncached_tokens`
- `decode_seconds_per_token(running: int, cfg: SimConfig) -> float`
  = `decode_base_s + decode_per_running_s * max(running - 1, 0)`

**Starting values** for `configs/sim.default.toml` (replaced by calibration in Task 22):
`prefill_base_s = 0.02`, `prefill_per_token_s = 0.0004`, `decode_base_s = 0.02`,
`decode_per_running_s = 0.002`, `max_running = 8`, `cache_capacity_blocks = 2000`,
`block_size = 16`, `max_tokens_default = 64`.

**Setup:** `uv init`, then `uv add fastapi uvicorn httpx` and
`uv add --dev pytest pytest-asyncio`. Set `asyncio_mode = "auto"` under `[tool.pytest.ini_options]`.

**Test cases:**

| Test | Input | Expect |
|---|---|---|
| prefill zero uncached | 0 tokens | == base |
| prefill linear | 1000 tokens, defaults | ≈ 0.42 |
| decode single request | running=1 | == decode_base |
| decode grows | running=8 | == base + 7×per_running |
| decode never below base | running=0 | == base |
| config loads | default toml | all fields typed correctly |

- [ ] Write tests → run `uv run pytest tests/simserver/test_timing.py -q` → FAIL
- [ ] Implement → PASS
- [ ] Commit `feat(simserver): config and timing model`

### Task 2: Prefix cache

**Concept:** vLLM caches KV in fixed-size blocks. A block is reusable only if **every block
before it** matches too, so each block's key is a hash that chains the previous block's key
with this block's tokens. Only full blocks are cached.

**Files:** Create `simserver/prefix_cache.py`, `tests/simserver/test_prefix_cache.py`

**Interfaces — produces:**
- `PrefixCache(capacity_blocks: int, block_size: int)`
- `.lookup_and_insert(tokens: list[str]) -> int` — returns the number of **cached tokens**
  (multiple of `block_size`) for the longest matching chain of leading full blocks, then inserts
  all full blocks of `tokens` and marks them most-recently-used.
- `.hits_tokens: int`, `.queried_tokens: int` — running totals for metrics.

**Test cases:**

| Test | Sequence of calls | Expect |
|---|---|---|
| cold | lookup 40 tokens | returns 0 |
| exact repeat | same 40 tokens twice | 2nd returns 32 (two full blocks) |
| shared prefix | A = 48 tokens; B = A[:32] + 16 different | B returns 32 |
| divergence in block 1 | B differs at token 5 | returns 0 |
| partial block not cached | 10 tokens twice | returns 0 |
| LRU eviction | capacity 2 blocks; insert X (2 blocks), then Y (2 blocks) | X returns 0 afterwards |
| counters | after two calls | `queried_tokens` = sum of lengths; `hits_tokens` = sum of returns |

**Hints:** `collections.OrderedDict` gives LRU (`move_to_end`, `popitem(last=False)`). Chain
key: `hash((prev_key, tuple(block)))` is fine inside one process.

- [ ] Tests → FAIL → implement → PASS → commit `feat(simserver): chained-block prefix cache`

### Task 3: Slots (running / waiting)

**Concept:** continuous batching simplified to N slots; extra requests wait in FIFO order.

**Files:** Create `simserver/slots.py`, `tests/simserver/test_slots.py`

**Interfaces — produces:**
- `Slots(max_running: int)`; `async with slots.hold():` acquires a slot for the block.
- `.running: int`, `.waiting: int` — exact at every moment, including when a waiting task is
  cancelled.

**Test cases (async):**

| Test | Setup | Expect |
|---|---|---|
| counts under limit | max 2, start 2 holders | running 2, waiting 0 |
| queueing | max 2, start 3 holders | running 2, waiting 1 |
| release admits | finish one of the 2 | running 2, waiting 0 |
| cancel while waiting | cancel the waiting task | waiting 0, no slot lost |
| cancel while running | cancel a running task | running decrements |

**Hint:** `asyncio.Semaphore` plus counters incremented/decremented in `try/finally`. The
cancel-while-waiting case is the one that breaks naive versions.

- [ ] Tests → FAIL → implement → PASS → commit `feat(simserver): slot accounting`

### Task 4: simserver HTTP surface (non-streaming, streaming, metrics, health)

**Files:** Create `simserver/app.py`, `simserver/__main__.py`, `tests/simserver/test_app.py`

**Interfaces — produces:**
- `create_sim_app(cfg: SimConfig, name: str) -> FastAPI`
- `POST /v1/chat/completions` — body `{"model", "messages": [{"role","content"}], "stream": bool, "max_tokens"?: int}`
  - Tokens = whitespace split of all message contents concatenated in order (the stand-in
    tokenizer; spec §3.1).
  - Flow: hold a slot → `cached = cache.lookup_and_insert(tokens)` →
    sleep `prefill_seconds(len(tokens) - cached)` → emit `max_tokens` tokens, sleeping
    `decode_seconds_per_token(slots.running)` before each.
  - Output tokens: deterministic words, e.g. `f"w{i}"`.
  - Non-streaming: OpenAI `chat.completion` JSON.
  - Streaming: SSE lines `data: {chunk}\n\n` where chunk is
    `{"id","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"w0 "},"finish_reason":null}]}`,
    a final chunk with `"finish_reason":"stop"`, then `data: [DONE]\n\n`.
  - Response header `x-sim-backend: <name>`.
- `GET /metrics` — Prometheus text:
  ```
  vllm:num_requests_running <n>
  vllm:num_requests_waiting <n>
  sim:prefix_cache_hit_tokens_total <n>
  sim:prefix_cache_queried_tokens_total <n>
  ```
- `GET /health` — `200` with empty body.

**Test cases** (use `httpx.AsyncClient(transport=httpx.ASGITransport(app=...))`; set all timing
values to 0 in a test config so tests are fast):

| Test | Expect |
|---|---|
| non-streaming shape | 200; `choices[0].message.content` has `max_tokens` words |
| streaming shape | lines start `data: `; last line `data: [DONE]`; `max_tokens` content chunks |
| metrics idle | running 0, waiting 0 |
| metrics during load | with max_running=1 and a slow config, 2 concurrent requests → waiting 1 observed |
| cache counters | same request twice → hit_tokens > 0 after the 2nd |
| health | 200 |
| header | `x-sim-backend` equals name |

**Hint:** `fastapi.responses.StreamingResponse` with an async generator and
`media_type="text/event-stream"`.

- [ ] Tests → FAIL → implement → PASS
- [ ] Manual: `uv run python -m simserver --port 9001 --name s1`, then
  `curl -N localhost:9001/v1/chat/completions -H 'content-type: application/json' -d '{"model":"m","messages":[{"role":"user","content":"hi there"}],"stream":true}'` — see tokens arrive over time.
- [ ] Commit `feat(simserver): OpenAI-compatible API, metrics, health`

### Task 5: Fault switches

**Files:** Create `simserver/faults.py`; modify `simserver/app.py`; test `tests/simserver/test_faults.py`

**Interfaces — produces:**
- `FaultState` with `mode: Literal["none","crash","hang","slow","midstream_error"]`,
  `slow_factor: float = 5.0`, `error_after_tokens: int = 3`
- `POST /admin/fault` body `{"mode": ..., "slow_factor"?: float, "error_after_tokens"?: int}`
- Behaviour:
  - `crash` → process exits (`os._exit(1)`); tested via subprocess in Task 21, not here.
  - `hang` → accept the request, never send a byte.
  - `slow` → multiply every sleep by `slow_factor`.
  - `midstream_error` → stream `error_after_tokens` chunks, then close the connection with no
    `[DONE]`.
  - `/health` stays `200` in every mode except `crash`. Real failures often pass health
    checks, so the gateway must cope with a "healthy" backend that hangs or breaks mid-stream.

**Test cases:** `none` unchanged; `slow` measurably slower (factor with nonzero timings);
`midstream_error` yields exactly N content chunks and no `[DONE]`; `hang` does not complete
within 0.5 s (use `asyncio.wait_for`); switching back to `none` restores behaviour.

- [ ] Tests → FAIL → implement → PASS → commit `feat(simserver): fault switches`

---

## Week 2 — gateway baseline + loadgen (Oct 5 → 11)

### Task 6: Backends registry + RoundRobin

**Files:** Create `gateway/config.py`, `gateway/backends.py`, `gateway/policies.py`,
`configs/gateway.default.toml`, tests `tests/gateway/test_policies_rr.py`

**Interfaces — produces:**
- `@dataclass Backend`: `name: str`, `url: str`, `healthy: bool = True`, `inflight: int = 0`,
  `reported_running: int = 0`, `reported_waiting: int = 0`
- `Registry(backends: list[Backend])`; `.all() -> list[Backend]`; `.healthy() -> list[Backend]`;
  `.get(name) -> Backend`
- `class Router(Protocol): def choose(self, request: ChatRequest, candidates: list[Backend], exclude: set[str] = frozenset()) -> Backend | None`
- `@dataclass ChatRequest`: `messages: list[dict]`, `stream: bool`, `raw: dict`
- `RoundRobin().choose(...)` cycles through candidates not in `exclude`; `None` if none.
- `GatewayConfig`: `backends: list[{name,url}]`, `connect_timeout_s = 1.0`,
  `first_byte_timeout_s = 10.0`, `health_interval_s = 0.5`, `breaker_failures = 3`,
  `breaker_cooldown_s = 5.0`, `shed_waiting_threshold = 16`, `max_inflight = 256`, `vnodes = 100`

**Test cases:** 3 backends → A,B,C,A order; `exclude={"B"}` skips B; empty candidates → `None`;
all excluded → `None`.

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): registry and round-robin`

### Task 7: Streaming proxy — open, then relay

**Concept (the key design decision, spec §3.2 failover rule):** the gateway may retry only
until the first byte reaches the client. So split the proxy into two phases:
1. **open**: send the upstream request, check the status, and **read the first SSE event** —
   all *before* the gateway returns a response. Any failure here is retryable.
2. **relay**: return a `StreamingResponse` that yields the already-read first event, then the
   rest. Failures here are not retryable.

**Files:** Create `gateway/proxy.py`; tests `tests/gateway/test_proxy.py`

**Interfaces — produces:**
- `class UpstreamError(Exception)` with `.retryable: bool`
- `async def open_upstream(client: httpx.AsyncClient, backend: Backend, body: dict, first_byte_timeout_s: float) -> OpenStream`
  — raises `UpstreamError(retryable=True)` on connect error, timeout before first event, or
  status ≥ 500; `retryable=False` on 4xx.
- `@dataclass OpenStream`: `backend: Backend`, `first_event: bytes`, `rest: AsyncIterator[bytes]`, `close: Callable[[], Awaitable[None]]`
- `async def relay(stream: OpenStream) -> AsyncIterator[bytes]` — yields first event, then the
  rest; if the upstream ends without `data: [DONE]` or raises mid-stream, yields
  `data: {"error": {"message": "upstream_interrupted", "backend": "<name>"}}\n\n` and stops.
  Always calls `stream.close()` and decrements `backend.inflight` in `finally`.

**Test cases** (run against real `create_sim_app` via `httpx.ASGITransport` — no mocks of your own code):

| Test | Expect |
|---|---|
| happy path | relayed bytes == direct simserver bytes |
| 500 upstream | `UpstreamError(retryable=True)` from open |
| hang before first byte | `open_upstream` raises retryable within `first_byte_timeout_s` (set 0.2) |
| inflight accounting | inflight 1 during relay, 0 after |

**Hints:** build the request with `client.build_request("POST", url, json=body)` and send it
with `await client.send(req, stream=True)` — that returns before the body is read, so you can
pull the first event from `resp.aiter_lines()` yourself, then hand the same iterator to
`relay`. Wrap the first read in `asyncio.timeout(first_byte_timeout_s)`. Remember
`await resp.aclose()` on every path.

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): two-phase streaming proxy`

### Task 8: Gateway app

**Files:** Create `gateway/app.py`, `gateway/__main__.py`; tests `tests/gateway/test_app.py`

**Interfaces — produces:**
- `create_gateway_app(cfg: GatewayConfig, router: Router, registry: Registry) -> FastAPI`
- `POST /v1/chat/completions` → choose backend → `backend.inflight += 1` → `open_upstream` →
  `StreamingResponse(relay(...))` with header `x-backend: <name>`. Non-streaming requests are
  proxied as a plain response (no relay).
- If `router.choose` returns `None` → `503` JSON `{"error":"no_healthy_backend"}`.
- `GET /metrics` (gateway's own, Prometheus text): `gw_requests_total{backend,outcome}`,
  `gw_retries_total`, `gw_shed_total`, `gw_breaker_open{backend}` (later tasks fill these).
- CLI: `python -m gateway --config configs/gateway.default.toml --policy rr|ll|prefix --port 8000`

**Test cases** (3 simservers mounted in-process with separate `ASGITransport`s, or run as
subprocesses on ports — your choice; state it in the test module docstring):

| Test | Expect |
|---|---|
| streams through | 200, SSE ends `[DONE]`, `x-backend` present |
| round-robin spread | 6 requests → 2 per backend |
| **Review Focus 2:** all unhealthy | mark all `healthy=False` → `503` immediately (< 100 ms) |
| **Review Focus 1:** client disconnect | open stream, read 1 event, close client → within 1 s backend `inflight == 0` and simserver `running == 0` |

- [ ] Tests → FAIL → implement → PASS
- [ ] Manual: 3 simservers (ports 9001–9003) + gateway on 8000; `curl -N` through it.
- [ ] Commit `feat(gateway): app with round-robin routing`

### Task 9: loadgen — workload

**Files:** Create `loadgen/workload.py`; tests `tests/loadgen/test_workload.py`

**Interfaces — produces:**
- `@dataclass PlannedTurn`: `conv_id: str`, `turn: int`, `start_offset_s: float | None` (only turn 0 has one), `user_text: str`, `system_text: str | None`, `max_tokens: int`
- `@dataclass Workload`: `conversations: list[list[PlannedTurn]]`, `singles: list[PlannedTurn]`
- `build_workload(seed: int, rate_per_s: float, duration_s: float, conv_fraction: float = 0.6, turns: tuple[int,int] = (3, 6), system_prompts: int = 3, think_time_s: float = 1.0) -> Workload`
  - Arrivals: Poisson (`random.expovariate(rate)`) until `duration_s`.
  - Each arrival is a conversation with probability `conv_fraction`, else a single.
  - System prompt picked from `system_prompts` fixed long texts (~200 words each, generated
    from the seed) — shared prefixes across conversations, like real apps.
  - User texts: seeded random words, 20–80 words.

**Test cases:** same seed → identical workload (compare `dataclasses.asdict`); different seed →
different; arrival count ≈ rate × duration (within 20% for rate 5, duration 60); turn counts
within bounds; exactly `system_prompts` distinct system texts.

- [ ] Tests → FAIL → implement → PASS → commit `feat(loadgen): seeded workload`

### Task 10: loadgen — runner + summary

**Files:** Create `loadgen/runner.py`, `loadgen/summary.py`, `loadgen/__main__.py`; tests
`tests/loadgen/test_runner.py`, `tests/loadgen/test_summary.py`

**Interfaces — produces:**
- `async def run_workload(url: str, wl: Workload, out_path: str) -> None` — one task per
  conversation / single, started at its offset; a conversation sends turn k+1 after turn k
  finishes plus `think_time_s`, including turn k's assistant reply in `messages`.
- JSON-lines record per request: `conv_id, turn, backend, status, t_send, ttft_s,
  itl_s (list), total_s, output_tokens, error`
- `summarize(records: list[dict]) -> dict` — `count, ok, errors, shed (status 429), ttft_p50,
  ttft_p99, itl_p99, throughput_tok_s, by_backend`
- `def paired_diff(a: list[dict], b: list[dict], key="ttft_s") -> dict` — join on
  `(conv_id, turn)`; return `n, median_diff, wins_a, wins_b, ties`.
- CLI: `python -m loadgen --url http://localhost:8000 --seed 1 --rate 5 --duration 60 --out experiments/results/x.jsonl`

**Test cases:** summary percentiles on a hand-made list (p50 of [1..100] == 50.5 using
`statistics.quantiles` or your chosen method — document which); `paired_diff` on two tiny
hand-made runs; runner against gateway+simservers with zero timings produces one record per
planned turn and `status == 200` everywhere.

- [ ] Tests → FAIL → implement → PASS → commit `feat(loadgen): runner, summary, paired diff`

### Task 11: Experiment harness + baseline run

**Files:** Create `experiments/harness.py`; tests `tests/experiments/test_harness.py`

**Interfaces — produces:**
- `class Stack` (context manager): starts N simservers on ports 9001.. and the gateway on 8000
  as subprocesses with given configs/policy; waits for `/health`; on exit terminates all.
- `.fault(backend_name: str, mode: str, **kw) -> None` — POSTs `/admin/fault`.
- `.snapshot() -> dict` — scrapes every simserver `/metrics` + gateway `/metrics`.
- `def run_condition(policy, seed, rate, duration, out_dir, faults=[]) -> dict` — starts a
  Stack, snapshot before, runs loadgen, snapshot after, writes `summary.json` with cache-hit
  rate = Δhit_tokens / Δqueried_tokens.

**Test:** `run_condition("rr", seed=1, rate=5, duration=5, ...)` with zero-timing config
completes, writes files, cache-hit rate is a number in [0, 1].

- [ ] Tests → FAIL → implement → PASS
- [ ] **Baseline:** `rr`, seed 1, rate 5, 60 s, default timings → `experiments/results/baseline-rr/`
- [ ] Commit `feat(experiments): harness and round-robin baseline`

### Task 12 (week 2, parallel): Read production-stack

Not code. ~3 h.
- [ ] Clone `https://github.com/vllm-project/production-stack`; read `CONTRIBUTING.md`: note
  whether commits need sign-off (`git commit -s`), the test commands, and any AI-assistance
  policy, in a new file `notes/upstream.md`.
- [ ] Run its router tests locally; record the command and result.
- [ ] Read the routing logic module (round-robin, session, prefix-aware, KV-aware) and write one
  paragraph per strategy comparing it with yours.
- [ ] Commit `docs: upstream reading notes`.

---

## Week 3 — health + LeastLoaded + PR #1 (Oct 12 → 18)

### Task 13: Metrics parsing, circuit breaker, health checker

**Files:** Create `gateway/metrics_parse.py`, `gateway/breaker.py`, `gateway/health.py`;
tests `tests/gateway/test_metrics_parse.py`, `test_breaker.py`, `test_health.py`

**Interfaces — produces:**
- `parse_prometheus(text: str) -> dict[str, float]` — metric name (labels stripped) → value;
  ignores `#` lines and malformed lines.
- `CircuitBreaker(failures_to_open: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic)`
  - `.state -> Literal["closed","open","half_open"]`
  - `.allow() -> bool` — closed: True; open: False until cooldown elapsed, then becomes
    half_open and returns True once; half_open: False while a probe is outstanding.
  - `.record_success()` → closed, counter reset; `.record_failure()` → counts; opens at N;
    failure in half_open → open again with a fresh cooldown.
- `HealthChecker(registry, client, interval_s, breakers: dict[str, CircuitBreaker])`
  - `async def run()` loop: for each backend, GET `/health` + `/metrics`; success → update
    `reported_running/waiting`, `record_success`; failure/timeout → `record_failure`.
  - Sets `backend.healthy = breakers[name].state != "open"`.
  - Proxy failures (Task 19) also call `record_failure` on the same breaker.

**Test cases:**

| Test | Expect |
|---|---|
| parse vLLM lines with labels | `vllm:num_requests_running{model_name="x"} 3.0` → 3.0 |
| parse junk | malformed lines skipped, no exception |
| breaker opens at N | 3 failures → open |
| breaker cooldown | fake clock +5 s → allow True once → half_open |
| half_open success closes | → closed |
| half_open failure reopens | → open, cooldown restarts |
| health marks crashed backend | stop one simserver → `healthy False` within 3 × interval |

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): health checks and circuit breaker`

### Task 14: LeastLoaded

**Files:** Modify `gateway/policies.py`; tests `tests/gateway/test_policies_ll.py`

**Interface:** `LeastLoaded().choose(...)` picks min of
`b.inflight + b.reported_waiting` (inflight is the gateway's own, fresher count; reported
waiting catches load from elsewhere); ties broken by name for determinism.

**Test cases:** picks lowest; inflight dominates a stale report; tie → alphabetical; exclude
honoured; empty → `None`.

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): least-loaded policy`
- [ ] Run `run_condition("ll", seed=1, rate=5, 60)`; commit results.

### Task 15: Upstream PR #1

Not gateway code. ~4 h.
- [ ] Pick one item: missing test, reproducible bug, or docs error hit in Task 12. Criteria:
  < ~100 lines, one concern, includes a test.
- [ ] Search open PRs for the same item; comment on the issue (or open one) saying you intend to
  fix it.
- [ ] Implement in your fork following `CONTRIBUTING.md` (sign-off, formatting, tests).
- [ ] Open the PR; link it in `notes/upstream.md`.
- [ ] From now on: reply to reviews within 48 h (~1 h/week).

---

## Week 4 — prefix-aware routing + E1 (Oct 19 → 25)

### Task 16: Hash ring

**Files:** Create `gateway/hashring.py`; tests `tests/gateway/test_hashring.py`

**Interfaces — produces:**
- `HashRing(nodes: list[str], vnodes: int = 100)`
- `.lookup(key: str, skip: set[str] = frozenset()) -> str | None` — first node clockwise from
  `hash(key)` not in `skip`.
- Use a stable hash (`hashlib.sha1(...).digest()` → int), never Python's built-in `hash()`
  (randomised per process).

**Test cases:**

| Test | Expect |
|---|---|
| deterministic | same key → same node across two ring instances |
| balance | 10,000 random keys over 3 nodes → each node 25–42% |
| minimal remap | remove 1 of 3 nodes → keys that moved ≈ those owned by the removed node only (every moved key was on it) |
| skip | skip owner → returns next distinct node |
| all skipped | → `None` |

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): consistent hash ring`

### Task 17: Conversation key + PrefixAware

**Files:** Create `gateway/prefix.py`; modify `gateway/policies.py`; tests
`tests/gateway/test_prefix.py`, `tests/gateway/test_policies_prefix.py`

**Interfaces — produces:**
- `conversation_key(messages: list[dict]) -> str` — sha256 hex of the JSON of
  `[m for m in messages if m["role"] == "system"]` + the **first** `user` message
  (see *Spec correction* above).
- `PrefixAware(ring: HashRing).choose(...)` — `ring.lookup(conversation_key(msgs), skip=exclude ∪ unhealthy names)`
  mapped back to a `Backend`.

**Test cases:**

| Test | Expect |
|---|---|
| stable across turns | turn-1 and turn-3 messages of one conversation → same key |
| distinct conversations | different first user message → different keys (with overwhelming probability) |
| **Review Focus 4:** single message, no system | key defined, non-empty; routing returns a backend |
| no user message at all | key from system messages only; no exception |
| unhealthy owner | owner marked unhealthy → next ring node chosen, and **same** fallback every time |

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): prefix-aware routing`

### Task 18: Experiment E1 — routing

- [ ] **Before running,** write `experiments/E1-routing.md`: hypothesis (prefix-aware lowers
  TTFT p50 vs round-robin at every load level because cache-hit rate rises; least-loaded lowers
  TTFT p99 at the highest load), workload (seed 1–3, rates chosen at ~30% / 60% / 90% of
  measured capacity — measure capacity first with a rising-rate run and record how), pass bars
  (e.g. prefix-aware paired median TTFT diff < 0 with wins > 60% of pairs at 60% load), and what
  would falsify it. Commit it.
- [ ] Run 3 policies × 3 loads × 3 seeds via `run_condition`.
- [ ] Append results: summary table, paired diffs vs round-robin, cache-hit rate per policy.
  Label every number **simulated**.
- [ ] Commit `exp(E1): routing results`.

---

## Week 5 — failover, shedding, E2, E3 (Oct 26 → Nov 1)

### Task 19: Failover in the request path

**Files:** Modify `gateway/app.py`; tests `tests/gateway/test_failover.py`

**Behaviour:**
- On `UpstreamError(retryable=True)` from `open_upstream`: `record_failure` on that backend's
  breaker, `gw_retries_total += 1`, choose again with `exclude={failed}`, try **once** more.
  Second failure → `502 {"error":"upstream_unavailable"}`.
- Non-retryable → pass the upstream status through.
- After relay has started, no retry (relay already emits the terminal error event).

**Test cases** (real simservers with fault switches):

| Test | Expect |
|---|---|
| crash-equivalent (connection refused) on chosen backend | 200 from another backend; retries_total 1 |
| **Review Focus 5:** `hang` on chosen backend | 200 from another backend within first_byte_timeout + normal latency |
| both picks bad | first and second chosen backends both refuse → `502` (only one retry, by design), even if a third backend is healthy |
| **Review Focus 3:** `midstream_error` | client receives N content events then the `upstream_interrupted` error event; **no** retry |
| **Review Focus 2:** all backends open-circuit | 503 at once |

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): single-retry failover before first byte`

### Task 20: Load shedding

**Files:** Create `gateway/shedding.py`; modify `gateway/app.py`; tests `tests/gateway/test_shedding.py`

**Interface:** `shed_decision(registry: Registry, cfg: GatewayConfig, global_inflight: int) -> bool`
— True if `global_inflight >= cfg.max_inflight`, or every healthy backend has
`reported_waiting + inflight >= cfg.shed_waiting_threshold`. App returns `429` with
`Retry-After: 1` and increments `gw_shed_total`; checked **before** choosing a backend.

**Test cases:** below thresholds → False; global cap hit → True; all healthy backends over
threshold → True; one healthy backend under threshold → False; unhealthy backends ignored.

- [ ] Tests → FAIL → implement → PASS → commit `feat(gateway): load shedding`

### Task 21: Experiments E2 and E3

- [ ] Write `experiments/E2-failover.md` **first**: kill (`crash`) / `hang` /
  `midstream_error` one of 3 backends 20 s into a 60 s run at 60% load, policy `prefix`.
  Pass bar: 0 requests lost among those whose failure happened before first byte; every
  mid-stream failure surfaces as the error event; recovery = time until breaker opens.
  Commit, then run, then append results.
- [ ] Write `experiments/E3-shedding.md` **first**: rate = 2× measured capacity, shedding on vs
  off (set thresholds to infinity for "off"), same seed. Pass bar: admitted-request TTFT p99
  with shedding ≤ ½ of p99 without. Commit, run, append.
- [ ] Commit `exp(E2,E3): failover and shedding results`.

---

## Week 6 — calibration + ship (Nov 2 → 8)

### Task 22: Lab calibration

**Prerequisite:** confirm with the lab that using the RTX 3060 for this project is fine.

- [ ] Write `lab/calibration-session.md` (run via AnyDesk, same pattern as the SQLAlchemy
  Rounds): exact commands to start vLLM with
  `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --max-model-len 4096 --gpu-memory-utilization 0.85`,
  a small script (you write it; stdlib + httpx) that measures:
  1. TTFT, cold vs warm (same prompt twice), at prompt lengths 256 / 1024 / 2048 / 3072 tokens,
     5 repeats each;
  2. inter-token latency at 1, 4, 8 concurrent streaming requests, 5 repeats each;
  3. `nvidia-smi` and `vllm --version` output.
  Results written to `lab/results/<date>/` as JSON.
- [ ] Run it at the lab; bring results back; commit raw results unedited.
- [ ] Fit `prefill_base_s`, `prefill_per_token_s` (least squares on cold TTFT vs length),
  `decode_base_s`, `decode_per_running_s` (ITL vs concurrency); write
  `configs/sim.calibrated.toml`; record the fit and residuals in `lab/results/<date>/fit.md`.
- [ ] Re-run E1 at 60% load with the calibrated config; append to E1 as "calibrated".
- [ ] Commit `lab: RTX 3060 calibration and calibrated E1`.
- [ ] *(Optional, first to cut)* two vLLM servers on the one GPU, small-scale E1, labelled
  "shared GPU — validates cache-hit behaviour, not production latency".

### Task 23: README, résumé, portfolio

- [ ] `README.md`: what it is, the diagram, how to run (3 commands), results tables for E1–E3
  with every number labelled simulated / calibrated / real, the failover rule and why,
  limitations (single gateway, simulated backends, one-GPU calibration), upstream PR link.
- [ ] Résumé bullet draft (numbers only from committed results) → Claude review → add to the
  résumé source and the portfolio site.
- [ ] Mock deep-dive with Claude: explain every component and result cold (spec success #5).
- [ ] Create the public GitHub repo and push (you do this).
- [ ] Commit `docs: README with results`.

---
