# LLM inference gateway — design

**Date:** 2026-09-27 · **Status:** approved in brainstorming, awaiting written-spec review
**Companion plan:** `interview-prep-2027` (spec in that repo). This project gets 12 h/week in
weeks 1–6 (Mon 2026-09-28 → Sun 2026-11-08); interview prep takes priority when time runs short.

---

## 1. Brief

**Goal.** A Python gateway in front of several LLM inference servers that routes each request,
streams the response, survives backend failure and sheds load — with every routing change
measured against the previous one on identical traffic. Followed by at least one pull request
to vLLM's production-stack router.

**Why this project** (researched 2026-09-27)
- Closes Viraj's largest evidence gaps for big-company roles: distributed systems /
  concurrency, and open-source contribution in a codebase he did not start.
- Sits in a growth area: AI/ML engineering +39% since 2022 while entry-level hiring fell
  (SignalFire 2026); Google advertises SWE roles for AI/ML serving infrastructure.
- Feeds interviews directly: "design an LLM serving system / load balancer / rate limiter",
  and the new read-and-extend-a-codebase rounds (via the upstream contribution).

**Stated by Viraj**
- Option A1: build a small gateway first, then contribute upstream.
- Python. Target: big companies.
- Smaller scope (option b) at 12 h/week for ~6 weeks.
- Lab GPU (RTX 3060) is available, as used in the SQLAlchemy project.

**Agreed constraints**
- **Viraj writes the gateway code himself.** Claude explains concepts and vLLM internals,
  designs experiments with him, and reviews code.
- Development runs against simulated backends; cost ~$0.
- Every published number is labelled **simulated** or **real (RTX 3060)**.
- Upstream contribution is a bonus: opening a PR is required, a merge is not.

**Out of scope (dropped for the smaller version)**
- Cache-vs-load trade-off routing (breaking prefix stickiness under hot-prefix load).
  Candidate follow-up after week 16.
- A full real-GPU experiment suite. Real hardware is used only for calibration (§6.1).
- Docker, Kubernetes, multi-replica gateway, auth, billing.

**Success criteria**
1. Public repo with the four components (§3), tests passing, README with results.
2. Experiments E1–E3 run with hypotheses and pass bars written before each run.
3. Simulator calibrated from at least one real vLLM run on the lab RTX 3060.
4. At least one PR opened to vllm-project/production-stack.
5. Viraj can explain every component and every result cold (checked in a Claude mock).

---

## 2. Concepts the design depends on

- **Prefill / decode.** The server reads the whole prompt once (prefill, compute-heavy), then
  generates tokens one at a time (decode).
- **KV cache.** Prefill produces per-token key/value tensors kept in GPU memory so decode need
  not reread the prompt.
- **Prefix caching.** A request whose prompt begins with an already-cached prefix reuses that
  KV and skips most prefill → lower time to first token. The cache lives on **one** server,
  which is why routing matters.
- **Continuous batching.** A server runs many requests at once and admits new ones as others
  finish; past capacity, requests queue and time to first token climbs.
- **Metrics.** TTFT (time to first token), inter-token latency, throughput (tokens/s),
  reported at p50 and p99.

---

## 3. Architecture

```
 loadgen ──HTTP──▶  gateway  ──HTTP stream──▶  simserver ×3
 (traffic,          ├ router policy            (vLLM-compatible API,
  timings)          ├ streaming proxy           timing model,
                    ├ health checker            prefix cache,
                    ├ failover                  fault switches)
                    └ load shedding
```

Each component runs as its own process on localhost. Units communicate only over HTTP.

### 3.1 `simserver` — simulated vLLM backend
- **API:** OpenAI-compatible `POST /v1/chat/completions` with `stream: true` (SSE) and
  non-streaming; `GET /metrics` in Prometheus text exposing vLLM's queue gauges
  (`vllm:num_requests_running`, `vllm:num_requests_waiting`); `GET /health`.
- **Timing model:**
  - `prefill_s = base + per_token × uncached_prompt_tokens`
  - `decode_s_per_token = f(running_requests)`, rising with concurrency
  - Fixed number of running slots; excess requests wait in a FIFO queue.
- **Prefix cache:** bounded LRU of prompt blocks (16 tokens, matching vLLM's default block
  size), keyed by cumulative block hash. Hits skip that portion of prefill.
- **Tokens:** a deterministic stand-in tokenizer (whitespace split) is sufficient; the gateway
  never depends on real tokenization.
- **Fault switches** (set via an admin endpoint or CLI flag): `crash` (process exits),
  `hang` (accept, never respond), `slow` (multiply timings), `midstream_error` (drop the
  connection after N tokens).
- **Parameters** live in one config file so calibration (§6.1) only edits numbers.

### 3.2 `gateway` — the core
- **Backend registry:** static list of backend URLs from config.
- **Router interface:** `choose(request, backends) -> Backend`, three implementations:
  - `RoundRobin` — baseline.
  - `LeastLoaded` — minimum of (backend-reported waiting + running) combined with the gateway's
    own in-flight count for that backend, which is fresher than a polled metric.
  - `PrefixAware` — hash of the conversation prefix (system prompt + all messages except the
    last) placed on a consistent-hash ring with virtual nodes; if the chosen backend is down,
    walk the ring to the next healthy node.
- **Streaming proxy:** forwards SSE chunks as they arrive, no buffering. If the client
  disconnects, the upstream request is cancelled.
- **Health checker:** background task polling `/health` and `/metrics` on an interval.
  Circuit breaker per backend: **closed** → **open** after N consecutive failures (removed from
  routing) → **half-open** probe after a cool-down → closed on success.
- **Failover rule:** on connect error, timeout or 5xx **before the first byte is sent to the
  client**, retry once on a different healthy backend. **After the first byte,** never retry;
  emit a terminal SSE error event and close. (A retry after streaming began would duplicate or
  contradict tokens the user already saw.)
- **Load shedding:** if every healthy backend's waiting count exceeds a threshold, or the
  gateway's global in-flight cap is reached, reply `429 Too Many Requests` with `Retry-After`.
- **Observability:** the gateway exposes its own `/metrics` (requests by backend and outcome,
  retries, sheds, circuit state).

### 3.3 `loadgen` — traffic generator
- Workload from a fixed seed: a mix of multi-turn conversations (shared, growing prefixes) and
  single-shot requests; Poisson arrivals at a configured rate.
- Records per request: send time, TTFT, inter-token latencies, total time, status, backend.
- Writes one JSON-lines file per run plus a summary (p50/p99 TTFT, p99 inter-token latency,
  throughput, error and shed counts).
- Same seed ⇒ identical request sequence, enabling paired comparison across policies.

### 3.4 `experiments` — measurement
Each experiment has a file written **before** it runs, containing hypothesis, workload, pass
bar and what would falsify it. Results are appended, never edited.

| ID | Question | Setup | Primary measures |
|---|---|---|---|
| **E1** | Does routing policy change TTFT and cache hit rate? | RoundRobin vs LeastLoaded vs PrefixAware; 3 backends; 3 load levels; same seed | TTFT p50/p99, backend cache-hit rate, throughput |
| **E2** | Does failover lose requests? | Kill 1 of 3 backends mid-run (each fault switch) | Requests lost before first token (target 0), errors after first token, recovery time |
| **E3** | Does shedding protect latency under overload? | Arrival rate at 2× measured capacity, shedding on vs off | p99 TTFT of admitted requests, shed rate |

Comparisons report paired per-request differences on identical traffic, not just averages.

---

## 4. Stack

Python 3.13 · uv · FastAPI + uvicorn · httpx (async streaming) · pytest + pytest-asyncio.
No Docker or Kubernetes in this scope.

## 5. Testing

- **Unit:** each router policy (deterministic); consistent-hash ring including the property
  that removing one of N nodes remaps ≈ 1/N of keys; circuit-breaker state transitions; the
  failover rule (retry before first byte, none after).
- **Integration:** gateway + three in-process simservers; streaming end to end; client
  disconnect cancels upstream.
- **Fault:** one test per fault switch.
- Tests are written first where natural, and Viraj writes them.

---

## 6. Plan (12 h/week)

| Week | Dates | Build | Output |
|---|---|---|---|
| 1 | Sep 28 → Oct 4 | Concepts; `simserver`: non-streaming → streaming → metrics → prefix cache | Backend that behaves like vLLM |
| 2 | Oct 5 → Oct 11 | Gateway streaming proxy + `RoundRobin`; `loadgen` v1; read production-stack router code (§7) | Baseline numbers |
| 3 | Oct 12 → Oct 18 | `LeastLoaded` + health checker; **open upstream PR #1** | E1 partial |
| 4 | Oct 19 → Oct 25 | `PrefixAware` + consistent-hash ring | E1 complete |
| 5 | Oct 26 → Nov 1 | Failover, circuit breaker, load shedding, fault switches | E2 + E3 |
| 6 | Nov 2 → Nov 8 | Lab calibration run (§6.1); README + results; résumé bullet; portfolio entry | Ship |

### 6.1 Lab calibration (week 6)
- **Hardware:** lab Dell XPS 8950, RTX 3060 12 GB, via AnyDesk, run as a written session file
  (same pattern as the SQLAlchemy "Rounds").
- **Run:** one vLLM server, small model (Qwen2.5-0.5B- or 1.5B-Instruct), prefix caching on.
  Measure TTFT on cache miss vs hit at 3–4 prompt lengths, and inter-token latency at 1, 4 and
  8 concurrent requests.
- **Use:** fit `simserver` timing parameters; README states the calibration source.
- **Optional, if time remains:** two vLLM servers sharing the one GPU, small-scale E1 on real
  hardware, labelled "shared GPU — validates cache-hit behaviour, not production latency".
- **Prerequisite:** Viraj confirms lab-GPU use for a personal portfolio project is acceptable.

---

## 7. Upstream contribution — vllm-project/production-stack

Evidence (GitHub, 2026-09-27): 21 PRs merged in the prior 60 days from 15 distinct authors,
median ~5.6 days open-to-merge among merged PRs; 121 open PRs; 3 `good first issue` labels, all
from 2025.

| When | Action | ~Hours |
|---|---|---|
| Week 2 | Clone; read CONTRIBUTING (sign-off, AI-use policy); run router tests; read routing code | 3 |
| Weeks 2–3 | Choose PR #1; comment on the issue first; confirm no open PR covers it | 1 |
| Week 3 | Open PR #1 | 3 |
| Weeks 4–10 | Respond to review within 48 h | 1/week |
| Weeks 5–6 (bonus) | PR #2 grounded in a gateway finding (e.g. remapping on backend removal) | 3–4 |

**PR #1 selection:** < ~100 lines, one concern, has a test — a missing test, a reproducible bug,
or a docs error actually hit during setup. No features, refactors or design-discussion items.

**Rules:** Viraj writes the PR code; follow the project's AI-disclosure policy; résumé mentions
only merged PRs; after 2 weeks of silence, one polite ping, then leave it.

**Fallback repository:** LiteLLM (Python, active; its router covers load balancing and
fallbacks).

---

## 8. Scope-cut order (when the week slips)

Proposed while writing this spec (not discussed in brainstorming — confirm in review). Per the shared stall rules, cut in this order:
1. Optional shared-GPU E1 run
2. Upstream PR #2
3. E3 (load shedding stays implemented; the formal experiment is dropped)
4. `LeastLoaded` health-metric polling (keep in-flight count only)

Never cut: E1, E2, tests, the week-6 calibration, PR #1.

## 9. Repository layout

Public repo `llm-inference-gateway`:

```
simserver/        simulated vLLM backend + config
gateway/          router policies, proxy, health, failover, shedding
loadgen/          workload generator + recorder
experiments/      E1–E3: pre-registered plan files, results, run scripts
lab/              calibration session files and raw results
tests/
docs/superpowers/specs/   this spec
README.md
```
