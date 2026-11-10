# Distributed LLM-Supervised HVAC Control System

A simulation-backed, multi-agent HVAC control system for a 5-zone office floor in which
**Large Language Models act as the supervisory layer** (one agent per zone plus a building
coordinator), while **deterministic physics and safety layers keep the system safe and
correct** regardless of what the LLM says.

```
                      ┌────────────────────────────────┐
                      │   CoordinatorAgent  (LLM)      │  demand limit, DR events,
                      │   peak shaving, power budgets   │  priority allocation
                      └──────────────┬─────────────────┘
                   budgets ▲         │  zone reports (need, hold power, priority)
                           │         ▼
              ┌────────────────────────────────────────────┐
              │                 MessageBus                 │  pub/sub (MQTT/BACnet stand-in)
              └───┬──────────┬──────────┬──────────┬────────┘
                  ▼          ▼          ▼          ▼
             ZoneAgent  ZoneAgent  ZoneAgent  ZoneAgent …   one per zone: observe → LLM → Safety Shield
                  │          │          │          │
                  ▼          ▼          ▼          ▼
               PI loop    PI loop    PI loop    PI loop      local terminal-unit controller (deadband PI)
                  └──────────┴────┬─────┴──────────┘
                                  ▼
                   Multi-zone RC building model (thermal_model.py)
```

---

## 1. Problem statement

HVAC is roughly 40–50 % of commercial-building energy. Conventional building management
systems (BMS) run **static schedules and fixed setpoints**: they do not know a meeting starts
in 40 minutes, that electricity triples in price at 14:00, or that the utility has asked the
building to cap demand this afternoon. Model-predictive control and reinforcement learning can
exploit such context but need accurate models or long training and are hard for operators to
trust or interrogate.

The project question: **can LLM agents, which natively reason over schedules, tariffs and
operator intent in language, run a *distributed* HVAC supervisory layer — and can that be made
safe and verifiable?**

## 2. What the research says (and how the design follows it)

| Finding from the literature | Design decision in this project |
|---|---|
| LLMs encode operational knowledge and context but are unreliable at tight numerical closed-loop optimisation; the recommended pattern is **hierarchical** — LLM supervisory, classical control at the actuator (arXiv 2603.26050; critical review arXiv 2609.05314). | LLM agents choose heating/cooling **setpoints** and a **mode** every 15 min. A **PI controller** on each terminal unit does the actual actuation at 60 s resolution. The LLM never touches the actuator. |
| Minimum deployment package: **hard-coded safety constraints outside the LLM**, a **fallback controller**, **audit logging** (2609.05314, Table 7). | `safety.py` is a pure-Python Safety Shield; every action passes through it. Garbage → rule-based fallback. `AuditLog` writes every decision, its reasoning and shield verdict to JSONL. |
| ThermoLLM (arXiv 2606.22911): structured action = one heating + one cooling setpoint per zone, validated against ranges before each 15-min simulator step. | Same action schema (`heating_setpoint_C`, `cooling_setpoint_C`), same 15-min step, range + deadband + rate-limit validation. |
| LLM-controlled HVAC needs no training phase, unlike DQN/RL (Ahn et al. 2023, *Buildings*). | No training loop anywhere; the mock backend and the Claude backend are drop-in. |
| Field study "Office-in-the-Loop" (Sawada et al. 2025, Cambridge Data-Centric Engineering): prompting generative AI with occupancy, weather and feedback yields setpoints that cut energy while improving comfort. | Prompts embed occupancy forecasts, outdoor forecast, tariff forecast and DR notices. |
| Distributed multi-agent architecture: **zone agents on local data + coordination layer + aggregator** (Astudillo & Koch 2025, arXiv 2502.15663; Michailidis et al. 2024 review, *Energies*). | `ZoneAgent` sees only its own sensors and bus messages; `CoordinatorAgent` only publishes budgets and never commands setpoints; agents run concurrently (asyncio). If the coordinator dies, zones continue with their last budget. |
| Guardrail layering: the LLM should steer indirectly; constraint satisfaction must be **guaranteed by construction**, with failures degrading to "less optimal", never "unsafe" (arXiv 2604.21316; "Safety Drift" arXiv 2607.18366). | Coordinator shield always enforces Σ budgets ≤ limit; zone shield clips to hard limits and comfort band; the plant itself clips HVAC power to the budget. |

Full notes: [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

## 3. How the system works, step by step (every 15 minutes)

1. **Coordinator** reads last step's zone reports (how much power each zone wants, how much it
   needs just to *hold* temperature, and whether the request is `comfort` or `discretionary`).
   It knows the contracted demand limit, any active/upcoming demand-response (DR) event and the
   **running billing-period peak**. It asks the LLM for an *effective limit* and per-zone
   budgets, the shield validates them, and the budgets are published on the bus. If a DR event
   is less than 2 h away it broadcasts a notice so zones can pre-condition.
2. **Zone agents** (concurrently) build an observation: temperature, current setpoints,
   occupants now and forecast for the next hour, comfort band, hard limits, outdoor forecast,
   tariff now and next 4 h, their power budget (and whether it is binding), the terminal unit's
   current regime and estimated steady-state load. The prompt contains a narrative *and* a
   `<state>` JSON block. The LLM returns JSON: heating setpoint, cooling setpoint, mode,
   request priority, confidence, reasoning.
3. **Safety Shield** parses, validates and (if necessary) clips or replaces the action, logging
   every intervention. The validated setpoints are written to the terminal unit.
4. **Physics** integrates 15 min with the PI loops active; the coordinator's budget caps each
   unit's electrical power.
5. **Metrics** record energy, cost, peak, discomfort and limit breaches.

### Where the "intelligence" shows up
Look in `results/audit_log.jsonl`; representative reasoning strings produced in a run:

* *"Unoccupied now but occupancy expected within 1 h: bring the zone just inside the comfort band
  for arrival (occupants' own heat does the rest)."*
* *"Occupied, currently cooling, and ON-PEAK starts soon: pre-cool toward the lower comfort bound
  so the zone can coast through it."* (`request_priority: discretionary`)
* *"Occupied during ON-PEAK: ride the energy-saving edge of the comfort band."*
* Coordinator: *"Capped at 9 500 W (running peak 9 500 W) so discretionary pre-conditioning does
  not set a new demand-charge peak; comfort requests untouched."*

## 4. Repository layout

```
hvac_llm_control/
├── run_simulation.py          CLI: runs baselines + LLM system, prints KPIs, saves figures/logs
├── requirements.txt
├── hvac_llm/
│   ├── config.py              building, zones, tariff, simulation settings (default 5-zone office)
│   ├── thermal_model.py       RC physics + deadband PI with anti-windup
│   ├── environment.py         weather presets, occupancy, tariff, DR events, forecasts
│   ├── llm_backend.py         AnthropicBackend (Claude API) / MockLLMBackend (offline, deterministic)
│   ├── prompts.py             system prompts + user-prompt builders
│   ├── safety.py              Safety Shield for zone actions and coordinator budgets
│   ├── agents.py              MessageBus, AuditLog, ZoneAgent, CoordinatorAgent
│   ├── baseline.py            FixedSetpoint and ScheduledSetpoint conventional controllers
│   ├── metrics.py             KPI definitions
│   └── simulation.py          closed-loop harness
├── tests/test_core.py         18 unit/integration tests
├── docs/RESEARCH_NOTES.md     literature summary and design rationale
└── results*/                  kpis.json, audit_log.jsonl, fig_*.png per scenario
```

## 5. Running it

```bash
pip install -r requirements.txt
pytest -q                               # 18 tests: physics, shield, protocol, full loop w/ injected LLM failures

python run_simulation.py                # 48 h, hot summer, offline mock LLM (no API key needed)
python run_simulation.py --verbose      # print a status line every 2 simulated hours
python run_simulation.py --hours 168    # a full week incl. weekend
python run_simulation.py --profile mild | cold_winter
python run_simulation.py --no-dr        # disable the demand-response event

# With a real Claude model (≈1 200 calls for 48 h; set HVAC_LLM_MODEL to pick the model):
export ANTHROPIC_API_KEY=...
python run_simulation.py --backend anthropic --hours 12
```

Outputs in `results/`: `kpis.json`, `audit_log.jsonl`, `fig_zone_temperatures.png`,
`fig_total_power.png`, `fig_kpis.png`.

## 6. Results

Baselines: **fixed setpoint** (21/23 °C, 24/7 — legacy BMS) and **scheduled setpoint** (23/25 °C
in scheduled hours, 19/29 °C setback — a good conventional BMS; the fair comparison). All three
controllers share the identical plant, PI loop, weather, occupancy and tariff. Cost = TOU energy
cost + $12/kW demand charge + $1.5/kWh penalty for energy above the demand limit.

| Scenario (vs. scheduled BMS) | Energy | Cost | Peak kW | Discomfort °C·h (LLM / BMS) | Limit breaches (LLM / BMS / fixed) |
|---|---|---|---|---|---|
| Hot summer, 48 h, DR cap 9 kW 15–17 h | −0.1 % | **−4.5 %** | **−5.4 %** | 0.75 / 0.47 | 0 / 0 / 7 |
| Hot summer, 48 h, no DR | −0.2 % | **−4.5 %** | **−5.4 %** | **0.00** / 0.47 | 0 / 0 / 0 |
| Hot summer, full week | −0.1 % | **−3.4 %** | **−5.5 %** | **0.81** / 1.24 | 0 / 0 / 16 |
| Mild climate, 48 h | +1.0 % | −1.8 % | −1.7 % | 0.03 / 0.04 | 0 / 0 / 0 |
| Cold winter, 48 h | +0.2 % | +10.1 % | +14.7 % | **0.51** / 4.54 | 0 / 0 / 0 |

In the main run the LLM layer made 1 152 decisions with **0 fallbacks and 0 shield clips**; in
the test suite a backend that returns garbage every 5th call is survived with hard limits never
violated.

### Reading the results honestly
* **Energy is roughly flat; the savings are in cost and peak.** Against a well-tuned deadband BMS
  there is little night-time waste left to remove. The LLM's gains come from *when* energy is
  used: pre-cooling before ON-PEAK, riding the warm edge of the band at peak prices, demand
  coordination and peak shaving. This is consistent with the 2026 critical review's conclusion
  that LLMs are best used as a semantic/supervisory layer.
* **Comfort is better in most scenarios** because agents condition zones *before* people arrive
  rather than when the schedule flips.
* **Winter is a trade-off, not a bug.** The BMS lets occupants arrive to a 20.2 °C room (below the
  22 °C band); the LLM pre-heats so they arrive to 21.9 °C, which cuts discomfort by 89 % but adds
  a morning peak and therefore a demand charge. An operator who prefers cost can change one line
  in the zone prompt (allow the band to be reached 30 min after arrival).
* **The first version reported −25 % energy.** That was a weak-baseline artefact (a single-setpoint
  baseline that heated empty rooms at night). Fixing the plant to dual setpoints strengthened the
  baseline and shrank the gap; the numbers above are the credible ones.

### Bugs the simulation caught during development (worth knowing when explaining the design)
1. **Integrator windup during coasting** — a cool-only unit whose setpoint sits above the room
   temperature "wants to heat", winds its integral up, and overshoots by ~2 °C when cooling is
   finally needed. Fixed with conditional integration and clamping.
2. **Resetting the PI integral on setpoint change** destroyed the controller's load estimate.
3. **Stale requests let the coordinator "unconstrain" zones** → demand limit exceeded. The shield
   now always enforces Σ budgets ≤ limit, and zones never request less than their hold power.
4. **Single setpoint + auto mode heated empty rooms to 29 °C.** Replaced by heating/cooling
   setpoints with a deadband, as in ThermoLLM and real BMS.
5. **Simultaneous pre-cooling created a new demand-charge peak**, erasing tariff savings. Fixed by
   peak shaving in the coordinator and a `comfort`/`discretionary` priority on zone requests.

## 7. Limitations and next steps
* The plant is an RC model, not EnergyPlus. Coupling to Sinergym/BOPTEST is the natural next step.
* Occupancy forecasts are schedule-based; a real deployment would use sensor-driven forecasts.
* The mock backend encodes policies a well-prompted LLM converges to in the cited studies, which
  makes runs reproducible; results with a live Claude model will vary and should be re-measured.
* Agents hold no memory across steps beyond the audit log; adding a rolling decision memory (as in
  recent agentic BMS prototypes) could reduce oscillation.
* Humidity, ventilation/CO₂ and equipment cycling constraints are not modelled.
