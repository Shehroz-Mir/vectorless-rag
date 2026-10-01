# Eval run 2026-10-01 18:40

24 answers (24 questions × 1); 23 passed (correct, key facts present, no wrong refusal).

Models: chat `gpt-5.6-sol`, judge `gpt-5.6-sol`, index `gpt-5.6-luna`, vision `gpt-5.6-luna`. Settings: `AGENT_MAX_STEPS=20`, `AGENT_TIMEOUT_S=120.0`, `AGENT_REASONING_SUMMARY=auto`, `VIEW_PAGES_MAX_CALLS=4`, `VIEW_PAGES_MAX_PAGES=3`, `MAX_IMAGE_SETS_IN_CONTEXT=2`, `VIEW_PAGES_IMAGE_DETAIL=high`, `RENDER_DPI=170`.

## Metrics

Averages over the answers each metric applies to; – means it applies to none.

| Metric | All | text (8) | figure (8) | multi_page (3) | cross_document (2) | unanswerable (3) |
|---|---|---|---|---|---|---|
| Page recall | 95% | 100% | 88% | 100% | 100% | – |
| Page precision | 64% | 75% | 59% | 61% | 45% | – |
| Hit@3 | 95% | 100% | 88% | 100% | 100% | – |
| MRR | 0.86 | 0.94 | 0.75 | 0.83 | 1.00 | – |
| Document hit rate | 100% | 100% | 100% | 100% | 100% | – |
| First document was gold | 100% | 100% | 100% | 100% | 100% | – |
| Figure hit rate | 75% | – | 75% | – | – | – |
| Dead ends per question | 0.1 | 0.0 | 0.2 | 0.3 | 0.0 | 0.0 |
| Key facts present | 100% | 100% | 100% | 100% | 100% | – |
| Correctness (judge) | 98% | 100% | 94% | 100% | 100% | 100% |
| Faithfulness | 99% | 100% | 98% | 97% | 100% | 100% |
| Answer relevance | 100% | 100% | 100% | 100% | 100% | 100% |
| Citation precision | 95% | 100% | 88% | 100% | 100% | – |
| Citation recall | 95% | 100% | 88% | 100% | 100% | – |
| Refusal accuracy (unanswerable) | 100% | – | – | – | – | 100% |
| False refusals (answerable) | 0% | 0% | 0% | 0% | 0% | – |
| No answer (run incomplete) | 0% | 0% | 0% | 0% | 0% | 0% |
| Tool calls | 3.7 | 3.0 | 4.0 | 4.3 | 5.0 | 3.3 |
| Pages read | 2.1 | 1.5 | 1.8 | 3.3 | 4.5 | 2.0 |
| Images viewed | 0.5 | 0.0 | 1.2 | 0.7 | 0.0 | 0.0 |
| Model calls | 4.1 | 4.0 | 4.1 | 4.3 | 4.0 | 4.3 |
| Input tokens | 39,054 | 34,535 | 41,486 | 41,273 | 60,873 | 27,858 |
| Output tokens | 360 | 234 | 321 | 694 | 462 | 399 |
| Time (s) | 15.0 | 12.9 | 14.9 | 20.0 | 17.0 | 14.6 |
| Answering cost | $0.163 | $0.143 | $0.172 | $0.179 | $0.253 | $0.119 |

## Questions

| Question | Type | Pass | Answer | Key facts | Page recall | Citation recall | Tools | Pages | Cost |
|---|---|---|---|---|---|---|---|---|---|
| tdi110-shutoff-temperature | text | pass | correct | yes | 100% | 100% | 3 | 1 | $0.113 |
| tdi110-amber-led | text | pass | correct | yes | 100% | 100% | 3 | 2 | $0.113 |
| navio-usb-cable-length | text | pass | correct | yes | 100% | 100% | 3 | 2 | $0.113 |
| navio-face-id | text | pass | correct | yes | 100% | 100% | 3 | 2 | $0.113 |
| iseries-battery-cycle-life | text | pass | correct | yes | 100% | 100% | 3 | 2 | $0.200 |
| iseries-i16-brightness | text | pass | correct | yes | 100% | 100% | 3 | 1 | $0.199 |
| pilot-dwell-time | text | pass | correct | yes | 100% | 100% | 3 | 1 | $0.146 |
| pilot-ipad-version | text | pass | correct | yes | 100% | 100% | 3 | 1 | $0.146 |
| iseries-calibration-results | figure | pass | correct | yes | 0% | 0% | 4 | 1 | $0.213 |
| iseries-feedback-colours | figure | pass | correct | yes | 100% | 100% | 3 | 2 | $0.201 |
| iseries-feedback-selected | figure | pass | correct | yes | 100% | 100% | 4 | 2 | $0.226 |
| tdi110-power-button-position | figure | pass | correct | yes | 100% | 100% | 5 | 4 | $0.156 |
| tdi110-hotspot-location | figure | pass | correct | yes | 100% | 100% | 4 | 1 | $0.126 |
| navio-mini-front-camera | figure | **fail** | partly | yes | 100% | 100% | 4 | 2 | $0.139 |
| pilot-calibration-screen | figure | pass | correct | yes | 100% | 100% | 4 | 1 | $0.159 |
| pilot-track-status-button | figure | pass | correct | yes | 100% | 100% | 4 | 1 | $0.160 |
| iseries-battery-replacement | multi_page | pass | correct | yes | 100% | 100% | 4 | 3 | $0.224 |
| navio-force-restart | multi_page | pass | correct | yes | 100% | 100% | 6 | 4 | $0.188 |
| tdi110-reset-duration | multi_page | pass | correct | yes | 100% | 100% | 3 | 3 | $0.125 |
| tdi110-vs-navio-maxi-weight | cross_document | pass | correct | yes | 100% | 100% | 5 | 4 | $0.190 |
| iseries-vs-pilot-battery-run-time | cross_document | pass | correct | yes | 100% | 100% | 5 | 5 | $0.316 |
| navio-warranty-years | unanswerable | pass | refused | – | – | – | 3 | 2 | $0.115 |
| tdi110-refresh-rate | unanswerable | pass | refused | – | – | – | 4 | 2 | $0.128 |
| navio-mini-brightness | unanswerable | pass | refused | – | – | – | 3 | 2 | $0.116 |

## Failures

1 answers did not pass.

### navio-mini-front-camera (figure)

**Question:** Where is the front camera on the TD Navio Mini, and how is that different from the TD Navio Midi?

**Answer:**
> - **TD Navio Mini:** The front camera is on the **left edge of the front bezel**, beside the upper-left area of the screen.
> - **TD Navio Midi:** The front camera is **centered above the screen**, beneath the handle and between the two speakers.
> 
> So, the Mini’s camera is side-mounted, while the Midi’s is top-center. [1]

**Reference:** On the Mini the front camera is on the left edge of the front frame, halfway down. On the Midi it is at the top centre, below the handle, between the volume buttons.

- Key facts present: yes
- Correctness (partly): The overall side-versus-top-center distinction is correct, but the Mini camera is halfway down the left edge, not by the upper-left screen area.
- Faithfulness (6/7 claims supported): All location claims are supported by the diagram except the characterization of the Mini camera as near the upper-left area.
- Relevance (yes): The answer directly compares the front-camera locations on the Mini and Midi.
- Refusal (refused: no, invents an answer: yes): The answer does not refuse.

## Spot-check these

Passes the judge accepted; read the answer and its reasons.

### tdi110-shutoff-temperature (text)

**Question:** At what temperature does the TD I-110 shut itself off to avoid harm?

> The TD I-110 shuts itself off at **60°C (140°F)** to prevent risk of harm. [1]

- Correctness (correct): The answer matches the reference: the TD I-110 shuts off at 60°C (140°F) to avoid harm.
- Faithfulness (1/1 claims supported): The cited page explicitly states that the device shuts off at 60°C/140°F to prevent risk of harm.
- Relevance (yes): The answer directly provides the requested shutoff temperature.
- Refusal (refused: no, invents an answer: no): The agent gave a specific, evidence-supported answer rather than refusing.

### iseries-calibration-results (figure)

**Question:** In the TD I-Series calibration results example, how many calibration points are rated Great, and where is the point with No data?

> There are **3 calibration points rated Great**. The **No data** point is in the **top-center position** of the 3×3 grid. [1]

- Correctness (correct): The answer matches the reference: three points are rated Great, and the No data point is at the top center.
- Faithfulness (2/2 claims supported): Both claims are directly supported by the calibration-results figure on the cited page.
- Relevance (yes): The answer directly addresses both the requested count and location.
- Refusal (refused: no, invents an answer: no): The agent provided a supported answer rather than refusing.

### iseries-battery-replacement (multi_page)

**Question:** How do I replace the battery in a TD I-Series, and how do I make sure the new battery sits correctly?

> ### Replacing the TD I‑Series battery
> 
> Use only a **Tobii Dynavox TDGG2 battery pack**; installing another type creates an explosion risk. [1]
> 
> 1. Shut down the TD I‑Series completely.
> 2. Disconnect its power cable.
> 3. Place the device **screen-side down** on a protected surface.
> 4. Using a Phillips screwdriver, remove the **five screws** securing the battery cover.
> 5. Remove the cover.
> 6. Pull the battery strap to lift out the old battery. [1]
> 7. Push the new battery fully into the compartment.
> 8. **Check its position:** locate the small battery icon on the battery and the matching icon insid …

- Correctness (correct): The answer matches the reference procedure and correctly explains that the two battery icons must align for proper positioning.
- Faithfulness (12/13 claims supported): All substantive instructions are supported, although the recommendation to use a protected surface is not stated in the cited pages.
- Relevance (yes): The answer directly and completely addresses battery replacement and correct seating.
- Refusal (refused: no, invents an answer: no): The agent provides a supported answer rather than refusing.

### tdi110-vs-navio-maxi-weight (cross_document)

**Question:** Which is heavier, the TD I-110 or the TD Navio Maxi?

> The **TD Navio Maxi is heavier**:
> 
> - **TD Navio Maxi:** 1.54 kg (3.4 lb) [1]
> - **TD I-110:** 1.4 kg (3.1 lb) [2]
> 
> The Navio Maxi is about **0.14 kg (0.3 lb) heavier**.

- Correctness (correct): The answer matches the reference, correctly identifying the TD Navio Maxi as heavier and giving both weights and the accurate difference.
- Faithfulness (4/4 claims supported): All claims, including both weights, the comparison, and the calculated difference, are supported by the cited specifications.
- Relevance (yes): The answer directly identifies which device is heavier and quantifies the difference.
- Refusal (refused: no, invents an answer: no): The agent provided a direct, evidence-based answer rather than refusing.

### navio-warranty-years (unanswerable)

**Question:** How many years of warranty come with the TD Navio?

> The TD Navio manual **does not specify the number of warranty years**. It directs users to Tobii Dynavox’s online warranty terms for the applicable coverage. [1]

- Correctness (correct): The answer matches the reference: the manual gives no warranty duration and directs readers to Tobii Dynavox’s online warranty page.
- Faithfulness (2/2 claims supported): Both claims are directly supported by the cited manual page’s warranty section.
- Relevance (yes): It directly answers whether the manual specifies how many warranty years apply.
- Refusal (refused: yes, invents an answer: no): It appropriately states that the manual does not specify the warranty period and points to the external warranty terms.

## Cost and time

| | Total | Per answer |
|---|---|---|
| Answering | $3.922 | $0.163 |
| Judging | $0.412 | $0.017 |
| All | $4.334 | $0.181 |
| Answering time | 360 s | 15.0 s |
| Whole run | 11.3 min | |

Costs use the full input price for every token (cached input is cheaper), so they are upper bounds.
Building the eval library is a one-off and not included.
