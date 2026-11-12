# Predict Roblox's Up-and-Coming top 10 — entry

A program that reads hourly snapshots of Roblox's Up-and-Coming chart
(`/input/history.ndjson` + `/input/task.json`) and writes one forecast
to `/output/forecast.json`: per-place probabilities for the top 10 and a
`top10` guess in order. Python 3.13, standard library only, deterministic,
no network, no model calls. Runs in ~0.1 s.

## Method

Rank momentum is the signal. In the archive, log-playing growth does not
predict the next rank change (corr −0.02), rank changes are mildly
autocorrelated (+0.17), and a game's recent rank trend persists across
the whole 72 h window (decaying the trend hurts badly).

1. For each of the 50 best-ranked games at the close, estimate rank
   velocity (ranks/hour) two ways and average the two forecasts:
   endpoint slope over the last 24 h, and least-squares slope over the
   last 36 h, both shrunk by 0.5.
2. At each hour h = 1..72 (step 3) the game's rank is a Gaussian kernel
   centred at `last_rank + velocity * h`, sigma 4.0. Kernels are averaged
   over the horizon and each place's column is normalised to 0.8.
3. The remaining 0.2 per place is left unnamed, so it is shared by every
   game on a future chart not named at that place — this covers brand-new
   entrants and far climbers.
4. Age-out: when a game's `created` date is known (house-recorder
   snapshots carry it in `details`), it contributes no mass at hours
   after `created + 28 days`, since the chart only lists games published
   in the last 28 days.

## Local backtest (exact scorer: log score, floor 0.0001)

72 h window after each cutoff snapshot of the kit archive:

| model | 5 folds (cuts 14/18/22/26/30) | 11 folds (cuts 10–32) |
|---|---|---|
| house baseline (starter) | −46.56 | −47.41 |
| this entry (`main.py`, end-to-end) | **−35.77** | **−36.49** |

The real practice window is easier than these folds (its house baseline
is −43.51 vs −47.41 here), and the house-model bar to beat is −37.2, so
this entry is competitive for a paid place if the pattern holds into
the final window (2026-11-13 06:00 → 2026-11-16 06:00 UTC).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=../kit/roblox-up-and-coming-forecast-01-kit/input \
  MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the entry (7,485 bytes)
- `LICENSE` — MIT
