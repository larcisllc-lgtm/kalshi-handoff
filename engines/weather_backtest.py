#!/usr/bin/env python3
"""
Backtest de los cortes (umbrales) de la REGLA DURA del motor de clima, y
chequeo de CONFIABILIDAD del ancho predictivo re-escalado.

Preguntas que responde:
  1. Cuando el desacuerdo entre modelos sube, ¿el pronóstico pierde skill?
     (si no, el corte por dispersión no sirve)
  2. ¿La distribución predictiva N(mediana corregida, sd calibrada) que usa el
     motor está calibrada? (--reliability: cobertura predicha vs real a ±1/±2/±3°)

Observado = máxima OFICIAL del reporte CLI (IEM), la misma con la que liquida
Kalshi — no el reanálisis del píxel.

LIMITACIÓN (medida, no supuesta): ensemble-api solo conserva ~3 días de
histórico, así que NO se puede reconstruir el ensemble de 119 miembros hacia
atrás. La dispersión ENTRE MODELOS (historical-forecast-api, 4 modelos) es el
proxy: mide la misma señal ("¿están de acuerdo?") en escala menor. Los umbrales
del motor están en escala de ensemble, traducidos por percentil.

Uso:
  python3 weather_backtest.py [dias] [--reliability]
"""
import sys, os, json, urllib.request, statistics, datetime, math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_engine import STATIONS, CALIBRATED_ON, phi  # bias/sd vigentes del motor

UA = {"User-Agent": "kalshi-weather-analyst larcisllc@gmail.com"}

MODELS = "ecmwf_ifs025,icon_seamless,gem_global,meteofrance_arpege_world"


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180))


def per_model_forecast(lat, lon, d0, d1):
    url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}"
           f"&daily=temperature_2m_max&models={MODELS}"
           f"&temperature_unit=fahrenheit&timezone=auto")
    d = get(url)["daily"]
    keys = [k for k in d if k.startswith("temperature_2m_max_")]
    out = {}
    for i, day in enumerate(d["time"]):
        vals = [d[k][i] for k in keys if d[k][i] is not None]
        if len(vals) >= 3:
            out[day] = vals
    return out


def observed_cli(station, years):
    """Máxima oficial del reporte CLI (IEM) — la que liquida Kalshi."""
    out = {}
    for y in years:
        d = get(f"https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={y}")
        for row in d.get("results", []):
            if row.get("high") is not None:
                out[row["valid"]] = row["high"]
    return out


def collect(days=60, end_offset=2):
    today = datetime.date.today()
    d1 = today - datetime.timedelta(days=end_offset)
    d0 = d1 - datetime.timedelta(days=days - 1)
    years = sorted({d0.year, d1.year})
    rows = []
    for city, cfg in STATIONS.items():
        try:
            fc = per_model_forecast(cfg["lat"], cfg["lon"], d0.isoformat(), d1.isoformat())
            ob = observed_cli(cfg["station"], years)
        except Exception as e:
            print(f"{city}: ERROR {e}", file=sys.stderr)
            continue
        for day, vals in fc.items():
            real = ob.get(day)
            if real is None or not (d0.isoformat() <= day <= d1.isoformat()):
                continue
            corr = sorted(v - cfg["bias"] for v in vals)
            med = statistics.median(corr)
            spread = corr[-1] - corr[0]
            rows.append({
                "city": city, "day": day, "spread": spread,
                "err": abs(med - real), "signed": med - real,
                "sd_ratio": statistics.pstdev(corr) / cfg["sd"],
                "median": med, "real": real, "sd_calib": cfg["sd"],
            })
    return rows, d0, d1


def band(rows, label):
    if not rows:
        print(f"  {label:28s}  (vacío)")
        return
    err = [r["err"] for r in rows]
    print(f"  {label:28s} n={len(rows):4d}  MAE={statistics.mean(err):5.2f}°  "
          f"p90_err={sorted(err)[int(.9 * (len(err) - 1))]:5.2f}°  "
          f"%err>3°={100 * sum(1 for e in err if e > 3) / len(err):5.1f}%")


def reliability(rows):
    """¿N(mediana corregida, sd calibrada) está calibrada contra el CLI real?

    Para cada banda ±k°: cobertura predicha = media de [Φ(k/sd)−Φ(−k/sd)],
    cobertura real = fracción de días con |mediana−real| ≤ k. Si predicha ≈
    real, el ancho predictivo del motor es honesto; predicha >> real = motor
    sobreconfiado; predicha << real = demasiado plano.
    """
    print("\n== Confiabilidad del ancho predictivo N(mediana, sd calibrada) ==")
    print(f"  {'banda':8s} {'predicho':>9s} {'real':>7s} {'gap':>7s}")
    for k in (1, 2, 3):
        pred = statistics.mean(2 * phi(k / r["sd_calib"]) - 1 for r in rows)
        real = sum(1 for r in rows if r["err"] <= k) / len(rows)
        print(f"  ±{k}°      {pred * 100:8.1f}% {real * 100:6.1f}% {100 * (real - pred):+6.1f}pp")
    print("  (gap positivo = el motor es conservador; negativo = sobreconfiado)")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
    rows, d0, d1 = collect(days)
    print(f"Backtest {days}d ({d0}..{d1}) — {len(rows)} días-ciudad, observado = CLI "
          f"oficial (IEM), calibración del motor: {CALIBRATED_ON}\n")

    sp = sorted(r["spread"] for r in rows)
    qs = {p: sp[int(p / 100 * (len(sp) - 1))] for p in [50, 75, 90, 95, 99]}
    print("Distribución del spread inter-modelo:")
    print("  " + "  ".join(f"p{p}={v:.1f}°" for p, v in qs.items()))
    print(f"  media={statistics.mean(sp):.2f}°  max={sp[-1]:.1f}°\n")

    print("== ¿El error crece con el spread? (si no, el corte por spread no sirve) ==")
    for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 99)]:
        band([r for r in rows if lo <= r["spread"] < hi], f"spread {lo}-{hi}°")

    print("\n== Correlación spread vs error ==")
    xs = [r["spread"] for r in rows]
    ys = [r["err"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    print(f"  Pearson r = {num / den:+.3f}" if den else "  n/a")

    print("\n== Calidad al descartar el top-X% de spread ==")
    for p in [100, 95, 90, 85, 80, 75]:
        thr = sp[int(p / 100 * (len(sp) - 1))]
        keep = [r for r in rows if r["spread"] <= thr]
        band(keep, f"quedarse con p<={p} (thr {thr:.1f}°)")

    print("\n== Por ciudad ==")
    by = defaultdict(list)
    for r in rows:
        by[r["city"]].append(r)
    for c in STATIONS:
        rs = by.get(c, [])
        if not rs:
            continue
        sps = [r["spread"] for r in rs]
        band(rs, f"{c} (spread medio {statistics.mean(sps):.1f}°)")

    if "--reliability" in sys.argv:
        reliability(rows)
