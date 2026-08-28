#!/usr/bin/env python3
"""Calibración del modelo de mlb_engine.py contra resultados reales.

La pregunta que responde: cuando el motor dice 70%, ¿gana el 70% de las veces?
Eso es lo único que decide si los picks valen algo, y es independiente de los precios.

Por qué existe: el motor emite ~331 picks con edge ≥5¢ por jornada contra un mercado
cuyo vig mediano es de 1-2¢ (medido 2026-08-05). Encontrar 331 oportunidades contra un
libro tan barato no es creíble — la explicación simple es que el modelo sobreestima
probabilidades. Esto lo mide en vez de suponerlo.

Cómo evita mirar el futuro (es donde este tipo de backtest se rompe callado):
  · Las stats de cada partido se piden con `endDate` = el DÍA ANTERIOR al partido, así
    que la ventana de 30d es la que el motor habría visto esa mañana. Verificado que
    `byDateRange` corta de verdad por fecha.
  · Los abridores salen de `hydrate=probablePitcher` del schedule histórico: el que se
    anunció, no el que terminó lanzando.
  · El ERA del abridor también se pide con corte el día anterior.

Contaminación conocida y NO corregible con esta API: el ratio de bullpen es de TEMPORADA
(statsapi no cruza sitCodes=rp con byDateRange), así que al backtestear un partido de
hace 30 días ese ratio incluye juegos posteriores. Es leve y acotado — se reporta en vez
de esconderse.

Qué NO mide: el edge contra el precio de Kalshi. La API pública solo sirve mercados
abiertos, así que los precios de cierre de hace 30 días no están disponibles. Esto
califica el MODELO; el edge se mide aparte cuando haya muestra en vivo.

Uso:
  python3 mlb_calibrate.py            # 30 días (default), Poisson independiente
  python3 mlb_calibrate.py 14         # ventana corta
  python3 mlb_calibrate.py 30 --csv   # vuelca las predicciones crudas
  python3 mlb_calibrate.py 30 --rho 0.02        # con corrección de covarianza (joint_matrix)
  python3 mlb_calibrate.py 30 --nbinom-r 8      # binomial negativa (sobredispersión)
  python3 mlb_calibrate.py 30 --rho 0.02 --nbinom-r 8   # ambas corrECCIONES a la vez

--rho: covarianza entre equipos (Dixon-Coles adaptado). Medido 2026-08-28 con rho
0.02/0.05/0.10/0.20 sobre 14 días: NO cerró el gap (SPREAD empeoró, TOTAL mejoró marginal).
Descartado como causa dominante.

--nbinom-r: sobredispersión DENTRO de cada equipo — la Poisson individual es más angosta
que la varianza real de carreras del equipo. r más chico = más ancho. r muy grande
converge a Poisson pura. Sin calibrar todavía; barrer valores (ej. 4, 8, 15, 30) y
comparar el gap por mercado antes de fijar NBINOM_R en mlb_engine.py.
"""
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Se IMPORTA el motor en vez de reimplementarlo: si el modelo cambia, la calibración
# cambia con él. Una copia de las fórmulas acá se desincroniza en la primera edición.
import mlb_engine as E

PT = E.PT


def get(url):
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "kalshi-mlb-calibrate/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def statsapi(path):
    return get("https://statsapi.mlb.com/api/v1" + path)


# ---------- caches: una llamada por (equipo, fecha de corte), no por partido ----------
_TEAM_CACHE = {}
_ERA_CACHE = {}
_BULLPEN_CACHE = {}


def team_stats(team_id, grupo, start, end):
    k = (team_id, grupo, start, end)
    if k in _TEAM_CACHE:
        return _TEAM_CACHE[k]
    try:
        d = statsapi(f"/teams/{team_id}/stats?stats=byDateRange&group={grupo}"
                     f"&startDate={start}&endDate={end}&season={end[:4]}")
        out = d["stats"][0]["splits"][0]["stat"]
    except Exception:
        out = None
    _TEAM_CACHE[k] = out
    return out


def starter_era(pid, start, end):
    k = (pid, start, end)
    if k in _ERA_CACHE:
        return _ERA_CACHE[k]
    try:
        d = statsapi(f"/people/{pid}/stats?stats=byDateRange&group=pitching"
                     f"&startDate={start}&endDate={end}&season={end[:4]}")
        out = float(d["stats"][0]["splits"][0]["stat"]["era"])
    except Exception:
        out = None
    _ERA_CACHE[k] = out
    return out


def bullpen_era(team_id, season):
    """ERA de relevo de temporada. Contaminado por definición — ver docstring del módulo."""
    if team_id in _BULLPEN_CACHE:
        return _BULLPEN_CACHE[team_id]
    try:
        d = statsapi(f"/teams/{team_id}/stats?stats=statSplits&sitCodes=rp"
                     f"&group=pitching&season={season}")
        out = float(d["stats"][0]["splits"][0]["stat"]["era"])
    except Exception:
        out = None
    _BULLPEN_CACHE[team_id] = out
    return out


def ratio_acotado(valor, referencia):
    """Mismo encogimiento (50% hacia 1.0) y cotas [0.75, 1.30] que usa el motor."""
    if valor is None or referencia is None or referencia <= 0:
        return 1.0
    raw = valor / max(0.1, referencia)
    return max(0.75, min(1.3, 1 + (raw - 1) * 0.5))


def predecir(g, fecha):
    """Reproduce el modelo del motor para un partido, con datos previos al partido.

    Devuelve lista de (mercado, prob_predicha, acerto) o None si falta algo.
    """
    corte = (datetime.strptime(fecha, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    inicio = (datetime.strptime(corte, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    season = int(fecha[:4])

    away, home = g["teams"]["away"], g["teams"]["home"]
    if away.get("score") is None or home.get("score") is None:
        return None
    ra_final, rh_final = int(away["score"]), int(home["score"])

    aid, hid = away["team"]["id"], home["team"]["id"]
    h_a = team_stats(aid, "hitting", inicio, corte)
    h_h = team_stats(hid, "hitting", inicio, corte)
    p_a = team_stats(aid, "pitching", inicio, corte)
    p_h = team_stats(hid, "pitching", inicio, corte)
    if not all((h_a, h_h, p_a, p_h)):
        return None

    def pg(st, key):
        try:
            return float(st.get(key, 0)) / max(1, int(st.get("gamesPlayed", 1)))
        except (TypeError, ValueError):
            return None

    rs_a, ra_a = pg(h_a, "runs"), pg(p_a, "runs")
    rs_h, ra_h = pg(h_h, "runs"), pg(p_h, "runs")
    if None in (rs_a, ra_a, rs_h, ra_h):
        return None

    def staff_era(st):
        try:
            return float(st.get("era", 4.0))
        except (TypeError, ValueError):
            return 4.0

    # abridores anunciados ese día (no el que terminó lanzando)
    def r_starter(team_side, team_pit):
        pp = team_side.get("probablePitcher")
        if not pp:
            return 1.0
        era = starter_era(pp["id"], inicio, corte)
        return ratio_acotado(era, staff_era(team_pit))

    r_a = r_starter(away, p_a)
    r_h = r_starter(home, p_h)
    b_a = ratio_acotado(bullpen_era(aid, season), staff_era(p_a))
    b_h = ratio_acotado(bullpen_era(hid, season), staff_era(p_h))

    venue = g.get("venue", {}).get("name", "?")
    pf = E.PARK.get(venue, 100)

    mu_a = ((rs_a + ra_h) / 2) * (pf / 100) * (0.6 * r_h + 0.4 * b_h)
    mu_h = ((rs_h + ra_a) / 2) * (pf / 100) * (0.6 * r_a + 0.4 * b_a)
    pa = E.carreras_dist(mu_a, r=NBINOM_R)
    ph = E.carreras_dist(mu_h, r=NBINOM_R)

    total_real = ra_final + rh_final
    out = []

    # ML — el motor usa pythagenpat para el ganador (no depende de la corrección de
    # covarianza: pythagenpat no pasa por pa/ph, así que RHO_MLB no le toca nada)
    win_a = E.pythagenpat(mu_a, mu_h)
    out.append(("ML", win_a, ra_final > rh_final))
    out.append(("ML", 1 - win_a, rh_final > ra_final))

    if RHO is None:
        # Poisson independiente — comportamiento histórico (RHO_MLB=0), para comparar.
        for line in (6.5, 7.5, 8.5, 9.5, 10.5):
            out.append(("TOTAL", E.p_total_over(pa, ph, line), total_real > line))
        for line in (2.5, 3.5, 4.5, 5.5):
            out.append(("TT", E.p_team_over(pa, line), ra_final > line))
            out.append(("TT", E.p_team_over(ph, line), rh_final > line))
        for line in (1.5, 2.5, 3.5):
            out.append(("SPREAD", E.p_margin_over(ph, pa, line), (rh_final - ra_final) > line))
            out.append(("SPREAD", E.p_margin_over(pa, ph, line), (ra_final - rh_final) > line))
    else:
        # Grilla conjunta con covarianza (ver joint_matrix en mlb_engine.py).
        grid = E.joint_matrix(pa, ph, mu_a, mu_h, rho=RHO)
        for line in (6.5, 7.5, 8.5, 9.5, 10.5):
            out.append(("TOTAL", E.p_total_over_corr(grid, line), total_real > line))
        for line in (2.5, 3.5, 4.5, 5.5):
            out.append(("TT", E.p_team_over_corr(grid, line, "a"), ra_final > line))
            out.append(("TT", E.p_team_over_corr(grid, line, "h"), rh_final > line))
        for line in (1.5, 2.5, 3.5):
            out.append(("SPREAD", E.p_margin_over_corr(grid, line, "h"),
                        (rh_final - ra_final) > line))
            out.append(("SPREAD", E.p_margin_over_corr(grid, line, "a"),
                        (ra_final - rh_final) > line))

    return out


def bucket(p):
    """Buckets de 10 puntos sobre la CONFIANZA (el lado que el modelo favorece).

    Se pliega en 50-100%: predecir 20% de que pase X es lo mismo que 80% de que no pase.
    Sin plegar, cada predicción se contaría dos veces y la tabla se vuelve simétrica y
    sin información.
    """
    q = p if p >= 0.5 else 1 - p
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        if q < lo + 0.1 or lo == 0.9:
            return lo
    return 0.9


RHO = None         # None = Poisson independiente (default); set por --rho en main()
NBINOM_R = None    # None = Poisson pura (default); set por --nbinom-r en main()


def main():
    global RHO, NBINOM_R
    dias = 30
    volcar = "--csv" in sys.argv
    argv = sys.argv[1:]
    consumidos = set()  # índices de valores pegados a un flag, no cuentan como "días"
    if "--rho" in argv:
        i = argv.index("--rho")
        RHO = float(argv[i + 1])
        consumidos.add(i + 1)
    if "--nbinom-r" in argv:
        i = argv.index("--nbinom-r")
        NBINOM_R = float(argv[i + 1])
        consumidos.add(i + 1)
    for idx, a in enumerate(argv):
        if idx not in consumidos and a.isdigit():
            dias = int(a)

    hoy = datetime.now(PT).date()
    # se termina AYER: los partidos de hoy pueden no haber terminado
    fin = hoy - timedelta(days=1)
    ini = fin - timedelta(days=dias - 1)

    partes_modo = []
    partes_modo.append("Poisson independiente" if RHO is None else f"grilla conjunta rho={RHO}")
    if NBINOM_R is not None:
        partes_modo.append(f"nbinom r={NBINOM_R}")
    modo = " + ".join(partes_modo)
    print(f"=== CALIBRACIÓN MLB — {ini} .. {fin} ({dias} días) — {modo} ===")
    print("Pregunta: cuando el modelo dice X%, ¿pasa el X% de las veces?")
    print("Stats point-in-time (corte el día anterior a cada partido).")
    print("OJO: el ratio de bullpen es de temporada — contaminación leve, ver docstring.\n")

    filas = []
    n_juegos = 0
    d = ini
    while d <= fin:
        fecha = d.strftime("%Y-%m-%d")
        try:
            sched = statsapi(f"/schedule?sportId=1&date={fecha}"
                             f"&hydrate=probablePitcher")
        except Exception as e:
            print(f"  {fecha}: error de schedule ({e})")
            d += timedelta(days=1)
            continue
        juegos = []
        for blk in sched.get("dates", []):
            juegos += blk.get("games", [])
        usados = 0
        for g in juegos:
            if g["status"]["detailedState"] != "Final":
                continue
            r = predecir(g, fecha)
            if not r:
                continue
            usados += 1
            for mercado, p, ok in r:
                filas.append((fecha, mercado, p, ok))
        n_juegos += usados
        print(f"  {fecha}: {usados}/{len(juegos)} partidos modelados", flush=True)
        d += timedelta(days=1)

    if not filas:
        print("\nSIN DATOS — no se pudo modelar ningún partido.")
        return

    print(f"\n{n_juegos} partidos · {len(filas)} predicciones\n")

    # ---------- tabla por bucket ----------
    agg = defaultdict(lambda: [0, 0, 0.0])
    for _, _, p, ok in filas:
        b = bucket(p)
        q = p if p >= 0.5 else 1 - p
        gano = ok if p >= 0.5 else (not ok)
        agg[b][0] += 1
        agg[b][1] += 1 if gano else 0
        agg[b][2] += q

    print("CALIBRACIÓN GLOBAL (plegada al lado que el modelo favorece)")
    print(f"{'bucket':>10s} {'n':>6s} {'predicho':>9s} {'real':>7s} {'gap':>8s}")
    for b in sorted(agg):
        n, w, sp = agg[b]
        if n == 0:
            continue
        pred, real = sp / n * 100, w / n * 100
        marca = "  <<<" if abs(pred - real) > 5 and n >= 30 else ""
        print(f"{b*100:>7.0f}-{b*100+10:<3.0f} {n:>6d} {pred:>8.1f}% {real:>6.1f}% "
              f"{real-pred:>+7.1f}pp{marca}")

    # ---------- por mercado ----------
    print("\nPOR MERCADO")
    print(f"{'mercado':>10s} {'n':>6s} {'predicho':>9s} {'real':>7s} {'gap':>8s}")
    porm = defaultdict(lambda: [0, 0, 0.0])
    for _, mercado, p, ok in filas:
        q = p if p >= 0.5 else 1 - p
        gano = ok if p >= 0.5 else (not ok)
        porm[mercado][0] += 1
        porm[mercado][1] += 1 if gano else 0
        porm[mercado][2] += q
    for m in sorted(porm):
        n, w, sp = porm[m]
        pred, real = sp / n * 100, w / n * 100
        marca = "  <<<" if abs(pred - real) > 5 and n >= 30 else ""
        print(f"{m:>10s} {n:>6d} {pred:>8.1f}% {real:>6.1f}% {real-pred:>+7.1f}pp{marca}")

    # ---------- extremos: donde se acumulan los picks caros ----------
    print("\nEXTREMOS (confianza ≥85%) — donde se acumulan los picks de 90¢+")
    for m in sorted(porm):
        sub = [(p, ok) for _, mm, p, ok in filas if mm == m
               and (p >= 0.85 or p <= 0.15)]
        if len(sub) < 20:
            continue
        n = len(sub)
        pred = sum(p if p >= 0.5 else 1 - p for p, _ in sub) / n * 100
        real = sum(1 for p, ok in sub if (ok if p >= 0.5 else not ok)) / n * 100
        print(f"{m:>10s} {n:>6d} {pred:>8.1f}% {real:>6.1f}% {real-pred:>+7.1f}pp")

    print("\nLectura: gap negativo = el modelo SOBREESTIMA (dice más de lo que pasa).")
    print("Un gap de −10pp en un bucket con n≥100 explica edges inflados sin bug de precio.")

    if volcar:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibracion_mlb.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fecha", "mercado", "prob_predicha", "acerto"])
            w.writerows(filas)
        print(f"\nCSV: {ruta}")


if __name__ == "__main__":
    main()
