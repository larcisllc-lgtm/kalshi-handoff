#!/usr/bin/env python3
"""Motor MLB para Kalshi — hace TODO el cálculo cuantitativo de forma determinista.

Uso:
  python3 mlb_engine.py               # jornada de hoy (fecha PT)
  python3 mlb_engine.py 2026-07-26    # fecha específica
  python3 mlb_engine.py --equipo NYY  # solo partidos de un equipo
  python3 mlb_engine.py --emit        # + escribe predicciones/, filas PAPER y manifiesto

Qué hace (para que el modelo NO lo haga a mano):
  1. Schedule de MLB (statsapi) con abridores y estado real.
  2. Stats de equipo ÚLTIMOS 30 DÍAS (byDateRange, nunca season) — RS/g, RA/g, ERA staff.
  3. ERA 30d del abridor del día → ajuste al RA/g del equipo (exige muestra mínima).
  4. Park factor del estadio (tabla local abajo).
  5. Modelo Poisson conjunto → prob. de ML, spread, total y team total.
  6. Precios de Kalshi (API pública) de las 4 series, matcheados por partido.
  7. Edge por mercado (mejor lado YES/NO), conviction, sizing ⅛ Kelly y
     ENTRADA/OBJ/STOP + MÁX-PAGABLE. Se vende al take-profit, no a settlement.

Filtros de emisión (todos medidos, ver los bloques de constantes):
  · entrada 50-80¢ — bajo 50¢ el modelo sobreestima al lado barato y pierde plata
  · máx 1 TT y 1 TOTAL por jornada — mercados delgados, no hay salida al TP
  · máx 2 filas por partido — las líneas de un juego son UNA tesis

Salida: texto listo para que el modelo redacte el reporte. El modelo no recalcula nada.
"""
import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")

# Banca, topes, Kelly y umbrales salen de _kalshi_core/config.json (Etapa 2): un solo
# lugar para los 4 motores. Editar ESE archivo cambia el sizing de todo el sistema.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from core import cfg as _cfg

C = _cfg("MLB")
BANKROLL = C.bankroll
MAX_TRADE = C.max_trade
MAX_OPEN = C.max_open      # informativo: el modelo verifica lo abierto en registro.md
KELLY_FRAC = C.kelly_frac
EDGE_MIN = C.edge_min      # centavos: debajo de esto, NO INVIERTE
EDGE_SUSPECT = C.edge_suspect  # edge así de grande = sospechoso, revisar a mano
MAXR = 30                  # soporte de la Poisson

# Combinadas (--combo): ELIMINADAS el 2026-08-09. 81 líneas de código y una sección de
# reporte que en toda la historia del registro no produjeron una sola posición. Además
# eran un camino falso: mostraban "edge estimado" calculado contra un precio inventado
# (producto de asks), no contra el precio real de Kalshi.
PICKS = []             # lo llena fmt_row; lo consume emit_seleccion()
BULLPEN_CACHE = {}     # team_id -> (ratio, nota); una llamada por equipo, no por partido
PRECIOS_TS = None      # ISO del instante exacto de la bajada de precios de Kalshi.
                       # Se congela UNA vez por corrida: los precios se mueven en vivo
                       # (medido 2026-08-05: un SPREAD pasó de +5¢ invertible a +4¢
                       # descartado entre dos corridas separadas por segundos).

# --- Calibración (Platt scaling) ---
# El modelo Poisson sobreestima, y el error crece con la confianza. Medido 2026-08-05 con
# mlb_calibrate.py sobre 357 partidos / 7497 predicciones point-in-time:
#   bucket 70-80% → pasaba el 64.4% (−10.6pp) · 80-90% → 72.2% (−12.6pp)
#   90-100% → 81.6% (−11.3pp) · TT en extremos ≥85% → dice 89.0%, entrega 71.1%
# Causa de fondo (sin arreglar): la Poisson asume independencia entre las carreras de los
# dos equipos y subestima la varianza real del total, así que produce probabilidades
# demasiado seguras. Esto es el PARCHE, no el arreglo — corrige el síntoma, no la causa.
#
#   p_calibrada = sigmoid(a + b * logit(p_cruda))
#
# b<1 aplasta hacia 50%: es exactamente lo que hacía falta. Verificado sobre los mismos
# datos: los gaps por bucket pasan de −12.6pp a ±0.5pp.
#
# Recalibrar corriendo `python3 mlb_calibrate.py 30 --csv` y reajustando estos pares.
CALIBRACION = {
    "ML":     (-0.0000, 0.3533),
    "SPREAD": (-0.3018, 0.5047),
    "TOTAL":  (-0.2714, 0.5155),
    "TT":     (-0.2651, 0.5568),
}
# F5, F5SPREAD, F5TOTAL, RFI y EXTRAS NO tienen curva: el backtest no los cubrió. No se
# les inventa una — se marcan SIN-CALIBRAR y no se dimensionan, porque su sesgo es
# desconocido y la evidencia dice que el sesgo por defecto de este motor es grande.
MERCADOS_SIN_CALIBRAR = ("F5", "F5SPREAD", "F5TOTAL", "RFI", "EXTRAS")

# --- Filtro mecánico para --emit ---
# Es el filtro que hasta ahora aplicaba el agente a mano, sin dejar rastro de por qué
# 212 candidatos limpios se volvían 3-7 picks. Aquí es determinista y auditable.
EMIT_MAX_POR_PARTIDO = C.max_por_partido  # filas de un partido = UNA tesis (config)
EMIT_ASK_MAX = 80          # sobre 80¢ el +22% no cabe (POCO-RECORRIDO)
EMIT_FLAGS_MALOS = ("SOSPECHOSO", "SIN-LIQUIDEZ", "SPREAD-ANCHO", "LIQUIDEZ-BAJA",
                    "CONF-BAJA", "POCO-RECORRIDO")

# --- Piso de entrada (medido 2026-08-09 sobre los 37 picks del 08-03..08-08) ---
# Es el hallazgo más fuerte de la semana y no depende del mercado ni del edge:
#   entrada <50¢ → 18 picks, 5W-13L, ROI −36%
#   entrada ≥50¢ → 19 picks, 13W-6L, ROI positivo en los 3 mercados con muestra
# El modelo sobreestima sistemáticamente al lado barato: cuando dice "este contrato de
# 43¢ vale 52¢", casi siempre el mercado tenía razón y el equipo simplemente es malo.
# Subir el umbral de edge NO arreglaba esto (a 8¢ el ROI empeoraba); el precio sí.
EMIT_ASK_MIN = 50

# --- Cupo por mercado según si se puede SALIR al take-profit ---
# MLB se vende al TP, no a settlement. Pero eso solo es posible si hay a quién venderle.
# Volumen mediano medido de los picks de la semana (2026-08-09):
#   ML     1,562,219  → profundo, la salida al TP es real
#   SPREAD   547,416  → profundo
#   TOTAL    120,296  → dudoso
#   TT        11,635  → delgado (mínimo observado: 528). Aquí NO se sale: te comes el
#                       resultado a settlement aunque el plan dijera TP.
# Por eso TT/TOTAL se racionan: son justo donde el motor más produce (28 de 37 picks) y
# donde peor le fue (−15.1% y −11.3%), y no se puede gestionar la posición.
EMIT_MAX_POR_MERCADO = {"TT": 1, "TOTAL": 1}

# Familia de mercado a partir del nombre que imprime el motor ("TT AZ 4.5" → "TT").
def _familia(mercado):
    m = mercado.split()[0]
    return {"SPREAD": "SPREAD", "TOTAL": "TOTAL", "TT": "TT", "ML": "ML"}.get(m, m)

# Park factors (runs, 100=neutral). Aproximación Statcast multi-año.
# Revisado 2026-07-25 — actualizar si cambia estadio o temporada.
PARK = {
    "Coors Field": 112, "Fenway Park": 104, "Great American Ball Park": 107,
    "Kauffman Stadium": 104, "Chase Field": 103, "Nationals Park": 102,
    "Citizens Bank Park": 103, "Wrigley Field": 101, "Truist Park": 101,
    "Rogers Centre": 101, "American Family Field": 102, "Rate Field": 101,
    "Guaranteed Rate Field": 101, "Yankee Stadium": 100, "Target Field": 100,
    "Oriole Park at Camden Yards": 99, "Globe Life Field": 99, "Angel Stadium": 99,
    "Daikin Park": 99, "Minute Maid Park": 99, "Dodger Stadium": 98,
    "Progressive Field": 98, "Comerica Park": 98, "Busch Stadium": 97,
    "PNC Park": 97, "loanDepot park": 97, "Citi Field": 96, "Oracle Park": 96,
    "Petco Park": 96, "T-Mobile Park": 92, "Sutter Health Park": 104,
    "George M. Steinbrenner Field": 105, "Tropicana Field": 96,
}

# Alias de abreviaturas statsapi → posibles códigos en tickers de Kalshi
ALIAS = {
    "AZ": ["AZ", "ARI"], "CWS": ["CWS", "CHW"], "WSH": ["WSH", "WAS"],
    "SF": ["SF", "SFG"], "SD": ["SD", "SDP"], "TB": ["TB", "TBR"],
    "KC": ["KC", "KCR"], "OAK": ["OAK", "ATH"], "ATH": ["ATH", "OAK"],
}

# Solo las 4 series con curva de calibración. Las 5 de F5/RFI/EXTRAS se quitaron el
# 2026-08-09: sin curva salían con CONF-BAJA, que es flag de descarte, así que eran 5
# llamadas de red por corrida para imprimir 224 líneas que nunca podían ser pick.
SERIES = ["KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBTEAMTOTAL"]


def get(url):
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "kalshi-mlb-skill/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def statsapi(path):
    return get("https://statsapi.mlb.com/api/v1" + path)


def kalshi_series(series):
    out, cursor = [], ""
    for _ in range(10):
        url = (f"https://api.elections.kalshi.com/trade-api/v2/markets"
               f"?series_ticker={series}&status=open&limit=200")
        if cursor:
            url += f"&cursor={cursor}"
        d = get(url)
        out += d.get("markets", [])
        cursor = d.get("cursor", "")
        if not cursor or not d.get("markets"):
            break
    return out


# ---------- Poisson ----------
def pois(mu):
    p, out = math.exp(-mu), []
    for k in range(MAXR + 1):
        out.append(p)
        p *= mu / (k + 1)
    return out


# ---------- Correlación entre carreras (Dixon-Coles adaptado a MLB) ----------
# Causa de fondo documentada en CALIBRACION: la Poisson conjunta pa[i]*ph[j] asume que
# las carreras de los dos equipos son independientes. No lo son — mismo clima, mismo
# parque, mismo umpire y el mismo bullpen débil de la noche empujan a AMBOS equipos hacia
# arriba o abajo juntos. En soccer eso se corrige con Dixon-Coles clásico (tau sobre las
# 4 esquinas 0-0/1-0/0-1/1-1, porque los goles son raros). En MLB las carreras rondan
# 3-6/equipo, no hay "marcador bajo especial": la corrección tiene que actuar sobre TODA
# la grilla, no solo la esquina. Se usa una covarianza lineal simple:
#
#   p_conjunta[i][j] = pa[i] * ph[j] * (1 + RHO_MLB * (i - mu_a) * (j - mu_h))
#
# RHO_MLB > 0 sube la probabilidad conjunta cuando ambos equipos se desvían de su media
# en el MISMO sentido (los dos altos o los dos bajos) y la baja cuando se desvían en
# sentido contrario — exactamente la covarianza positiva que el Poisson independiente
# ignora. Se renormaliza para que la grilla siga sumando 1. RHO_MLB se calibra con
# mlb_calibrate.py comparando el gap de TOTAL/TT en calibración contra la independencia
# pura (RHO_MLB=0 reproduce el comportamiento anterior exacto).
RHO_MLB = 0.02  # punto de partida; recalibrar con mlb_calibrate.py --rho antes de fiarse


def joint_matrix(pa, ph, mu_a, mu_h, rho=RHO_MLB):
    """Matriz conjunta P(runs_a=i, runs_b=j) con covarianza positiva entre equipos.

    Devuelve una matriz (len(pa) x len(ph)) que renormaliza a 1. rho=0 es exactamente
    el producto independiente pa[i]*ph[j] (comportamiento previo, para no romper nada
    que no dependa de esta corrección).
    """
    n, m = len(pa), len(ph)
    grid = [[0.0] * m for _ in range(n)]
    total = 0.0
    for i in range(n):
        for j in range(m):
            w = 1.0 + rho * (i - mu_a) * (j - mu_h)
            w = max(w, 0.0)  # nunca negativo: una probabilidad no puede serlo
            p = pa[i] * ph[j] * w
            grid[i][j] = p
            total += p
    if total <= 0:
        return [[pa[i] * ph[j] for j in range(m)] for i in range(n)]
    inv = 1.0 / total
    for i in range(n):
        for j in range(m):
            grid[i][j] *= inv
    return grid


def p_margin_over_corr(grid, line, eje_w):
    """P(equipo ganador gana por más de `line` carreras) sobre la grilla conjunta.

    grid está fijo en orientación (away=eje 0, home=eje 1). eje_w indica cuál de los
    dos es el "equipo que cubre" en esta apuesta: 'a' (away) o 'h' (home). line x.5 →
    margen mínimo entero = ceil(line).
    """
    n, m = len(grid), len(grid[0])
    k = math.ceil(line)
    if eje_w == "a":
        return sum(grid[i][j] for i in range(n) for j in range(m) if i - j >= k)
    return sum(grid[i][j] for i in range(n) for j in range(m) if j - i >= k)


def p_total_over_corr(grid, line):
    """P(total > line) sobre la grilla conjunta. line x.5 → total ≥ ceil(line)."""
    n, m = len(grid), len(grid[0])
    thresh = math.ceil(line)
    return sum(grid[i][j] for i in range(n) for j in range(m) if i + j >= thresh)


def p_team_over_corr(grid, line, eje):
    """P(equipo > line) marginalizando la grilla conjunta. eje 'a' (away) o 'h' (home)."""
    n, m = len(grid), len(grid[0])
    k = math.ceil(line)
    if eje == "a":
        return sum(grid[i][j] for i in range(n) for j in range(m) if i >= k)
    return sum(grid[i][j] for i in range(n) for j in range(m) if j >= k)


def p_margin_over(pw, pl, line):
    """P(equipo_w gana por más de `line` carreras). line siempre x.5 en Kalshi."""
    k = math.ceil(line)  # margen mínimo entero
    return sum(pw[i] * sum(pl[:max(0, i - k + 1)]) for i in range(len(pw)))


def p_total_over(pa, ph, line):
    """P(total > line). line x.5 → total ≥ ceil(line)."""
    n = math.ceil(line)
    return sum(pa[i] * ((1 - sum(ph[:n - i])) if i < n else 1.0) for i in range(len(pa)))


def p_team_over(p, line):
    """P(equipo > line). line x.5 → ≥ ceil(line) carreras."""
    return 1 - sum(p[:math.ceil(line)])


def pythagenpat(mu_w, mu_l):
    """Prob. de victoria vía Pythagenpat (mejor calibrado que Poisson para ML)."""
    x = (mu_w + mu_l) ** 0.287
    return mu_w ** x / (mu_w ** x + mu_l ** x)


# ---------- Kelly ----------
def sizing(prob, ask_c):
    """Kelly fraccional sobre la banca del config, con tope por trade."""
    return C.sizing(prob, ask_c)


def conviction(edge_c):
    return C.conviction(edge_c)


def calibrar(mercado, prob):
    """Aplica la curva de Platt del mercado. Devuelve (prob_calibrada, sin_curva).

    `mercado` viene con sufijos ("SPREAD TOR-1.5", "TT NYY 4.5"), así que se matchea por
    la primera palabra. El orden importa: F5SPREAD y F5TOTAL tienen que resolverse ANTES
    que SPREAD/TOTAL, o el prefijo corto se los roba.
    """
    fam = mercado.split()[0] if mercado else ""
    if fam in MERCADOS_SIN_CALIBRAR:
        return prob, True
    par = CALIBRACION.get(fam)
    if par is None:
        return prob, True
    a, b = par
    p = min(max(prob, 1e-6), 1 - 1e-6)
    x = math.log(p / (1 - p))
    return 1 / (1 + math.exp(-(a + b * x))), False


def fmt_row(mercado, ticker, lado, ask_c, prob, flags="", juego=None, hora=None, pred=None):
    # La probabilidad CRUDA del Poisson sobreestima (ver bloque CALIBRACION). Todo lo que
    # sigue —edge, sizing, MÁX-PAGABLE— usa la calibrada; la cruda solo se muestra.
    prob_cruda = prob
    prob, sin_curva = calibrar(mercado, prob)
    edge = round(prob * 100 - ask_c)
    conv = conviction(edge)
    if sin_curva:
        flags = (flags + " SIN-CALIBRAR").strip()
    if edge >= EDGE_SUSPECT:
        flags = (flags + " SOSPECHOSO(revisar-a-mano)").strip()
    ent = ask_c
    if ask_c > 80:
        flags = (flags + " POCO-RECORRIDO(+22% no cabe)").strip()
    # SIN-LIQUIDEZ / SPREAD-ANCHO: el número se muestra como referencia pero NUNCA se
    # dimensiona — a ese ask no hay contraparte, así que el edge no es cobrable y el stop
    # a −22% cae dentro del propio spread. Misma regla que ya tenía soccer (`mudo`); MLB
    # calculaba el sizing solo con el edge e ignoraba los flags (corregido 2026-08-05).
    # SIN-CALIBRAR entra al mute: sesgo desconocido y la evidencia dice que el sesgo por
    # defecto de este motor es grande. Se muestran como contexto, nunca se dimensionan.
    mudo = ("SIN-LIQUIDEZ" in flags or "SPREAD-ANCHO" in flags
            or "SIN-CALIBRAR" in flags)
    size = 0.0 if mudo else (sizing(prob, ask_c) if edge >= EDGE_MIN else 0.0)
    # MLB se vende al TAKE-PROFIT (restaurado 2026-08-03). Del 2026-07-29 al 2026-08-02 se
    # probó aguantar a settlement, por una medición de n=7 que resultó mal hecha: con la
    # muestra completa (n=25 desde 2026-07-26) la era TP dio ROI +3.2% / CLV +2.5¢ contra
    # ROI −23.0% / CLV −11.9¢ de la era settlement. TENIS, que nunca cambió, mide igual.
    # Se emite OBJ/STOP; MÁX-PAGABLE (= el fair, donde muere el edge) se mantiene porque
    # es de la regla de órdenes límite, independiente de la salida.
    obj, stop = min(round(ask_c * 1.22), 97), round(ask_c * 0.78)
    maxpag = round(prob * 100)
    tr = (f" | ${size} | ENTRADA {ent}¢ OBJ {obj}¢ STOP {stop}¢ | MÁX-PAGABLE {maxpag}¢"
          if size > 0 and edge >= EDGE_MIN else "")
    fl = f" | {flags}" if flags else ""
    # Registro estructurado que consume emit_seleccion().
    if juego is not None:
        PICKS.append({"juego": juego, "hora": hora[0] if hora else "", "mercado": mercado,
                      "inicio": hora[1] if hora else None,
                      "lado": lado, "ask": ask_c, "prob": prob, "edge": edge,
                      "flags": flags, "ticker": ticker,
                      "size": size, "obj": obj, "stop": stop, "maxpag": maxpag})
    cr = "" if sin_curva else f" (crudo {round(prob_cruda*100)}%)"
    return (f"  {mercado:<14} {lado:<10} ask {ask_c:>2}¢ | modelo {round(prob*100):>2}%{cr} | "
            f"edge {edge:+d}¢ {conv}{tr}{fl} | {ticker}")


def cents(m, side):
    v = m.get(f"{side}_ask_dollars")
    try:
        c = round(float(v) * 100)
        return c if 0 < c < 100 else None
    except (TypeError, ValueError):
        return None


def best_side(prob, m, mercado, ticker, label_yes, label_no, flags="", juego=None,
              hora=None, pred=None):
    """Evalúa YES y NO, devuelve la fila del mejor lado (o del YES si ninguno da).

    `pred` se acepta y se ignora: era el predicado del marcador que consumían las
    combinadas, eliminadas el 2026-08-09. Se mantiene en la firma para no tocar las ~15
    llamadas del cuerpo del motor, que lo pasan posicionalmente.
    """
    ya, na = cents(m, "yes"), cents(m, "no")
    # La comparación entre lados usa la prob CALIBRADA: con la cruda, el aplastamiento
    # hacia 50% podía elegir un lado que tras calibrar ya no era el de más edge.
    cal_y, _ = calibrar(mercado, prob)
    cal_n, _ = calibrar(mercado, 1 - prob)
    cands = []
    if ya is not None:
        cands.append((cal_y * 100 - ya, label_yes, ya, prob))
    if na is not None:
        cands.append((cal_n * 100 - na, label_no, na, 1 - prob))
    if not cands:
        return f"  {mercado:<14} sin precio ask publicado | {ticker}"
    _, lbl, ask, p = max(cands, key=lambda r: r[0])
    return fmt_row(mercado, ticker, lbl, ask, p, flags, juego, hora)




def emit_seleccion():
    """Aplica el filtro mecánico y devuelve (elegidos, descartados_con_motivo).

    Esto es exactamente lo que el agente hacía a mano y sin dejar rastro. Al bajarlo a
    código, dos corridas sobre los mismos precios eligen los mismos picks, y de los que
    NO eligió queda escrito el motivo — que es lo que el usuario pidió poder rebatir.

    Orden de los motivos: primero el flag (es del motor), después el rango de entrada,
    y de último la correlación (necesita saber quién ganó el cupo del partido).
    """
    elegidos, descartados = [], []
    candidatos = []

    for p in PICKS:
        if p["edge"] < EDGE_MIN:
            continue  # bajo el umbral: ni siquiera es candidato, no se reporta
        flag_malo = next((f for f in EMIT_FLAGS_MALOS if f in p["flags"]), None)
        if flag_malo:
            descartados.append((p, f"flag: {flag_malo}"))
        elif p["ask"] > EMIT_ASK_MAX:
            descartados.append((p, f"flag: POCO-RECORRIDO (ask {p['ask']}¢ >{EMIT_ASK_MAX}¢)"))
        elif p["ask"] < EMIT_ASK_MIN:
            descartados.append((p, f"entrada {p['ask']}¢ bajo el piso de {EMIT_ASK_MIN}¢"))
        elif p["size"] <= 0:
            descartados.append((p, "sizing 0 (Kelly no autoriza tamaño)"))
        else:
            candidatos.append(p)

    # Correlación: las filas de un partido son UNA tesis. Gana la de más edge; las demás
    # se descartan diciendo QUIÉN les ganó el cupo, no con una narrativa post-hoc.
    por_juego = {}
    for p in sorted(candidatos, key=lambda x: -x["edge"]):
        por_juego.setdefault(p["juego"], []).append(p)

    for juego, ps in por_juego.items():
        elegidos += ps[:EMIT_MAX_POR_PARTIDO]
        ganador = ps[0]
        for p in ps[EMIT_MAX_POR_PARTIDO:]:
            descartados.append(
                (p, f"correlación: {ganador['mercado']} ({ganador['edge']:+d}¢) "
                    f"ganó el cupo de {juego}"))

    # Cupo por mercado: TT y TOTAL son los más delgados (no hay salida al TP) y los que
    # más picks producen. Se quedan con los de más edge de la jornada; el resto cae con
    # el motivo explícito, no en silencio.
    elegidos.sort(key=lambda p: -p["edge"])
    vistos, filtrados = {}, []
    for p in elegidos:
        fam = _familia(p["mercado"])
        tope = EMIT_MAX_POR_MERCADO.get(fam)
        n = vistos.get(fam, 0)
        if tope is not None and n >= tope:
            descartados.append(
                (p, f"cupo de {fam}: máximo {tope}/jornada (mercado delgado, "
                    f"sin salida al TP)"))
            continue
        vistos[fam] = n + 1
        filtrados.append(p)
    elegidos = filtrados

    elegidos.sort(key=lambda p: -p["edge"])
    descartados.sort(key=lambda d: -d[0]["edge"])
    candidatos.sort(key=lambda p: -p["edge"])
    # `candidatos` = todo lo limpio (modo B: el research elige DENTRO de esto).
    # `elegidos`   = lo que el orden mecanico por edge escogeria sin research.
    return elegidos, descartados, candidatos


def emit_run(fecha, salida_motor):
    """Puente al contrato compartido de `_kalshi_core/emit.py`."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from emit import Pick, Descartado, emit as core_emit

    from emit import Candidato
    elegidos, descartados, candidatos = emit_seleccion()

    picks = [Pick(
        ticker=p["ticker"], lado=p["lado"], mercado=p["mercado"], partido=p["juego"],
        entrada=p["ask"], p_mod=round(p["prob"] * 100), edge=p["edge"],
        tp=p["obj"], stop=p["stop"], max_pagable=p["maxpag"], tamano=p["size"],
        flags=p["flags"],
        razonamiento=(f"{p['juego']} {p['hora']} — {p['mercado']} {p['lado']}; "
                      f"modelo {round(p['prob']*100)}% vs ask {p['ask']}¢, "
                      f"edge {p['edge']:+d}¢ {conviction(p['edge'])}"),
    ) for p in elegidos]

    desc = [Descartado(
        ticker=p["ticker"], mercado=p["mercado"], partido=p["juego"], entrada=p["ask"],
        p_mod=round(p["prob"] * 100), edge=p["edge"], motivo=motivo,
    ) for p, motivo in descartados]

    cands = [Candidato(
        ticker=p["ticker"], lado=p["lado"], mercado=p["mercado"], partido=p["juego"],
        entrada=p["ask"], p_mod=round(p["prob"] * 100), edge=p["edge"],
        tp=p["obj"], stop=p["stop"], max_pagable=p["maxpag"], tamano=p["size"],
        flags=p["flags"],
    ) for p in candidatos]

    core_emit(modelo="MLB", fecha=fecha, picks=picks, salida_motor=salida_motor,
              precios_ts=PRECIOS_TS, descartados=desc, candidatos=cands)

    # Modo B: el research trabaja sobre esta lista, no sobre los 714 renglones del motor.
    libres = [c for c in cands if c.ticker not in {p.ticker for p in picks}]
    if libres:
        print("\nCANDIDATOS LIMPIOS NO ELEGIDOS (research puede subirlos, nunca agregar")
        print("tickers fuera de esta lista). Si subes uno, di el motivo y vuelve a emitir:")
        for c in libres:
            print(f"  {c.partido:>9s} {c.mercado:<16} {c.lado:<18} ask {c.entrada:>2}c | "
                  f"mod {c.p_mod:>2}% | edge {c.edge:+d}c | ${c.tamano:.2f} | {c.ticker}")


SPREAD_MAX = 10   # ¢ entre yes_bid y yes_ask: arriba de esto el precio no es operable


def liq_flag(m):
    """'SIN-LIQUIDEZ' | 'SPREAD-ANCHO' | 'LIQUIDEZ-BAJA' | '' .

    El chequeo de spread se agregó el 2026-08-05: hasta entonces solo se miraba volumen y
    tamaño, y eso deja pasar libros vacíos con precio publicado. Caso medido ese día en
    KXMLBF5TOTAL: yes_ask 98¢ contra yes_bid 55¢ (spread 43¢) con volumen 34. El motor lo
    dimensionaba como pick normal, pero a 98¢ no hay contraparte: el edge es ficticio
    porque el precio no es ejecutable, y el stop a −22% cae dentro del propio spread.

    Medido sobre las 9 series (687 markets): spread mediano 1-3¢, así que 10¢ corta solo
    el 5.1% y no toca el mercado sano. KXMLBRFI es la excepción estructural — mediana 14¢
    y volumen 0 en toda la serie — y queda marcada entera a propósito.
    """
    try:
        vol = float(m.get("volume_fp") or 0)
        sz = float(m.get("yes_ask_size_fp") or 0)
    except ValueError:
        return "SIN-LIQUIDEZ"
    # Se mira el spread de AMBOS lados: el motor compra YES o NO según cuál dé más edge,
    # y en KXMLBRFI el lado NO es el ilíquido (medido 2026-08-05: la serie entera tiene
    # spread mediano de 14¢ con volumen 0). Mirar solo el YES lo dejaba pasar entero.
    def _bid(side):
        try:
            v = m.get(f"{side}_bid_dollars")
            return round(float(v) * 100) if v is not None else None
        except (TypeError, ValueError):
            return None

    for side in ("yes", "no"):
        a, b = cents(m, side), _bid(side)
        if a is not None and b is not None and (a - b) > SPREAD_MAX:
            return "SPREAD-ANCHO"
    if vol < 50 and sz < 50:
        return "LIQUIDEZ-BAJA"
    return ""


# ---------- main ----------
def _run():
    args = [a for a in sys.argv[1:]]
    do_emit = "--emit" in args
    if do_emit:
        args.remove("--emit")
    team_filter = None
    if "--equipo" in args:
        i = args.index("--equipo")
        team_filter = args[i + 1].upper()
        del args[i:i + 2]
    date = args[0] if args else datetime.now(PT).strftime("%Y-%m-%d")

    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    season = end.year

    teams = {t["id"]: t for t in statsapi(f"/teams?sportId=1&season={season}")["teams"]}

    def stats30(group):
        d = statsapi(f"/teams/stats?season={season}&group={group}&stats=byDateRange"
                     f"&startDate={start}&endDate={date}&sportId=1")
        return {s["team"]["id"]: s["stat"] for s in d["stats"][0]["splits"]}

    hit, pit = stats30("hitting"), stats30("pitching")

    sched = statsapi(f"/schedule?sportId=1&date={date}&hydrate=probablePitcher,venue")
    games = []
    for d in sched.get("dates", []):
        games += d.get("games", [])
    if not games:
        print(f"Sin partidos de MLB el {date}.")
        return

    # Kalshi: bajar las 9 series una sola vez
    global PRECIOS_TS
    kmkts = {}
    PRECIOS_TS = datetime.now(PT).strftime("%Y-%m-%dT%H:%M:%S%z")
    for s in SERIES:
        try:
            kmkts[s] = kalshi_series(s)
        except Exception as e:
            kmkts[s] = []
            print(f"AVISO: fallo bajando {s} de Kalshi ({e}) — mercados de esa serie no evaluados.")

    def cand(abbr):
        return ALIAS.get(abbr, [abbr])

    def match_event(series, away, home, et_date, et_hhmm):
        """Matchea markets de una serie al partido por teams+fecha ET (y hora si hay doble)."""
        outs = {}
        for m in kmkts.get(series, []):
            mm = re.match(r"^KX[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{4})([A-Z]+)$",
                          m["event_ticker"])
            if not mm:
                continue
            dd, tstr, hhmm = mm.group(3), mm.group(5), mm.group(4)
            if dd != et_date:
                continue
            if any(a + h == tstr for a in cand(away) for h in cand(home)):
                outs.setdefault(hhmm, []).append(m)
        if not outs:
            return []
        if len(outs) == 1:
            return next(iter(outs.values()))
        # doubleheader: la hora del ticker más cercana a la hora ET real
        best = min(outs, key=lambda h: abs(int(h) - int(et_hhmm)))
        return outs[best]

    print(f"=== MOTOR MLB — {date} | stats 30d: {start}..{date} | banca ${BANKROLL:.0f} "
          f"(⅛ Kelly, tope ${MAX_TRADE:.0f}/trade, ${MAX_OPEN:.0f} abierto) ===")
    print(f"Regla: edge <{EDGE_MIN}¢ = NO INVIERTE. Edge ≥{EDGE_SUSPECT}¢ = SOSPECHOSO, "
          "revisar a mano antes de reportar.\n")

    league_rs = sum(float(h.get("runs", 0)) / max(1, int(h.get("gamesPlayed", 1)))
                    for h in hit.values()) / max(1, len(hit))

    for g in games:
        away, home = g["teams"]["away"], g["teams"]["home"]
        aid, hid = away["team"]["id"], home["team"]["id"]
        aab = teams[aid].get("abbreviation", "?")
        hab = teams[hid].get("abbreviation", "?")
        if team_filter and team_filter not in (aab, hab):
            continue
        aname, hname = teams[aid]["name"], teams[hid]["name"]
        gdt = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
        pt_t = gdt.astimezone(PT).strftime("%I:%M %p PT").lstrip("0")
        et_dt = gdt.astimezone(ET)
        state = g["status"]["detailedState"]
        venue = g.get("venue", {}).get("name", "?")
        pf = PARK.get(venue)
        pf_note = f"PF {pf}" if pf else "PF NO LISTADO→100 (buscar y añadir a la tabla)"
        pf = pf or 100

        print(f"— {aname} @ {hname} | {pt_t} | {venue} ({pf_note}) | estado: {state}")
        gkey, ghora = f"{aab}@{hab}", (pt_t, gdt)
        # `a` y `h` en los predicados = carreras del visitante y del local al final.

        if state not in ("Scheduled", "Pre-Game", "Warmup"):
            print("  EN VIVO/FINAL → el modelo pre-partido NO aplica. Para edge en vivo usa"
                  f" winProbability oficial: statsapi /v1/game/{g['gamePk']}/winProbability\n")
            continue

        try:
            h_a, h_h = hit[aid], hit[hid]
            p_a, p_h = pit[aid], pit[hid]
        except KeyError:
            print("  SIN STATS 30d para un equipo — partido no modelado.\n")
            continue

        def pergame(st, key):
            return float(st.get(key, 0)) / max(1, int(st.get("gamesPlayed", 1)))

        rs_a, ra_a = pergame(h_a, "runs"), pergame(p_a, "runs")
        rs_h, ra_h = pergame(h_h, "runs"), pergame(p_h, "runs")

        def bullpen_ratio(team_id, side_team_pit):
            """Ratio del bullpen vs el staff del equipo, acotado igual que el abridor.

            OJO con la fuente: statsapi NO deja cruzar `sitCodes=rp` con `byDateRange`
            (probado 2026-08-05: statSplits ignora las fechas y devuelve temporada;
            byDateRange ignora el sitCode y devuelve el staff mezclado). Así que el ERA de
            relevo es de TEMPORADA, no de 30 días como el resto del motor. Es la
            inconsistencia menor: una temporada de bullpen son ~424 IP, muestra bastante
            más estable que las ~5 aperturas del abridor, así que el ruido es menor que la
            distorsión de tratar los 30 bullpens iguales. Si el bullpen cambió de forma
            reciente, este número lo va a ver con retraso.
            """
            if team_id in BULLPEN_CACHE:
                return BULLPEN_CACHE[team_id]
            staff_era = float(side_team_pit.get("era", 4.0))
            try:
                d = statsapi(f"/teams/{team_id}/stats?stats=statSplits&sitCodes=rp"
                             f"&group=pitching&season={season}")
                era = float(d["stats"][0]["splits"][0]["stat"]["era"])
                raw = era / max(0.1, staff_era)
                # Mismo encogimiento y cota que el abridor, por simetría de tratamiento.
                r = max(0.75, min(1.3, 1 + (raw - 1) * 0.5))
                out = (r, f"bullpen ERA {era:.2f} (staff {staff_era:.2f}, ratio {r:.2f})")
            except Exception:
                out = (1.0, "bullpen sin ERA disponible (ratio 1.0)")
            BULLPEN_CACHE[team_id] = out
            return out

        # abridores: ERA 30d vs ERA 30d del staff → ratio acotado [0.6, 1.5]
        # Muestra mínima (añadido 2026-08-09): antes esto leía el ERA sin mirar cuántas
        # entradas lo respaldaban, y un ERA de 0.00 salido de 1 IP de relevo se convertía
        # en ratio 0.75 — el MEJOR valor posible, o sea que la falta de datos se premiaba
        # como si fuera un as. Casos reales: Cade Povich (0.00 con 5 apariciones en toda la
        # temporada, 3 meses fuera por antebrazo) y Erik Miller (0.00, relevista puro de
        # 40 apariciones y 0 aperturas — juego de opener). Mismo bug que Legumina el
        # 2026-07-29. Ahora: sin IP suficientes, ratio 1.0 (neutro) y bandera visible.
        IP_MIN_ABRIDOR = 10.0   # ~2 aperturas; por debajo el ERA no dice nada
        GS_MIN_ABRIDOR = 1      # 0 aperturas en 30d = relevista/opener, no abridor

        def starter_ratio(side_team_pit, prob_pitcher):
            staff_era = float(side_team_pit.get("era", 4.0))
            if not prob_pitcher:
                return 1.0, "abridor SIN ANUNCIAR (ratio 1.0, confianza F5 más baja)"
            pid, pname = prob_pitcher["id"], prob_pitcher["fullName"]
            try:
                d = statsapi(f"/people/{pid}/stats?stats=byDateRange&group=pitching"
                             f"&startDate={start}&endDate={date}&season={season}")
                st = d["stats"][0]["splits"][0]["stat"]
                era = float(st["era"])
                ip = float(st.get("inningsPitched", 0) or 0)
                gs = int(st.get("gamesStarted", 0) or 0)
                if ip < IP_MIN_ABRIDOR:
                    return 1.0, (f"{pname} MUESTRA-INSUFICIENTE ({ip:.1f} IP en 30d, "
                                 f"ERA30 {era:.2f} no es señal) → ratio 1.0")
                if gs < GS_MIN_ABRIDOR:
                    return 1.0, (f"{pname} SIN-APERTURAS ({gs} GS en 30d, {ip:.1f} IP: "
                                 f"relevista/opener) → ratio 1.0")
                # ERA de 30d son ~5 aperturas: ruido alto → encoger 50% hacia 1.0 y acotar
                raw = era / max(0.1, staff_era)
                r = max(0.75, min(1.3, 1 + (raw - 1) * 0.5))
                return r, (f"{pname} ERA30 {era:.2f} en {ip:.1f} IP/{gs} GS "
                           f"(staff {staff_era:.2f}, ratio {r:.2f})")
            except Exception:
                return 1.0, f"{pname} sin ERA30 disponible (ratio 1.0)"

        r_a, note_a = starter_ratio(p_a, away.get("probablePitcher"))
        r_h, note_h = starter_ratio(p_h, home.get("probablePitcher"))
        b_a, bnote_a = bullpen_ratio(aid, p_a)
        b_h, bnote_h = bullpen_ratio(hid, p_h)

        # Carreras esperadas F9. El abridor cubre ~60% del juego y el bullpen el 40%
        # restante (~3.5 innings). Hasta el 2026-08-05 ese 40% era la CONSTANTE 0.4: los
        # 30 bullpens valían igual. Medido ese día: ERA de relevo va de 2.90 (BOS) a 5.67
        # (ATH), sd 0.59 — casi 2x entre extremos. Ahora escala con el ratio del bullpen
        # rival, misma mecánica que el abridor.
        mu_a = ((rs_a + ra_h) / 2) * (pf / 100) * (0.6 * r_h + 0.4 * b_h)
        mu_h = ((rs_h + ra_a) / 2) * (pf / 100) * (0.6 * r_a + 0.4 * b_a)
        # F5: solo abridor
        mu_a5 = (5 / 9) * ((rs_a + ra_h) / 2) * (pf / 100) * r_h
        mu_h5 = (5 / 9) * ((rs_h + ra_a) / 2) * (pf / 100) * r_a

        pa, ph = pois(mu_a), pois(mu_h)
        pa5, ph5 = pois(mu_a5), pois(mu_h5)
        # Grilla conjunta F9 con covarianza (SPREAD/TOTAL/TT) — ver joint_matrix(). El ML
        # sigue con Pythagenpat (no depende de esto) y F5 se queda en Poisson independiente
        # porque mlb_calibrate.py no cubre F5 (SIN-CALIBRAR, no se le inventa corrección).
        grid = joint_matrix(pa, ph, mu_a, mu_h)
        win_a = pythagenpat(mu_a, mu_h)
        win_h = 1 - win_a

        print(f"  30d: {aab} RS {rs_a:.2f} RA {ra_a:.2f} | {hab} RS {rs_h:.2f} RA {ra_h:.2f} "
              f"(liga RS {league_rs:.2f})")
        print(f"  Abridores: {aab} {note_a} | {hab} {note_h}")
        print(f"  Bullpen:   {aab} {bnote_a} | {hab} {bnote_h}")
        print(f"  Esperado: {aab} {mu_a:.2f} — {hab} {mu_h:.2f} (total {mu_a+mu_h:.2f}) | "
              f"ML modelo: {aab} {win_a*100:.0f}% / {hab} {win_h*100:.0f}%")

        et_date, et_hhmm = et_dt.strftime("%d"), et_dt.strftime("%H%M")
        rows = []

        def team_of(tk):
            suf = tk.split("-")[-1]
            return re.sub(r"\d+$", "", suf)

        # ML: solo lado YES de cada market (el NO es el espejo del otro — evita duplicar)
        for m in match_event("KXMLBGAME", aab, hab, et_date, et_hhmm):
            t = team_of(m["ticker"])
            prob = win_a if t in cand(aab) else win_h
            ya = cents(m, "yes")
            if ya is not None:
                es_away = t in cand(aab)
                pr = ((lambda a, h, g: g) if es_away else (lambda a, h, g: not g))
                rows.append(fmt_row("ML", m["ticker"], f"{t} gana", ya, prob, liq_flag(m),
                                    gkey, ghora, pr))

        for m in match_event("KXMLBSPREAD", aab, hab, et_date, et_hhmm):
            t, line = team_of(m["ticker"]), m.get("floor_strike")
            if line is None:
                continue
            es_home = t in cand(hab)
            prob = p_margin_over_corr(grid, line, "h" if es_home else "a")
            pr = ((lambda a, h, g, L=line: h - a > L) if es_home
                  else (lambda a, h, g, L=line: a - h > L))
            rows.append(best_side(prob, m, f"SPREAD {t}-{line}", m["ticker"],
                                  f"{t} por {line}+", "NO cubre", liq_flag(m), gkey, ghora, pr))

        for m in match_event("KXMLBTOTAL", aab, hab, et_date, et_hhmm):
            line = m.get("floor_strike")
            if line is None:
                continue
            prob = p_total_over_corr(grid, line)
            rows.append(best_side(prob, m, f"TOTAL {line}", m["ticker"],
                                  f"OVER {line}", f"UNDER {line} (NO)", liq_flag(m), gkey, ghora,
                                  (lambda a, h, g, L=line: a + h > L)))

        for m in match_event("KXMLBTEAMTOTAL", aab, hab, et_date, et_hhmm):
            t, line = team_of(m["ticker"]), m.get("floor_strike")
            if line is None:
                continue
            es_away_tt = t in cand(aab)
            prob = p_team_over_corr(grid, line, "a" if es_away_tt else "h")
            pr = ((lambda a, h, g, L=line: a > L) if es_away_tt
                  else (lambda a, h, g, L=line: h > L))
            rows.append(best_side(prob, m, f"TT {t} {line}", m["ticker"],
                                  f"{t} OVER {line}", f"{t} UNDER (NO)", liq_flag(m),
                                  gkey, ghora, pr))

        # F5, F5SPREAD, F5TOTAL, RFI y EXTRAS: ELIMINADOS de la salida el 2026-08-09.
        # No tienen curva de calibración (el backtest de 7497 predicciones no los cubrió),
        # así que el motor les ponía CONF-BAJA, y CONF-BAJA está en EMIT_FLAGS_MALOS.
        # Resultado: 224 de 686 líneas por jornada (33%) que era IMPOSIBLE que llegaran a
        # pick — se calculaban, se imprimían y había que leerlas para nada. Peor: mostraban
        # edges de +20¢ y +25¢ que invitaban a saltarse el filtro (el pick de Legumina del
        # 2026-07-29 salió justo de ahí).
        # Para reactivarlos hace falta primero calibrarlos con mlb_calibrate.py y quitarles
        # el CONF-BAJA; mientras no exista esa curva, imprimirlos es solo ruido.

        if rows:
            print("\n".join(rows))
        else:
            print("  Sin mercados de Kalshi matcheados para este partido "
                  "(revisar abreviaturas/fecha si Kalshi sí lo lista).")
        print()

    print(f"FIN. Candidatos = edge ≥+{EDGE_MIN}¢, entrada {EMIT_ASK_MIN}-{EMIT_ASK_MAX}¢, "
          "sin flags malos.\n"
          "OJO: las filas de un mismo partido están CORRELACIONADAS (una sola tesis).\n"
          f"Cupo por jornada: TT máx {EMIT_MAX_POR_MERCADO['TT']}, "
          f"TOTAL máx {EMIT_MAX_POR_MERCADO['TOTAL']} (mercados delgados, sin salida al TP).\n"
          "Research cualitativo (lesiones/bullpen) solo sobre candidatos.")


def main():
    """Envoltura: con --emit captura el stdout íntegro y lo persiste tras correr."""
    if "--emit" not in sys.argv[1:]:
        return _run()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from emit import capture
    args = [a for a in sys.argv[1:] if a != "--emit"]
    args = [a for a in args if not a.startswith("--")]
    # la fecha efectiva es la misma que resuelve _run(); se recalcula igual para el nombre
    fecha = None
    skip = False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False
            continue
        if a == "--equipo":
            skip = True
            continue
        if not a.startswith("--"):
            fecha = a
            break
    fecha = fecha or datetime.now(PT).strftime("%Y-%m-%d")
    with capture() as buf:
        _run()
    emit_run(fecha, buf.getvalue())


if __name__ == "__main__":
    main()
