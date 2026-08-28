#!/usr/bin/env python3
"""
Motor de clima para el skill kalshi-markets-analyst — loop completo, paridad con
mlb_engine.py: del ensemble al veredicto sin ningún cálculo a mano.

Qué hace, en orden:
  1. Coordenadas OFICIALES de la estación de liquidación (NWS/ASOS) de cada serie
     KXHIGH* de Kalshi (issuedby del reporte CLI). Día calendario LOCAL de cada
     ciudad (timezone=auto), que es el que liquida Kalshi.
  2. Baja el ensemble Open-Meteo (GFS+ECMWF+ICON, ~119 miembros).
  3. CORRECCIÓN DE SESGO por estación (BIAS), medida contra la máxima OFICIAL del
     reporte CLI (IEM) — el número exacto con el que liquida Kalshi, no el píxel
     del reanálisis. Recalibrar con scripts/weather_calibrate.py.
  4. RE-ESCALADO A ANCHO CALIBRADO: la dispersión del ensemble NO es el error
     real — contra la liquidación CLI oficial el error medido (sd 0.8-1.5°) suele
     ser MENOR que el spread de miembros. Sin corregir, la fracción cruda de
     miembros queda descalibrada (demasiado plana o demasiado picuda) y fabrica
     edges falsos en los buckets. El motor escala los miembros alrededor de la
     mediana y convoluciona con N(0, sd_k) para que el ancho predictivo total
     sea exactamente la sd calibrada de la estación, conservando la forma
     (asimetría) del ensemble. Validado con weather_backtest.py --reliability.
  5. REGLA DURA anti-edge-falso: distribución sesgada, ensemble disperso, forma
     bimodal o divergencia con NWS => confianza BAJA => edges capados, sin picks.
  6. Baja los mercados KXHIGH* de Kalshi para la fecha, calcula prob por bucket,
     edge del mejor lado (YES/NO), flags (SOSPECHOSO / LIQUIDEZ / POCO-RECORRIDO),
     conviction, sizing Kelly y ENTRADA/OBJETIVO(+22%)/STOP(−22%).
  7. UNA tesis por ciudad: imprime como pick solo la MEJOR EXPRESIÓN (máximo 1
     pick por ciudad/fecha; los buckets de una ciudad están 100% correlacionados).

Sizing: clima usa ¼ Kelly (un escalón POR ENCIMA del ⅛ de MLB/tenis/soccer — aun
calibrado, un grado de error mueve el bucket entero) y pide edge ≥7¢. Confianza MEDIA
recorta a la mitad. Banca, topes y fracción salen de _kalshi_core/config.json; el tope
global de abiertos se verifica contra registro.md antes de recomendar.

Uso:
  python3 weather_engine.py --all [YYYY-MM-DD]     # las 7 ciudades (default: hoy)
  python3 weather_engine.py CITY [YYYY-MM-DD]      # una ciudad
  (sin args -> calibración vigente)

NO coloca órdenes. Solo análisis. Requiere Python 3, stdlib, red.
"""
import sys, os, json, math, time, urllib.request, statistics, datetime
from zoneinfo import ZoneInfo

UA = {"User-Agent": "kalshi-weather-analyst larcisllc@gmail.com"}
PT = ZoneInfo("America/Los_Angeles")

# Banca, topes, Kelly y umbrales salen de _kalshi_core/config.json (Etapa 2): un solo
# lugar para los 4 motores. Editar ESE archivo cambia el sizing de todo el sistema.
# Clima conserva sus diferencias INTENCIONALES: ¼ Kelly (escalón extra sobre el ⅛ de MLB)
# y edge_min 7¢ (el modelo es más ruidoso, pide más edge). Confianza MEDIA recorta ×0.5,
# que es el `kelly_frac_conf_baja` = 0.125 del config.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from core import cfg as _cfg

C = _cfg("CLIMA")
BANKROLL = C.bankroll
MAX_TRADE = C.max_trade
MAX_OPEN = C.max_open      # informativo: lo abierto se verifica en registro.md
KELLY = C.kelly_frac
EDGE_MIN = C.edge_min      # centavos: debajo de esto, NO INVIERTE (clima pide más que MLB)
EDGE_SUSPECT = C.edge_suspect  # edge así de grande = sospechoso, revisar a mano
RECAL_DAYS = 45   # días de vigencia de la calibración antes de pedir recalibrar

# Estación de liquidación oficial por serie Kalshi (issuedby del CLI report).
# lat/lon verificadas contra api.weather.gov/stations/K***.
# BIAS/SD/MAE = mediana multi-modelo (ecmwf+icon+gem+arpege, día local) vs máxima
# OFICIAL del reporte CLI (IEM) — el número exacto con el que liquida Kalshi.
# bias>0 => el modelo corre CALIENTE => hay que RESTAR bias a cada miembro.
# Regenerar el bloque con: python3 weather_calibrate.py 60 --emit
CALIBRATED_ON = "2026-07-25"  # ventana 2026-05-25..2026-07-23, observado = CLI oficial (IEM)
STATIONS = {
    "LAX":  {"series":"KXHIGHLAX", "station":"KLAX", "lat":33.9381, "lon":-118.3889,
             "bias":+0.91, "sd":1.82, "mae":1.57},
    "CHI":  {"series":"KXHIGHCHI", "station":"KMDW", "lat":41.7842, "lon":-87.7553,
             "bias":-1.20, "sd":1.4, "mae":1.49},
    "NY":   {"series":"KXHIGHNY", "station":"KNYC", "lat":40.7833, "lon":-73.9667,
             "bias":+0.29, "sd":1.32, "mae":1.07},
    "MIA":  {"series":"KXHIGHMIA", "station":"KMIA", "lat":25.7906, "lon":-80.3164,
             "bias":-2.16, "sd":1.5, "mae":2.18},
    "DEN":  {"series":"KXHIGHDEN", "station":"KDEN", "lat":39.8466, "lon":-104.6562,
             "bias":-0.84, "sd":1.36, "mae":1.21},
    "AUS":  {"series":"KXHIGHAUS", "station":"KAUS", "lat":30.183, "lon":-97.6799,
             "bias":-1.44, "sd":1.74, "mae":1.73},
    "PHIL": {"series":"KXHIGHPHIL", "station":"KPHL", "lat":39.8733, "lon":-75.2268,
             "bias":-2.17, "sd":1.47, "mae":2.27},
}

# ---------------------------------------------------------------------------
# Umbrales de la REGLA DURA anti-edge-falso.
#
# BACKTEST (scripts/weather_backtest.py, 60d × 7 ciudades): el error del
# pronóstico crece de forma monótona con el desacuerdo entre modelos, así que
# el corte por dispersión tiene fundamento empírico. El backtest usa spread
# inter-modelo (4 modelos); el motor usa rango del ensemble (~119 miembros),
# que es ~1.6-1.7x más ancho — umbrales traducidos por percentil.
MAX_MEAN_MEDIAN_GAP = 2.5   # |media-mediana| mayor => distribución sesgada por cola
MAX_ENSEMBLE_RANGE  = 12.0  # rango (max-min) del ensemble, ~p88 del backtest
MAX_SD_RATIO        = 1.75  # sd del día / sd calibrada. Detecta la FORMA rota
                            # (bimodal) que el rango solo no ve: AUS 2026-07-25
                            # dio sd 3.70° vs 1.70° calibrado (ratio 2.18) con
                            # rango de apenas 13°. Las sd calibradas contra CLI
                            # multi-modelo (1.3-1.8°) quedaron en la misma escala
                            # que las viejas, así que el umbral se mantiene.
NWS_DIVERGENCE      = 3.0   # |mediana_corregida - NWS| mayor => confianza BAJA
BORDE_MAE_RATIO     = 0.5   # distancia al borde de decisión < este × mae de la
                            # estación => BORDE-CERCANO => sin pick. El contrato se
                            # decide dentro del error típico del modelo, así que la
                            # probabilidad emitida está sobrevendida (ver borde_cercano).


def get(url, intentos=3):
    """Reintenta fallos transitorios de red (visto en el sandbox cloud: 403 de
    Tunnel connection en llamadas sueltas que al reintentar sí pasan — no es
    rechazo real de la API, es inestabilidad del proxy de salida)."""
    ultimo_error = None
    for i in range(intentos):
        try:
            return json.load(
                urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception as e:
            ultimo_error = e
            if i < intentos - 1:
                time.sleep(1.5 * (i + 1))
    raise ultimo_error


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


def event_token(target_date):
    """2026-07-25 -> '26JUL25' (sufijo de event_ticker de Kalshi)."""
    d = datetime.date.fromisoformat(target_date)
    return f"{d.year % 100:02d}{d.strftime('%b').upper()}{d.day:02d}"


def ensemble_members(lat, lon, target_date):
    url = (f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max&models=gfs025,ecmwf_ifs025,icon_seamless"
           f"&temperature_unit=fahrenheit&timezone=auto&forecast_days=7")
    d = get(url)["daily"]
    if target_date not in d["time"]:
        return None
    i = d["time"].index(target_date)
    mem = [v[i] for k, v in d.items()
           if k.startswith("temperature_2m_max_member") and v[i] is not None]
    return mem


def nws_high(lat, lon, target_date):
    """Máxima pronosticada por NWS para target_date, o None."""
    try:
        pts = get(f"https://api.weather.gov/points/{lat},{lon}")
        fc = get(pts["properties"]["forecast"])
        want = datetime.date.fromisoformat(target_date)
        for p in fc["properties"]["periods"]:
            if not p["isDaytime"]:
                continue
            if datetime.date.fromisoformat(p["startTime"][:10]) == want:
                return p["temperature"]
    except Exception:
        return None
    return None


def lead_days(target_date):
    return (datetime.date.fromisoformat(target_date) - datetime.date.today()).days


# ---------- probabilidad por bucket (miembros suavizados) ----------
def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def scale_members(mem, median, sd_day, sd_target):
    """Escala los miembros para que el ancho predictivo total = sd calibrada.

    Kernel mínimo 0.5° (liquidación en grados enteros). Si el ensemble viene
    más ancho que el error calibrado, se contrae alrededor de la mediana; si
    viene más angosto, se deja y el kernel pone el ancho que falta.
    Devuelve (miembros_escalados, sd_k).
    """
    sd_target = max(sd_target, 0.5)
    if sd_day <= 0:
        return mem, sd_target
    if sd_day ** 2 >= sd_target ** 2 - 0.25:
        alpha = math.sqrt(max(sd_target ** 2 - 0.25, 0.01)) / sd_day
        return [median + alpha * (m - median) for m in mem], 0.5
    return mem, math.sqrt(sd_target ** 2 - sd_day ** 2)


def edge_distance(median, strike_type, floor_s, cap_s):
    """Grados entre la mediana pronosticada y el borde de decisión más cercano.

    Mismos umbrales que bucket_prob (Kalshi liquida en grados enteros, de ahí el
    ±0.5). Para 'between' hay dos bordes: manda el más cercano, porque es el que
    decide el contrato.
    """
    if strike_type == "greater":
        return abs(median - (floor_s + 0.5))
    if strike_type == "less":
        return abs(median - (cap_s - 0.5))
    return min(abs(median - (floor_s - 0.5)), abs(median - (cap_s + 0.5)))


def borde_cercano(median, strike_type, floor_s, cap_s, mae):
    """True si el contrato se decide DENTRO del error típico de la estación.

    Motivo (AUS 2026-07-29): banda 98-99°, el día cerró en 98.51° — el contrato se
    decidió por 0.51° cuando el mae calibrado de KAUS es 1.73°. El motor le había
    puesto 67% a ese lado, pero con un error típico 3× mayor que la distancia al
    borde no distingue 98.5 de 99.5: la probabilidad real está cerca de 50%. Un
    pick así no es edge, es una moneda al aire vendida como convicción.

    Umbral: distancia < 0.5 × mae. Afecta sobre todo bandas estrechas ('between',
    2°) en estaciones ruidosas (MIA mae 2.32°, PHIL 2.23°) — ahí el motor
    simplemente no tiene resolución y no debe emitir pick.
    """
    if not mae:
        return False
    return edge_distance(median, strike_type, floor_s, cap_s) < BORDE_MAE_RATIO * mae


def bucket_prob(members, sd_k, strike_type, floor_s, cap_s):
    """P(bucket) con cada miembro convolucionado con N(0, sd_k).

    Kalshi liquida en grados enteros: 'greater' floor=86 es '87° o más'
    (umbral continuo 86.5); 'less' cap=79 es '78° o menos' (umbral 78.5);
    'between' floor=85 cap=86 es '85° a 86°' (84.5..86.5).
    """
    n = len(members)
    if strike_type == "greater":
        t = floor_s + 0.5
        return sum(1.0 - phi((t - m) / sd_k) for m in members) / n
    if strike_type == "less":
        t = cap_s - 0.5
        return sum(phi((t - m) / sd_k) for m in members) / n
    lo, hi = floor_s - 0.5, cap_s + 0.5
    return sum(phi((hi - m) / sd_k) - phi((lo - m) / sd_k) for m in members) / n


# ---------- Kelly / formato (paridad con mlb_engine) ----------
def sizing(prob, ask_c, conf):
    """¼ Kelly del config; confianza MEDIA usa el escalón recortado (⅛)."""
    return C.sizing(prob, ask_c, conf_baja=(conf == "media"))


def conviction(edge_c):
    return "ALTO" if edge_c > 15 else ("MEDIO" if edge_c >= EDGE_MIN else "BAJO")


def cents(m, side):
    v = m.get(f"{side}_ask_dollars")
    try:
        c = round(float(v) * 100)
        return c if 0 < c < 100 else None
    except (TypeError, ValueError):
        return None


def liq_flag(m):
    try:
        vol = float(m.get("volume_fp") or 0)
        sz = float(m.get("yes_ask_size_fp") or 0)
    except ValueError:
        return "SIN-LIQUIDEZ"
    if vol < 50 and sz < 50:
        return "LIQUIDEZ-BAJA"
    return ""


def make_row(bucket_label, ticker, lado, ask_c, prob, conf, base_flags, dist=None):
    edge = round(prob * 100 - ask_c)
    flags = list(base_flags)
    if edge >= EDGE_SUSPECT:
        flags.append("SOSPECHOSO(revisar-a-mano)")
    if ask_c > 80:
        flags.append("POCO-RECORRIDO(+22% no cabe)")
    if dist is not None:
        flags.append(f"BORDE-CERCANO({dist:.2f}°-del-corte)")
    size = sizing(prob, ask_c, conf) if edge >= EDGE_MIN else 0.0
    obj, stop = min(round(ask_c * 1.22), 97), round(ask_c * 0.78)
    # SIN-RECORRIDO: el take-profit no supera la entrada, así que el pick no puede ganar
    # nada aunque acierte. Pasa en los buckets a 1-2¢ de mercados casi resueltos (Kalshi
    # cotiza todo a 1¢ cuando ya nadie los quiere), donde el ensemble sigue dando 26% y
    # el "edge +25¢" es contra un libro muerto, no contra un precio. Medido 2026-08-14:
    # 4 de 4 picks de clima salían así al correr de noche.
    if obj <= ask_c:
        flags.append(f"SIN-RECORRIDO(TP {obj}¢ = entrada {ask_c}¢)")
    return {
        "bucket": bucket_label, "ticker": ticker, "lado": lado, "ask": ask_c,
        "prob": prob, "edge": edge, "conv": conviction(edge), "size": size,
        "obj": obj, "stop": stop, "flags": flags,
        "pickable": (edge >= EDGE_MIN and size > 0
                     and not any(f.startswith(("SOSPECHOSO", "SIN-LIQUIDEZ",
                                               "BORDE-CERCANO", "SIN-RECORRIDO"))
                                 for f in flags)),
    }


def fmt_row(r):
    tr = (f" | ${r['size']} | ENTRADA {r['ask']}¢ OBJ {r['obj']}¢ STOP {r['stop']}¢"
          if r["size"] > 0 and r["edge"] >= EDGE_MIN else "")
    fl = f" | {' '.join(r['flags'])}" if r["flags"] else ""
    return (f"    {r['bucket']:<12} {r['lado']:<3} ask {r['ask']:>2}¢ | "
            f"modelo {round(r['prob'] * 100):>2}% | edge {r['edge']:+d}¢ {r['conv']}"
            f"{tr}{fl} | {r['ticker']}")


# ---------- análisis por ciudad ----------
def analyze(city, target_date, markets):
    cfg = STATIONS[city]
    mem_raw = ensemble_members(cfg["lat"], cfg["lon"], target_date)
    if not mem_raw:
        return {"city": city, "error": "sin datos de ensemble para esa fecha"}
    n = len(mem_raw)
    mem = sorted(m - cfg["bias"] for m in mem_raw)
    mean = statistics.mean(mem)
    median = mem[n // 2]
    rng = mem[-1] - mem[0]
    sd_day = statistics.pstdev(mem)
    sd_ratio = sd_day / cfg["sd"] if cfg["sd"] else 0.0
    mem_s, sd_k = scale_members(mem, median, sd_day, cfg["sd"])
    nws = nws_high(cfg["lat"], cfg["lon"], target_date)
    lead = lead_days(target_date)

    flags, conf = [], "alta"
    if abs(mean - median) > MAX_MEAN_MEDIAN_GAP:
        flags.append(f"distribución sesgada (media {mean:.1f} vs mediana {median:.1f})")
        conf = "baja"
    if rng > MAX_ENSEMBLE_RANGE:
        flags.append(f"ensemble disperso (rango {rng:.0f}°, modelos en desacuerdo)")
        conf = "baja"
    if sd_ratio > MAX_SD_RATIO:
        flags.append(f"dispersión anómala (sd {sd_day:.2f}° vs {cfg['sd']:.2f}° calibrada, "
                     f"ratio {sd_ratio:.2f}) — posible distribución bimodal")
        conf = "baja"
    if nws is not None and abs(median - nws) > NWS_DIVERGENCE:
        flags.append(f"diverge de NWS ({median:.1f} modelo vs {nws} NWS)")
        conf = "baja"
    if lead >= 3 and conf == "alta":
        conf = "media"
        flags.append(f"horizonte {lead} días (dispersión sube)")
    if cfg["mae"] >= 1.9 and conf == "alta":
        conf = "media"
        flags.append(f"estación ruidosa (mae calib {cfg['mae']}°)")

    # ---- mercados de Kalshi para esta ciudad/fecha ----
    token = event_token(target_date)
    mkts = [m for m in markets if m.get("event_ticker", "").endswith(token)]
    rows, close_pt = [], None
    for m in sorted(mkts, key=lambda x: (x.get("floor_strike") is None, x.get("floor_strike") or 0)):
        st = m.get("strike_type")
        if st not in ("greater", "less", "between"):
            continue
        prob = bucket_prob(mem_s, sd_k, st, m.get("floor_strike"), m.get("cap_strike"))
        label = m.get("yes_sub_title") or m.get("ticker")
        base = [f for f in [liq_flag(m)] if f]
        # ¿el contrato se decide dentro del error típico de la estación? Si sí, ningún
        # lado es operable: la probabilidad emitida no distingue los grados en juego.
        dist = None
        if borde_cercano(median, st, m.get("floor_strike"), m.get("cap_strike"), cfg["mae"]):
            dist = edge_distance(median, st, m.get("floor_strike"), m.get("cap_strike"))
        ya, na = cents(m, "yes"), cents(m, "no")
        cand = []
        if ya is not None:
            cand.append(make_row(label, m["ticker"], "YES", ya, prob, conf, base, dist))
        if na is not None:
            cand.append(make_row(label, m["ticker"], "NO", na, 1 - prob, conf, base, dist))
        if cand:
            rows.append(max(cand, key=lambda r: r["edge"]))
        if close_pt is None and m.get("close_time"):
            try:
                dt = datetime.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                close_pt = dt.astimezone(PT).strftime("%-I:%M %p PT %d-%b")
            except ValueError:
                pass

    cap = conf == "baja"
    best = None
    if not cap:
        pickables = [r for r in rows if r["pickable"]]
        if pickables:
            best = max(pickables, key=lambda r: r["edge"])

    return {
        "city": city, "station": cfg["station"], "date": target_date, "lead": lead,
        "n": n, "mean": round(mean, 1), "median": round(median, 1),
        "range": [round(mem[0]), round(mem[-1])], "nws": nws,
        "sd_day": round(sd_day, 2), "sd_calib": cfg["sd"], "sd_ratio": round(sd_ratio, 2),
        "sd_k": round(sd_k, 2), "bias_applied": cfg["bias"], "confidence": conf,
        "flags": flags, "cap_edges": cap, "rows": rows, "best": best, "close_pt": close_pt,
    }


def print_report(res):
    if res.get("error"):
        print(f"{res['city']}: {res['error']}")
        return
    cierre = f" — cierre {res['close_pt']}" if res["close_pt"] else ""
    print(f"\n=== {res['city']} ({res['station']}) — máxima {res['date']} "
          f"[lead {res['lead']}d]{cierre} ===")
    print(f"  ensemble n={res['n']}  media(corr)={res['mean']}°  mediana={res['median']}°  "
          f"rango[{res['range'][0]},{res['range'][1]}]  bias={res['bias_applied']:+.2f}°  "
          f"ancho predictivo={res['sd_calib']}° (re-escalado, sd_k={res['sd_k']}°)")
    print(f"  sd día={res['sd_day']}° vs calib {res['sd_calib']}° (ratio {res['sd_ratio']})  |  "
          f"NWS: {res['nws']}°  |  CONFIANZA: {res['confidence'].upper()}")
    for f in res["flags"]:
        print(f"    ⚠ {f}")
    if res["cap_edges"]:
        print("    → EDGES CAPADOS: confianza baja, no se reportan picks de esta ciudad.")
    if not res["rows"]:
        print("  (sin mercados abiertos en Kalshi para esa fecha)")
        return
    print("  buckets (mejor lado por contrato):")
    for r in res["rows"]:
        print(fmt_row(r))
    if res["cap_edges"]:
        return
    if res["best"]:
        b = res["best"]
        print(f"  → MEJOR EXPRESIÓN (única de la ciudad): {b['lado']} \"{b['bucket']}\" "
              f"a {b['ask']}¢ | edge {b['edge']:+d}¢ {b['conv']} | ${b['size']} | "
              f"ENTRADA {b['ask']}¢ OBJ {b['obj']}¢ STOP {b['stop']}¢ | {b['ticker']}")
    else:
        print(f"  → sin edge ≥{EDGE_MIN}¢ limpio: NO INVIERTE en esta ciudad.")


def staleness_warning():
    age = (datetime.date.today() - datetime.date.fromisoformat(CALIBRATED_ON)).days
    if age > RECAL_DAYS:
        print(f"⚠ CALIBRACIÓN VENCIDA: {age} días desde {CALIBRATED_ON} (tope {RECAL_DAYS}). "
              f"Corre: python3 weather_calibrate.py 60 --emit  y pega el bloque nuevo.")


RESULTADOS = []   # lo llena run(); lo consume --emit (Etapa 1 del blindaje)
PRECIOS_TS = None  # instante congelado de la bajada de precios de Kalshi


def emit_run(salida_motor, target_date):
    """Persiste el contrato verificable (Etapa 1 del blindaje).

    Filtro mecanico de clima: ya vive en analyze(). `best` es la MEJOR EXPRESION unica
    de la ciudad (los buckets son UNA tesis) y viene vacia si la confianza es baja
    (cap_edges) o si no hay edge limpio >= EDGE_MIN. Aqui no se re-decide nada: se
    persiste lo que el motor ya resolvio.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from emit import Pick, Candidato, Descartado, emit as core_emit

    picks, cands, desc = [], [], []
    for res in RESULTADOS:
        if res.get("error"):
            continue
        ciudad = res["city"]
        b = res.get("best")
        if res.get("cap_edges"):
            # confianza baja: el motor capa los edges a proposito. Queda el motivo escrito
            # para que nadie los "rescate a mano" despues (regla del SKILL.md).
            for r in res.get("rows", []):
                if r.get("edge", 0) >= EDGE_MIN:
                    desc.append(Descartado(
                        ticker=r["ticker"], mercado=r.get("bucket", ""), partido=ciudad,
                        entrada=r["ask"], p_mod=round(r.get("prob", 0) * 100),
                        edge=r["edge"], motivo="flag: CONFIANZA-BAJA (edges capados)"))
            continue
        if not b:
            continue
        base = dict(ticker=b["ticker"], lado=b["lado"], mercado=b["bucket"],
                    partido=ciudad, entrada=b["ask"], p_mod=round(b["prob"] * 100),
                    edge=b["edge"], tp=b["obj"], stop=b["stop"],
                    max_pagable=round(b["prob"] * 100), tamano=b["size"], flags="")
        cands.append(Candidato(**base))
        picks.append(Pick(razonamiento=(
            f"{ciudad} maxima {res['date']} — {b['lado']} \"{b['bucket']}\" a "
            f"{b['ask']}c; ensemble n={res['n']} media {res['mean']}deg, NWS "
            f"{res['nws']}deg, confianza {res['confidence']}; edge {b['edge']:+d}c"),
            **base))
        for r in res.get("rows", []):
            if r["ticker"] != b["ticker"] and r.get("edge", 0) >= EDGE_MIN:
                # El flag propio manda sobre la correlacion: si la fila ya era inoperable
                # (SIN-RECORRIDO, SOSPECHOSO, BORDE-CERCANO), ese es el motivo real.
                flag = next((f for f in r.get("flags", []) if f.startswith(
                    ("SOSPECHOSO", "SIN-LIQUIDEZ", "BORDE-CERCANO", "SIN-RECORRIDO"))), None)
                motivo = (f"flag: {flag}" if flag else
                          f"correlacion: {b['lado']} ({b['edge']:+d}c) es la unica "
                          f"expresion de {ciudad}")
                desc.append(Descartado(
                    ticker=r["ticker"], mercado=r.get("bucket", ""), partido=ciudad,
                    entrada=r["ask"], p_mod=round(r.get("prob", 0) * 100),
                    edge=r["edge"], motivo=motivo))

    core_emit(modelo="CLIMA", fecha=target_date, picks=picks,
              salida_motor=salida_motor, precios_ts=PRECIOS_TS or "sin-kalshi",
              descartados=desc, candidatos=cands)


def run(cities, target_date):
    staleness_warning()
    print(f"=== MOTOR CLIMA — máxima {target_date} | banca ${BANKROLL:.0f} | "
          f"¼ Kelly tope ${MAX_TRADE:.0f}/trade | calibrado {CALIBRATED_ON} (CLI oficial) ===")
    print(f"Regla: edge <{EDGE_MIN}¢ = NO INVIERTE. Edge ≥{EDGE_SUSPECT}¢ = SOSPECHOSO. "
          f"Máximo 1 pick por ciudad (los buckets son UNA tesis).")
    global PRECIOS_TS
    for city in cities:
        try:
            if PRECIOS_TS is None:
                PRECIOS_TS = datetime.datetime.now().astimezone().strftime(
                    "%Y-%m-%dT%H:%M:%S%z")
            res = analyze(city, target_date, kalshi_series(STATIONS[city]["series"]))
            RESULTADOS.append(res)
            print_report(res)
        except Exception as e:
            print(f"{city}: ERROR {e}")
    print(f"\nFIN. Tope global ${MAX_OPEN:.0f} abiertos: revisa registro.md y descuenta lo "
          f"abierto (MLB incluido) antes de recomendar. El usuario ejecuta en Kalshi/Robinhood.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(f"Calibración vigente por estación ({CALIBRATED_ON}, observado = CLI oficial IEM):")
        for c, v in STATIONS.items():
            print(f"  {c:5s} {v['station']}  bias={v['bias']:+5.2f}°  sd={v['sd']}  mae={v['mae']}")
        print(f"\nCortes: rango<={MAX_ENSEMBLE_RANGE}°  sd_ratio<={MAX_SD_RATIO}  "
              f"NWS<={NWS_DIVERGENCE}°  media-mediana<={MAX_MEAN_MEDIAN_GAP}°")
        staleness_warning()
        print("\nUso: weather_engine.py --all [YYYY-MM-DD]  |  CITY [YYYY-MM-DD]")
        print("     weather_calibrate.py 60 --emit  -> recalibrar bias")
        print("     weather_backtest.py  60         -> validar cortes y confiabilidad")
        sys.exit()
    today = datetime.date.today().isoformat()
    do_emit = "--emit" in args
    if do_emit:
        args.remove("--emit")
    ciudades = list(STATIONS) if args[0] == "--all" else [args[0].upper()]
    fecha = args[1] if len(args) > 1 else today
    if not do_emit:
        run(ciudades, fecha)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
        from emit import capture
        with capture() as buf:
            run(ciudades, fecha)
        emit_run(buf.getvalue(), fecha)
