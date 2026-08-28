#!/usr/bin/env python3
"""
Recalibración de bias/sd/mae por estación para weather_engine.py.

Compara, día a día sobre una ventana:
  - historical-forecast-api: la MEDIANA MULTI-MODELO que se había pronosticado
    (ecmwf+icon+gem+arpege — mismo proxy del backtest; gfs025 no tiene histórico
    en ese endpoint), con timezone LOCAL de cada estación (timezone=auto), que
    es el día calendario que liquida Kalshi. Se calibra la mediana multi-modelo
    y no el "best_match" porque el motor aplica el bias a la MEDIANA DEL
    ENSEMBLE — misma naturaleza de estadístico; calibrar sobre best_match dio
    sd irreales de 0.8° que la confiabilidad (backtest --reliability) reventó.
  - IEM CLI (mesonet.agron.iastate.edu): la máxima OFICIAL del reporte CLI de
    la estación de liquidación — el número exacto con el que liquida Kalshi.
    (Antes se usaba archive-api = reanálisis del píxel de grilla; eso mete el
    error píxel-vs-estación dentro del bias sin medirlo bien. La cadena que
    importa es forecast vs CLI, y eso es lo que se mide aquí.)

bias = media(forecast - observado CLI). bias>0 => el modelo corre CALIENTE =>
el motor resta ese bias a cada miembro del ensemble.

Uso:
  python3 weather_calibrate.py [dias]        # default 60
  python3 weather_calibrate.py 60 --emit     # imprime el bloque STATIONS listo para pegar
"""
import sys, json, urllib.request, statistics, datetime

UA = {"User-Agent": "kalshi-weather-analyst larcisllc@gmail.com"}

# mismas coordenadas/estaciones de liquidación que weather_engine.STATIONS
STATIONS = {
    "LAX":  {"series": "KXHIGHLAX",  "station": "KLAX", "lat": 33.9381, "lon": -118.3889},
    "CHI":  {"series": "KXHIGHCHI",  "station": "KMDW", "lat": 41.7842, "lon": -87.7553},
    "NY":   {"series": "KXHIGHNY",   "station": "KNYC", "lat": 40.7833, "lon": -73.9667},
    "MIA":  {"series": "KXHIGHMIA",  "station": "KMIA", "lat": 25.7906, "lon": -80.3164},
    "DEN":  {"series": "KXHIGHDEN",  "station": "KDEN", "lat": 39.8466, "lon": -104.6562},
    "AUS":  {"series": "KXHIGHAUS",  "station": "KAUS", "lat": 30.1830, "lon": -97.6799},
    "PHIL": {"series": "KXHIGHPHIL", "station": "KPHL", "lat": 39.8733, "lon": -75.2268},
}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120))


MODELS = "ecmwf_ifs025,icon_seamless,gem_global,meteofrance_arpege_world"


def series_forecast(lat, lon, d0, d1):
    """Mediana multi-modelo pronosticada, día calendario LOCAL."""
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
            out[day] = statistics.median(vals)
    return out


def cli_is_final(row):
    """True si la fila del CLI es el reporte FINAL del día, no un corte parcial.

    El NWS emite el CLI varias veces al día. El de la mañana lleva
    'VALID AS OF 0700 AM LOCAL TIME' y su MAXIMUM es solo el máximo de las
    primeras horas — de madrugada, antes de que la ciudad caliente. Ejemplo real
    (KAUS 2026-07-29): el CLI de las 07:48 daba high=79 (12:12 AM) cuando la
    máxima final del día fue 98.5°. Usar ese número como observación produce
    errores de 20°, y arruina cualquier calibración o auditoría que lo consuma.

    El reporte final se emite al DÍA SIGUIENTE del día que reporta: el campo
    `product` trae el timestamp de emisión (AAAAMMDDHHMM). Si la emisión cae el
    mismo día que reporta, es parcial → se descarta.
    """
    prod, valid = row.get("product") or "", row.get("valid") or ""
    if len(prod) < 8 or len(valid) < 10:
        return False  # sin metadatos para verificar: no se usa
    emitido = prod[:8]                       # AAAAMMDD de emisión
    reportado = valid.replace("-", "")       # AAAAMMDD del día reportado
    return emitido > reportado


def series_observed_cli(station, years, strict=True):
    """Máxima oficial del reporte CLI de la estación (IEM), la que liquida Kalshi.

    Con strict=True (default) descarta los CLI parciales del día en curso —
    ver cli_is_final(). NO pongas strict=False para "ver el dato de hoy": la
    máxima del día en curso NO EXISTE hasta el CLI nocturno.
    """
    out = {}
    for y in years:
        d = get(f"https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={y}")
        for row in d.get("results", []):
            if row.get("high") is None:
                continue
            if strict and not cli_is_final(row):
                continue
            out[row["valid"]] = row["high"]
    return out


def calibrate(days=60, end_offset=2):
    """end_offset: días atrás donde cortar (el CLI del día puede salir parcial)."""
    today = datetime.date.today()
    d1 = today - datetime.timedelta(days=end_offset)
    d0 = d1 - datetime.timedelta(days=days - 1)
    years = sorted({d0.year, d1.year})
    out = {}
    for city, cfg in STATIONS.items():
        try:
            fc = series_forecast(cfg["lat"], cfg["lon"], d0.isoformat(), d1.isoformat())
            ob = series_observed_cli(cfg["station"], years)
        except Exception as e:
            out[city] = {"error": str(e)}
            continue
        diffs = []
        for day, f in fc.items():
            o = ob.get(day)
            if f is None or o is None or not (d0.isoformat() <= day <= d1.isoformat()):
                continue
            diffs.append(f - o)
        if len(diffs) < 10:
            out[city] = {"error": f"muestra insuficiente (n={len(diffs)})"}
            continue
        out[city] = {
            "n": len(diffs),
            "bias": round(statistics.mean(diffs), 2),
            "sd": round(statistics.pstdev(diffs), 2),
            "mae": round(statistics.mean(abs(x) for x in diffs), 2),
            "window": f"{d0.isoformat()}..{d1.isoformat()}",
        }
    return out, d0, d1


def observed(city, day):
    """Máxima OFICIAL de liquidación de `city` en `day` (AAAA-MM-DD), o None.

    Único camino autorizado para preguntar "¿cuánto marcó de verdad?" al auditar
    un pick de clima. Devuelve None si el CLI final todavía no salió — None
    significa "no se sabe todavía", NUNCA se sustituye por el pronóstico NWS ni
    por el CLI parcial del día en curso.

    Uso:  python3 weather_calibrate.py --observed AUS 2026-07-29
    """
    if city not in STATIONS:
        raise ValueError(f"ciudad desconocida: {city} (válidas: {', '.join(STATIONS)})")
    year = int(day[:4])
    return series_observed_cli(STATIONS[city]["station"], [year]).get(day)


if __name__ == "__main__":
    if "--observed" in sys.argv:
        i = sys.argv.index("--observed")
        try:
            city, day = sys.argv[i + 1].upper(), sys.argv[i + 2]
        except IndexError:
            sys.exit("uso: weather_calibrate.py --observed CIUDAD AAAA-MM-DD")
        v = observed(city, day)
        est = STATIONS[city]["station"]
        if v is None:
            print(f"{city} ({est}) {day}: CLI FINAL NO DISPONIBLE todavía.")
            print("  La máxima del día en curso no existe hasta el CLI nocturno.")
            print("  NO uses el pronóstico NWS ni el CLI parcial como observación.")
        else:
            print(f"{city} ({est}) {day}: máxima oficial de liquidación {v}°F")
        sys.exit(0)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    days = int(args[0]) if args else 60
    res, d0, d1 = calibrate(days)
    print(f"Calibración sobre {days} días ({d0}..{d1}), observado = CLI oficial (IEM):\n")
    print(f"{'ciudad':6s} {'est':6s} {'n':>4s} {'bias':>7s} {'sd':>6s} {'mae':>6s}")
    for c, v in res.items():
        if "error" in v:
            print(f"{c:6s} {STATIONS[c]['station']:6s}  ERROR {v['error']}")
            continue
        print(f"{c:6s} {STATIONS[c]['station']:6s} {v['n']:4d} {v['bias']:+7.2f} {v['sd']:6.2f} {v['mae']:6.2f}")

    if "--emit" in sys.argv:
        print("\n# --- bloque para weather_engine.py ---")
        print(f'CALIBRATED_ON = "{datetime.date.today().isoformat()}"  '
              f'# ventana {d0}..{d1}, observado = CLI oficial (IEM)')
        print("STATIONS = {")
        for c, cfg in STATIONS.items():
            v = res.get(c, {})
            if "error" in v:
                continue
            print(f'    "{c}":{" " * (5 - len(c))}{{"series":"{cfg["series"]}", '
                  f'"station":"{cfg["station"]}", "lat":{cfg["lat"]}, "lon":{cfg["lon"]},')
            print(f'             "bias":{v["bias"]:+.2f}, "sd":{v["sd"]}, "mae":{v["mae"]}}},')
        print("}")
