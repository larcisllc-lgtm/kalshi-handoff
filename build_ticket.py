#!/usr/bin/env python3
"""Handoff Claude -> Kalshi Bot (Grok): lee el manifiesto del motor, cruza contra el
sharp (Pinnacle vía The Odds API), y escribe latest.json + copia con fecha al repo
kalshi-handoff para que Grok lo lea en su rutina (4am/8am/12pm PT) en vez de investigar.

Diseño acordado 2026-08-28 (ver memoria project_kalshibot-handoff-diseno-final):
  - Solo entra al ticket un pick si el motor tiene edge Y Pinnacle está del mismo lado.
  - Sin tamaño de posición: Grok decide size con los $21 reales, no con el Kelly de Claude.
  - MLB restringido a ML/SPREAD/TOTAL cuando se active (TT queda fuera del handoff aunque
    el motor lo emita — ver EMIT_HANDOFF_MERCADOS abajo).
  - MLB sigue BLOQUEADO del handoff hasta pasar el fix de covarianza (RHO_MLB) + resolver
    el payout simétrico. Este script corre en modo --dry-run por defecto para MLB.

Uso:
  python3 build_ticket.py clima 2026-08-28              # clima: activo, hace push real
  python3 build_ticket.py mlb 2026-08-28 --dry-run       # MLB: solo imprime, no hace push
  python3 build_ticket.py mlb 2026-08-28 --force-push    # MLB: ignora el bloqueo (explícito)
"""
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
REPO = os.path.dirname(os.path.abspath(__file__))
AUDITOR = os.path.join(REPO, "auditor")
MANIFIESTOS = os.path.join(AUDITOR, "manifiestos")
ODDS_KEY_PATH = os.path.join(REPO, "core", "odds_api_key.txt")

# MLB bloqueado del handoff hasta que RHO_MLB esté calibrado y el payout simétrico esté
# resuelto (ver project_mlb-pnl-negativo-post-platt). CLIMA no tiene ese bloqueo.
MOTORES_BLOQUEADOS = {"mlb"}

# Cuando MLB se active: solo estos mercados van al handoff (Grok pidió ML/spread/total,
# no team-total ni F5/RFI/extras).
EMIT_HANDOFF_MERCADOS = {"ML", "SPREAD", "TOTAL"}

SPORT_KEY = {"mlb": "baseball_mlb"}  # clima no tiene sharp book equivalente

# The Odds API devuelve nombres completos ("Los Angeles Dodgers"), no los códigos de
# 3 letras que usa statsapi/Kalshi ("LAD"). Mapa directo, sin heurística de substring
# (que falla: "AZ" no está contenido en "Arizona Diamondbacks").
EQUIPO_MLB = {
    "AZ": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "ATH": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}


def leer_key():
    if not os.path.exists(ODDS_KEY_PATH):
        return None
    return open(ODDS_KEY_PATH).read().strip()


def cargar_manifiesto(motor, fecha):
    ruta = os.path.join(MANIFIESTOS, f"{fecha}-{motor}.json")
    if not os.path.exists(ruta):
        print(f"No hay manifiesto en {ruta} — corre el motor con --emit primero.")
        return None
    return json.load(open(ruta, encoding="utf-8"))


def familia(mercado):
    return mercado.split()[0] if mercado else ""


def fetch_sharp_odds(sport_key, key):
    """Pinnacle vía The Odds API. Devuelve lista de eventos o [] si falla/no hay key."""
    if not key:
        return []
    url = (f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
           f"?apiKey={key}&regions=us&markets=h2h,spreads,totals&bookmakers=pinnacle")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
        if isinstance(d, dict) and "message" in d:
            print(f"  odds API error: {d['message']}")
            return []
        return d
    except Exception as e:
        print(f"  odds API falló: {e}")
        return []


def team_tokens(nombre_partido):
    """'LAD@DET' -> ('LAD', 'DET')."""
    partes = nombre_partido.split("@")
    return (partes[0], partes[1]) if len(partes) == 2 else (None, None)


def sharp_side_ml(evento, away_tok, home_tok):
    """Devuelve el lado (away/home) que Pinnacle favorece en h2h, o None si no matchea."""
    for bk in evento.get("bookmakers", []):
        if bk["key"] != "pinnacle":
            continue
        for mk in bk.get("markets", []):
            if mk["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mk["outcomes"]}
            if len(outcomes) != 2:
                continue
            # precio más bajo (decimal) = favorito
            fav = min(outcomes, key=outcomes.get)
            return fav
    return None


def matchea_evento(evento, away_tok, home_tok):
    away_full = EQUIPO_MLB.get(away_tok)
    home_full = EQUIPO_MLB.get(home_tok)
    if not away_full or not home_full:
        return False
    return evento.get("away_team") == away_full and evento.get("home_team") == home_full


def cruzar_con_sharp(pick, eventos):
    """True si Pinnacle coincide con el lado que el modelo eligió. False si discrepa
    o no hay dato (sin dato = no nace ticket, no se asume acuerdo por defecto)."""
    fam = familia(pick["mercado"])
    if fam not in EMIT_HANDOFF_MERCADOS:
        return False, "mercado fuera del handoff (solo ML/SPREAD/TOTAL)"
    away_tok, home_tok = team_tokens(pick.get("partido", ""))
    ev = next((e for e in eventos if matchea_evento(e, away_tok, home_tok)), None)
    if ev is None:
        return False, "sin evento Pinnacle casado"
    if fam == "ML":
        fav = sharp_side_ml(ev, away_tok, home_tok)
        if fav is None:
            return False, "sin h2h de Pinnacle"
        lado_modelo_es_away = pick["lado"].startswith(away_tok) if away_tok else None
        fav_es_away = fav == EQUIPO_MLB.get(away_tok)
        if lado_modelo_es_away is None:
            return False, "no se pudo determinar lado del modelo"
        if lado_modelo_es_away == fav_es_away:
            return True, f"Pinnacle coincide (favorito {fav})"
        return False, f"Pinnacle discrepa (favorito {fav}, modelo eligió otro lado)"
    # SPREAD/TOTAL: The Odds API sí trae spreads/totals de Pinnacle, pero casar la línea
    # exacta de Kalshi contra la línea de Pinnacle es trabajo adicional no resuelto en
    # esta primera versión. Se deja explícito en vez de fingir un cruce que no se hizo.
    return False, "SPREAD/TOTAL: cruce de línea Pinnacle pendiente de implementar"


def construir_ticket(motor, fecha, man, eventos):
    items = []
    for p in man.get("picks", []):
        ok, motivo = cruzar_con_sharp(p, eventos) if eventos or motor == "mlb" else (
            False, "sin fuente de sharp para este motor")
        if motor != "mlb":
            # CLIMA no tiene sharp book: la regla de acuerdo no aplica, pasa directo.
            ok, motivo = True, "sin sharp aplicable (clima)"
        item = {
            "ticker": p["ticker"],
            "mercado": p["mercado"],
            "lado": p["lado"],
            "entrada_sugerida_c": p["entrada"],
            "prob_modelo": p["p_mod"],
            "edge_c": p["edge"],
            "max_pagable_c": p["max_pagable"],
            "partido": p.get("partido", ""),
            "sharp_ok": ok,
            "sharp_nota": motivo,
        }
        if ok:
            items.append(item)
        else:
            print(f"  DESCARTADO del handoff: {p['ticker']} — {motivo}")
    return items


def escribir_y_push(motor, fecha, items, dry_run):
    ahora = datetime.now(PT)
    payload = {
        "generado_ts": ahora.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "motor": motor.upper(),
        "fecha": fecha,
        "tickets": items,
    }
    if dry_run:
        print(f"\n--dry-run: no se escribe ni se hace push. Contenido que se hubiera "
              f"generado:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
        return

    os.makedirs(REPO, exist_ok=True)
    latest_path = os.path.join(REPO, "latest.json")
    fechado_path = os.path.join(REPO, f"{fecha}-{motor}.json")

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(fechado_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    subprocess.run(["git", "-C", REPO, "add", "latest.json", os.path.basename(fechado_path)],
                    check=True)
    result = subprocess.run(
        ["git", "-C", REPO, "commit", "-m", f"{motor}: {fecha} ({len(items)} tickets)"],
        capture_output=True, text=True)
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        print(result.stdout, result.stderr)
    subprocess.run(["git", "-C", REPO, "push"], check=True)
    print(f"\nOK — {len(items)} ticket(s) escritos y empujados a kalshi-handoff.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    motor, fecha = sys.argv[1].lower(), sys.argv[2]
    dry_run = "--dry-run" in sys.argv
    force = "--force-push" in sys.argv

    if motor in MOTORES_BLOQUEADOS and not force:
        print(f"{motor.upper()} está BLOQUEADO del handoff (ver "
              f"project_mlb-pnl-negativo-post-platt). Corriendo en --dry-run.")
        dry_run = True

    man = cargar_manifiesto(motor, fecha)
    if man is None:
        sys.exit(1)

    eventos = []
    if motor in SPORT_KEY:
        key = leer_key()
        eventos = fetch_sharp_odds(SPORT_KEY[motor], key)
        print(f"  Pinnacle: {len(eventos)} evento(s) traídos para {motor.upper()}")

    items = construir_ticket(motor, fecha, man, eventos)
    escribir_y_push(motor, fecha, items, dry_run)


if __name__ == "__main__":
    main()
