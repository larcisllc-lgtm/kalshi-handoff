#!/usr/bin/env python3
"""Contrato de emisión compartido por los motores de Kalshi (Etapa 1 del blindaje).

Qué resuelve: hasta ahora el motor imprimía, y el AGENTE decidía a mano qué picks
sobrevivían, escribía `predicciones/` y las filas PAPER. Ese paso a mano se saltaba en
silencio (7 de 9 archivos sin salida íntegra, TP lleno al 47%). Aquí el motor hace las
tres cosas de una sola pasada y deja un manifiesto verificable.

Un motor que quiera `--emit` solo tiene que:
  1. capturar su propio stdout (`capture()`),
  2. llenar `Pick(...)` por cada fila que sobrevivió sus filtros mecánicos,
  3. llamar `emit(modelo=..., fecha=..., picks=[...], salida_motor=..., precios=...)`.

Lo que NO hace: no decide cuáles picks son buenos. El filtro mecánico vive en cada motor
(es distinto por deporte). Aquí solo se persiste, se congela y se deja rastro.
"""
import hashlib
import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")

AUDITOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "auditor")
PREDICCIONES = os.path.join(AUDITOR, "predicciones")
REGISTRO = os.path.join(AUDITOR, "registro.md")
MANIFIESTOS = os.path.join(AUDITOR, "manifiestos")

# Salida por modelo. Medido 2026-07-29 / corregido 2026-08-03: MLB y TENIS se venden al
# take-profit; el resto aguanta a settlement. No se cambia sin medición nueva.
SALIDA = {"MLB": "TP", "TENIS": "TP", "CLIMA": "settlement", "SOCCER": "settlement"}


@dataclass
class Pick:
    """Un pick que ya pasó los filtros mecánicos del motor.

    Todos los números vienen del motor, nunca recalculados aquí: este módulo los copia
    tal cual al archivo, al registro y al manifiesto, que es justamente el punto — que
    las tres salidas no puedan diferir entre sí.
    """
    ticker: str
    lado: str          # "KC over 3.5", "NO Rakhimova", etc.
    mercado: str       # "TOTAL 7.5", "ML", "TT KC 3.5"
    entrada: int       # ¢ — el ask congelado que se usó
    p_mod: int         # % entero del modelo
    edge: int          # ¢
    tp: int            # ¢ (OBJ del motor)
    stop: int          # ¢
    max_pagable: int   # ¢ (= fair = p_mod)
    tamano: float      # $
    razonamiento: str
    partido: str = ""
    flags: str = ""
    # Modo B: por qué este candidato llegó a pick. "edge" = lo eligió el orden mecánico;
    # cualquier otra cosa es research y queda auditable contra el CLV en la Etapa 3.
    motivo_seleccion: str = "edge"

    @property
    def fair(self):
        return self.p_mod


@dataclass
class Candidato:
    """Fila limpia que sobrevivió el filtro mecánico y queda disponible para el research.

    Modo B (decidido 2026-08-14): el motor emite el conjunto de candidatos limpios; el
    research puede REORDENAR y ELEGIR dentro de ellos, con motivo, pero NUNCA agregar un
    ticker que el motor no evaluó. Así se cierra el hueco real (nadie veía por qué morían
    205 de 212 candidatos) sin quitarle al research lo único que el motor no sabe hacer:
    leer que el abridor viene 1-4 con 6.50 de ERA.
    """
    ticker: str
    lado: str
    mercado: str
    entrada: int
    p_mod: int
    edge: int
    tp: int
    stop: int
    max_pagable: int
    tamano: float
    partido: str = ""
    flags: str = ""


@dataclass
class Descartado:
    """Fila con edge suficiente que NO llegó a pick, con su motivo.

    Va al manifiesto (no al registro): es la auditoría del filtro que hoy nadie podía
    revisar. Motivo en las categorías del SKILL.md: research / correlación / flag: X / tope.
    """
    ticker: str
    mercado: str
    entrada: int
    p_mod: int
    edge: int
    motivo: str
    partido: str = ""


@contextmanager
def capture():
    """Captura el stdout del motor sin perderlo de pantalla.

    El motor imprime igual que siempre; además guardamos el texto para escribirlo
    íntegro en `predicciones/`. Así la salida guardada es literalmente la que se vio,
    no una reconstrucción.
    """
    buf = io.StringIO()
    real = sys.stdout

    class Tee:
        def write(self, s):
            real.write(s)
            buf.write(s)

        def flush(self):
            real.flush()

    sys.stdout = Tee()
    try:
        yield buf
    finally:
        sys.stdout = real


def sha(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def _fila_registro(fecha, modelo, p):
    """16 columnas, en el orden del header de registro.md (verificado 2026-08-14)."""
    ref = f"{fecha}-{modelo.lower()}.md#PICK-{p.idx}"
    contrato = f"{p.ticker} {p.lado} — {p.razonamiento}"
    return (f"| {fecha} | {modelo} | {contrato} | {p.entrada}¢ | ${p.tamano:.2f} | PAPER "
            f"| | | | {p.p_mod}% | {p.fair}¢ | {p.edge:+d}¢ | {p.tp}¢ | | | {ref} |")


def emit(modelo, fecha, picks, salida_motor, precios_ts, descartados=None,
         candidatos=None, nombre_archivo=None):
    """Escribe predicciones/, filas PAPER y manifiesto. Devuelve el dict del manifiesto.

    modelo:       "MLB" | "TENIS" | "CLIMA" | "SOCCER"
    fecha:        "AAAA-MM-DD"
    picks:        [Pick]
    salida_motor: stdout íntegro del motor (de `capture()`)
    precios_ts:   ISO del momento EXACTO de la bajada de precios de Kalshi. No es la hora
                  de la corrida: dos corridas separadas por segundos difieren porque el
                  mercado se mueve (medido 2026-08-05, un SPREAD pasó de +5¢ a +4¢ y de
                  invertible a descartado). Congelarlo es lo que evita que verify.py
                  grite sin razón.
    """
    descartados = descartados or []
    candidatos = candidatos or []

    # Modo B, limite innegociable: el research elige DENTRO de los candidatos del motor.
    # Un ticker que el motor no evaluo no puede llegar a pantalla por ninguna via.
    if candidatos:
        validos = {c.ticker for c in candidatos}
        intrusos = [p.ticker for p in picks if p.ticker not in validos]
        if intrusos:
            raise ValueError(
                "PICK NO EVALUADO POR EL MOTOR: " + ", ".join(intrusos) +
                ". El research puede quitar y reordenar candidatos, nunca agregar "
                "tickers que el motor no evaluo.")

    os.makedirs(PREDICCIONES, exist_ok=True)
    os.makedirs(MANIFIESTOS, exist_ok=True)

    for i, p in enumerate(picks, 1):
        p.idx = i

    slug = nombre_archivo or f"{fecha}-{modelo.lower()}.md"
    ruta_pred = os.path.join(PREDICCIONES, slug)
    ahora = datetime.now(PT)
    corrida = ahora.strftime("%Y-%m-%dT%H:%M:%S%z")

    # --- (1) predicciones/ : salida íntegra + un ancla por pick ---
    partes = []
    if not os.path.exists(ruta_pred):
        partes.append(f"# Predicciones {modelo} — {fecha}\n")
    partes.append(f"\n## CORRIDA {corrida} (precios congelados {precios_ts})\n")
    partes.append("\n## SALIDA DEL MOTOR (íntegra)\n\n```\n" + salida_motor + "\n```\n")
    for p in picks:
        partes.append(
            f"\n### PICK-{p.idx}\n"
            f"Contrato: {p.ticker} {p.lado}\n"
            f"P_mod: {p.p_mod}% | Fair: {p.fair}¢ | Entrada: {p.entrada}¢ | "
            f"Edge: {p.edge:+d}¢ | TP (OBJ): {p.tp}¢ | Stop: {p.stop}¢ | "
            f"Máx. pagable: {p.max_pagable}¢ | Salida: {SALIDA.get(modelo, 'settlement')} | "
            f"Tamaño: ${p.tamano:.2f}\n"
            f"Seleccion: {p.motivo_seleccion}\n"
            f"Razonamiento: {p.razonamiento}\n")
    with open(ruta_pred, "a", encoding="utf-8") as f:
        f.write("".join(partes))

    # --- (2) registro central : una fila PAPER por pick ---
    # Se agrega al final del archivo. Las filas existentes NO se tocan: una posición
    # abierta (ej. la de $10.40 del 2026-08-02) queda intacta por construcción.
    if picks:
        with open(REGISTRO, "a", encoding="utf-8") as f:
            for p in picks:
                f.write(_fila_registro(fecha, modelo, p) + "\n")

    # --- (3) manifiesto : lo que verify.py va a exigir ---
    man = {
        "modelo": modelo,
        "fecha": fecha,
        "corrida_ts": corrida,
        "precios_ts": precios_ts,
        "hash_salida_motor": sha(salida_motor),
        "archivo_predicciones": slug,
        "tickers": [p.ticker for p in picks],
        "picks": [{**{k: v for k, v in asdict(p).items()}, "idx": p.idx} for p in picks],
        "descartados": [asdict(d) for d in descartados],
        "candidatos_limpios": [asdict(c) for c in candidatos],
    }
    ruta_man = os.path.join(MANIFIESTOS, f"{fecha}-{modelo.lower()}.json")
    with open(ruta_man, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 78}")
    print(f"EMIT OK — {modelo} {fecha}")
    print(f"{'=' * 78}")
    print(f"  predicciones : {ruta_pred}")
    print(f"  registro     : {len(picks)} fila(s) PAPER agregada(s)")
    print(f"  manifiesto   : {ruta_man}")
    print(f"  precios congelados: {precios_ts}")
    if picks:
        print("  tickers emitidos:")
        for p in picks:
            print(f"    PICK-{p.idx}  {p.ticker}  {p.lado}  {p.entrada}¢  "
                  f"edge {p.edge:+d}¢  ${p.tamano:.2f}")
    else:
        print("  SIN PICKS — no hay edge hoy. No se escribió fila al registro.")
    if candidatos:
        elegidos_tk = {p.ticker for p in picks}
        libres = [c for c in candidatos if c.ticker not in elegidos_tk]
        print(f"  candidatos limpios: {len(candidatos)} "
              f"({len(picks)} elegidos, {len(libres)} disponibles para research)")
    if descartados:
        print(f"  descartados registrados: {len(descartados)}")
    print(f"\nVerifica antes de imprimir el reporte:")
    print(f"  python3 core/verify.py {modelo.lower()} {fecha}")
    return man
