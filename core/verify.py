#!/usr/bin/env python3
"""Verificador del contrato de emisión (Etapa 1 del blindaje).

Compara el manifiesto de la corrida contra lo que quedó en disco, y —si se le pasa el
reporte que se va a imprimir— contra la pantalla. Falla RUIDOSAMENTE: exit 1 y bloque
grande de error. Ese es el punto entero. El modo de falla real del 2026-08-01 fue
silencioso: 4 contratos operados que el archivo no conservó, y nadie se enteró en 7 días.

Uso:
  python3 verify.py mlb 2026-08-14                    # archivo vs manifiesto
  python3 verify.py mlb 2026-08-14 --reporte rep.md   # + pantalla vs manifiesto

Qué exige:
  1. El manifiesto existe.
  2. `predicciones/` tiene el bloque `## SALIDA DEL MOTOR (íntegra)`.
  3. Hay exactamente un `### PICK-n` por ticker del manifiesto.
  4. Cada ancla lleva P_mod, Fair, Entrada, Edge, TP, Stop, Máx. pagable y Tamaño.
  5. El registro tiene una fila PAPER por ticker, con la ref correcta.
  6. Con --reporte: los tickers impresos son EXACTAMENTE los del manifiesto, en ambas
     direcciones (impreso sin escribir = pick fantasma; escrito sin imprimir = pick perdido).
"""
import json
import os
import re
import sys

AUDITOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "auditor")
PREDICCIONES = os.path.join(AUDITOR, "predicciones")
REGISTRO = os.path.join(AUDITOR, "registro.md")
MANIFIESTOS = os.path.join(AUDITOR, "manifiestos")

CAMPOS_ANCLA = ["P_mod:", "Fair:", "Entrada:", "Edge:", "TP (OBJ):", "Stop:",
                "Máx. pagable:", "Salida:", "Tamaño:"]

# Modo B: un pick elegido por research (motivo distinto de "edge") tiene que declararlo.
# Sin esto, "el research lo subió" se vuelve otra vez una afirmación no auditable.
MOTIVO_POR_DEFECTO = "edge"


def fallar(errores, modelo, fecha):
    print("\n" + "!" * 78)
    print(f"!!  VERIFICACIÓN FALLIDA — {modelo.upper()} {fecha}")
    print("!" * 78)
    for e in errores:
        print(f"  ✗ {e}")
    print("\n  NO IMPRIMAS EL REPORTE. Corrige y vuelve a correr el motor con --emit.")
    print("  Lo que se imprime y lo que queda en disco tienen que ser el mismo conjunto.")
    print("!" * 78 + "\n")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("uso: verify.py <modelo> <fecha> [--reporte archivo.md]")
        sys.exit(2)
    modelo, fecha = args[0].lower(), args[1]
    reporte = None
    if "--reporte" in args:
        reporte = args[args.index("--reporte") + 1]

    errores = []

    ruta_man = os.path.join(MANIFIESTOS, f"{fecha}-{modelo}.json")
    if not os.path.exists(ruta_man):
        fallar([f"No existe el manifiesto {ruta_man}. El motor no corrió con --emit, "
                f"o corrió con otra fecha."], modelo, fecha)
    with open(ruta_man, encoding="utf-8") as f:
        man = json.load(f)

    tickers_man = man["tickers"]
    ruta_pred = os.path.join(PREDICCIONES, man["archivo_predicciones"])

    # --- 1. archivo de predicciones ---
    if not os.path.exists(ruta_pred):
        errores.append(f"Falta el archivo de predicciones {ruta_pred}")
        pred = ""
    else:
        with open(ruta_pred, encoding="utf-8") as f:
            pred = f.read()
        if "## SALIDA DEL MOTOR (íntegra)" not in pred:
            errores.append(f"{man['archivo_predicciones']} no tiene el bloque "
                           "'## SALIDA DEL MOTOR (íntegra)'")

    # --- 2. anclas por pick ---
    anclas = re.findall(r"### PICK-(\d+)\nContrato: (\S+)", pred)
    tickers_ancla = [t for _, t in anclas]
    for tk in tickers_man:
        if tk not in tickers_ancla:
            errores.append(f"{tk} está en el manifiesto pero NO tiene ancla ### PICK-n "
                           f"en {man['archivo_predicciones']}")
    for tk in tickers_ancla:
        if tk not in tickers_man:
            errores.append(f"{tk} tiene ancla en el archivo pero NO está en el manifiesto")

    # cada ancla completa (esto es lo que estaba al 47% en TP)
    bloques = re.split(r"### PICK-\d+", pred)[1:]
    for b in bloques:
        m = re.search(r"Contrato: (\S+)", b)
        tk = m.group(1) if m else "?"
        faltan = [c for c in CAMPOS_ANCLA if c not in b]
        if faltan:
            errores.append(f"ancla de {tk} incompleta, faltan: {', '.join(faltan)}")

    # --- 2b. modo B: ningún pick fuera de los candidatos del motor ---
    cands = {c["ticker"] for c in man.get("candidatos_limpios", [])}
    if cands:
        for p in man["picks"]:
            if p["ticker"] not in cands:
                errores.append(
                    f"PICK NO EVALUADO: {p['ticker']} no está entre los candidatos "
                    f"limpios del motor. El research elige dentro de la lista, no agrega.")
            elif p.get("motivo_seleccion", MOTIVO_POR_DEFECTO) != MOTIVO_POR_DEFECTO:
                # elegido por research: válido, pero el motivo tiene que ser algo real
                if len(p.get("motivo_seleccion", "").strip()) < 10:
                    errores.append(
                        f"{p['ticker']} se eligió por research pero el motivo está vacío "
                        f"o es demasiado corto: '{p.get('motivo_seleccion')}'")

    # --- 3. filas del registro ---
    if os.path.exists(REGISTRO):
        with open(REGISTRO, encoding="utf-8") as f:
            reg = f.read()
        for p in man["picks"]:
            ref = f"{man['archivo_predicciones']}#PICK-{p['idx']}"
            filas = [l for l in reg.splitlines()
                     if p["ticker"] in l and ref in l and "| PAPER " in l]
            if not filas:
                errores.append(f"{p['ticker']} no tiene fila PAPER en registro.md con "
                               f"ref {ref}")
            elif len(filas) > 1:
                errores.append(f"{p['ticker']} tiene {len(filas)} filas duplicadas en "
                               f"registro.md")
            else:
                fila = filas[0]
                for etiqueta, val in (("P_mod", f"{p['p_mod']}%"),
                                      ("Entrada", f"{p['entrada']}¢"),
                                      ("TP", f"{p['tp']}¢")):
                    if val not in fila:
                        errores.append(f"{p['ticker']}: la fila del registro no lleva el "
                                       f"{etiqueta} del manifiesto ({val})")
    else:
        errores.append(f"No existe {REGISTRO}")

    # --- 4. pantalla vs manifiesto ---
    if reporte:
        if not os.path.exists(reporte):
            errores.append(f"No existe el reporte {reporte}")
        else:
            with open(reporte, encoding="utf-8") as f:
                txt = f.read()
            impresos = set(re.findall(r"\bKX[A-Z0-9]+-[A-Z0-9]+-?[A-Z0-9]*\b", txt))
            for tk in tickers_man:
                if tk not in txt:
                    errores.append(f"PICK PERDIDO: {tk} se escribió a disco pero NO "
                                   f"aparece en el reporte")
            for tk in impresos:
                if tk not in tickers_man:
                    errores.append(f"PICK FANTASMA: {tk} aparece en el reporte pero NO "
                                   f"se escribió a disco")

    if errores:
        fallar(errores, modelo, fecha)

    print(f"VERIFICACIÓN OK — {modelo.upper()} {fecha}")
    print(f"  {len(tickers_man)} pick(s), archivo íntegro, anclas completas, "
          f"registro consistente.")
    print(f"  precios congelados: {man['precios_ts']}")
    print(f"  hash salida motor : {man['hash_salida_motor']}")
    if reporte:
        print(f"  reporte verificado contra manifiesto: sin fantasmas ni perdidos.")
    if man.get("descartados"):
        print(f"  descartados con motivo: {len(man['descartados'])}")


if __name__ == "__main__":
    main()
