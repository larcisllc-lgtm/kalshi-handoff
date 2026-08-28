#!/usr/bin/env python3
"""Handoff Grok -> Claude: lee fills.json del repo kalshi-handoff (lo que Kalshi Bot
escribió tras decidir sobre latest.json) y actualiza el registro central de
trading-auditor — cambia la fila PAPER correspondiente a TOMADO con el fill real.

Diseño acordado 2026-08-28 (project_kalshibot-handoff-diseno-final): Claude escribe el
fill al registro, no el usuario. Este script corre por cron ~30min después de cada
ventana de Grok (4:30am/8:30am/12:30pm PT aprox) vía el skill `schedule`.

Contrato esperado de fills.json (que Grok escribe al repo, mismo mecanismo de
build_ticket.py en reversa):
{
  "fills": [
    {"ticker": "KXMLB...-X", "tomado": true, "entrada_real_c": 52, "tamano_usd": 2.0},
    {"ticker": "KXMLB...-Y", "tomado": false, "motivo": "sin liquidez"}
  ]
}
Un ticker con "tomado": false NO se escribe al registro — la fila PAPER se queda tal
cual (nunca fue una posición real, no hay nada que conciliar).

Uso:
  python3 record_fill.py              # lee fills.json del repo, actualiza registro.md
  python3 record_fill.py --dry-run    # solo muestra qué cambiaría
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
REGISTRO = os.path.join(REPO, "auditor", "registro.md")
FILLS_PATH = os.path.join(REPO, "fills.json")


def git_pull():
    """Trae lo último que Grok haya empujado antes de leer fills.json."""
    subprocess.run(["git", "-C", REPO, "pull", "--quiet"], check=True)


def cargar_fills():
    if not os.path.exists(FILLS_PATH):
        return []
    d = json.load(open(FILLS_PATH, encoding="utf-8"))
    return d.get("fills", [])


def actualizar_registro(fills, dry_run):
    """Busca cada ticker en registro.md (columna Contrato contiene el ticker) y, si
    está en estado PAPER, lo reescribe como TOMADO con el fill real. No toca filas que
    ya estén en TOMADO o EXCLUIDO (evita duplicar una conciliación manual previa)."""
    if not fills:
        print("Sin fills nuevos en fills.json.")
        return

    lineas = open(REGISTRO, encoding="utf-8").read().splitlines()
    cambios = 0

    for fill in fills:
        if not fill.get("tomado"):
            print(f"  {fill['ticker']}: NO tomado ({fill.get('motivo', 'sin motivo')}) "
                  "— fila PAPER se queda igual")
            continue
        ticker = fill["ticker"]
        entrada_c = fill["entrada_real_c"]
        tamano = fill["tamano_usd"]
        encontrado = False
        for i, linea in enumerate(lineas):
            if not linea.strip().startswith("|") or ticker not in linea:
                continue
            celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
            if len(celdas) < 6:
                continue
            if celdas[5].upper() != "PAPER":
                continue  # ya es TOMADO/EXCLUIDO, no se toca dos veces
            celdas[3] = f"{entrada_c}¢"
            celdas[4] = f"${tamano:.2f}"
            celdas[5] = "TOMADO"
            nueva = "| " + " | ".join(celdas) + " |"
            print(f"  {ticker}: PAPER -> TOMADO ({entrada_c}¢, ${tamano:.2f})")
            if not dry_run:
                lineas[i] = nueva
            cambios += 1
            encontrado = True
            break
        if not encontrado:
            print(f"  AVISO: {ticker} no encontrado como PAPER en registro.md "
                  "(¿ya se conciliaba a mano, o el ticker no matchea?)")

    if cambios and not dry_run:
        with open(REGISTRO, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        print(f"\n{cambios} fila(s) actualizadas en {REGISTRO}")
    elif cambios:
        print(f"\n--dry-run: {cambios} fila(s) se hubieran actualizado")


def marcar_fills_procesados(dry_run):
    """Renombra fills.json a fills-procesado-{ts}.json y hace push, para que la
    próxima corrida no vuelva a procesar los mismos fills dos veces."""
    if dry_run or not os.path.exists(FILLS_PATH):
        return
    import time
    ts = int(time.time())
    procesado = os.path.join(REPO, f"fills-procesado-{ts}.json")
    os.rename(FILLS_PATH, procesado)
    subprocess.run(["git", "-C", REPO, "add", "-A"], check=True)
    subprocess.run(["git", "-C", REPO, "commit", "-m", f"fills procesados {ts}"],
                    check=True)
    subprocess.run(["git", "-C", REPO, "push"], check=True)


def main():
    dry_run = "--dry-run" in sys.argv
    git_pull()
    fills = cargar_fills()
    actualizar_registro(fills, dry_run)
    marcar_fills_procesados(dry_run)


if __name__ == "__main__":
    main()
