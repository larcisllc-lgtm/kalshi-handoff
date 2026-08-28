#!/usr/bin/env python3
"""Núcleo compartido de los motores de Kalshi (Etapa 2 del blindaje).

Qué resuelve: banca, topes y fracción de Kelly vivían duplicados en 5 archivos. Subir de
banca obligaba a editar 5 números y nada avisaba si uno quedaba desincronizado. Ahora
todo sale de `config.json`, y las diferencias intencionales por deporte (clima pide ¼
Kelly y 7¢ de edge) son parámetros explícitos, no constantes regadas.

Uso desde un motor:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from core import cfg

    C = cfg("MLB")
    C.bankroll, C.max_trade, C.edge_min, C.kelly_frac
    C.sizing(prob, ask_c)              # ⅛ Kelly, tope aplicado
    C.sizing(prob, ask_c, conf_baja=True)
    C.conviction(edge_c)               # ALTO / MEDIO / BAJO

Lo que NO vive aquí: el modelo de cada deporte. Poisson, xG, ensemble y ranking siguen
siendo de su motor. Aquí solo el dinero y los umbrales.
"""
import json
import os

_RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

with open(_RUTA, encoding="utf-8") as _f:
    CONFIG = json.load(_f)


class ModelConfig:
    """Parámetros de riesgo de un modelo, ya resueltos contra la banca global."""

    def __init__(self, modelo):
        m = modelo.upper()
        if m not in CONFIG["modelos"]:
            raise KeyError(
                f"Modelo desconocido: {modelo}. Válidos: {', '.join(CONFIG['modelos'])}")
        b, p = CONFIG["banca"], CONFIG["modelos"][m]
        self.modelo = m
        self.bankroll = b["bankroll"]
        self.max_trade = b["max_trade"]
        self.max_open = b["max_open"]
        self.kelly_frac = p["kelly_frac"]
        self.kelly_frac_conf_baja = p["kelly_frac_conf_baja"]
        self.edge_min = p["edge_min"]
        self.edge_suspect = p["edge_suspect"]
        self.salida = p["salida"]
        self.max_por_partido = p["max_por_partido"]
        self._alto = CONFIG["conviction"]["alto_sobre"]

    def sizing(self, prob, ask_c, conf_baja=False):
        """Kelly fraccional sobre la banca, con tope por trade. `ask_c` en centavos.

        `prob` es la probabilidad del modelo en 0-1. Devuelve dólares redondeados a 2.
        Sin edge positivo devuelve 0.0: nunca dimensiona un pick que el modelo no favorece.
        """
        p = ask_c / 100.0
        if p <= 0 or p >= 1 or prob <= p:
            return 0.0
        f = (prob - p) / (1 - p)
        frac = self.kelly_frac_conf_baja if conf_baja else self.kelly_frac
        return round(min(frac * f * self.bankroll, self.max_trade), 2)

    def conviction(self, edge_c):
        return ("ALTO" if edge_c > self._alto
                else ("MEDIO" if edge_c >= self.edge_min else "BAJO"))


_cache = {}


def cfg(modelo):
    """Config de un modelo (cacheada). MLB | TENIS | SOCCER | CLIMA."""
    m = modelo.upper()
    if m not in _cache:
        _cache[m] = ModelConfig(m)
    return _cache[m]


# ---------- helpers de Kalshi compartidos ----------

def cents(m, side):
    """Precio ask de un lado en centavos enteros, o None si no hay precio publicado.

    Kalshi usa los campos con sufijo `_dollars`; los nombres viejos (`yes_ask`) devuelven
    null y hacen parecer que el mercado está muerto.
    """
    v = m.get(f"{side}_ask_dollars")
    try:
        c = round(float(v) * 100)
        return c if 0 < c < 100 else None
    except (TypeError, ValueError):
        return None


def liq_flag(m, vol_min=50, size_min=50):
    """'SIN-LIQUIDEZ' | 'LIQUIDEZ-BAJA' | '' según volumen y tamaño en el ask."""
    try:
        vol = float(m.get("volume_fp") or 0)
        sz = float(m.get("yes_ask_size_fp") or 0)
    except (TypeError, ValueError):
        return "SIN-LIQUIDEZ"
    if vol < vol_min and sz < size_min:
        return "LIQUIDEZ-BAJA"
    return ""


if __name__ == "__main__":
    print(f"config: {_RUTA}")
    print(f"banca ${CONFIG['banca']['bankroll']:.2f} | tope "
          f"${CONFIG['banca']['max_trade']:.2f}/trade | "
          f"${CONFIG['banca']['max_open']:.2f} abierto (global)\n")
    print(f"{'modelo':8s} {'Kelly':>7s} {'K-baja':>7s} {'edge':>5s} {'salida':>11s} "
          f"{'máx/partido':>12s}")
    for m in CONFIG["modelos"]:
        c = cfg(m)
        print(f"{m:8s} {c.kelly_frac:>7.4f} {c.kelly_frac_conf_baja:>7.4f} "
              f"{c.edge_min:>4d}¢ {c.salida:>11s} {c.max_por_partido:>12d}")
