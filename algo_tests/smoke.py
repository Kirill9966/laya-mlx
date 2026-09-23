"""Примитивные тесты Laya для ALGO GIT: работает ли, как быстро, читает ли числа рынка."""
import json, statistics, time
import laya_mlx as laya

agent = laya.load("models/laya-mlx", dtype="float16")

print("=== 1. Их пример (письмо в поддержку) ===")
r = agent.predict(
    "I was billed twice. Please refund the duplicate today.",
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": ["billing", "technical", "sales"]},
     "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}},
)
print(json.dumps(r["answers"], ensure_ascii=False))

print("\n=== 2. Скорость на M2 (один вопрос, 50 прогонов после 5 разогревочных) ===")
q1 = {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}
for _ in range(5):
    agent.predict("I was billed twice.", q1)
ms = []
for _ in range(50):
    t = time.perf_counter(); agent.predict("I was billed twice.", q1); ms.append((time.perf_counter() - t) * 1000)
ms.sort()
print(f"P50 {statistics.median(ms):.1f} мс · P95 {ms[int(len(ms)*0.95)-1]:.1f} мс")

print("\n=== 3. Состояние рынка → решение (без дообучения) ===")
Q = {
    "action": {"type": "choice", "instructions": "What should a trader do on the next 1h candle?",
               "criteria": {"long": "buy, expect price up", "short": "sell, expect price down",
                            "wait": "no clear edge, stay out"}},
    "up": {"type": "noul", "instructions": "Will the price be higher one hour from now?"},
}
states = {
    "бычье":     {"symbol": "BTCUSDT", "tf": "1h", "ret_1h_pct": 2.4, "ret_24h_pct": 6.1, "volume_z": 5.2,
                  "rsi14": 71, "close_vs_ema200_pct": 4.5, "trend": "up"},
    "медвежье":  {"symbol": "BTCUSDT", "tf": "1h", "ret_1h_pct": -2.4, "ret_24h_pct": -6.1, "volume_z": 5.2,
                  "rsi14": 29, "close_vs_ema200_pct": -4.5, "trend": "down"},
    "тихое":     {"symbol": "BTCUSDT", "tf": "1h", "ret_1h_pct": 0.05, "ret_24h_pct": 0.2, "volume_z": 0.1,
                  "rsi14": 50, "close_vs_ema200_pct": 0.1, "trend": "flat"},
    "бычье без слова trend": {"symbol": "BTCUSDT", "tf": "1h", "ret_1h_pct": 2.4, "ret_24h_pct": 6.1,
                  "volume_z": 5.2, "rsi14": 71, "close_vs_ema200_pct": 4.5},
    "медвежье без слова trend": {"symbol": "BTCUSDT", "tf": "1h", "ret_1h_pct": -2.4, "ret_24h_pct": -6.1,
                  "volume_z": 5.2, "rsi14": 29, "close_vs_ema200_pct": -4.5},
}
for name, s in states.items():
    a = agent.predict(s, Q)["answers"]
    p = a["action"]["probabilities"]
    print(f"{name:26s} long {p['long']:.2f} · short {p['short']:.2f} · wait {p['wait']:.2f} · P(вверх) {a['up']['noul']:.2f}")
