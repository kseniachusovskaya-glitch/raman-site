"""Создаёт синтетический демо-спектр Si в data/examples/synthetic_si.txt.

Запуск: uv run python scripts/make_example.py
"""

from pathlib import Path

import numpy as np

x = np.arange(100, 1000, 0.5)
rng = np.random.default_rng(0)  # фиксированное зерно → один и тот же шум
peak = 1000 / (1 + ((x - 520.7) / 2.0) ** 2)  # пик c-Si, лоренциан, FWHM 4 см⁻¹
background = 50 + 0.05 * x  # наклонный фон
y = peak + background + rng.normal(0, 5, x.size)

out = Path(__file__).parent.parent / "data" / "examples" / "synthetic_si.txt"
np.savetxt(out, np.column_stack([x, y]), delimiter="\t", fmt="%.2f",
           header="Raman shift (cm-1)\tIntensity")
print(f"Сохранено: {out}")
