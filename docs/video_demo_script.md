# FinAdvisor AI v3 — Suggested 3–5 Minute Demo

**0:00–0:35 — Problem and architecture**  
Explain that FinAdvisor separates five-day ML prediction from long-horizon risk allocation. Show the architecture or the Methodology tab.

**0:35–1:10 — Data provenance and asset universe**  
Show the 15 formal assets: the 12-ETF core plus AAPL, MSFT and NVDA. Explain that the stocks are a separate robustness extension, frozen and hashed independently, while all assets use adjusted-close-only features.

**1:10–2:00 — Live advisor**  
Select one ETF and one of AAPL/MSFT/NVDA. Show probability, Buy/Hold/Sell signal, regime, dynamic allocation and explanation.

**2:00–2:50 — Walk-forward methodology**  
Show the fold table. Explain expanding training history and the five-observation purge that prevents five-day target labels from reaching into the next test fold.

**2:50–3:50 — Empirical results**  
Show the regenerated 30-run result CSV/summary after the connected reproduction run. Emphasise that all losses and failures are retained. Contrast the diversified ETF core with the individual-equity extension rather than cherry-picking one stock.

**3:50–4:30 — Conclusion and limitations**  
State that the system is evaluated as decision support, not guaranteed financial advice. Mention price-only inputs, simplified execution costs and the need for separate intraday data before ORB can be tested honestly.
