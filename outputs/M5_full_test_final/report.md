# M5 – hodnotenie na kompletných pôvodných testovacích dátach

Vygenerované: 2026-10-25 16:27

Toto je samostatné hodnotenie bez ďalšieho trénovania. Model M5 (CoAtNet) bol vybraný v pôvodnom porovnaní podľa validačných výsledkov; použité sú váhy `M5_run3/weights.pt`.

## Dáta

Použitý bol celý pôvodný, vopred oddelený priečinok `test`: **10500 obrázkov**.

- Fire: 3500
- Smoke: 3500
- Non_Fire: 3500

## Metriky

- Loss: 0.0545
- Accuracy: 98.3333%
- Precision (macro): 0.9834
- Recall (macro): 0.9833
- F1 (macro): 0.9833

`confusion_matrix.png` obsahuje maticu zámien a `predictions.csv` všetky predikcie.