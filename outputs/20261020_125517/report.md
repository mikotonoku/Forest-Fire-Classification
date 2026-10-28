# HNS Projekt 1 - výsledky

Vygenerované: 2026-10-20 17:26

## Nastavenie

Transfer learning, 15 epoch, batch size 8, AdamW (lr 0.0003), 80/10/10 stratifikované rozdelenie. Použité dáta: náhodná vyvážená podvzorka 1000 obrázkov na triedu. Každý model bol spustený 3-krát.

## Súhrn (priemer a smerodajná odchýlka)

| model   |   ('accuracy', 'mean') |   ('accuracy', 'std') |   ('precision_macro', 'mean') |   ('precision_macro', 'std') |   ('recall_macro', 'mean') |   ('recall_macro', 'std') |   ('f1_macro', 'mean') |   ('f1_macro', 'std') |
|:--------|-----------------------:|----------------------:|------------------------------:|-----------------------------:|---------------------------:|--------------------------:|-----------------------:|----------------------:|
| M1      |                 0.9789 |                0.0051 |                        0.9793 |                       0.005  |                     0.9789 |                    0.0051 |                 0.9789 |                0.0051 |
| M2      |                 0.9767 |                0.0088 |                        0.9775 |                       0.0078 |                     0.9767 |                    0.0088 |                 0.9767 |                0.0087 |
| M3      |                 0.99   |                0.0033 |                        0.9901 |                       0.0033 |                     0.99   |                    0.0033 |                 0.99   |                0.0033 |
| M4      |                 0.9789 |                0.0084 |                        0.9792 |                       0.0084 |                     0.9789 |                    0.0084 |                 0.9789 |                0.0084 |
| M5      |                 0.9933 |                0.0067 |                        0.9934 |                       0.0065 |                     0.9933 |                    0.0067 |                 0.9933 |                0.0067 |

## Najlepší beh každého typu

`best_runs.csv` označuje beh s najnižšou validation loss; pre prezentáciu použite jeho `history.png`, `confusion_matrix.png` a Grad-CAM obrázky. Testovacie metriky nie sú použité na výber modelu.

| model   |   run |   accuracy |   precision_macro |   recall_macro |   f1_macro |   best_valid_loss |   epochs_completed | augment   | unfreeze   |
|:--------|------:|-----------:|------------------:|---------------:|-----------:|------------------:|-------------------:|:----------|:-----------|
| M1      |     3 |   0.98     |          0.980287 |       0.98     |   0.980031 |         0.0531387 |                 14 | True      | True       |
| M2      |     3 |   0.98     |          0.980287 |       0.98     |   0.980031 |         0.0674651 |                 12 | True      | True       |
| M3      |     3 |   0.993333 |          0.993333 |       0.993333 |   0.993333 |         0.0654581 |                 10 | True      | True       |
| M4      |     2 |   0.97     |          0.970154 |       0.97     |   0.969963 |         0.0408199 |                 15 | True      | True       |
| M5      |     3 |   1        |          1        |       1        |   1        |         0.0385241 |                 10 | True      | True       |

## Súbory

Pre každý beh sú dostupné `history.png`, `confusion_matrix.png`, váhy a predikcie testovacích dát. Interpretujte výsledky s ohľadom na odchýlku troch behov, nie len najvyššiu presnosť.

## Doplnkové experimenty

`runs.csv` obsahuje aj jeden beh M1 bez augmentácie a jeden beh M1 bez postupného rozmrazenia. Porovnajte ich s porovnateľným základným behom M1.