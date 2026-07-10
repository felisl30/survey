# Comparación final MC-500 S0-S3

Fuente: `docs/experimentos/resumen_comparacion_mc500_s0_s3.md`, integrado en el informe Overleaf.

- Modelo fijo para la extensión multi-dataset: `gpt-5-mini`.
- Datasets: MuSiQue-MC-500, HotpotQA-MC-500 y 2Wiki-MC-500.
- Sistemas: S0 directo, S1 RAG top-k fijo, S2 RAG adaptativo, S3 FLARE-like MC.
- Todos los sistemas reportados tienen `n=500`, `valid_format_rate=1.0` y outputs completos según el resumen de integración.

## Lectura usada en el informe

S3 obtiene la mayor accuracy en los tres datasets, pero también tiene el mayor costo relativo: 2.31x, 2.51x y 2.37x los tokens de S0 en MuSiQue, HotpotQA y 2Wiki. S1 queda como baseline simple y competitivo, especialmente en 2Wiki, donde 0.892 queda muy cerca de 0.898 de S3 con menor costo.

## Archivos

- `mc500_s0_s3_comparison_final.csv`: tabla completa.
- `mc500_s0_s3_comparison_final.md`: tabla en Markdown.
- `mc500_pivot_accuracy_final.csv`: accuracy por dataset/sistema.
- `mc500_pivot_tokens_final.csv`: tokens promedio por dataset/sistema.
- `mc500_best_by_dataset_final.csv`: mejor sistema por dataset.
