# Previsão da Nota Final de Alunos com Machine Learning

**Integrante:** Gustavo Mendes Ventieri Mariano

Repositório: https://github.com/gustavoventieriestudos/p3-ia

Projeto que cria uma base fictícia de 500 alunos e treina um modelo de regressão para prever a nota final a partir de horas de estudo, frequência, atividades entregues e nota anterior.

## Conteúdo
- `modelo_notas.py` – código completo (geração dos dados, modelo, estatísticas e gráficos)
- `base_alunos.csv` – base fictícia gerada
- `graficos/` – gráficos produzidos
- `resultados.json`, `estatistica_descritiva.csv`, `comparacao_modelos.csv` – resultados numéricos
- `Relatorio_Previsao_Notas.docx` – relatório com análise

## Como executar
```bash
pip install numpy pandas matplotlib scipy scikit-learn
python modelo_notas.py
```

## Resultados principais (conjunto de teste, 100 alunos)
| Indicador | Valor |
|---|---|
| R² | 0,681 |
| MAE | 0,498 |
| RMSE | 0,666 |
| Média do erro | 0,007 |
| Desvio padrão do erro | 0,669 |
| IC 95% da média do erro | [-0,126 ; 0,140] |
| Intervalo de erro individual 95% | [-1,30 ; +1,32] |

Conclusão: modelo sem viés, com erro típico de ~0,5 ponto; desempenho aceitável como apoio à análise pedagógica.
