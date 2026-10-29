# MELD 实验结果记录

MELD Dev（1,109 条），数值为百分比。类别 Acc 表示该类别的 Recall；Overall 的 wF1 和 mF1 分别为 Weighted-F1 和 Macro-F1。Test 尚未评估。

| Methods | Neutral Acc | Neutral F1 | Surprise Acc | Surprise F1 | Fear Acc | Fear F1 | Sadness Acc | Sadness F1 | Joy Acc | Joy F1 | Disgust Acc | Disgust F1 | Anger Acc | Anger F1 | Overall Acc | Overall wF1 | Overall mF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Zero-shot (T) | 71.49 | 71.79 | 72.67 | 50.70 | 30.00 | 33.33 | 44.14 | 47.57 | 44.17 | 51.61 | 13.64 | 22.22 | 49.02 | 55.97 | 59.15 | 59.00 | 47.60 |
| Qwen3-8B LoRA (T) | 87.45 | 81.23 | 69.33 | 66.67 | 22.50 | 31.03 | 46.85 | 55.61 | 64.42 | 66.04 | 40.91 | 36.73 | 60.78 | 65.96 | 70.60 | 69.66 | 57.61 |

在同一 MELD Dev 集上，纯文本 LoRA 相比 Zero-shot 的 Accuracy、Weighted-F1 和 Macro-F1 分别提高 11.45、10.66 和 10.01 个百分点；Disgust F1 从 22.22 提高到 36.73，Fear F1 从 33.33 降至 31.03。整体上 LoRA 为后续多模态实验提供了较强的纯文本基线，少数类的变化仍需结合后续实验观察。
