# AIA for DrugOOD IC50

기존 AIA의 causal/adversarial mask와 alternating optimization을 DrugOOD IC50의 dense node/edge feature에 맞게 확장한 구현이다.

## 적용 규약

- binary classification
- Virtual Node가 없는 4-layer GIN: 2-layer front + 2-layer back
- hidden dimension 128, dropout 0.5, sum pooling
- Adam, learning rate `1e-3`, batch size 128
- full-graph ERM pretraining 10 epochs
- AIA main training 최대 50 epochs
- OOD validation accuracy 기준 early-stopping patience 10
- seed `{1, 2, 3, 4}` 결과의 mean/std 보고

## 단일 실행

```bash
cd /workspace/baselines/AIA/DrugOOD

python3 train_ic50.py \
  --domain assay \
  --stable-feature-ratio 0.5 \
  --adversarial-penalty-weight 0.5 \
  --device cuda:0
```

빠른 동작 확인에서는 다음 두 인자를 추가한다.

```bash
--erm-pretrain-epochs 1 --epochs 1 --num-workers 0
```

각 실행은 `best.pt`, `history.json`, `summary.json`을 생성한다. ERM pretraining은 encoder와 predictor를 학습하며, AIA checkpoint 선택과 early stopping은 causal masker가 학습되는 main 단계부터 수행한다.

## 하이퍼파라미터 탐색

문서의 탐색 공간을 그대로 사용한다.

| 인자 | AIA 원 구현 대응 | 탐색 값 |
|---|---|---|
| `--stable-feature-ratio` | `cau_gamma` | `{0.1, 0.3, 0.5, 0.7, 0.9}` |
| `--adversarial-penalty-weight` | `adv_reg` | `{0.01, 0.1, 0.2, 0.5, 1.0, 3.0, 5.0}` |

명령 확인:

```bash
python3 sweep_ic50.py --domains assay --dry-run
```

실제 탐색:

```bash
python3 sweep_ic50.py --domains assay --device cuda:0
```

기본적으로 35개 하이퍼파라미터 조합을 seed 4개로 실행하므로 domain당 140 jobs이다. 완료 후 `sweeps/aggregate.json`에 조합별 mean/std와 validation 기준 best 조합을 기록한다.

## 판단한 항목

- AIA 원 구현은 OGB categorical feature encoder를 사용하므로 DrugOOD의 39차원 node feature와 10차원 edge feature용 linear encoder로 교체했다.
- 원 구현의 mean pooling은 공통 실험 규약에 맞춰 sum pooling으로 변경했다.
- DrugOOD hidden dimension은 공통 IC50 구현과 맞춰 128로 정했고, 문서에 없는 dropout은 AIA 공식 구현의 0.5를 유지했다.
- `stable feature ratio`는 causal mask의 목표 비율인 `cau_gamma`, `adversarial penalty weight`는 attacker mask regularizer 계수인 `adv_reg`로 해석했다.
- AIA 원 구현의 mask regularizer와 공격자 목적함수 부호는 그대로 유지했다.
- ERM pretraining 후 encoder/predictor weight는 유지하고 AIA optimizer는 새로 생성한다.
