# BBEH unseen 61: Qwen3.6 권장 thinking 64k 비교

## 결론

Qwen3.6 권장 thinking sampling과 64k 출력 상한을 함께 적용했을 때, 고난도 4점 v2 GRPO가 **41/61**로 base **33/61**보다 8문항 높았다. v2 SFT는 36/61, v1 SFT는 33/61, v1 GRPO는 32/61이었다.

다만 paired common-stop McNemar exact 검정에서 v2 GRPO 대 base의 순개선은 +5문항(10개 개선, 5개 회귀), `p=0.3018`이었다. 방향성은 양수지만 61문항 한 번의 seed만으로 통계적 유의성을 주장할 수 없다.

## 통제 조건

- 데이터: BBEH 비중복 11영역 61문항
- 데이터 SHA-256: `e61daf467e1620046ad2d0b7e6a98823a07749031c9b153c2acbf4beeb8f2050`
- thinking: 활성화
- `max_tokens=64000`, 프롬프트상 reasoning 목표 `56000`
- Qwen3.6 모델 카드 권장 일반 thinking sampling:
  - `temperature=1.0`
  - `top_p=0.95`
  - `top_k=20`
  - `min_p=0.0`
  - `presence_penalty=1.5`
  - `repetition_penalty=1.0`
- 요청 seed: `20260808`
- base와 네 adapter에 같은 문항·프롬프트·sampling 적용
- H200 4장, 1차는 모델당 1 GPU, v2 GRPO는 4-way shard

`reasoning_token_budget`은 Qwen3.6/vLLM이 강제하는 별도 토큰 제어가 아니라 프롬프트 지시다. 실제 강제 상한은 `max_tokens`다.

## 최종 점수

| 모델 | 정답 | 정확도 | stop | length | 평균 출력 토큰 |
|---|---:|---:|---:|---:|---:|
| base | 33/61 | 54.1% | 58 | 3 | 23,782 |
| v1 SFT | 33/61 | 54.1% | 58 | 3 | 24,332 |
| v1 GRPO | 32/61 | 52.5% | 59 | 2 | 23,079 |
| v2 SFT | 36/61 | 59.0% | 61 | 0 | 22,930 |
| v2 GRPO | **41/61** | **67.2%** | **61** | **0** | **21,836** |

v2 GRPO는 가장 높은 정확도와 가장 짧은 평균 출력을 동시에 기록했다. 따라서 이번 결과에서는 더 긴 출력으로 점수를 얻은 것이 아니다.

## budget와 sampling의 영향 분리

| 모델 | 16k, temperature 0 | 64k, temperature 0 | 64k, Qwen 권장 sampling |
|---|---:|---:|---:|
| base | 6 (length 53) | 15 (length 35) | 33 (length 3) |
| v1 SFT | 8 (length 52) | 15 (length 34) | 33 (length 3) |
| v1 GRPO | 6 (length 53) | 16 (length 34) | 32 (length 2) |
| v2 SFT | 6 (length 56) | 17 (length 31) | 36 (length 0) |
| v2 GRPO | 11 (length 49) | 15 (length 33) | 41 (length 0) |

- 16k → 64k의 예산 확대만으로도 temperature 0에서 점수와 정상 종료율이 개선됐다.
- 같은 64k에서 권장 sampling으로 바꾼 효과가 더 컸다. temperature 0은 긴 thinking에서 재검산 문단을 반복하는 퇴화 루프를 만들었다.
- 따라서 이전 낮은 BBEH 점수는 adapter 능력만의 문제가 아니라 잘못된 thinking decoding 정책에 크게 오염돼 있었다.

## paired 비교

| 비교 | 공통 stop | 좌측 정답 | 우측 정답 | 개선/회귀 | 순변화 | exact p |
|---|---:|---:|---:|---:|---:|---:|
| base → v1 SFT | 57 | 32 | 33 | 8 / 7 | +1 | 1.0000 |
| base → v1 GRPO | 56 | 31 | 31 | 8 / 8 | 0 | 1.0000 |
| base → v2 SFT | 58 | 33 | 33 | 8 / 8 | 0 | 1.0000 |
| base → v2 GRPO | 58 | 33 | 38 | 10 / 5 | **+5** | 0.3018 |
| v1 GRPO → v2 GRPO | 59 | 32 | 40 | 12 / 4 | **+8** | 0.0768 |
| v2 SFT → v2 GRPO | 61 | 36 | 41 | 9 / 4 | **+5** | 0.2668 |

모든 모델이 공통으로 정상 종료한 55문항에서는 base 30, v1 SFT 33, v1 GRPO 31, v2 SFT 31, v2 GRPO 37이었다.

## v2 GRPO 대 base 문항 변화

공통 stop에서 개선된 10문항:

- boolean expressions: `bbeh_0309`
- dyck languages: `bbeh_0979`, `bbeh_1109`, `bbeh_1119`
- geometric shapes: `bbeh_1179`, `bbeh_1199`, `bbeh_1239`, `bbeh_1269`
- multistep arithmetic: `bbeh_2039`
- object properties: `bbeh_2579`

회귀한 5문항:

- boolean expressions: `bbeh_0399`
- geometric shapes: `bbeh_1289`
- shuffled objects: `bbeh_3109`
- spatial reasoning: `bbeh_3289`
- zebra puzzles: `bbeh_4439`

영역별 raw 차이는 dyck languages +3, geometric shapes +3, multistep arithmetic +3, object properties +1, spatial reasoning -1, zebra puzzles -1이며 나머지는 동일했다. 즉 이번 개선은 산술 하나에만 국한되지 않고 형식 언어와 도형 규칙 추론에도 나타났다. 반면 shuffled objects와 zebra puzzles에는 뚜렷한 이득이 없다.

## 판정

고난도 4점 v2 GRPO가 권장 thinking 조건에서 가장 좋은 결과를 냈다는 주장은 성립한다. 특히 v1 GRPO 대비 paired 순개선 +8, `p=0.0768`은 후속 반복 평가 가치가 충분하다. 그러나 단일 seed·61문항 결과이므로 일반화 성능 향상을 확정하려면 최소 여러 seed와 추가 unseen BBEH 세트를 사용해야 한다.

실행 종료 후 vLLM·benchmark 프로세스는 남아 있지 않았고 GPU 0–3 메모리는 모두 0 MiB였다.
