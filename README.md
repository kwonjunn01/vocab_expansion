# vocab-expansion

사전학습된 dense LLM의 토크나이저를 **제자리에서 확장**하고(in-place tokenizer expansion), 모델을 새 vocab에 적응시키는 단일 파이프라인입니다.
Liquid AI의 방식(블로그 "Tokenizer Expansion", 논문 *In-Place Tokenizer Expansion for Pre-trained LLMs*, arXiv 2607.15232)을 따릅니다.

1. **기존 BPE merge를 이어서 학습**: 대상 언어 코퍼스를 *기존 토크나이저로* 자른 결과 위에서 BPE를 계속 학습하고, 새 merge를 원래 merge 뒤에 붙입니다.
   - 기존 토큰 ID는 하나도 바뀌지 않습니다.
   - 모든 새 토큰은 기존 토큰들의 이어붙임입니다.
   - 어떤 텍스트든 먼저 원래대로 토큰화된 뒤 그 위에 새 merge가 적용됩니다. 그래서 대상 언어와 상관없는 텍스트(예: 영어)의 토큰화는 거의 그대로입니다.
2. **새 임베딩 = 서브토큰 임베딩의 평균**: 무작위 초기화가 없습니다. 출력 임베딩이 tie되지 않은 모델은 `lm_head`도 같은 방식으로 초기화합니다.
3. **Stage 1, 임베딩만 학습**: 모델 본체를 얼리고 새 임베딩 행만 학습합니다.
4. **Stage 2, 전체 CPT**: 대상 언어와 replay 데이터를 섞어 모델 전체를 학습합니다.

## 사용법

```bash
pip install -r requirements.txt
python pipeline.py --config configs/example.yaml                  # 전체
python pipeline.py --config configs/example.yaml --stages tokenizer init   # 일부 단계만
python pipeline.py --config configs/example.yaml --from stage2    # 이 단계부터 재개
```

단계: `tokenizer → init → data → stage1 → stage2 → evaluate`. 끝난 단계는 `_DONE` 표시를 남기고 다음 실행에서 건너뜁니다. 다시 돌리려면 `--force`를 붙이세요.
학습 단계는 config의 `launcher`(`python`, `accelerate launch`, `torchrun --nproc_per_node 8` 등)로 `vocab_expansion.train`을 실행합니다.

| 단계 | 출력 (`output_dir/…`) | 내용 |
|---|---|---|
| tokenizer | `tokenizer/` | 확장된 토크나이저, `expansion.json`, `verification.json`(ID 보존·round-trip 검사, 언어별 압축률) |
| init | `model_init/` | resize와 평균 초기화를 마친 모델, `expansion_meta.json`. 원래 logits이 변하지 않았는지 검사합니다 |
| data | `data/train`, `data/eval` | 패킹된 학습 mixture(토큰 비율 `weight`)와 held-out 행 |
| stage1 | `stage1/` | 새 임베딩 행만 학습한 모델 |
| stage2 | `stage2/` | 전체 CPT를 마친 모델 |
| evaluate | `eval/results.json` | 원본과 각 단계 모델의 **bits per byte**, tokens/word |

## 주의할 점

- **토큰당 perplexity로 비교하면 안 됩니다.** 토크나이저가 바뀌어 토큰 수가 다르기 때문입니다. `evaluate`는 tokenizer와 무관한 bits per byte로 비교합니다.
- **Stage 1에서는 원래 행이 변하지 않도록 보장합니다.** 원래 행의 gradient는 0으로 막고, weight decay는 강제로 0으로 둡니다(decay가 있으면 얼린 행도 줄어들기 때문). Stage 1은 학습하는 파라미터가 적으므로 DDP로 충분합니다. gradient hook은 FSDP의 flat parameter와 맞지 않습니다.
- **Stage 2는 규모에 따라 FSDP나 DeepSpeed를 씁니다.** `accelerate config`로 설정하면 됩니다. 1–3B dense 모델이면 Megatron 없이 충분합니다.
- **지원하는 토크나이저**: byte-level BPE(GPT-2, Llama 3, Qwen, LFM 계열). SentencePiece Unigram이나 `##` prefix를 쓰는 WordPiece/BPE는 지원하지 않습니다.
- **padding 행**: `pad_to_multiple_of`로 생긴, 어떤 토큰도 쓰지 않는 행은 기존 임베딩의 평균값으로 채웁니다.
- **instruct 모델에서 시작했다면** CPT 이후 SFT 등으로 채팅 능력을 복구해야 합니다.

## 테스트

```bash
python -m pytest -q
```

작은 byte-level BPE 토크나이저와 2-layer Llama로 CPU에서 확인하는 항목:

- 기존 ID가 보존되고 새 ID는 끝에 붙는지
- 학습 시 분절 결과가 저장된 토크나이저의 추론 결과와 일치하는지
- 한국어 압축률이 2배 이상 좋아지고 영어 토큰화는 그대로인지
- 새 행이 서브토큰 평균인지(tied, untied 모두)
- 원래 logits이 변하지 않는지
- Stage 1은 새 행만, Stage 2는 본체까지 바꾸는지
- 전체 파이프라인이 끝까지 돌고 재실행 시 건너뛰는지
