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

## 아주 쉽게

모델과 데이터만 있으면 됩니다.

1. GPU 서버에 이 저장소를 받는다.
2. `configs/example.yaml`을 복사해서 **모델 이름**과 **데이터 경로**만 바꾼다.
3. `python pipeline.py --config <내 config>` 한 줄을 실행한다.
4. 끝나면 `<output_dir>/stage2`에 새 모델이 생긴다.

> 아직 실제 모델과 GPU로는 돌려보지 않았습니다(작은 모델로 CPU 테스트만 통과). 처음에는 반드시 작은 설정으로 스모크 테스트를 하세요.

### AI 에이전트에게 맡길 때

아래를 그대로 붙여넣고 `< >`만 바꾸세요.

```
https://github.com/kwonjunn01/vocab_expansion 를 clone해서 README를 읽고 실행해줘.

- 베이스 모델: <모델 경로 또는 HF 이름>
- 추가할 언어 데이터: <jsonl 경로 또는 HF 데이터셋>, 텍스트 필드 이름: <text>
- 기존 언어(replay) 데이터: <경로>
- GPU: <예: H100 x 8>

순서:
1. 토크나이저가 BPE(byte-level 또는 byte-fallback)인지 먼저 확인
2. configs/example.yaml을 복사해서 위 정보로 수정
3. 아주 작은 설정으로 스모크 테스트를 끝까지 돌리고 결과(verification.json, logit 변화, eval/results.json)를 보고해줘
4. 문제 없으면 실제 규모로 실행
Stage 1은 DDP, Stage 2는 FSDP로 돌려줘.
```

스모크 테스트 예시 설정: `num_new_tokens: 1000`, 소스별 `max_docs: 5000`, stage별 `args.max_steps: 20`, `train_tokens: 5_000_000`.
확인할 것: `tokenizer/verification.json`에서 대상 언어 압축률이 1보다 큰지, init 로그의 "Max change of original logits"가 거의 0인지, `eval/results.json`의 bits/byte.

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
- **지원하는 토크나이저**: `tokenizer.json`의 모델 타입이 `BPE`인 두 계열입니다.
  - byte-level BPE (GPT-2, Llama 3, Qwen, LFM 계열)
  - byte-fallback BPE (Llama 2, Mistral 등 SentencePiece 계열)

  byte-fallback BPE에서는 vocab에 없는 글자를 `<0xXX>` 바이트 토큰으로 표현합니다. 그래서 대상 언어에서 자주 나오는 글자를 먼저 글자 토큰으로 추가하고(임베딩은 그 글자의 바이트 토큰 평균), 그다음 글자 위에서 merge를 학습합니다. 바이트 토큰끼리는 merge하지 않고(`<0xEB><0x8B>` 같은 토큰은 디코딩이 안 되기 때문), `▁`를 사이에 둔 두 단어를 하나로 합친 토큰도 만들지 않습니다. `min_frequency`보다 드물게 나오는 글자는 그대로 바이트 토큰으로 남습니다.

  SentencePiece Unigram과 `##` prefix를 쓰는 WordPiece/BPE는 지원하지 않습니다. 확인 방법:
  ```bash
  python -c "from transformers import AutoTokenizer as T; import json; t=T.from_pretrained('<모델>'); m=json.loads(t.backend_tokenizer.to_str())['model']; print(m['type'], 'byte_fallback' if m.get('byte_fallback') else '')"
  ```
- **padding 행**: `pad_to_multiple_of`로 생긴, 어떤 토큰도 쓰지 않는 행은 기존 임베딩의 평균값으로 채웁니다.
- **instruct 모델에서 시작했다면** CPT 이후 SFT 등으로 채팅 능력을 복구해야 합니다.

## 테스트

```bash
python -m pytest -q
```

작은 byte-level BPE와 byte-fallback BPE 토크나이저, 2-layer Llama로 CPU에서 확인하는 항목:

- 기존 ID가 보존되고 새 ID는 끝에 붙는지
- 학습 시 분절 결과가 저장된 토크나이저의 추론 결과와 일치하는지
- 한국어 압축률이 2배 이상 좋아지고 영어 토큰화는 그대로인지
- 새 행이 서브토큰 평균인지(tied, untied 모두)
- byte-fallback에서 글자 토큰이 추가되고, 바이트 토큰은 merge되지 않는지
- 원래 logits이 변하지 않는지
- Stage 1은 새 행만, Stage 2는 본체까지 바꾸는지
- 전체 파이프라인이 끝까지 돌고 재실행 시 건너뛰는지
