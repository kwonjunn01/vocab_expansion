import random

import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ENGLISH = [
    "The quick brown fox jumps over the lazy dog.",
    "Language models learn to predict the next token from large amounts of text.",
    "A tokenizer splits text into tokens before the model reads it.",
    "Continued pre-training adapts a model to a new domain or language.",
]
SUBJECTS = ["언어 모델은", "토크나이저는", "학생들은", "연구자들이", "우리는"]
OBJECTS = ["새로운 단어를", "한국어 문장을", "긴 문서를", "다양한 데이터를"]
VERBS = ["학습합니다.", "이해합니다.", "생성했습니다.", "분석하고 있습니다."]


def korean_texts(n, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(p) for p in (SUBJECTS, OBJECTS, VERBS)) for _ in range(n)]


def build_tokenizer(texts, vocab_size=400):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>")


def build_byte_fallback_tokenizer(texts, vocab_size=500):
    """SentencePiece-style BPE (Llama 2, Mistral): "▁" for spaces, <0xXX> tokens for missing characters."""
    tok = Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>", fuse_unk=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    byte_tokens = [f"<0x{i:02X}>" for i in range(256)]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"] + byte_tokens)
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")


@pytest.fixture
def english_tokenizer(tmp_path):
    tok = build_tokenizer(ENGLISH * 20)
    tok.save_pretrained(tmp_path / "base_tokenizer")
    return tok


@pytest.fixture(params=["byte_level", "byte_fallback"])
def any_tokenizer(request):
    """An English-only tokenizer of each supported kind: Korean is unknown to both."""
    if request.param == "byte_level":
        return build_tokenizer(ENGLISH * 20)
    return build_byte_fallback_tokenizer(ENGLISH * 20)


def build_model(tokenizer, tied, rows=None):
    config = LlamaConfig(
        vocab_size=rows or len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return LlamaForCausalLM(config)
