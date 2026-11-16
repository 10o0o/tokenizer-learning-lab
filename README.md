# 토크나이저 실습

UTF-8 바이트에서 BPE 토큰을 만드는 과정을 직접 구현하고, 그 결과를 실제 Qwen의 채팅 템플릿과 모델 입력에 연결해 본다. 주자료는 Karpathy의 [강의 영상](https://www.youtube.com/watch?v=zduSFxRajkE)과 [minbpe 실습](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md)이다.

## 바로 시작

1. Karpathy 강의의 [18:15 UTF-8과 바이트](https://www.youtube.com/watch?v=zduSFxRajkE&t=1095s)를 본다. [minbpe Step 1](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md#step-1)을 열고, [basic_tokenizer.py](basic_tokenizer.py)에서 `vocab_size=256`일 때 영어·한글·이모지를 바이트 ID로 바꿨다가 원문으로 복원한다.
2. 첫 확인용 테스트를 실행한다. 이 단계의 목표는 병합 0회에서 UTF-8 왕복이 되는 것이다.

```bash
python3 -m unittest discover -s tests -v -k utf8_bytes
```

첫 단계의 명령은 저장소 루트에서 실행하며 Python 표준 라이브러리만 사용한다. 초기 골격에서는 `NotImplementedError`가 나온다. 기본 바이트 ID는 `0`부터 `255`까지이며, 새 토큰 ID는 `256`부터 순서대로 붙인다.

UTF-8 왕복이 되면 [23:50 BPE 원리와 구현](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s)부터 57:36의 정규식 전처리 구간 직전까지 보면서 병합을 구현한다. 글로 따라가려면 [Hugging Face의 BPE 학습과 적용](https://huggingface.co/learn/llm-course/en/chapter6/5#training-algorithm)을 참고한다. `python3 -m unittest discover -s tests -v`로 [학습용 확인 항목](tests/test_basic_tokenizer.py) 세 개를 실행할 수 있다.

## 이어갈 순서

1. **문자열과 바이트** — `.encode("utf-8")`과 `.decode("utf-8")`로 영어, 한글, 이모지의 바이트를 확인한다. 토큰 경계가 글자 경계와 어긋날 수 있으므로, 복원할 때는 토큰에 해당하는 바이트를 모두 모은 뒤 문자열로 바꾼다.
2. **BPE 학습과 적용** — `train`에서 인접 쌍의 빈도로 병합 순서를 배우고, `encode`에서 저장한 순서를 새 입력에 적용한다. `decode(encode(text))`로 원문 복원을 확인한다. [minbpe Step 1](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md#step-1)
3. **전처리와 특수 토큰** — 공백, 숫자, 문장부호에서 어디를 미리 나누는지 본다. 다음으로 `messages`가 채팅 형식과 `input_ids`로 바뀌는 모습을 Qwen에서 관찰한다. [minbpe Step 2](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md#step-2) · [토크나이저의 구성 요소](https://huggingface.co/learn/llm-course/en/chapter6/8) · [채팅 템플릿](https://huggingface.co/docs/transformers/en/chat_templating)
4. **토큰 ID에서 모델 입력으로** — ID가 임베딩 행을 선택하는 방식과 다음 토큰 예측용 `input`·`target`이 한 칸씩 밀리는 구조를 본다. [LLMs from Scratch 2장](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb)의 데이터 샘플링, 토큰 임베딩 구간을 참고한다.

minbpe의 [Step 3](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md#step-3)는 GPT-4의 `cl100k_base` 인코딩과 같은 결과를 내는 실습이다. 토크나이저 방식 자체를 더 비교하고 싶으면 Hugging Face의 [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6)와 [Unigram](https://huggingface.co/learn/llm-course/en/chapter6/7)을 참고한다.

## Qwen에서 관찰하기

[inspect_qwen.py](inspect_qwen.py)는 공개 Qwen2.5 토크나이저로 공백·한글·특수 토큰, 템플릿 문자열, `input_ids`, `attention_mask`를 출력한다. `uv`가 스크립트의 Python 3.12와 Transformers 버전으로 환경을 구성한다. 토크나이저 파일은 허브 캐시를 사용하고 필요할 때 공개 저장소에서 받는다.

```bash
uv run inspect_qwen.py
```

`Hi`와 ` Hi`에 붙는 ID를 비교한다. 이어서 `messages`의 사용자 문구를 바꾸고, 템플릿 문자열과 실제 토큰 배열에서 달라진 위치를 찾는다.
