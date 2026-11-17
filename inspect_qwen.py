# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.22.2", "jinja2==3.1.6"]
# ///

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=False)

    for text in ("Hi", " Hi", "안녕하세요", "<|im_start|>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"{text!r} -> {ids} -> {tokenizer.convert_ids_to_tokens(ids)}")

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Hi"},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )

    print("\n템플릿 적용 결과")
    print(formatted)
    print("input_ids", tokenized["input_ids"])
    print("attention_mask", tokenized["attention_mask"])


if __name__ == "__main__":
    main()
