import heapq


class BasicTokenizer:
    """minbpe Step 1에서 직접 구현할 바이트 단위 토크나이저."""

    def init(self):
        self.vocab = {n: bytes([n]) for n in range(256)}
        self.merges = {}
        self.next_id = 256

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """학습 텍스트에서 병합 규칙과 토큰 사전을 만든다."""
        self.init()
        encoded_text = self.encode(text)

        for _ in range(vocab_size - len(self.vocab)):
            cnts = {}

            for i in range(1, len(encoded_text)):
                pair = (encoded_text[i - 1], encoded_text[i])
                if cnts.get(pair) is None:
                    cnts[pair] = 0

                cnts[pair] += 1

            pair = max(cnts, key=cnts.get)  # type: ignore
            self.merges[pair] = self.next_id
            self.vocab[self.next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            encoded_text = self.merge_pair(encoded_text, pair, self.next_id)
            self.next_id += 1

    def merge_pair(self, ids, pair, new_id):
        n = len(ids)

        i = 0
        tokens = []

        while i < n:
            if i + 1 < n and (ids[i], ids[i + 1]) == pair:
                tokens.append(new_id)
                i += 1
            else:
                tokens.append(ids[i])

            i += 1

        return tokens

    def encode(self, text: str) -> list[int]:
        """문자열을 토큰 ID 목록으로 변환한다."""
        raw = text.encode("utf-8")
        encoded = list(raw)
        if not encoded:
            return []

        prev_idx = [i - 1 for i in range(len(encoded))]
        next_idx = [-1 if i == len(encoded) - 1 else i + 1 for i in range(len(encoded))]

        cand = []

        for i in range(len(encoded)):
            if (
                i + 1 < len(encoded)
                and self.merges.get((encoded[i], encoded[i + 1])) is not None
            ):
                heapq.heappush(
                    cand,
                    (self.merges[(encoded[i], encoded[i + 1])], i, i + 1),
                )

        while cand:
            new_id, left, right = heapq.heappop(cand)

            if next_idx[left] != right:
                continue

            if self.merges.get((encoded[left], encoded[right])) != new_id:
                continue

            encoded[left] = new_id
            next_idx[left] = next_idx[right]

            if next_idx[right] != -1:
                prev_idx[next_idx[right]] = left

            next_idx[right] = -1

            if prev_idx[left] != -1:
                cand_id = self.merges.get((encoded[prev_idx[left]], encoded[left]))
                if cand_id is not None:
                    heapq.heappush(cand, (cand_id, prev_idx[left], left))

            if next_idx[left] != -1:
                cand_id = self.merges.get((encoded[left], encoded[next_idx[left]]))
                if cand_id is not None:
                    heapq.heappush(cand, (cand_id, left, next_idx[left]))

        ret = []
        pos = 0

        while pos != -1:
            ret.append(encoded[pos])
            pos = next_idx[pos]

        return ret

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 목록을 문자열로 복원한다."""
        return (b"".join([self.vocab[id] for id in ids])).decode("utf-8")


tokenizer = BasicTokenizer()
tokenizer.train("abab", vocab_size=258)
print(tokenizer.encode("abab"))
