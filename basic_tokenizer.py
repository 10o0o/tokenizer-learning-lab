class BasicTokenizer:
    """minbpe Step 1에서 직접 구현할 바이트 단위 토크나이저."""

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """학습 텍스트에서 병합 규칙과 토큰 사전을 만든다."""
        raise NotImplementedError("Step 1: train을 구현하세요.")

    def encode(self, text: str) -> list[int]:
        """문자열을 토큰 ID 목록으로 변환한다."""
        raise NotImplementedError("Step 1: encode를 구현하세요.")

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 목록을 문자열로 복원한다."""
        raise NotImplementedError("Step 1: decode를 구현하세요.")
