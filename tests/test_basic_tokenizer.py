import unittest

from basic_tokenizer import BasicTokenizer


class BasicTokenizerTests(unittest.TestCase):
    def test_01_utf8_bytes_round_trip_without_merges(self):
        tokenizer = BasicTokenizer()
        tokenizer.train("abc", vocab_size=256)
        text = "Hi, 한글 🙂\n"

        ids = tokenizer.encode(text)

        self.assertEqual(ids, list(text.encode("utf-8")))
        self.assertEqual(tokenizer.decode(ids), text)

    def test_02_learned_merge_applies_to_a_new_input(self):
        tokenizer = BasicTokenizer()
        tokenizer.train("abababab", vocab_size=257)

        ids = tokenizer.encode("zab")

        self.assertEqual(ids, [ord("z"), 256])
        self.assertEqual(tokenizer.decode(ids), "zab")

    def test_03_merge_can_combine_earlier_tokens(self):
        tokenizer = BasicTokenizer()
        tokenizer.train("abababab", vocab_size=258)

        ids = tokenizer.encode("abababab")

        self.assertEqual(ids, [257, 257])
        self.assertEqual(tokenizer.decode(ids), "abababab")
