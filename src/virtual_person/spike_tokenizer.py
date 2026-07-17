from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True, frozen=True)
class ByteTokenizer:
    """
    Deterministic UTF-8 byte tokenizer.

    It needs no pretrained vocabulary, cannot produce an unknown token, and lets
    a model trained from scratch consume dictionaries, stories, procedures, and
    action records without a separate tokenizer-training phase.
    """

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    SEP: int = 3
    BYTE_OFFSET: int = 4
    BYTE_COUNT: int = 256

    @property
    def vocab_size(self) -> int:
        return self.BYTE_OFFSET + self.BYTE_COUNT

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS)
        ids.extend(self.BYTE_OFFSET + value for value in text.encode("utf-8"))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        buffer = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id >= self.BYTE_OFFSET:
                byte_value = token_id - self.BYTE_OFFSET
                if 0 <= byte_value < 256:
                    buffer.append(byte_value)
            elif not skip_special:
                marker = {
                    self.PAD: b"<PAD>",
                    self.BOS: b"<BOS>",
                    self.EOS: b"<EOS>",
                    self.SEP: b"<SEP>",
                }.get(token_id, b"<SPECIAL>")
                buffer.extend(marker)
        return buffer.decode("utf-8", errors="replace")

    def encode_file(self, path: str | Path) -> list[int]:
        return self.encode(Path(path).read_text(encoding="utf-8"))
