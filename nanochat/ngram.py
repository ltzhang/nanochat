"""
Utilities for working with flattened n-gram ID vocabularies.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


class _TrieNode:
    __slots__ = ("children", "ngram_id")

    def __init__(self):
        self.children = {}
        self.ngram_id = 0


@dataclass
class NgramLexiconEntry:
    ngram_id: int
    token_ids: tuple[int, ...]


class NgramLexicon:
    """
    Longest-suffix n-gram lookup over a flattened global ID space.

    File format:
    - UTF-8 text
    - empty lines and lines starting with '#' are ignored
    - each data line is one of:
        <global_id>\t<token0 token1 ...>
        <global_id>\t<n>\t<token0 token1 ...>
        <global_id>\t<n>\t<token0 token1 ...>\t<count>\t<display_text>
    """

    def __init__(self, entries):
        self.entries = list(entries)
        self.root = _TrieNode()
        self.max_order = 0
        self.max_ngram_id = 0
        for entry in self.entries:
            self._insert(entry)
            self.max_order = max(self.max_order, len(entry.token_ids))
            self.max_ngram_id = max(self.max_ngram_id, entry.ngram_id)

    @property
    def vocab_size(self):
        """Backward-compatible alias for the maximum assigned n-gram ID."""
        return self.max_ngram_id

    @classmethod
    def from_file(cls, path):
        entries = []
        for lineno, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if not fields or not fields[0].strip().isdigit():
                continue
            ngram_id = int(fields[0])
            if ngram_id <= 0:
                raise ValueError(f"Line {lineno}: n-gram IDs must start at 1, got {ngram_id}")
            if len(fields) == 2:
                token_field = fields[1]
                n_expected = None
            elif len(fields) >= 3:
                n_expected = int(fields[1])
                token_field = fields[2]
            else:
                raise ValueError(f"Line {lineno}: expected 2 or 3 tab-separated fields")
            token_ids = tuple(int(tok) for tok in token_field.split())
            if not token_ids:
                raise ValueError(f"Line {lineno}: empty token sequence")
            if n_expected is not None and len(token_ids) != n_expected:
                raise ValueError(f"Line {lineno}: declared n={n_expected}, got {len(token_ids)} tokens")
            entries.append(NgramLexiconEntry(ngram_id=ngram_id, token_ids=token_ids))
        return cls(entries)

    def _insert(self, entry):
        node = self.root
        for token_id in reversed(entry.token_ids):
            node = node.children.setdefault(token_id, _TrieNode())
        if node.ngram_id != 0 and node.ngram_id != entry.ngram_id:
            raise ValueError(f"Duplicate n-gram token sequence with conflicting IDs: {entry.token_ids}")
        node.ngram_id = entry.ngram_id

    def longest_suffix_id(self, tokens):
        node = self.root
        best = 0
        steps = 0
        for token_id in reversed(tokens):
            child = node.children.get(int(token_id))
            if child is None:
                break
            node = child
            steps += 1
            if node.ngram_id != 0:
                best = node.ngram_id
            if steps >= self.max_order:
                break
        return best

    def encode_sequence(self, tokens):
        token_list = [int(tok) for tok in tokens]
        ids = []
        for end_idx in range(len(token_list)):
            start_idx = max(0, end_idx + 1 - self.max_order)
            ids.append(self.longest_suffix_id(token_list[start_idx:end_idx + 1]))
        return ids

    def encode_tensor(self, token_ids, device=None):
        assert token_ids.ndim in {1, 2}, f"Expected 1D or 2D tensor, got shape {tuple(token_ids.shape)}"
        target_device = token_ids.device if device is None else torch.device(device)
        if token_ids.ndim == 1:
            ids = self.encode_sequence(token_ids.tolist())
            return torch.tensor(ids, dtype=torch.long, device=target_device)
        rows = [self.encode_sequence(row.tolist()) for row in token_ids]
        return torch.tensor(rows, dtype=torch.long, device=target_device)
