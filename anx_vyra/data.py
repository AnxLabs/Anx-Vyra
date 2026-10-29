"""Synthetic corpora for ANX-VYRA.

Two kinds of data, both generated from a seed so a run is reproducible from its config alone:

  * general bodies  - the five task families the model is pretrained on (copy, lookup, addition,
                      periodic, plus random filler), used as the background mix.
  * needle rows     - a synthetic retrieval task: one or more MEM: records placed early, then
                      filler, then a QUERY:. The model must emit the payload of the matching
                      record. `distance` is the exact token gap from the last byte of the record
                      to the first byte of the query marker, and it is what the retrieval
                      experiments varied.

Retrieval geometry
------------------
For a needle row, one record occupies `piece = RECORD_LEN + filler` tokens and the needle-to-query
distance is `filler + RECORD_LEN`. So choosing a distance fixes the filler length, and the number
of records that fit in a row follows from the sequence length:

    piece             = 27 + filler
    distance          = filler + 8
    copies_per_row    = min(8, 510 // piece)

Every row is padded to SEQ tokens regardless of how many records it holds, which is why measured
memory is flat across distances even as retrieval accuracy changes.
"""
from __future__ import annotations

import random

import torch

from .model import BOS, EOS, N_NEEDLE, N_TRAIN, N_VALID, PAD, SEQ, VOCAB

RECORD_LEN = 8          # payload bytes per record
MEM_MARK = b"MEM:"
QUERY_MARK = b"QUERY:"
SEP = b"\n"
# a record is b"MEM:" + 8 payload bytes -> 12 bytes; the 27 below accounts for that header plus the
# separator and the query marker, so piece = 27 + filler is the exact token stride between records.
RECORD_OVERHEAD = 27


def geometry(distance: int) -> dict:
    """Row layout implied by a needle-to-query distance."""
    if distance < RECORD_LEN:
        raise ValueError(f"distance {distance} is shorter than one record ({RECORD_LEN})")
    filler = distance - RECORD_LEN
    piece = RECORD_OVERHEAD + filler
    copies = min(8, (SEQ - 2) // piece)
    return {"distance": distance, "filler": filler, "piece_len": piece,
            "copies_per_row": copies, "row_bytes": piece * copies,
            "inside_window": True}   # filled in by the caller against the arm's window


def make_body(rng: random.Random) -> list[int]:
    """One general-corpus row: exactly the five task families used in pretraining."""
    kind = rng.randrange(10)
    body = [0] * (SEQ - 2)
    if kind <= 3:                                   # copy / lookup
        pairs = [(rng.randrange(256), rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(8)]
        flat = [byte for pair in pairs for byte in pair]
        chosen = pairs[rng.randrange(8)]
        body[:32] = flat
        body[32] = 0x1E
        body[33:470] = [rng.randrange(256) for _ in range(437)]
        body[470:474] = list(chosen)
    elif kind <= 6:                                 # periodic / repeat
        chunk = [rng.randrange(256) for _ in range(32)]
        body[:32] = chunk
        body[32] = 0x1F
        body[33:65] = chunk
        body[65:] = [rng.randrange(256) for _ in range(SEQ - 2 - 65)]
    elif kind <= 8:                                 # addition
        left, right = rng.randrange(50), rng.randrange(50)
        text = f"{left}+{right}={left + right};".encode("ascii")
        body[:len(text)] = list(text)
        body[len(text):] = [rng.randrange(128) for _ in range(SEQ - 2 - len(text))]
    else:                                           # random filler
        pattern = [rng.randrange(256) for _ in range(4)]
        body = (pattern * SEQ)[:SEQ - 2]
    return [BOS, *body, EOS]


def build_split(seed: int, count: int) -> torch.Tensor:
    rng = random.Random(seed)
    return torch.tensor([make_body(rng) for _ in range(count)], dtype=torch.int16)


def build_needles(train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed 400-token-filler needle probe from the pretraining corpus (distance 408)."""
    blob = bytes(train[:, 1:-1].reshape(-1).tolist())
    rng = random.Random(909)
    prompts, targets = [], []
    for _ in range(N_NEEDLE):
        payload = None
        for _try in range(100):
            candidate = bytes(rng.randrange(1, 256) for _ in range(RECORD_LEN))
            if candidate not in blob and MEM_MARK + candidate not in blob:
                payload = candidate
                break
        if payload is None:
            raise RuntimeError("needle payload collided with the training bytes")
        filler = bytes(rng.randrange(97, 123) for _ in range(400))
        prompts.append([BOS, *MEM_MARK, *payload, *SEP, *filler, *QUERY_MARK])
        targets.append(list(payload))
    return torch.tensor(prompts, dtype=torch.int16), torch.tensor(targets, dtype=torch.int16)


def needle_row(payloads: list[bytes], filler_bytes: bytes) -> tuple[list[int], list[int]]:
    """Build one retrieval row and the per-position mask that scores its answer records.

    Layout:  MEM:payload\\n MEM:payload\\n ... filler QUERY:  ->  pad to (SEQ - 2)  ->  BOS/EOS.

    The returned mask is length SEQ - 2: it marks the payload bytes of each record, which are the
    only positions that carry a training signal. Predicting the payload requires having kept the
    record across the filler, so the mask is exactly the retrieval test.
    """
    body: list[int] = []
    record_spans: list[tuple[int, int]] = []
    for payload in payloads:
        start = len(body) + len(MEM_MARK)
        body.extend(MEM_MARK)
        body.extend(payload)
        record_spans.append((start, start + len(payload)))
        body.extend(SEP)
    body.extend(filler_bytes)
    body.extend(QUERY_MARK)
    if len(body) > SEQ - 2:
        raise ValueError(f"row needs {len(body)} tokens, only {SEQ - 2} available")
    mask = [0] * (SEQ - 2)
    for start, stop in record_spans:
        for i in range(start, stop):
            mask[i] = 1
    body.extend([PAD] * (SEQ - 2 - len(body)))
    return [BOS, *body, EOS], mask


def build_distance_set(distance: int, n_rows: int, seed: int,
                       ban: frozenset = frozenset()) -> dict:
    """A train/unseen pair for one needle-to-query distance, with no payload reused anywhere.

    `ban` carries payloads from other datasets so a release corpus can prove its splits are
    disjoint from everything trained on before.
    """
    geo = geometry(distance)
    rng = random.Random(seed)
    seen = set(ban)

    def fresh() -> bytes:
        for _try in range(10_000):
            candidate = bytes(rng.randrange(1, 256) for _ in range(RECORD_LEN))
            if candidate not in seen:
                seen.add(candidate)
                return candidate
        raise RuntimeError("could not draw a fresh payload")

    def rows_for(count: int) -> tuple[torch.Tensor, torch.Tensor, list[bytes]]:
        rows, masks, used = [], [], []
        for _ in range(count):
            payloads = [fresh() for _ in range(geo["copies_per_row"])]
            filler = bytes(rng.randrange(97, 123) for _ in range(geo["filler"]))
            row, mask = needle_row(payloads, filler)
            rows.append(row)
            masks.append(mask)
            used.extend(payloads)
        return (torch.tensor(rows, dtype=torch.int16),
                torch.tensor(masks, dtype=torch.int16), used)

    train_rows, train_mask, train_payloads = rows_for(n_rows)
    unseen_rows, unseen_mask, unseen_payloads = rows_for(n_rows)
    return {"rows": train_rows, "mask": train_mask, "payloads": train_payloads,
            "unseen": unseen_rows, "unseen_mask": unseen_mask,
            "unseen_payloads": unseen_payloads,
            "geometry": geo, "copies": geo["copies_per_row"]}


def build_corpus(seed: int = 20261026) -> dict:
    """The frozen pretraining corpus: general rows plus the 400-filler needle probe."""
    train = build_split(seed, N_TRAIN)
    valid = build_split(seed + 1, N_VALID)
    needle_prompt, needle_target = build_needles(train)
    perm_rng = random.Random(424242)
    perm = list(range(N_TRAIN))
    perm_rng.shuffle(perm)
    return {"train": train, "valid": valid, "needle_prompt": needle_prompt,
            "needle_target": needle_target, "perm": torch.tensor(perm, dtype=torch.int64)}


__all__ = ["RECORD_LEN", "geometry", "make_body", "build_split", "build_needles",
           "needle_row", "build_distance_set", "build_corpus", "VOCAB", "SEQ"]
