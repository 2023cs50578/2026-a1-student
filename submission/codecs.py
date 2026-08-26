"""
submission/codecs.py — variable-byte (VByte / VInt) integer coding.

This is the compression layer the "index size" leaderboard component
(assignment Section 7) is really about, and it follows the convention from
the Retrieval-II lecture notes ("Variable Byte (VB) codes") exactly:

    Split a value G into 7-bit groups, most significant group first, and put
    each group in the low 7 bits of one byte. The continuation bit (the top
    bit, 0x80) is 0 on every byte except the LAST byte of a number, where it
    is 1. That makes the byte stream uniquely prefix-decodable — you know a
    number has ended the moment you see a byte >= 128.

    The worked example from the lecture slide, reproduced byte for byte:

        docIDs  824       829        215406
        gaps    824       5          214577
        bytes   00000110 10111000 | 10000101 | 00001101 00001100 10110001

Why it pays off here: postings lists are stored as *d-gaps* (differences
between consecutive doc ids) rather than absolute doc ids, and gaps in a long
postings list are small — a term appearing in 30% of documents has an average
gap of ~3, which VByte stores in one byte instead of the four an int32 array
would use. Term frequencies are even more skewed (the overwhelming majority
are 1 or 2), so they compress to one byte essentially always.

Both directions are vectorised with NumPy. That matters twice over:
  - encoding runs over every posting in the collection during build_index(),
    which is charged against the index-build-time efficiency metric;
  - decoding runs inside retrieve() on the postings of every query term,
    which is charged against per-query latency. A naive Python byte-at-a-time
    decoder costs ~100x more, which would hand back the latency advantage
    that compression is supposed to buy.

A pure-Python fallback is kept for both directions so the module still works
if NumPy is unavailable; it is exercised by the unit tests.
"""
from typing import List, Sequence

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - NumPy is in requirements.txt
    _HAVE_NUMPY = False

# Widest value we support: 5 VByte bytes * 7 bits = 35 bits, comfortably more
# than any doc id, gap, or term frequency in a corpus of this size.
_MAX_VBYTE_BYTES = 5


def vbyte_encode_py(values: Sequence[int]) -> bytes:
    """Pure-Python VByte encoder. Reference implementation / fallback."""
    out = bytearray()
    for v in values:
        if v < 0:
            raise ValueError(f"VByte codes are for non-negative integers; got {v}")
        groups = [v & 127]
        v >>= 7
        while v:
            groups.append(v & 127)
            v >>= 7
        groups.reverse()  # most significant 7-bit group first
        groups[-1] |= 128  # continuation bit set => last byte of this number
        out.extend(groups)
    return bytes(out)


def vbyte_decode_py(buf: bytes, offset: int = 0, count: int = -1) -> List[int]:
    """Pure-Python VByte decoder. Reference implementation / fallback.

    Decodes `count` integers starting at `buf[offset]`, or to the end of the
    buffer when `count` is negative.
    """
    out: List[int] = []
    pos = offset
    end = len(buf)
    value = 0
    while pos < end and (count < 0 or len(out) < count):
        byte = buf[pos]
        pos += 1
        if byte < 128:
            value = (value << 7) | byte
        else:
            out.append((value << 7) | (byte & 127))
            value = 0
    return out


def vbyte_encode(values) -> bytes:
    """Encode a sequence of non-negative integers as a VByte byte string.

    Vectorised: instead of looping over values, we compute how many bytes each
    value needs, lay out the output buffer in one shot, and then fill byte
    position j for every value at once — at most 5 passes total, regardless of
    how many millions of postings we are encoding.
    """
    if not _HAVE_NUMPY:
        return vbyte_encode_py(values)

    a = np.asarray(values, dtype=np.int64)
    if a.size == 0:
        return b""
    if a.min() < 0:
        raise ValueError("VByte codes are for non-negative integers")

    # n_bytes[i] = how many 7-bit groups value i needs (at least one).
    n_bytes = np.ones(a.size, dtype=np.int64)
    remaining = a >> 7
    while remaining.any():
        n_bytes += remaining > 0
        remaining >>= 7

    ends = np.cumsum(n_bytes)
    starts = ends - n_bytes
    out = np.empty(int(ends[-1]), dtype=np.uint8)

    # Byte j of a number holds 7-bit group (n_bytes - 1 - j), i.e. most
    # significant group first, matching the lecture's worked example.
    for j in range(int(n_bytes.max())):
        wide_enough = n_bytes > j
        shift = 7 * (n_bytes[wide_enough] - 1 - j)
        out[starts[wide_enough] + j] = (a[wide_enough] >> shift) & 127

    out[ends - 1] |= 128  # mark the final byte of every number
    return out.tobytes()


def vbyte_decode(buf: bytes, offset: int = 0, length: int = -1) -> "np.ndarray":
    """Decode the VByte stream in `buf[offset : offset + length]`.

    Returns a NumPy int64 array (a Python list when NumPy is unavailable).
    `length` is a byte count, not a value count — postings lists are stored
    with a known byte extent, so slicing by bytes is what callers actually
    have.

    The vectorised decode works like this:
      - a byte >= 128 terminates a number, so `is_last` marks the boundaries;
      - the exclusive cumulative sum of `is_last` labels every byte with the
        index of the number it belongs to;
      - a byte's distance from the END of its number gives its 7-bit group
        position, hence its weight 2^(7 * distance) (most significant first);
      - `np.add.reduceat` sums each group's weighted 7-bit payloads in one go.
    """
    if length < 0:
        length = len(buf) - offset
    if length == 0:
        return np.empty(0, dtype=np.int64) if _HAVE_NUMPY else []
    if not _HAVE_NUMPY:
        return vbyte_decode_py(buf, offset, -1)[: length]

    b = np.frombuffer(buf, dtype=np.uint8, count=length, offset=offset)
    is_last = b >= 128
    # A byte range can end part-way through a number (a caller decoding a
    # fixed-size window rather than a whole postings list). Drop the trailing
    # incomplete number rather than mis-decoding it.
    complete = np.flatnonzero(is_last)
    if complete.size == 0:
        return np.empty(0, dtype=np.int64)
    if complete[-1] != b.size - 1:
        b = b[: complete[-1] + 1]
        is_last = is_last[: complete[-1] + 1]

    # Byte i belongs to number `group[i]`.
    group = np.empty(b.size, dtype=np.int64)
    group[0] = 0
    np.cumsum(is_last[:-1], out=group[1:])

    # Start byte offset of each number.
    starts = np.empty(int(group[-1]) + 1, dtype=np.int64)
    starts[0] = 0
    np.add(np.flatnonzero(is_last)[:-1], 1, out=starts[1:])

    last_byte_of_group = np.flatnonzero(is_last)
    shift = (last_byte_of_group[group] - np.arange(b.size, dtype=np.int64)) * 7
    payload = (b & 127).astype(np.int64) << shift
    return np.add.reduceat(payload, starts)


def encode_postings(doc_ids, freqs) -> bytes:
    """Encode one postings list as interleaved d-gaps and term frequencies.

    `doc_ids` must be ascending internal integer doc ids. The stream is
    [gap_1, tf_1, gap_2, tf_2, ...] where gap_1 is the first (absolute) doc id
    and gap_i = doc_id_i - doc_id_{i-1} thereafter. Interleaving keeps a
    term's doc ids and frequencies in one contiguous run, so scoring a query
    term is a single sequential read.
    """
    if _HAVE_NUMPY:
        d = np.asarray(doc_ids, dtype=np.int64)
        f = np.asarray(freqs, dtype=np.int64)
        gaps = d.copy()
        gaps[1:] -= d[:-1]
        interleaved = np.empty(d.size * 2, dtype=np.int64)
        interleaved[0::2] = gaps
        interleaved[1::2] = f
        return vbyte_encode(interleaved)

    interleaved = []
    previous = 0
    for doc_id, freq in zip(doc_ids, freqs):
        interleaved.append(doc_id - previous)
        interleaved.append(freq)
        previous = doc_id
    return vbyte_encode_py(interleaved)


def decode_postings(buf: bytes, offset: int, length: int):
    """Inverse of `encode_postings`: returns (doc_ids, freqs) arrays."""
    flat = vbyte_decode(buf, offset, length)
    if _HAVE_NUMPY:
        doc_ids = np.cumsum(flat[0::2])
        return doc_ids, flat[1::2]
    doc_ids, freqs, running = [], [], 0
    for i in range(0, len(flat), 2):
        running += flat[i]
        doc_ids.append(running)
        freqs.append(flat[i + 1])
    return doc_ids, freqs


def vbyte_byte_widths(values) -> "np.ndarray":
    """How many bytes `vbyte_encode` will spend on each value, without
    actually encoding.

    `InvertedIndex.save()` uses this to work out each term's byte extent in
    `postings.bin` (so `load()` can seek straight to a term's list) while
    still encoding the whole collection's postings in one vectorised call
    rather than one call per term.
    """
    a = np.asarray(values, dtype=np.int64)
    widths = np.ones(a.size, dtype=np.int64)
    remaining = a >> 7
    while remaining.any():
        widths += remaining > 0
        remaining >>= 7
    return widths
