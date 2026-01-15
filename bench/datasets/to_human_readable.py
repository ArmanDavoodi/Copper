import struct
import argparse
import math
import hashlib

# ---------------- Correct HyperLogLog ---------------- #

class HyperLogLog:
    def __init__(self, p=12):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m

        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

        self.max_rank = 64 - p

    def add(self, value_bytes):
        # 64-bit hash
        h = hashlib.blake2b(value_bytes, digest_size=8).digest()
        x = int.from_bytes(h, "big")

        # Use top p bits for index
        idx = x >> (64 - self.p)

        # Remaining bits for rank
        w = x & ((1 << (64 - self.p)) - 1)

        # Count leading zeros in w (limited width)
        if w == 0:
            rank = self.max_rank + 1
        else:
            rank = self.max_rank - w.bit_length() + 1

        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def count(self):
        indicator = sum(2.0 ** -r for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # Small-range correction
        zeros = self.registers.count(0)
        if estimate <= 2.5 * self.m and zeros > 0:
            estimate = self.m * math.log(self.m / zeros)

        return int(estimate)

# ---------------- Streaming Reader ---------------- #

DTYPE_INFO = {
    "uint8":  1,
    "uint16": 2,
    "float":  4,
    "float32": 4,
}

def estimate_unique_vectors(filename, dtype, p=12):
    elem_size = DTYPE_INFO[dtype]
    hll = HyperLogLog(p=p)

    with open(filename, "rb") as f:
        num_vectors, dim = struct.unpack("<II", f.read(8))
        vector_size = dim * elem_size

        for i in range(num_vectors):
            raw = f.read(vector_size)
            if len(raw) != vector_size:
                raise ValueError("Unexpected EOF")
            hll.add(raw)

    return num_vectors, hll.count()

# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("dtype")
    parser.add_argument("--precision", type=int, default=12)

    args = parser.parse_args()

    total, unique = estimate_unique_vectors(
        args.input, args.dtype, args.precision
    )

    print("=== Results ===")
    print(f"Total vectors           : {total}")
    print(f"Estimated unique vectors: {unique}")
    print(f"Estimated duplicates    : {total - unique}")

if __name__ == "__main__":
    main()