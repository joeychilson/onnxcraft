"""Generate tiny ONNX fixtures using only Python's standard library.

The protobuf field numbers come from ONNX's onnx.proto3. These graphs have no
trained weights.
"""
from pathlib import Path
import struct

ROOT = Path(__file__).parent


def varint(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def integer(field, value):
    return varint(field << 3) + varint(value)


def data(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def value(name, dtype, shape, sequence=False):
    dims = b"".join(data(1, data(2, d) if isinstance(d, str) else integer(1, d)) for d in shape)
    tensor = data(1, integer(1, dtype) + data(2, dims))
    kind = data(4, data(1, tensor)) if sequence else tensor
    return data(1, name) + data(2, kind)


def node(op, inputs, outputs):
    return b"".join(data(1, n) for n in inputs) + b"".join(data(2, n) for n in outputs) + data(4, op)


def model(name, nodes, inputs, outputs):
    graph = b"".join(data(1, n) for n in nodes) + data(2, name)
    graph += b"".join(data(11, v) for v in inputs) + b"".join(data(12, v) for v in outputs)
    (ROOT / f"{name}.onnx").write_bytes(integer(1, 10) + data(7, graph) + data(8, integer(2, 21)))


for name, dtype in [("float32", 1), ("uint8", 2), ("int8", 3), ("uint16", 4),
                    ("int16", 5), ("int32", 6), ("int64", 7), ("string", 8),
                    ("bool", 9), ("float16", 10), ("float64", 11),
                    ("uint32", 12), ("uint64", 13), ("bfloat16", 16)]:
    model(f"identity_{name}", [node("Identity", ["x"], ["y"])],
          [value("x", dtype, ["n"])], [value("y", dtype, ["n"])])

model("scalar", [node("Identity", ["x"], ["y"])], [value("x", 1, [])], [value("y", 1, [])])
model("add", [node("Add", ["a", "b"], ["sum"])],
      [value("a", 1, ["batch", 3]), value("b", 1, ["batch", 3])], [value("sum", 1, ["batch", 3])])
model("mixed", [node("Identity", ["x"], ["y"]), node("Identity", ["i"], ["j"])],
      [value("x", 1, ["n"]), value("i", 7, ["n"])], [value("y", 1, ["n"]), value("j", 7, ["n"])])
model("fanout", [node("Identity", ["x"], [f"y{i}"]) for i in range(20)],
      [value("x", 1, ["n"])], [value(f"y{i}", 1, ["n"]) for i in range(20)])
model("sequence", [node("Identity", ["x"], ["y"])],
      [value("x", 1, ["n"], True)], [value("y", 1, ["n"], True)])
model("matmul", [node("MatMul", ["x" if i == 0 else f"y{i-1}", "x"], [f"y{i}"]) for i in range(32)],
      [value("x", 1, ["n", "n"])], [value("y31", 1, ["n", "n"])])
constant = integer(1, 3) + integer(2, 1) + data(9, struct.pack("<3f", 1, 2, 3))
attribute = data(1, "value") + data(5, constant) + integer(20, 4)
model("constant", [node("Constant", [], ["y"]) + data(5, attribute)], [], [value("y", 1, [3])])

axes = integer(1, 1) + integer(2, 7) + data(9, struct.pack("<q", 1))
attribute = data(1, "value") + data(5, axes) + integer(20, 4)
keepdims = data(1, "keepdims") + integer(3, 0) + integer(20, 2)
model("sum", [node("Constant", [], ["axes"]) + data(5, attribute),
              node("ReduceSum", ["x", "axes"], ["y"]) + data(5, keepdims)],
      [value("x", 1, ["batch", 10])], [value("y", 1, ["batch"])])
