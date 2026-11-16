import sys, re, json, html, math, argparse
import numpy as np
import tflite

OPN = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
SUPPORTED = ["CONCATENATION", "MUL", "SUB", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D",
             "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "LOGISTIC"]
OPC = {n: i for i, n in enumerate(SUPPORTED)}


def tround(x):
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def quantize_multiplier(d):
    if d == 0.0:
        return 0, 0
    q, shift = math.frexp(d)
    qf = int(tround(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        shift, qf = 0, 0
    if shift > 30:
        shift, qf = 30, (1 << 31) - 1
    return qf, shift


def act_range(act, scale, zp):
    qmin, qmax = -128, 127
    q = lambda f: zp + int(tround(float(np.float32(f) / np.float32(scale))))
    if act == 0:
        return qmin, qmax
    if act == 1:
        return max(qmin, q(0.0)), qmax
    if act == 2:
        return max(qmin, q(-1.0)), min(qmax, q(1.0))
    if act == 3:
        return max(qmin, q(0.0)), min(qmax, q(6.0))
    raise SystemExit("unsupported activation %d" % act)


def same_pad(inp, k, s, out, padding):
    if padding == 1:
        return 0
    total = max((out - 1) * s + k - inp, 0)
    return total // 2


def read_plan(path):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    i = s.index("Machine-readable plan")
    j = s.index("{", i)
    plan, _ = json.JSONDecoder().raw_decode(s[j:])
    return plan


class Model:
    def __init__(self, path):
        buf = open(path, "rb").read()
        self.m = tflite.Model.GetRootAsModel(buf, 0)
        self.g = self.m.Subgraphs(0)

    def tensor(self, i):
        t = self.g.Tensors(i)
        q = t.Quantization()
        sc = q.ScaleAsNumpy() if q is not None and q.ScaleLength() else np.array([], np.float32)
        zp = q.ZeroPointAsNumpy() if q is not None and q.ZeroPointLength() else np.array([], np.int64)
        b = self.m.Buffers(t.Buffer())
        data = None
        if b is not None and b.DataLength():
            raw = b.DataAsNumpy().tobytes()
            dt = {9: np.int8, 2: np.int32, 3: np.uint8, 0: np.float32, 4: np.int64}[t.Type()]
            data = np.frombuffer(raw, dt).reshape(t.ShapeAsNumpy() if t.ShapeLength() else ())
        shape = [int(x) for x in t.ShapeAsNumpy()] if t.ShapeLength() else []
        return dict(shape=shape, scale=np.asarray(sc, np.float32), zp=np.asarray(zp, np.int64),
                    data=data, type=t.Type())

    def ops(self):
        res = []
        for i in range(self.g.OperatorsLength()):
            o = self.g.Operators(i)
            oc = self.m.OperatorCodes(o.OpcodeIndex())
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            res.append(dict(name=OPN[code], ins=[int(x) for x in o.InputsAsNumpy()],
                            outs=[int(x) for x in o.OutputsAsNumpy()], op=o))
        return res


def opts(op, cls):
    t = op.BuiltinOptions()
    if t is None:
        return None
    o = cls()
    o.Init(t.Bytes, t.Pos)
    return o


def hwc(shape):
    if len(shape) == 4:
        return shape[1], shape[2], shape[3]
    if len(shape) == 2:
        return 1, 1, shape[1]
    if len(shape) == 1:
        return 1, 1, shape[0]
    raise SystemExit("shape %s" % shape)


def carr(name, arr, ctype, per_line=24):
    arr = list(int(x) for x in np.asarray(arr).reshape(-1))
    lines = []
    for i in range(0, len(arr), per_line):
        lines.append("    " + ", ".join(str(v) for v in arr[i:i + per_line]) + ",")
    return "const %s %s[%d] = {\n%s\n};\n" % (ctype, name, len(arr), "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("blueprint")
    ap.add_argument("--out", default=".")
    ap.add_argument("--ram", type=int, default=98304)
    ap.add_argument("--reserve", type=int, default=3072)
    a = ap.parse_args()

    M = Model(a.tflite)
    ops = M.ops()
    plan = read_plan(a.blueprint)
    pl = plan["layers"]
    if len(pl) != len(ops):
        raise SystemExit("plan has %d layers, model has %d ops" % (len(pl), len(ops)))
    for i, (p, o) in enumerate(zip(pl, ops)):
        if p["op"] != o["name"]:
            raise SystemExit("layer %d: plan %s model %s" % (i, p["op"], o["name"]))
        if o["name"] not in OPC:
            raise SystemExit("unsupported op %s" % o["name"])

    model_in = int(M.g.InputsAsNumpy()[0])
    act_ids = [model_in] + [o["outs"][0] for o in ops]
    tix = {t: i for i, t in enumerate(act_ids)}
    T = {t: M.tensor(t) for t in act_ids}
    producer = {o["outs"][0]: i for i, o in enumerate(ops)}

    lgroup = {}
    for gi, g in enumerate(plan["groups"]):
        for l in range(g["first_layer"], g["last_layer"] + 1):
            lgroup[l] = gi
    ng = len(plan["groups"])

    consumers = {t: [] for t in act_ids}
    for i, o in enumerate(ops):
        for t in o["ins"]:
            if t in consumers and i not in consumers[t]:
                consumers[t].append(i)

    data_c = ['#include "model_data.h"\n']
    data_h = ["#ifndef MODEL_DATA_H", "#define MODEL_DATA_H", "#include <stdint.h>"]
    layers_c = []

    def emit(name, arr, ctype):
        data_c.append(carr(name, arr, ctype))
        data_h.append("extern const %s %s[%d];" % (ctype, name, np.asarray(arr).size))
        return name

    for i, o in enumerate(ops):
        n = o["name"]
        out = o["outs"][0]
        To = T[out]
        oh, ow, oc = hwc(To["shape"])
        L = dict(op="OP_" + n.replace("_2D", "").replace("DEPTHWISE_CONV", "DW").replace("CONCATENATION", "CONCAT")
                 .replace("FULLY_CONNECTED", "FC").replace("AVERAGE_POOL", "AVGPOOL"),
                 n_in=1, ins=[0, 0, 0], out=tix[out], kh=1, kw=1, sh=1, sw=1, pt=0, pl=0,
                 amin=-128, amax=127, w="0", b="0", mult="0", shift="0", cst="0", cst_zp=0, cst_first=0,
                 m1=0, s1=0, m2=0, s2=0, mo=0, so=0, ls=0, lut="0")
        act_in = [t for t in o["ins"] if t in tix]
        L["n_in"] = len(act_in)
        for k, t in enumerate(act_in[:3]):
            L["ins"][k] = tix[t]
        Ti = T[act_in[0]]
        ih, iw, ic = hwc(Ti["shape"])
        si, so_ = float(Ti["scale"][0]), float(To["scale"][0])

        if n in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            cls = tflite.Conv2DOptions if n == "CONV_2D" else tflite.DepthwiseConv2DOptions
            op_ = opts(o["op"], cls)
            if op_.DilationHFactor() != 1 or op_.DilationWFactor() != 1:
                raise SystemExit("dilation not supported")
            if n == "DEPTHWISE_CONV_2D" and op_.DepthMultiplier() not in (0, 1) and ic != oc:
                raise SystemExit("depth multiplier not supported")
            W = M.tensor(o["ins"][1])
            B = M.tensor(o["ins"][2]) if len(o["ins"]) > 2 and o["ins"][2] >= 0 else None
            kh, kw = W["shape"][1], W["shape"][2]
            L.update(kh=kh, kw=kw, sh=op_.StrideH(), sw=op_.StrideW(),
                     pt=same_pad(ih, kh, op_.StrideH(), oh, op_.Padding()),
                     pl=same_pad(iw, kw, op_.StrideW(), ow, op_.Padding()))
            L["amin"], L["amax"] = act_range(op_.FusedActivationFunction(), To["scale"][0], int(To["zp"][0]))
            fs = W["scale"] if W["scale"].size == oc else np.repeat(W["scale"][:1], oc)
            mults, shifts = zip(*[quantize_multiplier(float(np.float64(Ti["scale"][0]) * np.float64(fs[c]) / np.float64(To["scale"][0]))) for c in range(oc)])
            L["w"] = emit("w%d" % i, W["data"], "int8_t")
            L["b"] = emit("b%d" % i, B["data"] if B is not None else np.zeros(oc, np.int32), "int32_t")
            L["mult"] = emit("m%d" % i, mults, "int32_t")
            L["shift"] = emit("s%d" % i, shifts, "int8_t")
        elif n == "FULLY_CONNECTED":
            op_ = opts(o["op"], tflite.FullyConnectedOptions)
            W = M.tensor(o["ins"][1])
            B = M.tensor(o["ins"][2]) if len(o["ins"]) > 2 and o["ins"][2] >= 0 else None
            L["amin"], L["amax"] = act_range(op_.FusedActivationFunction(), To["scale"][0], int(To["zp"][0]))
            fs = W["scale"] if W["scale"].size == oc else np.repeat(W["scale"][:1], oc)
            mults, shifts = zip(*[quantize_multiplier(float(np.float64(Ti["scale"][0]) * np.float64(fs[c]) / np.float64(To["scale"][0]))) for c in range(oc)])
            if W["data"].shape[1] != ih * iw * ic:
                raise SystemExit("FC input size mismatch")
            L["w"] = emit("w%d" % i, W["data"], "int8_t")
            L["b"] = emit("b%d" % i, B["data"] if B is not None else np.zeros(oc, np.int32), "int32_t")
            L["mult"] = emit("m%d" % i, mults, "int32_t")
            L["shift"] = emit("s%d" % i, shifts, "int8_t")
        elif n in ("ADD", "SUB", "MUL"):
            cls = {"ADD": tflite.AddOptions, "SUB": tflite.SubOptions, "MUL": tflite.MulOptions}[n]
            op_ = opts(o["op"], cls)
            act = op_.FusedActivationFunction() if op_ is not None else 0
            L["amin"], L["amax"] = act_range(act, To["scale"][0], int(To["zp"][0]))
            t1, t2 = o["ins"][0], o["ins"][1]
            Q1 = T[t1] if t1 in T else M.tensor(t1)
            Q2 = T[t2] if t2 in T else M.tensor(t2)
            for t, Q in ((t1, Q1), (t2, Q2)):
                if t not in T:
                    c = np.broadcast_to(Q["data"].reshape(-1), (oc,)) if Q["data"].size in (1, oc) else None
                    if c is None:
                        raise SystemExit("layer %d: constant shape %s not supported" % (i, Q["shape"]))
                    L["cst"] = emit("c%d" % i, c, "int8_t")
                    L["cst_zp"] = int(Q["zp"][0])
                    L["cst_first"] = 1 if t == t1 else 0
            s1, s2 = np.float32(Q1["scale"][0]), np.float32(Q2["scale"][0])
            if n == "MUL":
                rm = float(np.float64(s1) * np.float64(s2) / np.float64(To["scale"][0]))
                L["mo"], L["so"] = quantize_multiplier(rm)
            else:
                ls = 20
                tw = 2.0 * float(max(s1, s2))
                r1 = float(s1) / tw
                r2 = float(s2) / tw
                ro = tw / ((1 << ls) * float(To["scale"][0]))
                L["m1"], L["s1"] = quantize_multiplier(r1)
                L["m2"], L["s2"] = quantize_multiplier(r2)
                L["mo"], L["so"] = quantize_multiplier(ro)
                L["ls"] = ls
        elif n == "AVERAGE_POOL_2D":
            op_ = opts(o["op"], tflite.Pool2DOptions)
            kh, kw = op_.FilterHeight(), op_.FilterWidth()
            L.update(kh=kh, kw=kw, sh=op_.StrideH(), sw=op_.StrideW(),
                     pt=same_pad(ih, kh, op_.StrideH(), oh, op_.Padding()),
                     pl=same_pad(iw, kw, op_.StrideW(), ow, op_.Padding()))
            L["amin"], L["amax"] = act_range(op_.FusedActivationFunction(), To["scale"][0], int(To["zp"][0]))
            if Ti["scale"][0] != To["scale"][0] or Ti["zp"][0] != To["zp"][0]:
                raise SystemExit("avgpool requant not supported")
        elif n == "CONCATENATION":
            op_ = opts(o["op"], tflite.ConcatenationOptions)
            ax = op_.Axis()
            if ax not in (-1, 3):
                raise SystemExit("concat axis %d not supported" % ax)
            if len(act_in) > 3:
                raise SystemExit("concat with >3 inputs")
            for t in act_in:
                if T[t]["scale"][0] != To["scale"][0] or T[t]["zp"][0] != To["zp"][0]:
                    raise SystemExit("concat requant not supported")
        elif n == "LOGISTIC":
            lut = []
            for k in range(256):
                v = k if k < 128 else k - 256
                dq = np.float32(Ti["scale"][0]) * np.float32(v - int(Ti["zp"][0]))
                tr = np.float32(1.0) / (np.float32(1.0) + np.exp(-dq, dtype=np.float32))
                rs = tround(float(np.float32(tr * (np.float32(1.0) / np.float32(To["scale"][0])))))
                lut.append(max(-128, min(127, int(rs) + int(To["zp"][0]))))
            L["lut"] = emit("lut%d" % i, lut, "int8_t")
        layers_c.append(L)

    tens_c = []
    for t in act_ids:
        h, w, c = hwc(T[t]["shape"])
        tens_c.append(dict(id=t, h=h, w=w, c=c, zp=int(T[t]["zp"][0]), rb=w * c,
                           prod=producer.get(t, -1), pg=lgroup.get(producer.get(t, -1), -1)))

    ring_of = {}
    for gi, g in enumerate(plan["groups"]):
        for b in g.get("buffers", []):
            ring_of[(gi, b["tensor"])] = b
    whole = []
    for t in act_ids:
        if t == model_in:
            continue
        pg = lgroup[producer[t]]
        if (pg, t) in ring_of:
            continue
        last = max([lgroup[c] for c in consumers[t]] or [pg])
        whole.append((t, pg, last))

    def align(x):
        return (x + 3) & ~3

    def try_layout(A):
        live = []
        woff = {}
        gs, gd = [], []
        for gi, g in enumerate(plan["groups"]):
            live = [x for x in live if x[3] >= gi]
            for t, pg, last in whole:
                if pg != gi:
                    continue
                sz = align(tens_c[tix[t]]["h"] * tens_c[tix[t]]["rb"])
                occ = sorted(((x[1], x[1] + x[2]) for x in live), key=lambda z: -z[0])
                end, off = A, None
                for s_, e_ in occ:
                    if end - e_ >= sz:
                        off = end - sz
                        break
                    end = min(end, s_)
                if off is None:
                    if end >= sz:
                        off = end - sz
                    else:
                        return None
                live.append((t, off, sz, last))
                woff[t] = off
            reqs = []
            for b in g.get("buffers", []):
                role = {"int": "MEM_RING", "in-sd": "MEM_SD_IN", "out-sd": "MEM_SD_OUT"}[b["role"]]
                t = b["tensor"]
                if b["row_bytes"] != tens_c[tix[t]]["rb"]:
                    raise SystemExit("t%d row bytes plan %d model %d" % (t, b["row_bytes"], tens_c[tix[t]]["rb"]))
                reqs.append(("s", dict(t=tix[t], mem=role, rows=b["rows"]), align(b["rows"] * b["row_bytes"])))
            for d in plan.get("sd_transfers", []):
                if d["group"] == gi:
                    reqs.append(("d", dict(t=tix[d["tensor"]], dir=0 if d["dir"] == "read" else 1,
                                           chunk=d["tile_bytes"], nbuf=d["buffers"]), align(d["tile_bytes"] * d["buffers"])))
            used = sorted((x[1], x[1] + x[2]) for x in live)
            for kind, r, sz in sorted(reqs, key=lambda z: -z[2]):
                pos = 0
                placed = None
                for s_, e_ in sorted(used):
                    if s_ - pos >= sz:
                        placed = pos
                        break
                    pos = max(pos, e_)
                if placed is None:
                    if A - pos >= sz:
                        placed = pos
                    else:
                        return None
                r["off"] = placed
                used.append((placed, placed + sz))
            gs.append([r for k, r, _ in reqs if k == "s"])
            gd.append([r for k, r, _ in reqs if k == "d"])
        return woff, gs, gd

    arena = max(g["ram_bytes"] for g in plan["groups"])
    arena = (arena + 3) & ~3
    while True:
        res = try_layout(arena)
        if res:
            break
        arena += 32
    whole_off, group_slots, group_dma = res

    spill = {}
    parts = {p["name"]: p for p in plan["partitions"]}
    for s in plan.get("spilled", []):
        spill[s["tensor"]] = parts[s["partition"]]["offset"] + s["offset"]
    inp = plan.get("input")
    if isinstance(inp, dict):
        in_part = parts[inp["location"]]["offset"] + inp.get("offset", 0)
    else:
        in_part = parts[inp]["offset"] if inp in parts else 0
    spill_end = max([spill[t] + tens_c[tix[t]]["h"] * tens_c[tix[t]]["rb"] for t in spill] or [0])
    for t in spill:
        pg = lgroup[producer[t]]
        if not any(sl["t"] == tix[t] and sl["mem"] == "MEM_SD_OUT" for sl in group_slots[pg]):
            raise SystemExit("t%d is spilled but not streamed out row by row in its group; not supported" % t)
        for c in consumers[t]:
            cg = lgroup[c]
            if not any(sl["t"] == tix[t] and sl["mem"] == "MEM_SD_IN" for sl in group_slots[cg]):
                raise SystemExit("t%d is read whole from SD by group %d; not supported" % (t, cg))
    for gi in range(ng):
        if sum(1 for d in group_dma[gi] if d["dir"] == 0) > 1 or sum(1 for d in group_dma[gi] if d["dir"] == 1) > 1:
            raise SystemExit("group %d needs more than one SD reader or writer; not supported" % gi)
        for d in group_dma[gi]:
            if d["chunk"] % 512 or d["nbuf"] > 4:
                raise SystemExit("group %d: DMA chunk must be a multiple of 512 and at most 4 buffers" % gi)
    for i, p in enumerate(pl):
        if (p.get("tile_rows") or 1) > 1:
            print("note: layer %d tile_rows %d, engine runs it one row at a time" % (i, p["tile_rows"]))

    gl = []
    si = di = 0
    slot_rows = []
    dma_rows = []
    for gi, g in enumerate(plan["groups"]):
        gl.append("    { %d, %d, %d, %d, %d, %d, %d }," % (g["first_layer"], g["last_layer"], 1 if g["whole"] else 0,
                                                       si, len(group_slots[gi]), di, len(group_dma[gi])))
        for s in group_slots[gi]:
            slot_rows.append("    { %d, %s, %d, %d }," % (s["t"], s["mem"], s["rows"], s["off"]))
        for d in group_dma[gi]:
            dma_rows.append("    { %d, %d, %d, %d, %d }," % (d["t"], d["dir"], d["nbuf"], d["chunk"], d["off"]))
        si += len(group_slots[gi])
        di += len(group_dma[gi])

    tl = []
    for k, tc in enumerate(tens_c):
        sd = spill.get(tc["id"], 0xFFFFFFFF)
        tl.append("    { %d, %d, %d, %d, %d, %d, %d, %d, %d, 0x%08Xu }, /* t%d */" % (
            tc["h"], tc["w"], tc["c"], tc["zp"], tc["rb"], tc["prod"], tc["pg"],
            whole_off.get(tc["id"], -1), 1 if tc["id"] in spill else 0, sd & 0xFFFFFFFF, tc["id"]))

    ll = []
    for i, L in enumerate(layers_c):
        ll.append("    { %s, %d, {%d, %d, %d}, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s, %s, %s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s }, /* #%d */" % (
            L["op"], L["n_in"], L["ins"][0], L["ins"][1], L["ins"][2], L["out"], L["kh"], L["kw"], L["sh"], L["sw"],
            L["pt"], L["pl"], L["amin"], L["amax"], L["w"], L["b"], L["mult"], L["shift"], L["cst"],
            L["cst_zp"], L["cst_first"], L["m1"], L["s1"], L["m2"], L["s2"], L["mo"], L["so"], L["ls"], L["lut"], i))

    data_h.append("#endif")
    open(a.out + "/model_data.h", "w").write("\n".join(data_h) + "\n")
    open(a.out + "/model_data.c", "w").write("\n".join(data_c))

    ph = ["#ifndef MODEL_PLAN_H", "#define MODEL_PLAN_H", '#include "nn_types.h"',
          "#define NN_N_TENSORS %d" % len(tens_c), "#define NN_N_LAYERS %d" % len(layers_c),
          "#define NN_N_GROUPS %d" % ng, "#define NN_ARENA_BYTES %du" % arena,
          "#define NN_INPUT_TENSOR 0", "#define NN_OUTPUT_TENSOR %d" % (len(tens_c) - 1),
          "#define NN_INPUT_BYTES %du" % (tens_c[0]["h"] * tens_c[0]["rb"]),
          "#define NN_INPUT_PART_OFFSET 0x%08Xu" % in_part,
          "#define NN_SPILL_END_BYTE 0x%08Xu" % spill_end,
          "#define NN_PLAN_LOADS %du" % sum(d.get("transfers", 0) for d in plan.get("sd_transfers", []) if d["dir"] == "read"),
          "#define NN_PLAN_STORES %du" % sum(d.get("transfers", 0) for d in plan.get("sd_transfers", []) if d["dir"] == "write"),
          "#define NN_PLAN_RD_BYTES %du" % sum(d["bytes"] for d in plan.get("sd_transfers", []) if d["dir"] == "read"),
          "#define NN_PLAN_WR_BYTES %du" % sum(d["bytes"] for d in plan.get("sd_transfers", []) if d["dir"] == "write"),
          "extern const nn_tensor_t nn_tensors[NN_N_TENSORS];",
          "extern const nn_layer_t nn_layers[NN_N_LAYERS];",
          "extern const nn_group_t nn_groups[NN_N_GROUPS];",
          "extern const nn_slot_t nn_slots[];", "extern const nn_dma_t nn_dmas[];",
          "extern const float nn_out_scale;", "extern const int32_t nn_out_zp;",
          "#endif"]
    open(a.out + "/model_plan.h", "w").write("\n".join(ph) + "\n")
    Tout = T[act_ids[-1]]
    pc = ['#include "model_plan.h"', '#include "model_data.h"', "",
          "const nn_tensor_t nn_tensors[NN_N_TENSORS] = {", *tl, "};", "",
          "const nn_layer_t nn_layers[NN_N_LAYERS] = {", *ll, "};", "",
          "const nn_group_t nn_groups[NN_N_GROUPS] = {", *gl, "};", "",
          "const nn_slot_t nn_slots[] = {", *(slot_rows or ["    {0}"]), "};", "",
          "const nn_dma_t nn_dmas[] = {", *(dma_rows or ["    {0}"]), "};", "",
          "const float nn_out_scale = %.9ef;" % float(Tout["scale"][0]),
          "const int32_t nn_out_zp = %d;" % int(Tout["zp"][0]), ""]
    open(a.out + "/model_plan.c", "w").write("\n".join(pc))

    flash = sum(np.asarray(M.tensor(t)["data"]).nbytes for o in ops for t in o["ins"]
                if t >= 0 and t not in T and M.tensor(t)["data"] is not None)
    print("layers %d  groups %d  tensors %d" % (len(layers_c), ng, len(tens_c)))
    print("arena %d B (plan peak %d B)  weights ~%d B in flash" % (arena, plan["totals"]["peak_ram"], flash))
    print("whole tensors: " + ", ".join("t%d@%d(%dB)" % (t, whole_off[t], tens_c[tix[t]]["h"] * tens_c[tix[t]]["rb"]) for t, _, _ in whole))
    print("spilled: " + ", ".join("t%d @ SD byte 0x%X" % kv for kv in spill.items()))
    free = a.ram - arena
    print("spill area ends at SD byte 0x%X (inputs must start after it)" % spill_end)
    print("RAM left for stack + globals: %d B (want >= %d)" % (free, a.reserve))
    if free < a.reserve:
        print("WARNING: plan leaves too little RAM. Re-run Tiling Lab with 'Keep free' >= %d B." % a.reserve)
    for i, o in enumerate(ops):
        h = hwc(T[o["outs"][0]]["shape"])
        L = layers_c[i]
        print("#%-2d %-18s out %3dx%3dx%3d k%dx%d s%d pad %d,%d act[%d,%d]" % (
            i, o["name"], *h, L["kh"], L["kw"], L["sh"], L["pt"], L["pl"], L["amin"], L["amax"]))


if __name__ == "__main__":
    main()
