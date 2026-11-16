#include "kernels.h"
#include "nn_dsp.h"

static inline int32_t srdhm(int32_t a, int32_t b)
{
    if (a == b && a == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / (1ll << 31));
}

static inline int32_t rdpot(int32_t x, int e)
{
    int32_t mask = (int32_t)((1u << e) - 1u);
    int32_t rem = x & mask;
    int32_t th = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> e) + (rem > th ? 1 : 0);
}

static inline int32_t mbqm(int32_t x, int32_t m, int s)
{
    int ls = s > 0 ? s : 0;
    int rs = s > 0 ? 0 : -s;
    return rdpot(srdhm((int32_t)((uint32_t)x << ls), m), rs);
}

static inline int8_t clampq(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int8_t)v;
}

void k_conv_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out)
{
    const int cin = ti->c, cout = to->c, iw = ti->w, ow = to->w;
    const int kh = L->k_h, kw = L->k_w;
    const int32_t zp = ti->zp, ozp = to->zp;
    for (int ox = 0; ox < ow; ox++) {
        const int ix0 = ox * L->s_w - L->pad_l;
        for (int co = 0; co < cout; co++) {
            int32_t acc = L->b[co];
            const int8_t *wc = L->w + co * kh * kw * cin;
            for (int ky = 0; ky < kh; ky++) {
                const int8_t *r = rows[ky];
                if (!r) continue;
                for (int kx = 0; kx < kw; kx++) {
                    const int ix = ix0 + kx;
                    if (ix < 0 || ix >= iw) continue;
                    const int8_t *px = r + ix * cin;
                    const int8_t *wp = wc + (ky * kw + kx) * cin;
                    for (int ci = 0; ci < cin; ci++) acc += (px[ci] - zp) * wp[ci];
                }
            }
            out[ox * cout + co] = clampq(mbqm(acc, L->mult[co], L->shift[co]) + ozp, L->act_min, L->act_max);
        }
    }
}

void k_pw_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out)
{
    const int cin = ti->c, cout = to->c, w = to->w;
    const int32_t ozp = to->zp;
    const uint32_t off = ((uint32_t)(uint16_t)(-ti->zp)) * 0x00010001u;
    const int words = cin >> 2;
    for (int x = 0; x < w; x++) {
        const int8_t *px = in + x * cin;
        int8_t *po = out + x * cout;
        for (int co = 0; co < cout; co += 4) {
            const int8_t *w0 = L->w + co * cin;
            const int8_t *w1 = w0 + cin;
            const int8_t *w2 = w1 + cin;
            const int8_t *w3 = w2 + cin;
            int32_t a0 = L->b[co], a1 = L->b[co + 1], a2 = L->b[co + 2], a3 = L->b[co + 3];
            for (int i = 0; i < words; i++) {
                const uint32_t xv = d_rd32(px + 4 * i);
                const uint32_t xe = d_sxtab16(off, xv);
                const uint32_t xo = d_sxtab16_ror8(off, xv);
                uint32_t wv = d_rd32(w0 + 4 * i);
                a0 = d_smlad(xe, d_sxtb16(wv), a0);
                a0 = d_smlad(xo, d_sxtb16_ror8(wv), a0);
                wv = d_rd32(w1 + 4 * i);
                a1 = d_smlad(xe, d_sxtb16(wv), a1);
                a1 = d_smlad(xo, d_sxtb16_ror8(wv), a1);
                wv = d_rd32(w2 + 4 * i);
                a2 = d_smlad(xe, d_sxtb16(wv), a2);
                a2 = d_smlad(xo, d_sxtb16_ror8(wv), a2);
                wv = d_rd32(w3 + 4 * i);
                a3 = d_smlad(xe, d_sxtb16(wv), a3);
                a3 = d_smlad(xo, d_sxtb16_ror8(wv), a3);
            }
            po[co] = clampq(mbqm(a0, L->mult[co], L->shift[co]) + ozp, L->act_min, L->act_max);
            po[co + 1] = clampq(mbqm(a1, L->mult[co + 1], L->shift[co + 1]) + ozp, L->act_min, L->act_max);
            po[co + 2] = clampq(mbqm(a2, L->mult[co + 2], L->shift[co + 2]) + ozp, L->act_min, L->act_max);
            po[co + 3] = clampq(mbqm(a3, L->mult[co + 3], L->shift[co + 3]) + ozp, L->act_min, L->act_max);
        }
    }
}

void k_dw_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out)
{
    const int C = to->c, iw = ti->w, ow = to->w;
    const int kh = L->k_h, kw = L->k_w;
    const int32_t zp = ti->zp, ozp = to->zp;
    for (int ox = 0; ox < ow; ox++) {
        const int ix0 = ox * L->s_w - L->pad_l;
        const int kx0 = ix0 < 0 ? -ix0 : 0;
        const int kx1 = ix0 + kw > iw ? iw - ix0 : kw;
        for (int c = 0; c < C; c++) {
            int32_t acc = L->b[c];
            for (int ky = 0; ky < kh; ky++) {
                const int8_t *r = rows[ky];
                if (!r) continue;
                const int8_t *wp = L->w + (ky * kw + kx0) * C + c;
                const int8_t *px = r + (ix0 + kx0) * C + c;
                for (int k = 0; k < kx1 - kx0; k++) acc += (px[k * C] - zp) * wp[k * C];
            }
            out[ox * C + c] = clampq(mbqm(acc, L->mult[c], L->shift[c]) + ozp, L->act_min, L->act_max);
        }
    }
}

void k_avgpool_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out)
{
    const int C = to->c, iw = ti->w, ow = to->w;
    const int kh = L->k_h, kw = L->k_w;
    for (int ox = 0; ox < ow; ox++) {
        const int ix0 = ox * L->s_w - L->pad_l;
        for (int c = 0; c < C; c++) {
            int32_t acc = 0, n = 0;
            for (int ky = 0; ky < kh; ky++) {
                const int8_t *r = rows[ky];
                if (!r) continue;
                for (int kx = 0; kx < kw; kx++) {
                    const int ix = ix0 + kx;
                    if (ix < 0 || ix >= iw) continue;
                    acc += r[ix * C + c];
                    n++;
                }
            }
            if (n == 0) { out[ox * C + c] = 0; continue; }
            acc = acc > 0 ? (acc + n / 2) / n : (acc - n / 2) / n;
            out[ox * C + c] = clampq(acc, L->act_min, L->act_max);
        }
    }
}

void k_add_row(const nn_layer_t *L, const nn_tensor_t *t1, const nn_tensor_t *t2, const nn_tensor_t *to, const int8_t *a, const int8_t *b, int8_t *out)
{
    const uint32_t n = to->row_bytes;
    const int32_t z1 = t1->zp, z2 = t2->zp, ozp = to->zp, ls = L->left_shift;
    for (uint32_t i = 0; i < n; i++) {
        int32_t x1 = (a[i] - z1) * (1 << ls);
        int32_t x2 = (b[i] - z2) * (1 << ls);
        int32_t s1 = rdpot(srdhm(x1, L->m1), -L->s1);
        int32_t s2 = rdpot(srdhm(x2, L->m2), -L->s2);
        int32_t r = L->op == OP_SUB ? s1 - s2 : s1 + s2;
        r = rdpot(srdhm(r, L->mo), -L->so) + ozp;
        out[i] = clampq(r, L->act_min, L->act_max);
    }
}

void k_cst_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out)
{
    const int C = to->c, w = to->w;
    const int32_t zi = ti->zp, zc = L->cst_zp, ozp = to->zp, ls = L->left_shift;
    for (int x = 0; x < w; x++) {
        for (int c = 0; c < C; c++) {
            int32_t xv = in[x * C + c] - zi;
            int32_t cv = L->cst[c] - zc;
            int32_t v1 = L->cst_first ? cv : xv;
            int32_t v2 = L->cst_first ? xv : cv;
            int32_t r;
            if (L->op == OP_MUL) {
                r = mbqm(v1 * v2, L->mo, L->so) + ozp;
            } else {
                int32_t s1 = rdpot(srdhm(v1 * (1 << ls), L->m1), -L->s1);
                int32_t s2 = rdpot(srdhm(v2 * (1 << ls), L->m2), -L->s2);
                r = L->op == OP_SUB ? s1 - s2 : s1 + s2;
                r = rdpot(srdhm(r, L->mo), -L->so) + ozp;
            }
            out[x * C + c] = clampq(r, L->act_min, L->act_max);
        }
    }
}

void k_concat_row(const nn_layer_t *L, const nn_tensor_t *const *ti, const int8_t *const *in, const nn_tensor_t *to, int8_t *out)
{
    const int w = to->w;
    for (int x = 0; x < w; x++) {
        int8_t *po = out + x * to->c;
        for (int k = 0; k < L->n_in; k++) {
            const int c = ti[k]->c;
            const int8_t *pi = in[k] + x * c;
            for (int j = 0; j < c; j++) *po++ = pi[j];
        }
    }
}

void k_fc(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out)
{
    const int n = ti->h * ti->row_bytes, m = to->c;
    const int32_t zp = ti->zp, ozp = to->zp;
    for (int o = 0; o < m; o++) {
        const int8_t *w = L->w + o * n;
        int32_t acc = L->b[o];
        for (int i = 0; i < n; i++) acc += (in[i] - zp) * w[i];
        out[o] = clampq(mbqm(acc, L->mult[o], L->shift[o]) + ozp, L->act_min, L->act_max);
    }
}

void k_logistic(const nn_layer_t *L, const int8_t *in, int8_t *out, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) out[i] = L->lut[(uint8_t)in[i]];
}
