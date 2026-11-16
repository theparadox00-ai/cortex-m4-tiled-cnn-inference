#include "engine.h"
#include "kernels.h"
#include "sd_stream.h"
#include <string.h>

static uint8_t arena[NN_ARENA_BYTES] __attribute__((aligned(4)));

typedef struct {
    int8_t *base;
    uint16_t cap;
    uint16_t done;
    uint8_t mem;
} rt_t;

static rt_t rt[NN_N_TENSORS];
static uint16_t nxt[NN_N_LAYERS];
static uint32_t gcyc[NN_N_GROUPS];
static uint32_t gwait[NN_N_GROUPS];
static int fail_layer;

__attribute__((weak)) uint32_t nn_now(void) { return 0; }
__attribute__((weak)) void nn_row_hook(int tensor, int row, const int8_t *p, uint32_t n)
{
    (void)tensor;
    (void)row;
    (void)p;
    (void)n;
}

static inline int8_t *rowp(int t, int r)
{
    const rt_t *x = &rt[t];
    uint32_t rb = nn_tensors[t].row_bytes;
    return x->base + (x->cap ? (uint32_t)(r % x->cap) : (uint32_t)r) * rb;
}

static inline int windowed(const nn_layer_t *L)
{
    return L->op == OP_CONV || L->op == OP_DW || L->op == OP_AVGPOOL;
}

static inline int lo_row(const nn_layer_t *L, int r)
{
    if (!windowed(L)) return r;
    int lo = r * L->s_h - L->pad_t;
    return lo < 0 ? 0 : lo;
}

static inline int hi_row(const nn_layer_t *L, int r, int ih)
{
    if (!windowed(L)) return r;
    int hi = r * L->s_h - L->pad_t + L->k_h - 1;
    return hi >= ih ? ih - 1 : hi;
}

static int min_needed(int t, const nn_group_t *g)
{
    int m = 0x7FFF;
    for (int l = g->first; l <= g->last; l++) {
        const nn_layer_t *L = &nn_layers[l];
        if (nxt[l] >= nn_tensors[L->out].h) continue;
        for (int k = 0; k < L->n_in; k++) {
            if (L->in[k] != t) continue;
            int lo = lo_row(L, nxt[l]);
            if (lo < m) m = lo;
        }
    }
    return m;
}

static int can_run(int l, const nn_group_t *g)
{
    const nn_layer_t *L = &nn_layers[l];
    int r = nxt[l];
    for (int k = 0; k < L->n_in; k++) {
        int t = L->in[k];
        if (rt[t].done <= hi_row(L, r, nn_tensors[t].h)) return 0;
    }
    const rt_t *o = &rt[L->out];
    if (o->cap && o->mem == MEM_RING && r >= min_needed(L->out, g) + o->cap) return 0;
    return 1;
}

static void run_row(int l, int r)
{
    const nn_layer_t *L = &nn_layers[l];
    const nn_tensor_t *to = &nn_tensors[L->out];
    const nn_tensor_t *ti = &nn_tensors[L->in[0]];
    int8_t *out = rowp(L->out, r);
    const int8_t *rows[8];

    if (windowed(L)) {
        int base = r * L->s_h - L->pad_t;
        for (int ky = 0; ky < L->k_h && ky < 8; ky++) {
            int y = base + ky;
            rows[ky] = (y < 0 || y >= ti->h) ? 0 : rowp(L->in[0], y);
        }
    }

    switch (L->op) {
    case OP_CONV:
        if (L->k_h == 1 && L->k_w == 1 && L->s_w == 1 && (ti->c & 3) == 0 && (to->c & 3) == 0)
            k_pw_row(L, ti, to, rows[0], out);
        else
            k_conv_row(L, ti, to, rows, out);
        break;
    case OP_DW:
        k_dw_row(L, ti, to, rows, out);
        break;
    case OP_AVGPOOL:
        k_avgpool_row(L, ti, to, rows, out);
        break;
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
        if (L->cst)
            k_cst_row(L, ti, to, rowp(L->in[0], r), out);
        else
            k_add_row(L, ti, &nn_tensors[L->in[1]], to, rowp(L->in[0], r), rowp(L->in[1], r), out);
        break;
    case OP_CONCAT: {
        const nn_tensor_t *tis[3];
        const int8_t *ins[3];
        for (int k = 0; k < L->n_in; k++) {
            tis[k] = &nn_tensors[L->in[k]];
            ins[k] = rowp(L->in[k], r);
        }
        k_concat_row(L, tis, ins, to, out);
        break;
    }
    default:
        break;
    }
}

static void run_whole(int l)
{
    const nn_layer_t *L = &nn_layers[l];
    const nn_tensor_t *ti = &nn_tensors[L->in[0]];
    const nn_tensor_t *to = &nn_tensors[L->out];
    const int8_t *in = rowp(L->in[0], 0);
    int8_t *out = rowp(L->out, 0);
    switch (L->op) {
    case OP_RESHAPE:
        memmove(out, in, ti->h * ti->row_bytes);
        break;
    case OP_FC:
        k_fc(L, ti, to, in, out);
        break;
    case OP_LOGISTIC:
        k_logistic(L, in, out, to->h * to->row_bytes);
        break;
    default:
        for (int r = 0; r < to->h; r++) run_row(l, r);
        break;
    }
    for (int r = 0; r < to->h; r++) nn_row_hook(L->out, r, rowp(L->out, r), to->row_bytes);
}

static uint32_t sd_sector_of(int t, uint32_t input_sector)
{
    if (t == NN_INPUT_TENSOR) return input_sector;
    return NN_SD_BASE_SECTOR + nn_tensors[t].sd_off / 512u;
}

static void group_setup(int gi, uint32_t input_sector)
{
    const nn_group_t *g = &nn_groups[gi];
    for (int t = 0; t < NN_N_TENSORS; t++) {
        const nn_tensor_t *T = &nn_tensors[t];
        rt[t].cap = 0;
        rt[t].mem = 0xFF;
        if (T->whole_off >= 0) {
            rt[t].base = (int8_t *)(arena + T->whole_off);
            rt[t].done = (T->prod_group >= 0 && T->prod_group < gi) ? T->h : 0;
        } else {
            rt[t].base = 0;
            rt[t].done = 0;
        }
    }
    for (int s = g->slot_first; s < g->slot_first + g->slot_count; s++) {
        const nn_slot_t *S = &nn_slots[s];
        rt[S->tensor].base = (int8_t *)(arena + S->off);
        rt[S->tensor].cap = S->rows;
        rt[S->tensor].done = 0;
        rt[S->tensor].mem = S->mem;
    }
    for (int l = g->first; l <= g->last; l++) nxt[l] = 0;
    for (int d = g->dma_first; d < g->dma_first + g->dma_count; d++) {
        const nn_dma_t *D = &nn_dmas[d];
        const nn_tensor_t *T = &nn_tensors[D->tensor];
        uint32_t sec = sd_sector_of(D->tensor, input_sector);
        if (D->dir == 0)
            sio_reader_open(sec, T->h * T->row_bytes, arena + D->off, D->chunk, D->nbuf);
        else
            sio_writer_open(sec, arena + D->off, D->chunk, D->nbuf);
    }
}

static int pull_sd_rows(const nn_group_t *g, int block)
{
    int prog = 0;
    for (int s = g->slot_first; s < g->slot_first + g->slot_count; s++) {
        const nn_slot_t *S = &nn_slots[s];
        if (S->mem != MEM_SD_IN) continue;
        int t = S->tensor;
        const nn_tensor_t *T = &nn_tensors[t];
        while (rt[t].done < T->h && rt[t].done < min_needed(t, g) + rt[t].cap) {
            if (!block && sio_reader_ready() < T->row_bytes) break;
            if (sio_read((uint8_t *)rowp(t, rt[t].done), T->row_bytes) < 0) return -1;
            nn_row_hook(t, rt[t].done, rowp(t, rt[t].done), T->row_bytes);
            rt[t].done++;
            prog = 1;
            if (block) return 1;
        }
    }
    return prog;
}

static int run_group(int gi)
{
    const nn_group_t *g = &nn_groups[gi];
    if (g->whole) {
        for (int l = g->first; l <= g->last; l++) run_whole(l);
        return NN_OK;
    }
    for (;;) {
        int prog = 0, all = 1;
        int p = pull_sd_rows(g, 0);
        if (p < 0) return NN_ERR_SD;
        prog |= p;
        for (int l = g->first; l <= g->last; l++) {
            const nn_layer_t *L = &nn_layers[l];
            const nn_tensor_t *to = &nn_tensors[L->out];
            while (nxt[l] < to->h && can_run(l, g)) {
                int r = nxt[l];
                run_row(l, r);
                int8_t *o = rowp(L->out, r);
                nn_row_hook(L->out, r, o, to->row_bytes);
                if (rt[L->out].mem == MEM_SD_OUT && sio_write((const uint8_t *)o, to->row_bytes) < 0) return NN_ERR_SD;
                nxt[l]++;
                rt[L->out].done++;
                prog = 1;
                if (sio_poll() < 0) return NN_ERR_SD;
            }
            if (nxt[l] < to->h) all = 0;
        }
        if (all) break;
        if (!prog) {
            p = pull_sd_rows(g, 1);
            if (p < 0) return NN_ERR_SD;
            if (!p) {
                for (int l = g->first; l <= g->last; l++)
                    if (nxt[l] < nn_tensors[nn_layers[l].out].h) {
                        fail_layer = l;
                        break;
                    }
                return NN_ERR_DEADLOCK;
            }
        }
    }
    for (int d = g->dma_first; d < g->dma_first + g->dma_count; d++) {
        if (nn_dmas[d].dir == 0)
            sio_reader_close();
        else if (sio_writer_close() < 0)
            return NN_ERR_SD;
    }
    return NN_OK;
}

int nn_run(uint32_t input_sector, int8_t *out, uint32_t out_len)
{
    fail_layer = -1;
    sio_reset_stats();
    for (int gi = 0; gi < NN_N_GROUPS; gi++) {
        uint32_t t0 = nn_now();
        uint32_t w0 = sio_stall_cycles();
        group_setup(gi, input_sector);
        int r = run_group(gi);
        gcyc[gi] = nn_now() - t0;
        gwait[gi] = sio_stall_cycles() - w0;
        if (r != NN_OK) return r;
    }
    const nn_tensor_t *T = &nn_tensors[NN_OUTPUT_TENSOR];
    uint32_t n = T->h * T->row_bytes;
    if (n > out_len) n = out_len;
    memcpy(out, rowp(NN_OUTPUT_TENSOR, 0), n);
    return NN_OK;
}

uint32_t nn_group_cycles(int g) { return gcyc[g]; }
uint32_t nn_group_wait(int g) { return gwait[g]; }
uint32_t nn_io_stall_cycles(void) { return sio_stall_cycles(); }
int nn_fail_layer(void) { return fail_layer; }
uint8_t *nn_scratch(void) { return arena; }
