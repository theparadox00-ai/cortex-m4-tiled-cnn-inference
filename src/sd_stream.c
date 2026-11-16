#include "sd_stream.h"
#include "sd.h"
#include <string.h>

#define MAXBUF 4

enum { B_EMPTY, B_BUSY, B_FULL };
enum { ACT_NONE, ACT_RD, ACT_WR };

extern uint32_t nn_now(void);

static struct {
    uint8_t on, nbuf, head, tail;
    uint8_t *buf;
    uint32_t chunk, sector, total, issued, pos;
    uint8_t st[MAXBUF];
    uint32_t len[MAXBUF];
} rd;

static struct {
    uint8_t on, nbuf, head, tail;
    uint8_t *buf;
    uint32_t chunk, sector, written, pos;
    uint8_t st[MAXBUF];
    uint32_t len[MAXBUF];
} wr;

static uint8_t act, act_idx;
static int err;
static uint32_t stall;
static sio_stats_t st;

static void finish(void)
{
    if (act == ACT_RD) rd.st[act_idx] = B_FULL;
    if (act == ACT_WR) {
        wr.st[act_idx] = B_EMPTY;
        wr.tail = (uint8_t)((wr.tail + 1) % wr.nbuf);
    }
    act = ACT_NONE;
}

static int rd_can_issue(void) { return rd.on && rd.issued < rd.total && rd.st[rd.tail] == B_EMPTY; }
static int wr_pending(void) { return wr.on && wr.st[wr.tail] == B_FULL; }

static int wr_starved(void)
{
    for (int i = 0; i < wr.nbuf; i++)
        if (wr.st[i] == B_EMPTY) return 0;
    return 1;
}

static void issue_read(void)
{
    uint32_t n = rd.total - rd.issued;
    if (n > rd.chunk) n = rd.chunk;
    uint8_t i = rd.tail;
    if (sd_start_read(rd.sector + rd.issued / 512u, rd.buf + i * rd.chunk, (n + 511u) / 512u) != SD_OK) {
        err = 1;
        return;
    }
    st.rd_cmds++;
    st.rd_sectors += (n + 511u) / 512u;
    rd.st[i] = B_BUSY;
    rd.len[i] = n;
    rd.issued += n;
    rd.tail = (uint8_t)((i + 1) % rd.nbuf);
    act = ACT_RD;
    act_idx = i;
}

static void issue_write(void)
{
    uint8_t i = wr.tail;
    uint32_t nb = (wr.len[i] + 511u) / 512u;
    if (sd_start_write(wr.sector + wr.written / 512u, wr.buf + i * wr.chunk, nb) != SD_OK) {
        err = 1;
        return;
    }
    st.wr_cmds++;
    st.wr_sectors += nb;
    wr.st[i] = B_BUSY;
    wr.written += nb * 512u;
    act = ACT_WR;
    act_idx = i;
}

int sio_poll(void)
{
    if (err) return -1;
    if (act != ACT_NONE) {
        int r = sd_service();
        if (r == SD_ERR) {
            err = 1;
            return -1;
        }
        if (r == SD_BUSY) return 1;
        finish();
    }
    int w = wr_pending(), r = rd_can_issue();
    if (w && (wr_starved() || !r)) issue_write();
    else if (r) issue_read();
    else if (w) issue_write();
    return err ? -1 : (act != ACT_NONE);
}

void sio_reader_open(uint32_t sector, uint32_t bytes, uint8_t *buf, uint32_t chunk, uint8_t nbuf)
{
    rd.on = 1;
    rd.sector = sector;
    rd.total = bytes;
    rd.buf = buf;
    rd.chunk = chunk;
    rd.nbuf = nbuf > MAXBUF ? MAXBUF : nbuf;
    rd.head = rd.tail = 0;
    rd.issued = rd.pos = 0;
    for (int i = 0; i < MAXBUF; i++) rd.st[i] = B_EMPTY;
    sio_poll();
}

uint32_t sio_reader_ready(void)
{
    uint32_t n = 0;
    uint8_t i = rd.head;
    uint32_t pos = rd.pos;
    for (int k = 0; k < rd.nbuf && rd.st[i] == B_FULL; k++) {
        n += rd.len[i] - pos;
        pos = 0;
        i = (uint8_t)((i + 1) % rd.nbuf);
    }
    return n;
}

int sio_read(uint8_t *dst, uint32_t n)
{
    uint32_t t0 = 0;
    int waited = 0;
    while (n) {
        if (rd.st[rd.head] != B_FULL) {
            if (!waited) {
                t0 = nn_now();
                waited = 1;
            }
            if (sio_poll() < 0) return -1;
            continue;
        }
        uint32_t k = rd.len[rd.head] - rd.pos;
        if (k > n) k = n;
        memcpy(dst, rd.buf + rd.head * rd.chunk + rd.pos, k);
        dst += k;
        n -= k;
        rd.pos += k;
        if (rd.pos == rd.len[rd.head]) {
            rd.st[rd.head] = B_EMPTY;
            rd.head = (uint8_t)((rd.head + 1) % rd.nbuf);
            rd.pos = 0;
            sio_poll();
        }
    }
    if (waited) stall += nn_now() - t0;
    return 0;
}

void sio_reader_close(void)
{
    while (act == ACT_RD && sio_poll() > 0) {}
    rd.on = 0;
}

void sio_writer_open(uint32_t sector, uint8_t *buf, uint32_t chunk, uint8_t nbuf)
{
    wr.on = 1;
    wr.sector = sector;
    wr.buf = buf;
    wr.chunk = chunk;
    wr.nbuf = nbuf > MAXBUF ? MAXBUF : nbuf;
    wr.head = wr.tail = 0;
    wr.written = wr.pos = 0;
    for (int i = 0; i < MAXBUF; i++) wr.st[i] = B_EMPTY;
}

int sio_write(const uint8_t *src, uint32_t n)
{
    uint32_t t0 = 0;
    int waited = 0;
    while (n) {
        if (wr.st[wr.head] != B_EMPTY) {
            if (!waited) {
                t0 = nn_now();
                waited = 1;
            }
            if (sio_poll() < 0) return -1;
            continue;
        }
        uint32_t k = wr.chunk - wr.pos;
        if (k > n) k = n;
        memcpy(wr.buf + wr.head * wr.chunk + wr.pos, src, k);
        src += k;
        n -= k;
        wr.pos += k;
        if (wr.pos == wr.chunk) {
            wr.len[wr.head] = wr.chunk;
            wr.st[wr.head] = B_FULL;
            wr.head = (uint8_t)((wr.head + 1) % wr.nbuf);
            wr.pos = 0;
            sio_poll();
        }
    }
    if (waited) stall += nn_now() - t0;
    return 0;
}

int sio_writer_close(void)
{
    uint32_t t0 = nn_now();
    if (wr.pos) {
        uint32_t padded = (wr.pos + 511u) & ~511u;
        while (wr.st[wr.head] != B_EMPTY)
            if (sio_poll() < 0) return -1;
        memset(wr.buf + wr.head * wr.chunk + wr.pos, 0, padded - wr.pos);
        wr.len[wr.head] = padded;
        wr.st[wr.head] = B_FULL;
        wr.head = (uint8_t)((wr.head + 1) % wr.nbuf);
        wr.pos = 0;
    }
    for (;;) {
        int busy = act != ACT_NONE;
        for (int i = 0; i < wr.nbuf; i++)
            if (wr.st[i] != B_EMPTY) busy = 1;
        if (!busy) break;
        if (sio_poll() < 0) return -1;
    }
    wr.on = 0;
    stall += nn_now() - t0;
    return 0;
}

int sio_error(void) { return err; }
uint32_t sio_stall_cycles(void) { return stall; }
void sio_reset_stats(void)
{
    stall = 0;
    st.rd_cmds = st.wr_cmds = st.rd_sectors = st.wr_sectors = 0;
    err = 0;
}

void sio_get_stats(sio_stats_t *s) { *s = st; }
