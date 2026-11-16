#ifndef SD_STREAM_H
#define SD_STREAM_H

#include <stdint.h>

void sio_reader_open(uint32_t sector, uint32_t bytes, uint8_t *buf, uint32_t chunk, uint8_t nbuf);
uint32_t sio_reader_ready(void);
int sio_read(uint8_t *dst, uint32_t n);
void sio_reader_close(void);

void sio_writer_open(uint32_t sector, uint8_t *buf, uint32_t chunk, uint8_t nbuf);
int sio_write(const uint8_t *src, uint32_t n);
int sio_writer_close(void);

int sio_poll(void);
int sio_error(void);
uint32_t sio_stall_cycles(void);
void sio_reset_stats(void);

typedef struct {
    uint32_t rd_cmds, wr_cmds;
    uint32_t rd_sectors, wr_sectors;
} sio_stats_t;

void sio_get_stats(sio_stats_t *s);

#endif
