#include "sd.h"
#include "stm32f4xx.h"

#define CS_LOW() (GPIOB->BSRR = GPIO_BSRR_BR6)
#define CS_HIGH() (GPIOB->BSRR = GPIO_BSRR_BS6)
#define SPI_BR_SLOW (7u << SPI_CR1_BR_Pos)
#define SPI_BR_FAST (1u << SPI_CR1_BR_Pos)
#define POLL_PER_CALL 32u
#define POLL_LIMIT 2000000u

enum { ST_IDLE, ST_R_TOKEN, ST_R_DMA, ST_W_DMA, ST_W_BUSY, ST_END_BUSY, ST_ERR };

static volatile uint8_t ff_byte = 0xFF;
static volatile uint8_t sink_byte;
static uint8_t hc;
static struct {
    uint8_t st;
    uint8_t *buf;
    uint32_t left;
    uint32_t polls;
} q;

static uint8_t spi_x(uint8_t b)
{
    while (!(SPI1->SR & SPI_SR_TXE)) {}
    *(volatile uint8_t *)&SPI1->DR = b;
    while (!(SPI1->SR & SPI_SR_RXNE)) {}
    return *(volatile uint8_t *)&SPI1->DR;
}

static uint8_t sd_cmd(uint8_t cmd, uint32_t arg, uint8_t crc)
{
    spi_x(0xFF);
    spi_x(0x40 | cmd);
    spi_x(arg >> 24);
    spi_x(arg >> 16);
    spi_x(arg >> 8);
    spi_x(arg);
    spi_x(crc);
    if (cmd == 12) spi_x(0xFF);
    for (int i = 0; i < 10; i++) {
        uint8_t r = spi_x(0xFF);
        if (!(r & 0x80)) return r;
    }
    return 0xFF;
}

static void dma_go(uint8_t *rx, int rx_inc, const uint8_t *tx, int tx_inc, uint32_t n)
{
    DMA2_Stream0->CR = 0;
    DMA2_Stream3->CR = 0;
    while ((DMA2_Stream0->CR & DMA_SxCR_EN) || (DMA2_Stream3->CR & DMA_SxCR_EN)) {}
    DMA2->LIFCR = 0x3Du | (0x3Du << 22);
    DMA2_Stream0->PAR = (uint32_t)&SPI1->DR;
    DMA2_Stream0->M0AR = (uint32_t)rx;
    DMA2_Stream0->NDTR = n;
    DMA2_Stream0->FCR = 0;
    DMA2_Stream0->CR = (3u << DMA_SxCR_CHSEL_Pos) | (3u << DMA_SxCR_PL_Pos) | (rx_inc ? DMA_SxCR_MINC : 0);
    DMA2_Stream3->PAR = (uint32_t)&SPI1->DR;
    DMA2_Stream3->M0AR = (uint32_t)tx;
    DMA2_Stream3->NDTR = n;
    DMA2_Stream3->FCR = 0;
    DMA2_Stream3->CR = (3u << DMA_SxCR_CHSEL_Pos) | (1u << DMA_SxCR_PL_Pos) | DMA_SxCR_DIR_0 | (tx_inc ? DMA_SxCR_MINC : 0);
    SPI1->CR2 |= SPI_CR2_RXDMAEN;
    DMA2_Stream0->CR |= DMA_SxCR_EN;
    DMA2_Stream3->CR |= DMA_SxCR_EN;
    SPI1->CR2 |= SPI_CR2_TXDMAEN;
}

static int dma_done(void)
{
    uint32_t isr = DMA2->LISR;
    if (isr & (DMA_LISR_TEIF0 | DMA_LISR_TEIF3)) return -1;
    if (!(isr & DMA_LISR_TCIF0)) return 0;
    while (SPI1->SR & SPI_SR_BSY) {}
    SPI1->CR2 &= ~(SPI_CR2_RXDMAEN | SPI_CR2_TXDMAEN);
    DMA2_Stream0->CR = 0;
    DMA2_Stream3->CR = 0;
    DMA2->LIFCR = 0x3Du | (0x3Du << 22);
    return 1;
}

static int fail(void)
{
    SPI1->CR2 &= ~(SPI_CR2_RXDMAEN | SPI_CR2_TXDMAEN);
    DMA2_Stream0->CR = 0;
    DMA2_Stream3->CR = 0;
    CS_HIGH();
    spi_x(0xFF);
    q.st = ST_ERR;
    return SD_ERR;
}

int sd_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_DMA2EN;
    RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
    (void)RCC->APB2ENR;
    GPIOA->MODER = (GPIOA->MODER & ~(0x3Fu << 10)) | (0x2Au << 10);
    GPIOA->OSPEEDR |= (0x3Fu << 10);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFFFu << 20)) | (0x555u << 20);
    GPIOA->PUPDR = (GPIOA->PUPDR & ~(3u << 12)) | (1u << 12);
    GPIOB->MODER = (GPIOB->MODER & ~(3u << 12)) | (1u << 12);
    GPIOB->OSPEEDR |= (3u << 12);
    CS_HIGH();
    SPI1->CR1 = 0;
    SPI1->CR1 = SPI_CR1_MSTR | SPI_CR1_SSM | SPI_CR1_SSI | SPI_BR_SLOW;
    SPI1->CR1 |= SPI_CR1_SPE;
    q.st = ST_IDLE;

    for (int i = 0; i < 10; i++) spi_x(0xFF);
    CS_LOW();
    uint8_t r = 0xFF;
    for (int i = 0; i < 100 && r != 0x01; i++) r = sd_cmd(0, 0, 0x95);
    if (r != 0x01) return fail();
    r = sd_cmd(8, 0x1AA, 0x87);
    int v2 = 0;
    if (r == 0x01) {
        uint8_t b[4];
        for (int i = 0; i < 4; i++) b[i] = spi_x(0xFF);
        if (b[2] != 0x01 || b[3] != 0xAA) return fail();
        v2 = 1;
    }
    uint32_t n = 0;
    do {
        sd_cmd(55, 0, 0x01);
        r = sd_cmd(41, v2 ? 0x40000000u : 0, 0x01);
    } while (r != 0x00 && ++n < 200000u);
    if (r != 0x00) return fail();
    hc = 0;
    if (v2) {
        if (sd_cmd(58, 0, 0x01) != 0x00) return fail();
        uint8_t ocr0 = spi_x(0xFF);
        spi_x(0xFF);
        spi_x(0xFF);
        spi_x(0xFF);
        hc = (ocr0 & 0x40) ? 1 : 0;
    }
    if (!hc && sd_cmd(16, 512, 0x01) != 0x00) return fail();
    CS_HIGH();
    spi_x(0xFF);
    SPI1->CR1 &= ~SPI_CR1_SPE;
    SPI1->CR1 = (SPI1->CR1 & ~SPI_CR1_BR) | SPI_BR_FAST;
    SPI1->CR1 |= SPI_CR1_SPE;
    q.st = ST_IDLE;
    return SD_OK;
}

static uint32_t addr(uint32_t sector) { return hc ? sector : sector * 512u; }

int sd_start_read(uint32_t sector, uint8_t *buf, uint32_t nblocks)
{
    if (q.st != ST_IDLE || nblocks == 0) return SD_ERR;
    CS_LOW();
    if (sd_cmd(18, addr(sector), 0x01) != 0x00) return fail();
    q.buf = buf;
    q.left = nblocks;
    q.polls = 0;
    q.st = ST_R_TOKEN;
    return SD_OK;
}

int sd_start_write(uint32_t sector, const uint8_t *buf, uint32_t nblocks)
{
    if (q.st != ST_IDLE || nblocks == 0) return SD_ERR;
    CS_LOW();
    if (sd_cmd(25, addr(sector), 0x01) != 0x00) return fail();
    spi_x(0xFF);
    q.buf = (uint8_t *)buf;
    q.left = nblocks;
    q.polls = 0;
    spi_x(0xFC);
    dma_go((uint8_t *)&sink_byte, 0, q.buf, 1, 512);
    q.st = ST_W_DMA;
    return SD_OK;
}

int sd_service(void)
{
    for (;;) {
        switch (q.st) {
        case ST_IDLE:
            return SD_OK;
        case ST_ERR:
            return SD_ERR;
        case ST_R_TOKEN:
            for (uint32_t i = 0; i < POLL_PER_CALL; i++) {
                uint8_t b = spi_x(0xFF);
                if (b == 0xFE) {
                    dma_go(q.buf, 1, (const uint8_t *)&ff_byte, 0, 512);
                    q.st = ST_R_DMA;
                    break;
                }
                if (b != 0xFF) return fail();
            }
            if (q.st == ST_R_TOKEN) {
                q.polls += POLL_PER_CALL;
                if (q.polls > POLL_LIMIT) return fail();
                return SD_BUSY;
            }
            break;
        case ST_R_DMA: {
            int d = dma_done();
            if (d < 0) return fail();
            if (!d) return SD_BUSY;
            spi_x(0xFF);
            spi_x(0xFF);
            q.buf += 512;
            q.polls = 0;
            if (--q.left) {
                q.st = ST_R_TOKEN;
            } else {
                sd_cmd(12, 0, 0x01);
                q.st = ST_END_BUSY;
            }
            break;
        }
        case ST_W_DMA: {
            int d = dma_done();
            if (d < 0) return fail();
            if (!d) return SD_BUSY;
            spi_x(0xFF);
            spi_x(0xFF);
            uint8_t resp = 0xFF;
            for (int i = 0; i < 8 && resp == 0xFF; i++) resp = spi_x(0xFF);
            if ((resp & 0x1F) != 0x05) return fail();
            q.polls = 0;
            q.st = ST_W_BUSY;
            break;
        }
        case ST_W_BUSY:
            for (uint32_t i = 0; i < POLL_PER_CALL; i++) {
                if (spi_x(0xFF) == 0xFF) {
                    q.buf += 512;
                    q.polls = 0;
                    if (--q.left) {
                        spi_x(0xFC);
                        dma_go((uint8_t *)&sink_byte, 0, q.buf, 1, 512);
                        q.st = ST_W_DMA;
                    } else {
                        spi_x(0xFD);
                        spi_x(0xFF);
                        q.st = ST_END_BUSY;
                    }
                    break;
                }
            }
            if (q.st == ST_W_BUSY) {
                q.polls += POLL_PER_CALL;
                if (q.polls > POLL_LIMIT) return fail();
                return SD_BUSY;
            }
            break;
        case ST_END_BUSY:
            for (uint32_t i = 0; i < POLL_PER_CALL; i++) {
                if (spi_x(0xFF) == 0xFF) {
                    CS_HIGH();
                    spi_x(0xFF);
                    q.st = ST_IDLE;
                    return SD_OK;
                }
            }
            q.polls += POLL_PER_CALL;
            if (q.polls > POLL_LIMIT) return fail();
            return SD_BUSY;
        default:
            return fail();
        }
    }
}

int sd_read(uint32_t sector, uint8_t *buf, uint32_t nblocks)
{
    if (sd_start_read(sector, buf, nblocks) != SD_OK) return SD_ERR;
    int r;
    while ((r = sd_service()) == SD_BUSY) {}
    return r;
}

int sd_write(uint32_t sector, const uint8_t *buf, uint32_t nblocks)
{
    if (sd_start_write(sector, buf, nblocks) != SD_OK) return SD_ERR;
    int r;
    while ((r = sd_service()) == SD_BUSY) {}
    return r;
}
