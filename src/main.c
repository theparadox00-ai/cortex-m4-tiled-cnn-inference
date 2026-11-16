#include "stm32f4xx.h"
#include "usart.h"
#include "sd.h"
#include "sd_stream.h"
#include "engine.h"
#include <string.h>

#define WAIT_BUTTON 0
#define GROUP_TABLE_EVERY_SAMPLE 0

uint32_t nn_now(void) { return DWT->CYCCNT; }

static void clock_84mhz(void)
{
    SCB->CPACR |= (0xFu << 20);
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) {}
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;
    PWR->CR = (PWR->CR & ~PWR_CR_VOS) | PWR_CR_VOS_1;
    FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN | FLASH_ACR_LATENCY_2WS;
    RCC->PLLCFGR = (16u << RCC_PLLCFGR_PLLM_Pos) | (336u << RCC_PLLCFGR_PLLN_Pos) | (1u << RCC_PLLCFGR_PLLP_Pos) | (7u << RCC_PLLCFGR_PLLQ_Pos);
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) {}
    RCC->CFGR = RCC_CFGR_PPRE1_DIV2 | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
    SystemCoreClock = 84000000u;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void button_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
    (void)RCC->AHB1ENR;
    GPIOC->MODER &= ~(3u << 26);
}

static void wait_button(void)
{
    while (GPIOC->IDR & (1u << 13)) {}
    for (volatile uint32_t i = 0; i < 400000u; i++) {}
    while (!(GPIOC->IDR & (1u << 13))) {}
}

static void halt(const char *msg)
{
    uart_puts(msg);
    uart_puts("\n");
    for (;;) {}
}

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

static void pad(uint32_t v, int w)
{
    uint32_t d = 1;
    for (uint32_t x = v; x >= 10; x /= 10) d++;
    for (int i = (int)d; i < w; i++) uart_putc(' ');
    uart_putu(v);
}

static void pad_ms(uint32_t cycles, int w)
{
    uint32_t ms = cycles / 84000u;
    uint32_t d = 1;
    for (uint32_t x = ms; x >= 10; x /= 10) d++;
    for (int i = (int)d + 3; i < w; i++) uart_putc(' ');
    uart_ms(cycles);
}

static void print_method(void)
{
    uart_puts("\nHow latency is measured\n");
    uart_puts("  clock      : 84 MHz, DWT->CYCCNT counts every CPU cycle\n");
    uart_puts("  latency    : CYCCNT after nn_run() - CYCCNT before nn_run(), / 84000 = ms\n");
    uart_puts("               covers all 54 layers + every SD load/store of the plan\n");
    uart_puts("               input copy B -> A+0 is done before and is NOT included\n");
    uart_puts("  SD wait    : cycles the CPU sat inside sd_stream waiting for a DMA buffer\n");
    uart_puts("               (input chunk not ready, write buffer still busy, final flush)\n");
    uart_puts("  compute    : latency - SD wait\n");
    uart_puts("               DMA transfers that finish while layers compute cost 0 wait\n");
    uart_puts("  group time : CYCCNT around each group, sums to the latency\n");
    uart_puts("  loads/stores: SD commands issued by the plan's streams (CMD18 / CMD25),\n");
    uart_puts("               one per DMA tile, sectors = 512 B blocks moved\n\n");
}

int main(void)
{
    clock_84mhz();
    uart_init();
    button_init();
    uart_puts("\n==== Tiling-plan engine, final_model_int8, NUCLEO-F401RE ====\n");
    uart_puts("arena ");
    uart_putu(NN_ARENA_BYTES);
    uart_puts(" B  layers ");
    uart_putu(NN_N_LAYERS);
    uart_puts("  groups ");
    uart_putu(NN_N_GROUPS);
    uart_puts("  plan: ");
    uart_putu(NN_PLAN_LOADS);
    uart_puts(" loads, ");
    uart_putu(NN_PLAN_STORES);
    uart_puts(" stores\n");
    print_method();

    if (sd_init() != SD_OK) halt("SD init FAIL");
    uint8_t *hdr = nn_scratch();
    if (sd_read(NN_SD_BASE_SECTOR, hdr, 1) != SD_OK || memcmp(hdr, "TLIN", 4) != 0) halt("SD header FAIL");
    uint32_t n = rd32(hdr + 4), nbytes = rd32(hdr + 8), first = rd32(hdr + 12), spb = rd32(hdr + 16);
    if (nbytes != NN_INPUT_BYTES) halt("input size mismatch");
    if (first * 512u < NN_SPILL_END_BYTE) halt("samples overlap the spill area, rebuild the SD image");
    uart_puts("samples on SD: ");
    uart_putu(n);
    uart_puts("\n");

    const uint32_t in_sector = NN_SD_BASE_SECTOR + NN_INPUT_PART_OFFSET / 512u;
    uint32_t pass = 0, sum_ms = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (WAIT_BUTTON) wait_button();
        if (sd_read(NN_SD_BASE_SECTOR, hdr, 1) != SD_OK) halt("SD header FAIL");
        uint8_t label = hdr[32 + i];
        int8_t exp = (int8_t)hdr[256 + i];

        uint32_t c0 = DWT->CYCCNT;
        uint32_t src = NN_SD_BASE_SECTOR + first + i * spb;
        for (uint32_t k = 0; k < spb; k += 8) {
            uint32_t m = spb - k < 8 ? spb - k : 8;
            if (sd_read(src + k, hdr, m) != SD_OK || sd_write(in_sector + k, hdr, m) != SD_OK) halt("input copy FAIL");
        }
        uint32_t copy_cyc = DWT->CYCCNT - c0;

        int8_t out = 0;
        uint32_t t0 = DWT->CYCCNT;
        int r = nn_run(in_sector, &out, 1);
        uint32_t cyc = DWT->CYCCNT - t0;
        if (r != NN_OK) {
            uart_puts(r == NN_ERR_SD ? "SD error during inference\n" : "deadlock at layer ");
            if (r != NN_ERR_SD) uart_puti(nn_fail_layer());
            halt("");
        }
        uint32_t wait = nn_io_stall_cycles();
        sio_stats_t st;
        sio_get_stats(&st);
        int32_t pm = (int32_t)(((float)out - (float)nn_out_zp) * nn_out_scale * 1000.0f + 0.5f);

        uart_puts("---- sample ");
        uart_putu(i);
        uart_puts("  label ");
        uart_putu(label);
        uart_puts(" ----\n");
        uart_puts("input copy B -> A+0 : ");
        uart_ms(copy_cyc);
        uart_puts(" ms (not in latency)\n");
        uart_puts("result      : q ");
        uart_puti(out);
        uart_puts("  p 0.");
        if (pm < 100) uart_putc('0');
        if (pm < 10) uart_putc('0');
        uart_putu((uint32_t)pm);
        uart_puts("  tflite ");
        uart_puti(exp);
        uart_puts(out == exp ? "  PASS\n" : "  FAIL\n");
        uart_puts("latency     : ");
        uart_ms(cyc);
        uart_puts(" ms = compute ");
        uart_ms(cyc - wait);
        uart_puts(" ms + SD wait ");
        uart_ms(wait);
        uart_puts(" ms   (");
        uart_putu(cyc);
        uart_puts(" cycles / 84000)\n");
        uart_puts("SD loads    : ");
        uart_putu(st.rd_cmds);
        uart_puts(" (plan ");
        uart_putu(NN_PLAN_LOADS);
        uart_puts(")  sectors ");
        uart_putu(st.rd_sectors);
        uart_puts("  bytes ");
        uart_putu(st.rd_sectors * 512u);
        uart_puts(" (plan ");
        uart_putu(NN_PLAN_RD_BYTES);
        uart_puts(")\n");
        uart_puts("SD stores   : ");
        uart_putu(st.wr_cmds);
        uart_puts(" (plan ");
        uart_putu(NN_PLAN_STORES);
        uart_puts(")  sectors ");
        uart_putu(st.wr_sectors);
        uart_puts("  bytes ");
        uart_putu(st.wr_sectors * 512u);
        uart_puts(" (plan ");
        uart_putu(NN_PLAN_WR_BYTES);
        uart_puts(")\n");

        if (i == 0 || GROUP_TABLE_EVERY_SAMPLE) {
            uart_puts("group  layers      total ms   SD wait ms\n");
            for (int g = 0; g < NN_N_GROUPS; g++) {
                pad((uint32_t)g + 1u, 5);
                uart_puts("  #");
                uart_putu(nn_groups[g].first);
                uart_puts("-#");
                uart_putu(nn_groups[g].last);
                uart_puts(nn_groups[g].first < 10 && nn_groups[g].last < 10 ? "   " : (nn_groups[g].first < 10 || nn_groups[g].last < 10 ? "  " : " "));
                pad_ms(nn_group_cycles(g), 11);
                pad_ms(nn_group_wait(g), 13);
                uart_puts("\n");
            }
        }
        pass += out == exp;
        sum_ms += cyc / 84000u;
    }
    uart_puts("\n==== done: match ");
    uart_putu(pass);
    uart_puts("/");
    uart_putu(n);
    uart_puts("  average latency ");
    uart_putu(n ? sum_ms / n : 0);
    uart_puts(" ms ====\n");
    for (;;) {}
}
