//!
//! \file       ssd1306_i2c.cpp
//! \author     Abdelrahman Ali
//! \date       2025-01-31
//!
//! \brief      ssd1306 driver.
//!

#include "ssd1306_i2c.h"


uint8_t font[] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Nothing
0x78, 0x14, 0x12, 0x11, 0x12, 0x14, 0x78, 0x00, //A
0x7f, 0x49, 0x49, 0x49, 0x49, 0x49, 0x7f, 0x00, //B
0x7e, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x00, //C
0x7f, 0x41, 0x41, 0x41, 0x41, 0x41, 0x7e, 0x00, //D
0x7f, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x00, //E
0x7f, 0x09, 0x09, 0x09, 0x09, 0x01, 0x01, 0x00, //F
0x7f, 0x41, 0x41, 0x41, 0x51, 0x51, 0x73, 0x00, //G
0x7f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x7f, 0x00, //H
0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x00, //I
0x21, 0x41, 0x41, 0x3f, 0x01, 0x01, 0x01, 0x00, //J
0x00, 0x7f, 0x08, 0x08, 0x14, 0x22, 0x41, 0x00, //K
0x7f, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x00, //L
0x7f, 0x02, 0x04, 0x08, 0x04, 0x02, 0x7f, 0x00, //M
0x7f, 0x02, 0x04, 0x08, 0x10, 0x20, 0x7f, 0x00, //N
0x3e, 0x41, 0x41, 0x41, 0x41, 0x41, 0x3e, 0x00, //O
0x7f, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e, 0x00, //P
0x3e, 0x41, 0x41, 0x49, 0x51, 0x61, 0x7e, 0x00, //Q
0x7f, 0x11, 0x11, 0x11, 0x31, 0x51, 0x0e, 0x00, //R
0x46, 0x49, 0x49, 0x49, 0x49, 0x30, 0x00, 0x00, //S
0x01, 0x01, 0x01, 0x7f, 0x01, 0x01, 0x01, 0x00, //T
0x3f, 0x40, 0x40, 0x40, 0x40, 0x40, 0x3f, 0x00, //U
0x0f, 0x10, 0x20, 0x40, 0x20, 0x10, 0x0f, 0x00, //V
0x7f, 0x20, 0x10, 0x08, 0x10, 0x20, 0x7f, 0x00, //W
0x00, 0x41, 0x22, 0x14, 0x14, 0x22, 0x41, 0x00, //X
0x01, 0x02, 0x04, 0x78, 0x04, 0x02, 0x01, 0x00, //Y
0x41, 0x61, 0x59, 0x45, 0x43, 0x41, 0x00, 0x00, //Z
0x3e, 0x41, 0x41, 0x49, 0x41, 0x41, 0x3e, 0x00, //0
0x00, 0x00, 0x42, 0x7f, 0x40, 0x00, 0x00, 0x00, //1
0x30, 0x49, 0x49, 0x49, 0x49, 0x46, 0x00, 0x00, //2
0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x36, 0x00, //3
0x3f, 0x20, 0x20, 0x78, 0x20, 0x20, 0x00, 0x00, //4
0x4f, 0x49, 0x49, 0x49, 0x49, 0x30, 0x00, 0x00, //5
0x3f, 0x48, 0x48, 0x48, 0x48, 0x48, 0x30, 0x00, //6
0x01, 0x01, 0x01, 0x61, 0x31, 0x0d, 0x03, 0x00, //7
0x36, 0x49, 0x49, 0x49, 0x49, 0x49, 0x36, 0x00, //8
0x06, 0x09, 0x09, 0x09, 0x09, 0x09, 0x7f, 0x00, //9
};


struct render_area {
    uint8_t start_col;
    uint8_t end_col;
    uint8_t start_page;
    uint8_t end_page;

    int buflen;
};

struct render_area frame_area = {
    start_col: 0,
    end_col : SSD1306_WIDTH - 1,
    start_page : 0,
    end_page : SSD1306_NUM_PAGES - 1
};

uint8_t buf[SSD1306_BUF_LEN];

void calc_render_area_buflen(struct render_area *area) {
    // calculate how long the flattened buffer will be for a render area
    area->buflen = (area->end_col - area->start_col + 1) * (area->end_page - area->start_page + 1);
}



void SSD1306_send_cmd(uint8_t cmd) {
    uint8_t buf[2] = {0x80, cmd};
    i2c_write_blocking(I2C_PORT, SSD1306_I2C_ADDR, buf, 2, false);
}

void SSD1306_send_cmd_list(uint8_t *buf, int num) {
    for (int i=0;i<num;i++)
        SSD1306_send_cmd(buf[i]);
}

void SSD1306_send_buf(uint8_t buf[], int buflen) {
    uint8_t *temp_buf = (uint8_t *)malloc(buflen + 1);

    temp_buf[0] = 0x40;
    memcpy(temp_buf+1, buf, buflen);

    i2c_write_blocking(I2C_PORT, SSD1306_I2C_ADDR, temp_buf, buflen + 1, false);

    free(temp_buf);
}

void SSD1306_init() {
    uint8_t cmds[] = {
        SSD1306_SET_DISP,               // set display off
        /* memory mapping */
        SSD1306_SET_MEM_MODE,           // set memory address mode 0 = horizontal, 1 = vertical, 2 = page
        0x00,                           // horizontal addressing mode
        /* resolution and layout */
        SSD1306_SET_DISP_START_LINE,    // set display start line to 0
        SSD1306_SET_SEG_REMAP | 0x01,   // set segment re-map, column address 127 is mapped to SEG0
        SSD1306_SET_MUX_RATIO,          // set multiplex ratio
        SSD1306_HEIGHT - 1,             // Display height - 1
        SSD1306_SET_COM_OUT_DIR | 0x08, // set COM (common) output scan direction. Scan from bottom up, COM[N-1] to COM0
        SSD1306_SET_DISP_OFFSET,        // set display offset
        0x00,                           // no offset
        SSD1306_SET_COM_PIN_CFG,        // set COM (common) pins hardware configuration. Board specific magic number.
                                        // 0x02 Works for 128x32, 0x12 Possibly works for 128x64. Other options 0x22, 0x32
        0x02,
        /* timing and driving scheme */
        SSD1306_SET_DISP_CLK_DIV,       // set display clock divide ratio
        0x80,                           // div ratio of 1, standard freq
        SSD1306_SET_PRECHARGE,          // set pre-charge period
        0xF1,                           // Vcc internally generated on our board
        SSD1306_SET_VCOM_DESEL,         // set VCOMH deselect level
        0x30,                           // 0.83xVcc
        /* display */
        SSD1306_SET_CONTRAST,           // set contrast control
        0xFF,
        SSD1306_SET_ENTIRE_ON,          // set entire display on to follow RAM content
        SSD1306_SET_NORM_DISP,           // set normal (not inverted) display
        SSD1306_SET_CHARGE_PUMP,        // set charge pump
        0x14,                           // Vcc internally generated on our board
        SSD1306_SET_SCROLL | 0x00,      // deactivate horizontal scrolling if set. This is necessary as memory writes will corrupt if scrolling was enabled
        SSD1306_SET_DISP | 0x01, // turn display on
    };

    SSD1306_send_cmd_list(cmds, count_of(cmds));
}


void render(uint8_t *buf, struct render_area *area) {
    // update a portion of the display with a render area
    uint8_t cmds[] = {
        SSD1306_SET_COL_ADDR,
        area->start_col,
        area->end_col,
        SSD1306_SET_PAGE_ADDR,
        area->start_page,
        area->end_page
    };

    SSD1306_send_cmd_list(cmds, count_of(cmds));
    SSD1306_send_buf(buf, area->buflen);
}

static inline int GetFontIndex(uint8_t ch) {
    if (ch >= 'A' && ch <='Z') {
        return  ch - 'A' + 1;
    }
    else if (ch >= '0' && ch <='9') {
        return  ch - '0' + 27;
    }
    else return  0; // Not got that char so space.
}

static void WriteChar(uint8_t *buf, int16_t x, int16_t y, uint8_t ch) {
    if (x > SSD1306_WIDTH - 8 || y > SSD1306_HEIGHT - 8)
        return;

    // For the moment, only write on Y row boundaries (every 8 vertical pixels)
    y = y/8;

    ch = toupper(ch);
    int idx = GetFontIndex(ch);
    int fb_idx = y * 128 + x;

    for (int i=0;i<8;i++) {
        buf[fb_idx++] = font[idx * 8 + i];
    }
}

static void WriteString(uint8_t *buf, int16_t x, int16_t y, char *str) {
    // Cull out any string off the screen
    if (x > SSD1306_WIDTH - 8 || y > SSD1306_HEIGHT - 8)
        return;

    while (*str) {
        WriteChar(buf, x, y, *str++);
        x+=8;
    }
}


void ssd1306_main_init() {
    SSD1306_init();
    ssd1306_clear();
}

void ssd1306_clear(void) {
    calc_render_area_buflen(&frame_area);
    memset(buf, 0, SSD1306_BUF_LEN);
    render(buf, &frame_area);
}

void ssd1306_display_text_line(char *text) {
    ssd1306_clear();
    WriteString(buf, 5, 0, text);
    render(buf, &frame_area);
}
