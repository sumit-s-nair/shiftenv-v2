#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_c(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value -= ctx->offset;
}