#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_b(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value *= 2;
}