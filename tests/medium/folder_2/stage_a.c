#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_a(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value += ctx->offset;
}