#include "pipeline.h"
#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void run_pipeline(void *ptr) {
    Context *ctx = (Context *)ptr;

    stage_a(ctx);
    stage_b(ctx);
    stage_c(ctx);
}