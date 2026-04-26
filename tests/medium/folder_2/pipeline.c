<<<<<<< HEAD
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
=======
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
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}