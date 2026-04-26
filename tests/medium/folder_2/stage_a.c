<<<<<<< HEAD
#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_a(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value += ctx->offset;
=======
#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_a(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value += ctx->offset;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}