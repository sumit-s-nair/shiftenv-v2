<<<<<<< HEAD
#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_b(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value *= 2;
=======
#include "stage.h"

typedef struct {
    int value;
    int offset;
} Context;

void stage_b(void *ptr) {
    Context *ctx = (Context *)ptr;
    ctx->value *= 2;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}