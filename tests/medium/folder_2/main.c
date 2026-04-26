#include <stdio.h>
#include "pipeline.h"

typedef struct {
    int value;
    int offset;
} Context;

int main() {
    Context ctx = {10, 3};

    run_pipeline(&ctx);

    printf("result=%d\n", ctx.value);
    return 0;
}