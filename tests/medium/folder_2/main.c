<<<<<<< HEAD
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
=======
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
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}