#include "c.h"
#include "b.h"

int stage_c(int x) {
    return stage_b(x) - 5;
}