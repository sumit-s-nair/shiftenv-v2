#include "d.h"
#include "b.h"
#include "c.h"

int stage_d(int x) {
    return stage_b(x) + stage_c(x);
}