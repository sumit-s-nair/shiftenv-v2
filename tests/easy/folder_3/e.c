#include "e.h"
#include "d.h"

int stage_e(int x) {
    int v = stage_d(x);
    return v * v;
}