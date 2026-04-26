#include "b.h"
#include "a.h"

int stage_b(int x) {
    return stage_a(x) * 2;
}