#include "state.h"
#include "math_ops.h"

void compute_sum(void* state) {
    int sum = get_a(state) + get_b(state);
    set_result(state, sum);
}
