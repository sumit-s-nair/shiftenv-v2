#include "state.h"
#include "math_ops.h"
#include "formatter.h"

int main() {
    void* session = init_state(40, 2);

    compute_sum(session);
    print_state_result(session);

    destroy_state(session);
    return 0;
}
