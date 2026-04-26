#include <stdio.h>
#include "state.h"
#include "formatter.h"

void print_state_result(void* state) {
    printf("Computed Result: %d\n", get_result(state));
}
