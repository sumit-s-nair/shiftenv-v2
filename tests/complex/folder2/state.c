#include <stdlib.h>
#include "state.h"

struct DataState {
    int a;
    int b;
    int result;
};

void* init_state(int a, int b) {
    struct DataState* state = (struct DataState*)malloc(sizeof(struct DataState));
    if (state) {
        state->a = a;
        state->b = b;
        state->result = 0;
    }
    return state;
}

int get_a(void* state) { return ((struct DataState*)state)->a; }
int get_b(void* state) { return ((struct DataState*)state)->b; }
void set_result(void* state, int res) { ((struct DataState*)state)->result = res; }
int get_result(void* state) { return ((struct DataState*)state)->result; }
void destroy_state(void* state) { free(state); }
