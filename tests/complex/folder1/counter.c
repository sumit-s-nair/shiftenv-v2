#include <stdlib.h>
#include "counter.h"

/* Struct is defined entirely within the C file */
struct Counter {
    int value;
};

void* create_counter(int initial_value) {
    struct Counter* c = (struct Counter*)malloc(sizeof(struct Counter));
    if (c) {
        c->value = initial_value;
    }
    return c;
}

int get_value(void* counter_handle) {
    if (!counter_handle) return 0;
    return ((struct Counter*)counter_handle)->value;
}

void set_value(void* counter_handle, int value) {
    if (counter_handle) {
        ((struct Counter*)counter_handle)->value = value;
    }
}

void free_counter(void* counter_handle) {
    free(counter_handle);
}
