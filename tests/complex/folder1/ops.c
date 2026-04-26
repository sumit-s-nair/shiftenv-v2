#include "counter.h"
#include "ops.h"

void increment_counter(void* counter_handle, int amount) {
    /* Must use getters/setters because struct internals are invisible here */
    int current = get_value(counter_handle);
    set_value(counter_handle, current + amount);
}
