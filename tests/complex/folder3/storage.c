#include <stdlib.h>
#include "storage.h"

struct Storage {
    int sum;
    int count;
};

void* create_storage(void) {
    struct Storage* st = (struct Storage*)malloc(sizeof(struct Storage));
    if (st) {
        st->sum = 0;
        st->count = 0;
    }
    return st;
}

void save_value(void* storage_handle, int value) {
    struct Storage* st = (struct Storage*)storage_handle;
    st->sum += value;
    st->count += 1;
}

int get_total_sum(void* storage_handle) {
    return ((struct Storage*)storage_handle)->sum;
}

void free_storage(void* storage_handle) {
    free(storage_handle);
}
