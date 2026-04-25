#include <stdio.h>
#include "cache.h"

void cache_init() {
    printf("Cache initialized\n");
}

void cache_put(const char *data) {
    printf("Cached: %s\n", data);
}

void cache_clear() {
    printf("Cache cleared\n");
}
