#include "storage.h"
#include "cache.h"
#include "../common/log.h"

void storage_init() {
    cache_init();
    log_info("Storage init");
}

void storage_save(const char *data) {
    cache_put(data);
    log_info("Storage save");
}

void storage_shutdown() {
    cache_clear();
    log_info("Storage shutdown");
}
