#include "engine.h"
#include "../module1/processor.h"
#include "../module2/storage.h"
#include "../module3/network.h"
#include "../common/log.h"

void engine_init(engine_t *engine) {
    log_info("Engine init");
    processor_init();
    storage_init();
    network_init();
    engine->initialized = 1;
}

void engine_run(engine_t *engine) {
    if (!engine->initialized) return;

    log_info("Engine run");

    processor_process();
    storage_save("data");
    network_send("hello");
}

void engine_shutdown(engine_t *engine) {
    log_info("Engine shutdown");
    network_shutdown();
    storage_shutdown();
    processor_shutdown();
}
