#include "core/engine.h"
#include "common/log.h"

int main() {
    log_info("Starting application");

    engine_t engine;
    engine_init(&engine);
    engine_run(&engine);
    engine_shutdown(&engine);

    log_info("Shutting down");
    return 0;
}
