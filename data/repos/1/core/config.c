#include "config.h"

config_t load_config() {
    config_t cfg;
    cfg.max_connections = 10;
    return cfg;
}
