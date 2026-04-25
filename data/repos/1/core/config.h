#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int max_connections;
} config_t;

config_t load_config();

#endif
