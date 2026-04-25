#ifndef ENGINE_H
#define ENGINE_H

typedef struct {
    int initialized;
} engine_t;

void engine_init(engine_t *engine);
void engine_run(engine_t *engine);
void engine_shutdown(engine_t *engine);

#endif
