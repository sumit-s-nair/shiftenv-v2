#ifndef COUNTER_H
#define COUNTER_H

/* ONLY function declarations. Struct is hidden. Returns an opaque handle. */
void* create_counter(int initial_value);
int get_value(void* counter_handle);
void set_value(void* counter_handle, int value);
void free_counter(void* counter_handle);

#endif
