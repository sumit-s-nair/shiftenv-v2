#ifndef STATE_H
#define STATE_H

void* init_state(int a, int b);
int get_a(void* state);
int get_b(void* state);
void set_result(void* state, int res);
int get_result(void* state);
void destroy_state(void* state);

#endif
