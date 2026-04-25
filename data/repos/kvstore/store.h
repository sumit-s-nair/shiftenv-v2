#ifndef STORE_H
#define STORE_H

#include <stddef.h>

#define BUCKET_COUNT 64

typedef struct Entry {
    char        *key;
    char        *value;
    struct Entry *next;
} Entry;

typedef struct {
    Entry  *buckets[BUCKET_COUNT];
    size_t  count;
} KVStore;

KVStore    *store_new(void);
void        store_free(KVStore *store);
int         store_put(KVStore *store, const char *key, const char *value);
const char *store_get(const KVStore *store, const char *key);
int         store_delete(KVStore *store, const char *key);
void        store_list(const KVStore *store);

#endif
