#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "store.h"

static unsigned int hash(const char *key) {
    unsigned int h = 5381;
    while (*key)
        h = ((h << 5) + h) + (unsigned char)*key++;
    return h % BUCKET_COUNT;
}

KVStore *store_new(void) {
    return calloc(1, sizeof(KVStore));
}

void store_free(KVStore *store) {
    for (int i = 0; i < BUCKET_COUNT; i++) {
        Entry *e = store->buckets[i];
        while (e) {
            Entry *next = e->next;
            free(e->key);
            free(e->value);
            free(e);
            e = next;
        }
    }
    free(store);
}

int store_put(KVStore *store, const char *key, const char *value) {
    unsigned int idx = hash(key);
    for (Entry *e = store->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            char *nv = strdup(value);
            if (!nv) return -1;
            free(e->value);
            e->value = nv;
            return 0;
        }
    }
    Entry *ne = malloc(sizeof(Entry));
    if (!ne) return -1;
    ne->key   = strdup(key);
    ne->value = strdup(value);
    ne->next  = store->buckets[idx];
    store->buckets[idx] = ne;
    store->count++;
    return 0;
}

const char *store_get(const KVStore *store, const char *key) {
    unsigned int idx = hash(key);
    for (Entry *e = store->buckets[idx]; e; e = e->next)
        if (strcmp(e->key, key) == 0) return e->value;
    return NULL;
}

int store_delete(KVStore *store, const char *key) {
    unsigned int idx = hash(key);
    Entry **ep = &store->buckets[idx];
    while (*ep) {
        if (strcmp((*ep)->key, key) == 0) {
            Entry *dead = *ep;
            *ep = dead->next;
            free(dead->key);
            free(dead->value);
            free(dead);
            store->count--;
            return 0;
        }
        ep = &(*ep)->next;
    }
    return -1;
}

void store_list(const KVStore *store) {
    for (int i = 0; i < BUCKET_COUNT; i++)
        for (Entry *e = store->buckets[i]; e; e = e->next)
            printf("%s = %s\n", e->key, e->value);
}
