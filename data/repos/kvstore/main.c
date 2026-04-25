#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "store.h"
#include "parser.h"

#define LINE_MAX 1024

static void dispatch(KVStore *store, Command *cmd) {
    switch (cmd->type) {
    case CMD_PUT:
        if (cmd->argc < 2) { printf("usage: put <key> <value>\n"); return; }
        store_put(store, cmd->args[0], cmd->args[1]);
        printf("OK\n");
        break;
    case CMD_GET: {
        if (cmd->argc < 1) { printf("usage: get <key>\n"); return; }
        const char *val = store_get(store, cmd->args[0]);
        printf("%s\n", val ? val : "(nil)");
        break;
    }
    case CMD_DELETE:
        if (cmd->argc < 1) { printf("usage: delete <key>\n"); return; }
        printf("%s\n", store_delete(store, cmd->args[0]) == 0 ? "OK" : "(nil)");
        break;
    case CMD_LIST:
        store_list(store);
        break;
    case CMD_UNKNOWN:
    default:
        printf("unknown command\n");
    }
}

int main(void) {
    KVStore *store = store_new();
    char line[LINE_MAX];

    while (fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\n")] = '\0';
        if (!*line) continue;

        Command cmd = parse_command(line);
        if (cmd.type == CMD_EXIT) {
            command_free(&cmd);
            break;
        }
        dispatch(store, &cmd);
        command_free(&cmd);
    }

    store_free(store);
    return 0;
}
