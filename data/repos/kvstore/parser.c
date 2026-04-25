#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "parser.h"

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

Command parse_command(const char *line) {
    Command cmd = {CMD_UNKNOWN, {NULL}, 0};
    char *buf = strdup(line);
    if (!buf) return cmd;

    char *verb = strtok(trim(buf), " \t");
    if (!verb) { free(buf); return cmd; }

    if      (strcmp(verb, "put")    == 0) cmd.type = CMD_PUT;
    else if (strcmp(verb, "get")    == 0) cmd.type = CMD_GET;
    else if (strcmp(verb, "delete") == 0) cmd.type = CMD_DELETE;
    else if (strcmp(verb, "list")   == 0) cmd.type = CMD_LIST;
    else if (strcmp(verb, "exit")   == 0) cmd.type = CMD_EXIT;

    char *tok;
    while ((tok = strtok(NULL, " \t")) != NULL && cmd.argc < MAX_ARGS)
        cmd.args[cmd.argc++] = strdup(tok);

    free(buf);
    return cmd;
}

void command_free(Command *cmd) {
    for (int i = 0; i < cmd->argc; i++) {
        free(cmd->args[i]);
        cmd->args[i] = NULL;
    }
    cmd->argc = 0;
}
