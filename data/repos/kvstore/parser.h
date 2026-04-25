#ifndef PARSER_H
#define PARSER_H

#define MAX_ARGS 4

typedef enum {
    CMD_PUT,
    CMD_GET,
    CMD_DELETE,
    CMD_LIST,
    CMD_EXIT,
    CMD_UNKNOWN,
} CmdType;

typedef struct {
    CmdType  type;
    char    *args[MAX_ARGS];
    int      argc;
} Command;

Command parse_command(const char *line);
void    command_free(Command *cmd);

#endif
