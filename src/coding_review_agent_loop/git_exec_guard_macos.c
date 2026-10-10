/* Loaded by dyld before Git main.  The already-running Git gets no exception
 * for its own executable path, so a later same-path relaunch is denied. */
#include <sandbox.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((constructor)) static void agent_loop_confine_git(void) {
    static const char profile[] =
        "(version 1)\n"
        "(allow default)\n"
        "(deny process-exec)\n"
        "(deny process-fork)\n"
        "(deny network*)\n";
    char *error = NULL;
    if (sandbox_init(profile, 0, &error) != 0) {
        if (error != NULL) sandbox_free_error(error);
        _exit(125);
    }
    const char *fd_text = getenv("AGENT_LOOP_GIT_GUARD_PROBE_FD");
    if (fd_text != NULL) {
        char *end = NULL;
        long fd = strtol(fd_text, &end, 10);
        if (end != fd_text && *end == '\0' && fd >= 0 && fd <= 1024) {
            (void)write((int)fd, "1", 1);
        }
    }
}
