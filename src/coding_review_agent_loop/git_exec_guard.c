/* Installed before Git's main through LD_PRELOAD.  Deny child execution and
 * network access even when repository config changes after validation. */
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/syscall.h>
#include <unistd.h>

#define DENY(number) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, number, 0, 1), \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)

#if defined(__x86_64__)
#define EXPECTED_ARCH AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define EXPECTED_ARCH AUDIT_ARCH_AARCH64
#else
#error Unsupported Linux architecture for Git execution denial
#endif

__attribute__((constructor)) static void agent_loop_confine_git(void) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, EXPECTED_ARCH, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
#if defined(__x86_64__)
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, ~0x40000000U),
#endif
        DENY(__NR_execve),
#ifdef __NR_execveat
        DENY(__NR_execveat),
#endif
        DENY(__NR_socket),
#ifdef __NR_socketpair
        DENY(__NR_socketpair),
#endif
        DENY(__NR_connect),
#ifdef __NR_bind
        DENY(__NR_bind),
#endif
#ifdef __NR_listen
        DENY(__NR_listen),
#endif
#ifdef __NR_accept
        DENY(__NR_accept),
#endif
#ifdef __NR_accept4
        DENY(__NR_accept4),
#endif
#ifdef __NR_sendto
        DENY(__NR_sendto),
#endif
#ifdef __NR_sendmsg
        DENY(__NR_sendmsg),
#endif
#ifdef __NR_sendmmsg
        DENY(__NR_sendmmsg),
#endif
#ifdef __NR_recvfrom
        DENY(__NR_recvfrom),
#endif
#ifdef __NR_recvmsg
        DENY(__NR_recvmsg),
#endif
#ifdef __NR_recvmmsg
        DENY(__NR_recvmmsg),
#endif
#ifdef __NR_shutdown
        DENY(__NR_shutdown),
#endif
#ifdef __NR_io_uring_setup
        DENY(__NR_io_uring_setup),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {sizeof(filter) / sizeof(filter[0]), filter};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
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
