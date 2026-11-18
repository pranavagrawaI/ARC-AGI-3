/* flare_ra.so: read-ahead cache for one large file on a slow FUSE mount (Kaggle datasets = gcsfuse).
 *
 * Flare opens its package with O_DIRECT and reads it in small synchronous pieces. On gcsfuse that is
 * one storage round trip per piece: smoke 3 loaded 4.0 GB in 40 min (1.7 MB/s). Preloaded into Flare,
 * this library recognizes the package (by st_dev/st_ino), drops O_DIRECT, and serves every read on it
 * from a block cache. Blocks are fetched by worker threads, each fetch being one large buffered pread
 * on a freshly opened handle, so gcsfuse sees plain sequential streams. Reads ahead of demand are
 * prefetched. Other files and mmap pass through untouched.
 *
 * Multi-source mode: FLARE_RA_SOURCES="path@offset;path@offset;..." serves the target's bytes from
 * several files that tile [0, size) in order (e.g. six staged notebook-output parts on local disk),
 * so the target can be a sparse placeholder of the right size. Without it the target is its own source.
 *
 * Environment: FLARE_RA_PATH (required; inert without it), FLARE_RA_SOURCES (optional), FLARE_RA_BLOCK_MB (64), FLARE_RA_AHEAD (16),
 * FLARE_RA_THREADS (16), FLARE_RA_CAP (blocks resident, 48), FLARE_RA_LOG (stats file, appended every 10 s).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

typedef int (*open_t)(const char *, int, ...);
typedef int (*openat_t)(int, const char *, int, ...);
typedef ssize_t (*pread_t)(int, void *, size_t, off_t);
typedef ssize_t (*read_t)(int, void *, size_t);
typedef ssize_t (*preadv_t)(int, const struct iovec *, int, off_t);
typedef ssize_t (*readv_t)(int, const struct iovec *, int);
typedef int (*close_t)(int);
typedef off_t (*lseek_t)(int, off_t, int);

static open_t r_open, r_open64;
static openat_t r_openat, r_openat64;
static pread_t r_pread, r_pread64;
static read_t r_read;
static preadv_t r_preadv, r_preadv64;
static readv_t r_readv;
static close_t r_close;
static lseek_t r_lseek;

enum { EMPTY = 0, QUEUED, LOADING, READY, FAILED };
struct block { int state; int refs; uint64_t last_use; char *buf; };

static int active;
static char target[PATH_MAX];
#define MAXSRC 16
static char src_path[MAXSRC][PATH_MAX];
static off_t src_off[MAXSRC], src_end[MAXSRC];
static int nsrc;
static dev_t t_dev;
static ino_t t_ino;
static off_t t_size;
static size_t bsize;
static long nblocks, ahead, cap, nthreads;
static struct block *blocks;
static long *queue, q_head, q_len;  /* deque of block indices */
static long resident;
static uint64_t tick;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready_cv = PTHREAD_COND_INITIALIZER, work_cv = PTHREAD_COND_INITIALIZER;
#define MAXFD 4096
static unsigned char tracked[MAXFD];
static char logpath[PATH_MAX];
/* stats */
static uint64_t st_calls, st_bytes, st_fetch_bytes, st_fetches, st_waits, st_wait_ns, st_fetch_ns, st_evict;
static uint64_t st_mmap_small, st_mmap_large, st_copy_calls;
static uint64_t st_opens, st_direct_stripped, st_fail, st_hist[8]; /* read sizes: <4K,<64K,<1M,<4M,<16M,<64M,<256M,>= */
static double t0;

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static void resolve(void) {
    if (r_pread64) return;
    r_open = (open_t)dlsym(RTLD_NEXT, "open");
    r_open64 = (open_t)dlsym(RTLD_NEXT, "open64");
    r_openat = (openat_t)dlsym(RTLD_NEXT, "openat");
    r_openat64 = (openat_t)dlsym(RTLD_NEXT, "openat64");
    r_pread = (pread_t)dlsym(RTLD_NEXT, "pread");
    r_read = (read_t)dlsym(RTLD_NEXT, "read");
    r_preadv = (preadv_t)dlsym(RTLD_NEXT, "preadv");
    r_preadv64 = (preadv_t)dlsym(RTLD_NEXT, "preadv64");
    r_readv = (readv_t)dlsym(RTLD_NEXT, "readv");
    r_close = (close_t)dlsym(RTLD_NEXT, "close");
    r_lseek = (lseek_t)dlsym(RTLD_NEXT, "lseek");
    r_pread64 = (pread_t)dlsym(RTLD_NEXT, "pread64");
}

static void push(long b, int front) {  /* caller holds mu; b is EMPTY */
    blocks[b].state = QUEUED;
    if (front) { q_head = (q_head - 1 + nblocks) % nblocks; queue[q_head] = b; }
    else queue[(q_head + q_len) % nblocks] = b;
    q_len++;
    pthread_cond_signal(&work_cv);
}

static void evict_locked(void) {
    while (resident > cap) {
        long victim = -1; uint64_t best = UINT64_MAX;
        for (long i = 0; i < nblocks; i++)
            if (blocks[i].state == READY && blocks[i].refs == 0 && blocks[i].last_use < best) { best = blocks[i].last_use; victim = i; }
        if (victim < 0) return;
        free(blocks[victim].buf); blocks[victim].buf = NULL; blocks[victim].state = EMPTY; resident--; st_evict++;
    }
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (q_len == 0) pthread_cond_wait(&work_cv, &mu);
        long b = queue[q_head]; q_head = (q_head + 1) % nblocks; q_len--;
        if (blocks[b].state != QUEUED) { pthread_mutex_unlock(&mu); continue; }
        blocks[b].state = LOADING;
        pthread_mutex_unlock(&mu);

        off_t off = (off_t)b * bsize;
        size_t want = (size_t)((off + (off_t)bsize > t_size) ? t_size - off : (off_t)bsize);
        char *buf = NULL;
        int ok = 0;
        double t = now();
        for (int attempt = 0; attempt < 6 && !ok; attempt++) {
            if (!buf && posix_memalign((void **)&buf, 4096, bsize)) { buf = NULL; break; }
            size_t got = 0;
            for (int si = 0; si < nsrc && got < want; si++) {
                off_t pos = off + (off_t)got;
                if (pos < src_off[si] || pos >= src_end[si]) continue;
                int fd = r_open(src_path[si], O_RDONLY | O_CLOEXEC);
                if (fd < 0) break;
                size_t lim = (size_t)(src_end[si] - pos < (off_t)(want - got) ? src_end[si] - pos : (off_t)(want - got));
                size_t part = 0;
                while (part < lim) {
                    ssize_t n = r_pread64(fd, buf + got + part, lim - part, pos - src_off[si] + (off_t)part);
                    if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
                    part += (size_t)n;
                }
                r_close(fd);
                got += part;
                if (part < lim) break;
            }
            ok = got == want;
            if (!ok) usleep(200000 << attempt);
        }
        pthread_mutex_lock(&mu);
        st_fetch_ns += (uint64_t)((now() - t) * 1e9);
        if (ok) { blocks[b].buf = buf; blocks[b].state = READY; blocks[b].last_use = ++tick; resident++; st_fetches++; st_fetch_bytes += want; evict_locked(); }
        else { free(buf); blocks[b].state = FAILED; st_fail++; }
        pthread_cond_broadcast(&ready_cv);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

static void write_stats(void) {
    FILE *f = fopen(logpath, "a");
    if (!f) return;
    pthread_mutex_lock(&mu);
    double el = now() - t0;
    fprintf(f, "{\"t\": %.1f, \"calls\": %lu, \"bytes\": %lu, \"fetches\": %lu, \"fetch_bytes\": %lu, "
               "\"fetch_MBps_avg\": %.1f, \"fetch_s_per_block\": %.3f, \"waits\": %lu, \"wait_s\": %.1f, "
               "\"resident\": %ld, \"queued\": %ld, \"evicted\": %lu, \"failed\": %lu, \"opens\": %lu, "
               "\"direct_stripped\": %lu, \"mmap_small\": %lu, \"mmap_large\": %lu, \"copy_calls\": %lu, \"hist\": [%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu]}\n",
            el, st_calls, st_bytes, st_fetches, st_fetch_bytes, el > 0 ? st_fetch_bytes / 1e6 / el : 0.0,
            st_fetches ? st_fetch_ns / 1e9 / st_fetches : 0.0, st_waits, st_wait_ns / 1e9, resident, q_len,
            st_evict, st_fail, st_opens, st_direct_stripped, st_mmap_small, st_mmap_large, st_copy_calls, st_hist[0], st_hist[1], st_hist[2], st_hist[3],
            st_hist[4], st_hist[5], st_hist[6], st_hist[7]);
    pthread_mutex_unlock(&mu);
    fclose(f);
}

static void *logger(void *arg) {
    (void)arg;
    for (;;) { sleep(10); write_stats(); }
    return NULL;
}

__attribute__((destructor)) static void fini(void) { if (active) write_stats(); }

__attribute__((constructor)) static void init(void) {
    resolve();
    const char *p = getenv("FLARE_RA_PATH");
    if (!p || !*p) return;
    struct stat sb;
    if (!realpath(p, target) || stat(target, &sb) != 0) return;
    t_dev = sb.st_dev; t_ino = sb.st_ino; t_size = sb.st_size;
    const char *srcs = getenv("FLARE_RA_SOURCES");
    if (srcs && *srcs) {
        char tmp[MAXSRC * (PATH_MAX + 24)];
        snprintf(tmp, sizeof tmp, "%s", srcs);
        off_t expect = 0;
        for (char *save = NULL, *tok = strtok_r(tmp, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
            char *at = strrchr(tok, '@');
            struct stat ss;
            if (!at || nsrc >= MAXSRC) return;
            *at = 0;
            if (stat(tok, &ss) != 0) return;
            off_t o = (off_t)strtoll(at + 1, NULL, 10);
            if (o != expect) return;  /* sources must tile the target in order */
            snprintf(src_path[nsrc], PATH_MAX, "%s", tok);
            src_off[nsrc] = o; src_end[nsrc] = o + ss.st_size; expect = src_end[nsrc]; nsrc++;
        }
        if (expect != t_size) return;
    } else {
        snprintf(src_path[0], PATH_MAX, "%s", target); src_off[0] = 0; src_end[0] = t_size; nsrc = 1;
    }
    const char *e;
    bsize = (size_t)((e = getenv("FLARE_RA_BLOCK_MB")) ? atol(e) : 64) << 20;
    ahead = (e = getenv("FLARE_RA_AHEAD")) ? atol(e) : 16;
    nthreads = (e = getenv("FLARE_RA_THREADS")) ? atol(e) : 16;
    cap = (e = getenv("FLARE_RA_CAP")) ? atol(e) : 48;
    if (cap < ahead + 4) cap = ahead + 4;
    snprintf(logpath, sizeof logpath, "%s", (e = getenv("FLARE_RA_LOG")) ? e : "/tmp/flare_ra.log");
    nblocks = (long)((t_size + (off_t)bsize - 1) / (off_t)bsize);
    blocks = calloc((size_t)nblocks, sizeof *blocks);
    queue = calloc((size_t)nblocks, sizeof *queue);
    if (!blocks || !queue) return;
    t0 = now();
    pthread_t th;
    for (long i = 0; i < nthreads; i++) { pthread_create(&th, NULL, worker, NULL); pthread_detach(th); }
    pthread_create(&th, NULL, logger, NULL); pthread_detach(th);
    active = 1;
}

static int is_target_path(const char *path) {
    char rp[PATH_MAX];
    return active && path && realpath(path, rp) && strcmp(rp, target) == 0;
}

static int after_open(int fd) {
    struct stat sb;
    if (active && fd >= 0 && fd < MAXFD && fstat(fd, &sb) == 0 && sb.st_dev == t_dev && sb.st_ino == t_ino) {
        __atomic_store_n(&tracked[fd], 1, __ATOMIC_RELEASE);
        pthread_mutex_lock(&mu); st_opens++; pthread_mutex_unlock(&mu);
        write_stats();  /* early evidence that the package is tracked */
    }
    return fd;
}

static int strip(const char *path, int flags) {
    if ((flags & O_DIRECT) && is_target_path(path)) {
        pthread_mutex_lock(&mu); st_direct_stripped++; pthread_mutex_unlock(&mu);
        return flags & ~O_DIRECT;
    }
    return flags;
}

#define MODE_ARG mode_t mode = 0; if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
int open(const char *path, int flags, ...) { resolve(); MODE_ARG; return after_open(r_open(path, strip(path, flags), mode)); }
int open64(const char *path, int flags, ...) { resolve(); MODE_ARG; return after_open(r_open64(path, strip(path, flags), mode)); }
int openat(int dirfd, const char *path, int flags, ...) {
    resolve(); MODE_ARG;
    if (path && path[0] != '/') return after_open(r_openat(dirfd, path, flags, mode));
    return after_open(r_openat(dirfd, path, strip(path, flags), mode));
}
int openat64(int dirfd, const char *path, int flags, ...) {
    resolve(); MODE_ARG;
    if (path && path[0] != '/') return after_open(r_openat64(dirfd, path, flags, mode));
    return after_open(r_openat64(dirfd, path, strip(path, flags), mode));
}
int close(int fd) {
    resolve();
    if (fd >= 0 && fd < MAXFD) __atomic_store_n(&tracked[fd], 0, __ATOMIC_RELEASE);
    return r_close(fd);
}

static int is_tracked(int fd) { return active && fd >= 0 && fd < MAXFD && __atomic_load_n(&tracked[fd], __ATOMIC_ACQUIRE); }

static void want_locked(long b, int front) {
    if (b < nblocks && (blocks[b].state == EMPTY || blocks[b].state == FAILED)) push(b, front);
}

static ssize_t ra_pread(void *buf, size_t count, off_t off) {
    if (off < 0) { errno = EINVAL; return -1; }
    if (off >= t_size || count == 0) return 0;
    if ((off_t)count > t_size - off) count = (size_t)(t_size - off);
    int bucket = count < 4096 ? 0 : count < 65536 ? 1 : count < (1 << 20) ? 2 : count < (4 << 20) ? 3 :
                 count < (16 << 20) ? 4 : count < (64 << 20) ? 5 : count < (256 << 20) ? 6 : 7;
    size_t done = 0;
    pthread_mutex_lock(&mu);
    st_calls++; st_hist[bucket]++;
    while (done < count) {
        off_t pos = off + (off_t)done;
        long b = (long)(pos / (off_t)bsize);
        want_locked(b, 1);
        for (long k = 1; k <= ahead; k++) want_locked(b + k, 0);
        if (blocks[b].state != READY) {
            double t = now(); st_waits++;
            while (blocks[b].state != READY && blocks[b].state != FAILED) pthread_cond_wait(&ready_cv, &mu);
            st_wait_ns += (uint64_t)((now() - t) * 1e9);
            if (blocks[b].state == FAILED) { pthread_mutex_unlock(&mu); errno = EIO; return done ? (ssize_t)done : -1; }
        }
        blocks[b].refs++; blocks[b].last_use = ++tick;
        size_t in = (size_t)(pos - (off_t)b * (off_t)bsize);
        size_t n = bsize - in;
        if (n > count - done) n = count - done;
        pthread_mutex_unlock(&mu);
        memcpy((char *)buf + done, blocks[b].buf + in, n);
        pthread_mutex_lock(&mu);
        blocks[b].refs--;
        done += n;
    }
    st_bytes += done;
    pthread_mutex_unlock(&mu);
    return (ssize_t)done;
}

ssize_t pread64(int fd, void *buf, size_t count, off_t off) { resolve(); return is_tracked(fd) ? ra_pread(buf, count, off) : r_pread64(fd, buf, count, off); }
ssize_t pread(int fd, void *buf, size_t count, off_t off) { resolve(); return is_tracked(fd) ? ra_pread(buf, count, off) : r_pread(fd, buf, count, off); }

static ssize_t ra_preadv(const struct iovec *iov, int n, off_t off) {
    ssize_t total = 0;
    for (int i = 0; i < n; i++) {
        ssize_t r = ra_pread(iov[i].iov_base, iov[i].iov_len, off + total);
        if (r < 0) return total ? total : -1;
        total += r;
        if ((size_t)r < iov[i].iov_len) break;
    }
    return total;
}
ssize_t preadv(int fd, const struct iovec *iov, int n, off_t off) { resolve(); return is_tracked(fd) ? ra_preadv(iov, n, off) : r_preadv(fd, iov, n, off); }
ssize_t preadv64(int fd, const struct iovec *iov, int n, off_t off) { resolve(); return is_tracked(fd) ? ra_preadv(iov, n, off) : r_preadv64(fd, iov, n, off); }

ssize_t read(int fd, void *buf, size_t count) {
    resolve();
    if (!is_tracked(fd)) return r_read(fd, buf, count);
    off_t pos = r_lseek(fd, 0, SEEK_CUR);
    ssize_t n = ra_pread(buf, count, pos);
    if (n > 0) r_lseek(fd, pos + n, SEEK_SET);
    return n;
}
ssize_t readv(int fd, const struct iovec *iov, int n) {
    resolve();
    if (!is_tracked(fd)) return r_readv(fd, iov, n);
    off_t pos = r_lseek(fd, 0, SEEK_CUR);
    ssize_t got = ra_preadv(iov, n, pos);
    if (got > 0) r_lseek(fd, pos + got, SEEK_SET);
    return got;
}

/* ---- the rest of the read surface Flare imports, so multi-source mode never leaks placeholder zeros ---- */
typedef ssize_t (*preadv2_t)(int, const struct iovec *, int, off_t, int);
typedef ssize_t (*cfr_t)(int, off_t *, int, off_t *, size_t, unsigned int);
typedef ssize_t (*sendfile_t)(int, int, off_t *, size_t);
typedef ssize_t (*splice_t)(int, off_t *, int, off_t *, size_t, unsigned int);
typedef void *(*mmap_t)(void *, size_t, int, int, int, off_t);
typedef ssize_t (*write_t)(int, const void *, size_t);
typedef ssize_t (*pwrite_t)(int, const void *, size_t, off_t);
#define SYM(T, name) static T r_##name; if (!r_##name) r_##name = (T)dlsym(RTLD_NEXT, #name)

ssize_t preadv2(int fd, const struct iovec *iov, int n, off_t off, int flags) {
    resolve(); SYM(preadv2_t, preadv2);
    if (!is_tracked(fd)) return r_preadv2(fd, iov, n, off, flags);
    if (off == -1) {  /* current position */
        off_t pos = r_lseek(fd, 0, SEEK_CUR);
        ssize_t got = ra_preadv(iov, n, pos);
        if (got > 0) r_lseek(fd, pos + got, SEEK_SET);
        return got;
    }
    return ra_preadv(iov, n, off);
}
ssize_t preadv64v2(int fd, const struct iovec *iov, int n, off_t off, int flags) { return preadv2(fd, iov, n, off, flags); }

/* read len bytes of the target at *in_off (or the fd position) and write them to out (at *out_off or its position) */
static ssize_t emulate_copy(int in_fd, off_t *in_off, int out_fd, off_t *out_off, size_t len) {
    SYM(write_t, write); SYM(pwrite_t, pwrite64);
    pthread_mutex_lock(&mu); st_copy_calls++; pthread_mutex_unlock(&mu);
    off_t pos = in_off ? *in_off : r_lseek(in_fd, 0, SEEK_CUR);
    size_t chunk = len < (8 << 20) ? len : (8 << 20);
    char *tmp = malloc(chunk ? chunk : 1);
    if (!tmp) { errno = ENOMEM; return -1; }
    size_t done = 0;
    while (done < len) {
        size_t want = len - done < chunk ? len - done : chunk;
        ssize_t got = ra_pread(tmp, want, pos + (off_t)done);
        if (got <= 0) break;
        size_t w = 0;
        while (w < (size_t)got) {
            ssize_t k = out_off ? r_pwrite64(out_fd, tmp + w, (size_t)got - w, *out_off + (off_t)(done + w))
                                : r_write(out_fd, tmp + w, (size_t)got - w);
            if (k <= 0) { if (k < 0 && errno == EINTR) continue; free(tmp); return done + w ? (ssize_t)(done + w) : -1; }
            w += (size_t)k;
        }
        done += (size_t)got;
        if ((size_t)got < want) break;
    }
    free(tmp);
    if (in_off) *in_off = pos + (off_t)done; else r_lseek(in_fd, pos + (off_t)done, SEEK_SET);
    if (out_off) *out_off += (off_t)done;
    return (ssize_t)done;
}
ssize_t copy_file_range(int in_fd, off_t *in_off, int out_fd, off_t *out_off, size_t len, unsigned int flags) {
    resolve(); SYM(cfr_t, copy_file_range);
    return is_tracked(in_fd) ? emulate_copy(in_fd, in_off, out_fd, out_off, len) : r_copy_file_range(in_fd, in_off, out_fd, out_off, len, flags);
}
ssize_t sendfile(int out_fd, int in_fd, off_t *off, size_t count) {
    resolve(); SYM(sendfile_t, sendfile);
    return is_tracked(in_fd) ? emulate_copy(in_fd, off, out_fd, NULL, count) : r_sendfile(out_fd, in_fd, off, count);
}
ssize_t sendfile64(int out_fd, int in_fd, off_t *off, size_t count) {
    resolve(); SYM(sendfile_t, sendfile64);
    return is_tracked(in_fd) ? emulate_copy(in_fd, off, out_fd, NULL, count) : r_sendfile64(out_fd, in_fd, off, count);
}
ssize_t splice(int in_fd, off_t *in_off, int out_fd, off_t *out_off, size_t len, unsigned int flags) {
    resolve(); SYM(splice_t, splice);
    return is_tracked(in_fd) ? emulate_copy(in_fd, in_off, out_fd, out_off, len) : r_splice(in_fd, in_off, out_fd, out_off, len, flags);
}

/* read-only maps of the target: up to 256 MiB are filled from the cache; larger ones pass through and
   are counted (in multi-source mode they would show placeholder zeros, so a nonzero count flags the run) */
static void *ra_mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off, mmap_t real) {
    if (!is_tracked(fd) || (prot & PROT_WRITE && flags & MAP_SHARED)) return real(addr, len, prot, flags, fd, off);
    if (len > (256u << 20)) { pthread_mutex_lock(&mu); st_mmap_large++; pthread_mutex_unlock(&mu); return real(addr, len, prot, flags, fd, off); }
    void *m = real(addr, len, PROT_READ | PROT_WRITE, (flags & ~(MAP_SHARED | MAP_SHARED_VALIDATE)) | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return m;
    ssize_t got = ra_pread(m, len, off);
    (void)got;
    mprotect(m, len, prot);
    pthread_mutex_lock(&mu); st_mmap_small++; pthread_mutex_unlock(&mu);
    return m;
}
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    SYM(mmap_t, mmap); return ra_mmap(addr, len, prot, flags, fd, off, r_mmap);
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    SYM(mmap_t, mmap64); return ra_mmap(addr, len, prot, flags, fd, off, r_mmap64);
}

/* ---- CUDA graph attribute shim (stage 3c) ----------------------------------------------------------
 * Smoke 3b failed in "Prepare execution": cuGraphKernelNodeSetAttribute -> CUDA_ERROR_INVALID_VALUE on
 * Kaggle's driver 580 (CUDA 13.0). Flare resolves CUDA entrypoints with dlopen("libcuda.so.1") + dlsym,
 * so this dlsym override hands it a wrapper for that one function. The wrapper logs every attribute id
 * it sees (calls, failures, the first failing value's bytes) to FLARE_CU_LOG, and when FLARE_CU_SKIP is
 * "all" or lists the id, reports success for an attribute the driver rejected. Inert without FLARE_CU_LOG.
 */
typedef int (*kn_setattr_t)(void *, int, const void *);
typedef void *(*dlsym_t)(void *, const char *);
static kn_setattr_t real_kn_setattr;
static dlsym_t real_dlsym_fn;
static char cu_log[PATH_MAX], cu_skip[256];
static int cu_on = -1;
static uint64_t cu_calls[64], cu_fail[64], cu_skipped[64];
static int cu_first_rc[64];
static unsigned char cu_first_val[64][32];
static pthread_mutex_t cu_mu = PTHREAD_MUTEX_INITIALIZER;

static void cu_write(void) {
    FILE *f = fopen(cu_log, "a");
    if (!f) return;
    pthread_mutex_lock(&cu_mu);
    fprintf(f, "{\"t\": %.1f, \"attrs\": {", now() - t0);
    int first = 1;
    for (int a = 0; a < 64; a++) {
        if (!cu_calls[a]) continue;
        fprintf(f, "%s\"%d\": {\"calls\": %lu, \"fail\": %lu, \"skipped\": %lu, \"first_rc\": %d, \"first_value_hex\": \"",
                first ? "" : ", ", a, cu_calls[a], cu_fail[a], cu_skipped[a], cu_first_rc[a]);
        for (int i = 0; i < 32; i++) fprintf(f, "%02x", cu_first_val[a][i]);
        fprintf(f, "\"}");
        first = 0;
    }
    fprintf(f, "}}\n");
    pthread_mutex_unlock(&cu_mu);
    fclose(f);
}

static int cu_should_skip(int attr) {
    if (!cu_skip[0]) return 0;
    if (strcmp(cu_skip, "all") == 0) return 1;
    char buf[256]; snprintf(buf, sizeof buf, "%s", cu_skip);
    for (char *save = NULL, *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
        if (atoi(tok) == attr) return 1;
    return 0;
}

static int kn_setattr_wrap(void *node, int attr, const void *value) {
    int rc = real_kn_setattr(node, attr, value);
    int a = attr & 63, skip = 0, first_fail = 0;
    pthread_mutex_lock(&cu_mu);
    cu_calls[a]++;
    if (rc) {
        if (!cu_fail[a]++) { first_fail = 1; cu_first_rc[a] = rc; if (value) memcpy(cu_first_val[a], value, 32); }
        if (cu_should_skip(attr)) { skip = 1; cu_skipped[a]++; }
    }
    pthread_mutex_unlock(&cu_mu);
    if (first_fail) cu_write();
    return skip ? 0 : rc;
}

static dlsym_t get_real_dlsym(void) {
    if (!real_dlsym_fn) {
        real_dlsym_fn = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (!real_dlsym_fn) real_dlsym_fn = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    return real_dlsym_fn;
}

void *dlsym(void *handle, const char *name) {
    void *p = get_real_dlsym()(handle, name);
    if (cu_on < 0) {
        const char *e = getenv("FLARE_CU_LOG");
        cu_on = e && *e;
        if (cu_on) {
            snprintf(cu_log, sizeof cu_log, "%s", e);
            snprintf(cu_skip, sizeof cu_skip, "%s", (e = getenv("FLARE_CU_SKIP")) ? e : "");
            if (t0 == 0) t0 = now();
        }
    }
    if (cu_on && p && name && strcmp(name, "cuGraphKernelNodeSetAttribute") == 0) {
        real_kn_setattr = (kn_setattr_t)p;
        return (void *)kn_setattr_wrap;
    }
    return p;
}

__attribute__((destructor)) static void cu_fini(void) { if (cu_on > 0) cu_write(); }
