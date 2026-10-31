// Pinned staging ring for host -> device copies. See copy-stage.cuh for why.
//
// One ring per device, built on the first copy that is worth staging: a few slots of anonymous
// pinned memory, each with the event of the copy that last used it. A copy takes the next slot,
// waits for that slot's event (the flow control: at most n_slots of data may be in flight), fills
// it with `threads` workers and queues one linear DMA — or one 2D DMA, whose rows the fill lays
// out packed. The submitting thread takes part in the filling, so threads = 1 costs no extra thread.

#include "copy-stage.cuh"
#include "common.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// MUSA's runtime has no cudaMemoryTypeUnregistered: the source's pinned-ness cannot be told apart,
// so the ring stays off there and the backend copies as it used to.
#if defined(GGML_USE_MUSA)

bool ggml_cuda_stage_h2d(int device, void * dst, const void * src, size_t bytes, void * stream) {
    (void) device; (void) dst; (void) src; (void) bytes; (void) stream;
    return false;
}

bool ggml_cuda_stage_2d_h2d(int device, void * dst, size_t dpitch, const void * src, size_t spitch,
                            size_t width, size_t height, void * stream) {
    (void) device; (void) dst; (void) dpitch; (void) src; (void) spitch;
    (void) width; (void) height; (void) stream;
    return false;
}

#else

static int stage_env_int(const char * name, int dflt) {
    const char * text = getenv(name);
    return text && text[0] ? atoi(text) : dflt;
}

// what to fill: a byte range, or rows of `width` bytes taken `spitch` apart
struct stage_fill {
    const char * src;
    char *       dst;
    size_t       width;
    size_t       spitch;
    bool         two_d;
};

static void stage_fill_run(const stage_fill & f, size_t lo, size_t hi) {
    if (!f.two_d) {
        memcpy(f.dst + lo, f.src + lo, hi - lo);
        return;
    }
    for (size_t r = lo; r < hi; r++) {
        memcpy(f.dst + r * f.width, f.src + r * f.spitch, f.width);
    }
}

// The workers claim chunks of the current job until the range is done. One job runs at a time (the
// ring's mutex is held across the whole fill). `gen`, `pending` and `stop` are guarded by `mu`, and
// the worker that finishes the last chunk notifies while holding it — a notification outside the
// mutex can land between the submitter's predicate test and its wait, which then never wakes up.
struct stage_pool {
    std::mutex mu;
    std::condition_variable cv, ready;
    std::vector<std::thread> threads;
    const stage_fill * job = nullptr;
    size_t total = 0, chunk = 0;
    std::atomic<size_t> next{0};
    int      pending = 0;
    int      started = 0;      // workers that reached run() once (see start())
    uint64_t gen = 0;
    bool     stop = false;

    void fill(const stage_fill & f, size_t total_, size_t chunk_) {
        while (true) {
            const size_t lo = next.fetch_add(chunk_, std::memory_order_relaxed);
            if (lo >= total_) {
                return;
            }
            stage_fill_run(f, lo, std::min(lo + chunk_, total_));
        }
    }

    void run() {
        std::unique_lock<std::mutex> lock(mu);
        if (++started == (int) threads.size()) {
            ready.notify_all();          // the last one tells start() that all are waiting
        }
        uint64_t seen = gen;             // safe only because start() waited for `started`
        while (!stop) {
            cv.wait(lock, [&] { return stop || gen != seen; });
            if (stop) {
                return;
            }
            seen = gen;
            const stage_fill * f = job;
            const size_t t = total, c = chunk;
            lock.unlock();
            fill(*f, t, c);
            lock.lock();
            if (--pending == 0) {
                cv.notify_all();
            }
        }
    }

    // The workers must sit in their wait before the first job is published: one that starts later
    // would read the generation that is already running, skip it and leave `pending` unmet.
    void start(int n) {
        std::unique_lock<std::mutex> lock(mu);
        for (int i = 0; i < n; i++) {
            threads.emplace_back([this] { run(); });
        }
        ready.wait(lock, [&] { return started == (int) threads.size(); });
    }

    // the calling thread takes part in the fill; the call returns once every chunk is copied
    void submit(const stage_fill & f, size_t total_, size_t chunk_, int n_workers) {
        const int helpers = std::min((int) threads.size(), std::max(0, n_workers - 1));
        if (helpers == 0) {
            stage_fill_run(f, 0, total_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            job   = &f;
            total = total_;
            chunk = chunk_;
            next  = 0;
            pending = helpers;      // only the workers count themselves: the submitter waits on them
            gen++;
        }
        cv.notify_all();
        fill(f, total_, chunk_);
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return pending == 0; });
    }
};

// A/B of the ring: bytes, fill time, wait-for-free-slot time and enqueue time, logged every
// GGML_CUDA_STAGE_STATS calls (0 = off).
struct stage_stats {
    std::atomic<long long> calls{0}, bytes{0}, fill_us{0}, wait_us{0}, enq_us{0};
    void log_if(int device, bool on) {
        if (!on) {
            return;
        }
        const long long n = ++calls;
        if (n % 2000 != 0) {
            return;
        }
        const long long by = bytes.load(), fi = fill_us.load(), wa = wait_us.load(), en = enq_us.load();
        fprintf(stderr, "[stage] CUDA%d: %lld calls, %.2f GB, fill %.1f ms, wait %.1f ms, enqueue %.1f ms"
                        " -> fill %.2f GB/s\n", device, (long long) n, by / 1e9,
                fi / 1e3, wa / 1e3, en / 1e3, by / (1e3 * (double) std::max(1LL, fi)));
    }
};

// One ring per device. `failed` latches the device off after a failed allocation, so a machine
// without pinned memory does not try again on every copy.
struct stage_ring {
    std::mutex mu;
    char * base = nullptr;
    size_t slot_bytes = 0;
    size_t chunk_bytes = 0;
    int    n_slots = 0;
    int    cur = 0;
    int    n_threads = 1;
    std::vector<cudaEvent_t> ev;
    std::vector<char>        used;
    bool   ready = false;
    bool   failed = false;
    int    device = -1;
    stage_stats st;
    stage_pool pool;

    ~stage_ring() {
        {
            std::lock_guard<std::mutex> lock(pool.mu);
            pool.stop = true;
            pool.gen++;
        }
        pool.cv.notify_all();
        for (auto & t : pool.threads) {
            t.join();
        }
        for (cudaEvent_t e : ev) {
            if (e) {
                cudaEventDestroy(e);
            }
        }
        if (base) {
            cudaFreeHost(base);
        }
    }

    bool init() {
        const int mb = stage_env_int("GGML_CUDA_STAGE_MB", 64);
        const int slots = stage_env_int("GGML_CUDA_STAGE_SLOTS", 12);
        const int threads = stage_env_int("GGML_CUDA_STAGE_THREADS", 3);
        if (mb <= 0 || slots < 2) {
            failed = true;
            return false;
        }
        slot_bytes  = (size_t) mb << 20;
        chunk_bytes = std::min(slot_bytes, (size_t) 2 << 20);
        n_slots     = slots;
        n_threads   = std::max(1, threads);
        if (cudaMallocHost((void **) &base, slot_bytes * n_slots) != cudaSuccess) {
            (void) cudaGetLastError();
            base = nullptr;
            failed = true;
            return false;
        }
        ev.assign(n_slots, nullptr);
        used.assign(n_slots, 0);
        for (int i = 0; i < n_slots; i++) {
            // ohne BlockingSync: gewartet wird auf dem Weg durch den Prefill hundertfach pro
            // Graph — ein Schlaf mit Aufwachlatenz bremst den einsendenden Thread mehr als die
            // Kopie selbst; das Warten auf ein Ereignis kostet hier einen Kern, und der ist leer
            if (cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming) != cudaSuccess) {
                (void) cudaGetLastError();
                ev[i] = nullptr;
            }
        }
        pool.start(n_threads - 1);   // the submitting thread does the first chunk itself
        ready = true;
        return true;
    }

    // the next slot, waiting for the copy that last used it; `publish` records the new one
    int take(char ** out) {
        const int i = cur;
        cur = (cur + 1) % n_slots;
        if (used[i] && ev[i]) {
            const long long t0 = ggml_time_us();
            cudaEventSynchronize(ev[i]);
            st.wait_us += ggml_time_us() - t0;
        }
        used[i] = 0;
        *out = base + (size_t) i * slot_bytes;
        return i;
    }

    void publish(int i, cudaStream_t s) {
        if (ev[i]) {
            cudaEventRecord(ev[i], s);
            used[i] = 1;
        }
    }
};

static stage_ring * g_stage[GGML_CUDA_MAX_DEVICES] = {};
static std::mutex   g_stage_mu;

static stage_ring * stage_get(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }
    if (g_stage[device]) {
        return g_stage[device]->ready ? g_stage[device] : nullptr;
    }
    std::lock_guard<std::mutex> lock(g_stage_mu);
    if (!g_stage[device]) {
        stage_ring * r = new stage_ring();
        r->device = device;
        r->init();                       // latches `failed`, so this runs once
        g_stage[device] = r;
    }
    return g_stage[device]->ready ? g_stage[device] : nullptr;
}

// Memory the driver already knows: registered host memory copies at the ring's rate anyway, and a
// CPU memcpy of managed or device memory would be wrong. Unknown pointers (an mmap'ed file) answer
// with an error or with cudaMemoryTypeUnregistered.
static bool stage_skip_src(const void * p) {
    static thread_local const void * last_p = nullptr;
    static thread_local bool         last_v = false;
    if (p == last_p) {
        return last_v;
    }
    bool skip = false;
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) == cudaSuccess) {
        skip = a.type != cudaMemoryTypeUnregistered;
    }
    (void) cudaGetLastError();   // querying an unknown pointer sets an error that must not linger
    last_p = p;
    last_v = skip;
    return skip;
}

bool ggml_cuda_stage_h2d(int device, void * dst, const void * src, size_t bytes, void * stream) {
    // Aus auf L28 gemessenen Grund: der Treiber-Stoffwechsel einer unpinnten Kopie ist dort schon
    // so gut, dass der Umweg über den Ring (1.548 t/s mit 8 Fäden gegen 1.558 ohne) nichts holt.
    // An bleibt er, wo `cudaHostRegister` zwar funktioniert, die Kopien aber langsam sind -
    // GGML_CUDA_STAGE=1 schaltet ihn an, die Sonde tools/pinprobe.cu zeigt, was die Maschine hergibt.
    static const bool  enabled = stage_env_int("GGML_CUDA_STAGE", 0) != 0;
    static const bool  stats = stage_env_int("GGML_CUDA_STAGE_STATS", 0) != 0;
    static const size_t min_bytes = (size_t) stage_env_int("GGML_CUDA_STAGE_MIN_KB", 1024) << 10;
    if (!enabled || bytes < min_bytes || stage_skip_src(src)) {
        return false;
    }
    stage_ring * r = stage_get(device);
    if (!r) {
        return false;
    }
    std::lock_guard<std::mutex> lock(r->mu);
    const cudaStream_t s = (cudaStream_t) stream;
    for (size_t off = 0; off < bytes; ) {
        const size_t n = std::min(bytes - off, r->slot_bytes);
        char * slot = nullptr;
        const int i = r->take(&slot);
        const long long t0 = ggml_time_us();
        const stage_fill f{ (const char *) src + off, slot, 0, 0, false };
        r->pool.submit(f, n, r->chunk_bytes, r->n_threads);
        const long long t1 = ggml_time_us();
        if (cudaMemcpyAsync((char *) dst + off, slot, n, cudaMemcpyHostToDevice, s) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        r->publish(i, s);
        const long long t2 = ggml_time_us();
        r->st.bytes  += n;
        r->st.fill_us += t1 - t0;
        r->st.enq_us += t2 - t1;
        off += n;
    }
    r->st.log_if(r->device, stats);
    return true;
}

bool ggml_cuda_stage_2d_h2d(int device, void * dst, size_t dpitch, const void * src, size_t spitch,
                            size_t width, size_t height, void * stream) {
    static const bool  enabled = stage_env_int("GGML_CUDA_STAGE", 0) != 0;
    static const bool  stats = stage_env_int("GGML_CUDA_STAGE_STATS", 0) != 0;
    static const size_t min_bytes = (size_t) stage_env_int("GGML_CUDA_STAGE_MIN_KB", 1024) << 10;
    if (!enabled || width == 0 || width * height < min_bytes || stage_skip_src(src)) {
        return false;
    }
    stage_ring * r = stage_get(device);
    if (!r) {
        return false;
    }
    std::lock_guard<std::mutex> lock(r->mu);
    const cudaStream_t s = (cudaStream_t) stream;
    // the rows go into the slot packed; the DMA spreads them to their pitch again
    const size_t rows_per_slot = std::max<size_t>(1, r->slot_bytes / width);
    const size_t rows_per_chunk = std::max<size_t>(1, r->chunk_bytes / width);
    for (size_t r0 = 0; r0 < height; r0 += rows_per_slot) {
        const size_t rows = std::min(height - r0, rows_per_slot);
        char * slot = nullptr;
        const int i = r->take(&slot);
        const long long t0 = ggml_time_us();
        const stage_fill f{ (const char *) src + r0 * spitch, slot, width, spitch, true };
        r->pool.submit(f, rows, rows_per_chunk, r->n_threads);
        const long long t1 = ggml_time_us();
        if (cudaMemcpy2DAsync((char *) dst + r0 * dpitch, dpitch, slot, width, width, rows,
                              cudaMemcpyHostToDevice, s) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        r->publish(i, s);
        const long long t2 = ggml_time_us();
        r->st.bytes   += rows * width;
        r->st.fill_us += t1 - t0;
        r->st.enq_us  += t2 - t1;
    }
    r->st.log_if(r->device, stats);
    return true;
}

#endif  // GGML_USE_MUSA
