// phase3_pipeline/handoff.h — the one-file, two-slot chunk handoff (P3.1).
//
// Both stages (GPU producer, NPU consumer) mmap the SAME file MAP_SHARED and
// exchange one residual-stream chunk at a time. Nothing here touches the INT8
// weight artifact: this file carries activations only, so the one-copy rule
// (Gate 4) is untouched by construction.
//
//   offset 0                     : SLOTS x SlotHdr, padded to one page so every
//                                  payload slot starts page-aligned for msync
//   offset HEADER_REGION + s*STRIDE : payload of slot s, SLOT_STRIDE bytes
//
// Ordering: the producer writes the payload, msyncs it, stamps t_prod_ns, then
// __atomic_store_n(&hdr->seq, chunk, RELEASE). The consumer spins with ACQUIRE
// and reads t_cons_ns as the first thing after the load, so
// (t_cons_ns - t_prod_ns) is the publish->observe latency. The producer may not
// reuse a slot until the consumer has released it (done >= chunk - SLOTS), which
// is what makes a torn payload impossible to observe rather than unlikely.
//
// Chunk indices are 1-based; 0 means "slot empty" in `seq`/`done`.

#pragma once

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hv {

// One Preset-A chunk's residual stream: M = 1530 tokens x d = 1536, fp16.
constexpr std::size_t CHUNK_BYTES = 4'700'160;
constexpr std::size_t PAGE = 4096;
constexpr std::size_t HEADER_REGION = PAGE;   // headers at 0..127, rest padding
constexpr int SLOTS = 2;
constexpr std::size_t SLOT_STRIDE =
    ((CHUNK_BYTES + PAGE - 1) / PAGE) * PAGE;                       // 4,702,208
constexpr std::size_t FILE_BYTES = HEADER_REGION + SLOTS * SLOT_STRIDE;

constexpr std::uint32_t FLAG_STREAM_DONE = 1u << 0;

struct alignas(8) SlotHdr {
    std::uint64_t seq;        // chunk index published by the producer (0 = empty)
    std::uint64_t done;       // chunk index released by the consumer
    std::uint32_t bytes;      // payload length actually written
    std::uint32_t flags;      // bit0 = producer finished the stream
    std::uint64_t t_prod_ns;  // CLOCK_MONOTONIC at publish
    std::uint64_t t_cons_ns;  // CLOCK_MONOTONIC at consume
    std::uint64_t payload_off;
    std::uint64_t reserved[2];
};
static_assert(sizeof(SlotHdr) == 64, "SlotHdr must be 64 B (one cache line)");

inline constexpr std::size_t slot_payload_off(int s) {
    return HEADER_REGION + static_cast<std::size_t>(s) * SLOT_STRIDE;
}

inline std::uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

struct Ring {
    int fd = -1;
    std::uint8_t* base = nullptr;
    std::size_t bytes = 0;
    bool created = false;  // file was absent/empty -> headers need a reset

    SlotHdr* hdr(int s) const {
        return reinterpret_cast<SlotHdr*>(base + static_cast<std::size_t>(s) * sizeof(SlotHdr));
    }
    std::uint8_t* payload(int s) const { return base + slot_payload_off(s); }
    bool ok() const { return base != nullptr && base != MAP_FAILED; }
};

// Open (creating if `create`) and map the ring. Both sides must agree on size;
// the file is truncated up only, never down, so a rerun keeps the ring intact.
// `created` is set when the file was absent or zero-length, i.e. when the caller
// is the one that owns initialising the headers.
inline Ring map_ring(const char* path, bool create) {
    Ring r;
    int flags = create ? (O_RDWR | O_CREAT) : O_RDWR;
    r.fd = ::open(path, flags, 0666);
    if (r.fd < 0) return r;

    struct stat st {};
    if (fstat(r.fd, &st) != 0) { ::close(r.fd); r.fd = -1; return r; }
    r.created = (st.st_size == 0);
    if (static_cast<std::size_t>(st.st_size) < FILE_BYTES) {
        if (!create) { ::close(r.fd); r.fd = -1; return r; }
        if (ftruncate(r.fd, static_cast<off_t>(FILE_BYTES)) != 0) {
            ::close(r.fd); r.fd = -1; return r;
        }
    }
    void* p = mmap(nullptr, FILE_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd, 0);
    if (p == MAP_FAILED) { ::close(r.fd); r.fd = -1; return r; }
    r.base = static_cast<std::uint8_t*>(p);
    r.bytes = FILE_BYTES;
    for (int s = 0; s < SLOTS; ++s) r.hdr(s)->payload_off = slot_payload_off(s);
    return r;
}

inline void unmap_ring(Ring& r) {
    if (r.ok()) munmap(r.base, r.bytes);
    if (r.fd >= 0) ::close(r.fd);
    r.base = nullptr;
    r.fd = -1;
}

// Zero the header region so a rerun starts from "empty" rather than from a
// stale publish. Payloads are not cleared (the selftest stamps every byte it
// reads, so stale bytes are detectable).
inline void reset_headers(Ring& r) {
    std::memset(r.base, 0, HEADER_REGION);
    for (int s = 0; s < SLOTS; ++s) r.hdr(s)->payload_off = slot_payload_off(s);
    msync(r.base, HEADER_REGION, MS_SYNC);
}

// msync policy: full payload, first page only, or none. On Linux a MAP_SHARED
// mapping is coherent across processes through the page cache, so `none` is
// correct for visibility; `first-page` is the cheap ordering/durability barrier
// the pipeline plan specifies (cost is recorded so it is a measurement, not a
// hope), and `full` exists to price the pessimistic choice.
enum class Msync { None, FirstPage, Full };

inline long msync_payload(Ring& r, int s, std::size_t bytes, Msync mode) {
    if (mode == Msync::None) return 0;
    std::size_t len = (mode == Msync::Full) ? bytes : PAGE;
    long t0 = static_cast<long>(now_ns());
    if (msync(r.payload(s), len, MS_SYNC) != 0) return -errno;
    return static_cast<long>(now_ns()) - t0;
}

// Producer side: payload must already be written. Publishes with RELEASE.
inline void publish(Ring& r, int s, std::uint64_t chunk, std::uint32_t bytes,
                    Msync mode, long* msync_ns) {
    long ms = msync_payload(r, s, bytes, mode);
    if (msync_ns) *msync_ns = ms;
    SlotHdr* h = r.hdr(s);
    __atomic_store_n(&h->bytes, bytes, __ATOMIC_RELAXED);
    __atomic_store_n(&h->t_prod_ns, now_ns(), __ATOMIC_RELAXED);
    __atomic_store_n(&h->seq, chunk, __ATOMIC_RELEASE);
}

// Consumer side: spin until slot `s` holds `chunk` or the timeout expires.
inline bool wait_publish(Ring& r, int s, std::uint64_t chunk,
                         std::uint64_t timeout_ns, std::uint64_t* t_seen_ns) {
    const std::uint64_t deadline = now_ns() + timeout_ns;
    SlotHdr* h = r.hdr(s);
    for (;;) {
        std::uint64_t v = __atomic_load_n(&h->seq, __ATOMIC_ACQUIRE);
        if (v == chunk) {
            if (t_seen_ns) *t_seen_ns = now_ns();
            return true;
        }
        if (v > chunk) return false;  // producer skipped/overwrote: protocol error
        if (now_ns() > deadline) return false;
    }
}

// Consumer side: release the slot back to the producer.
inline void release(Ring& r, int s, std::uint64_t chunk) {
    SlotHdr* h = r.hdr(s);
    __atomic_store_n(&h->t_cons_ns, now_ns(), __ATOMIC_RELAXED);
    __atomic_store_n(&h->done, chunk, __ATOMIC_RELEASE);
}

// Producer side: block (bounded) until slot `s` is free to reuse for `chunk`.
inline bool wait_free(Ring& r, int s, std::uint64_t chunk,
                      std::uint64_t timeout_ns, std::uint64_t* waited_ns) {
    if (chunk <= static_cast<std::uint64_t>(SLOTS)) {
        if (waited_ns) *waited_ns = 0;
        return true;
    }
    const std::uint64_t need = chunk - static_cast<std::uint64_t>(SLOTS);
    const std::uint64_t deadline = now_ns() + timeout_ns;
    const std::uint64_t t0 = now_ns();
    SlotHdr* h = r.hdr(s);
    for (;;) {
        if (__atomic_load_n(&h->done, __ATOMIC_ACQUIRE) >= need) {
            if (waited_ns) *waited_ns = now_ns() - t0;
            return true;
        }
        if (now_ns() > deadline) return false;
    }
}

}  // namespace hv
