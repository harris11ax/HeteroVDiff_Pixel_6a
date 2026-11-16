// tools/android/pipe/handoff_selftest.cpp — P3.1: the handoff primitive, no compute.
//
// Two processes, one file-mapped ring (phase3_pipeline/handoff.h). The producer
// stamps every 64-bit word of each payload with a chunk-derived pattern and
// publishes; the consumer verifies every word, in order, and checks
// t_cons_ns > t_prod_ns. Any torn or out-of-order chunk shows up as a mismatch
// count, not as a crash.
//
//   handoff_selftest --path /data/local/tmp/hv_handoff.bin --chunks 200
//   handoff_selftest --path ... --chunks 200 --stall-us 1000     # back-pressure
//   handoff_selftest --role producer --path ... --chunks 200     # split launch
//
// Default role is `fork`: the parent creates the ring, then forks one producer
// and one consumer, so the handoff is proven across two real PIDs sharing one
// mapping without needing two adb shells.
//
// Exit: 0 = PASS, 1 = verification failure, 2 = setup/protocol failure.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "handoff.h"

namespace {

struct Args {
    std::string path = "/data/local/tmp/hv_handoff.bin";
    std::string role = "fork";  // fork | producer | consumer
    std::uint64_t chunks = 200;
    std::size_t bytes = hv::CHUNK_BYTES;
    std::uint64_t stall_us = 0;
    std::string msync = "first";  // none | first | full
    std::uint64_t timeout_s = 60;
};

constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kMul = 0x100000001B3ull;

inline std::uint64_t word_for(std::uint64_t chunk, std::uint64_t j) {
    return (chunk * kSeed) ^ (j * kMul);
}

hv::Msync parse_msync(const std::string& s) {
    if (s == "none") return hv::Msync::None;
    if (s == "full") return hv::Msync::Full;
    return hv::Msync::FirstPage;
}

void fill_payload(std::uint8_t* p, std::size_t bytes, std::uint64_t chunk) {
    const std::size_t words = bytes / 8;
    auto* w = reinterpret_cast<std::uint64_t*>(p);
    for (std::size_t j = 0; j < words; ++j) w[j] = word_for(chunk, j);
    // Tail (bytes % 8) is stamped with the chunk index so a short/long payload
    // is still detectable.
    for (std::size_t i = words * 8; i < bytes; ++i) p[i] = static_cast<std::uint8_t>(chunk + i);
}

// Returns the number of mismatching 64-bit words; sets *first_off to the byte
// offset of the first mismatch.
std::uint64_t verify_payload(const std::uint8_t* p, std::size_t bytes, std::uint64_t chunk,
                             std::size_t* first_off, std::uint64_t* expect_word,
                             std::uint64_t* got_word) {
    const std::size_t words = bytes / 8;
    auto* w = reinterpret_cast<const std::uint64_t*>(p);
    std::uint64_t bad = 0;
    if (first_off) *first_off = 0;
    for (std::size_t j = 0; j < words; ++j) {
        const std::uint64_t e = word_for(chunk, j);
        if (w[j] != e) {
            if (bad == 0 && first_off) {
                *first_off = j * 8;
                if (expect_word) *expect_word = e;
                if (got_word) *got_word = w[j];
            }
            ++bad;
        }
    }
    for (std::size_t i = words * 8; i < bytes; ++i) {
        if (p[i] != static_cast<std::uint8_t>(chunk + i)) ++bad;
    }
    return bad;
}

int run_producer(const Args& a, hv::Ring& r) {
    const hv::Msync mode = parse_msync(a.msync);
    const std::uint64_t t0 = hv::now_ns();
    std::uint64_t waits = 0, wait_ns_total = 0, wait_ns_max = 0, msync_max = 0;
    std::uint64_t publishes = 0;

    for (std::uint64_t chunk = 1; chunk <= a.chunks; ++chunk) {
        const int s = static_cast<int>((chunk - 1) % hv::SLOTS);
        std::uint64_t waited = 0;
        if (!hv::wait_free(r, s, chunk, a.timeout_s * 1'000'000'000ull, &waited)) {
            std::fprintf(stderr, "FATAL|producer|slot %d never freed for chunk %" PRIu64 "\n", s,
                         chunk);
            return 2;
        }
        if (waited) {
            ++waits;
            wait_ns_total += waited;
            if (waited > wait_ns_max) wait_ns_max = waited;
        }
        fill_payload(r.payload(s), a.bytes, chunk);
        long ms = 0;
        hv::publish(r, s, chunk, static_cast<std::uint32_t>(a.bytes), mode, &ms);
        if (ms > 0 && static_cast<std::uint64_t>(ms) > msync_max) msync_max = static_cast<std::uint64_t>(ms);
        ++publishes;
        std::printf("TS|producer|chunk=%" PRIu64 "|slot=%d|t_ns=%" PRIu64 "|msync_ns=%ld|blocked_ns=%" PRIu64
                    "\n",
                    chunk, s, hv::now_ns(), ms, waited);
    }
    // Signal end-of-stream on the last published slot.
    const int last = static_cast<int>((a.chunks - 1) % hv::SLOTS);
    __atomic_store_n(&r.hdr(last)->flags, hv::FLAG_STREAM_DONE, __ATOMIC_RELEASE);

    const std::uint64_t el = hv::now_ns() - t0;
    std::printf("SUMMARY|producer|pid=%d|chunks=%" PRIu64 "|payload_bytes=%zu|elapsed_s=%.3f|"
                "chunks_blocked=%" PRIu64 "|blocked_ms=%.3f|blocked_max_us=%" PRIu64 "|msync_max_us=%" PRIu64
                "\n",
                static_cast<int>(getpid()), publishes, a.bytes, el / 1e9, waits,
                wait_ns_total / 1e6, wait_ns_max / 1000, msync_max / 1000);
    return 0;
}

int run_consumer(const Args& a, hv::Ring& r, std::uint64_t* out_bad_words,
                 std::uint64_t* out_bad_chunks, std::uint64_t* out_order) {
    const std::uint64_t t0 = hv::now_ns();
    std::uint64_t bad_words = 0, bad_chunks = 0, order_errors = 0, short_payloads = 0;
    std::uint64_t lat_min = ~0ull, lat_max = 0, lat_sum = 0, seen = 0;

    for (std::uint64_t chunk = 1; chunk <= a.chunks; ++chunk) {
        const int s = static_cast<int>((chunk - 1) % hv::SLOTS);
        std::uint64_t t_seen = 0;
        if (!hv::wait_publish(r, s, chunk, a.timeout_s * 1'000'000'000ull, &t_seen)) {
            const std::uint64_t got = __atomic_load_n(&r.hdr(s)->seq, __ATOMIC_ACQUIRE);
            std::fprintf(stderr, "FATAL|consumer|chunk %" PRIu64 " never published (slot %d holds %" PRIu64 ")\n",
                         chunk, s, got);
            return 2;
        }
        hv::SlotHdr* h = r.hdr(s);
        const std::uint32_t bytes = __atomic_load_n(&h->bytes, __ATOMIC_RELAXED);
        const std::uint64_t t_prod = __atomic_load_n(&h->t_prod_ns, __ATOMIC_RELAXED);

        std::size_t first_off = 0;
        std::uint64_t e_word = 0, g_word = 0;
        std::uint64_t bad = 0;
        if (bytes != a.bytes) {
            ++short_payloads;
            ++bad_chunks;
        } else {
            bad = verify_payload(r.payload(s), bytes, chunk, &first_off, &e_word, &g_word);
        }
        if (bad) {
            ++bad_chunks;
            bad_words += bad;
            std::printf("BAD|chunk=%" PRIu64 "|words=%" PRIu64 "|first_off=%zu|expect=0x%016" PRIx64
                        "|got=0x%016" PRIx64 "\n",
                        chunk, bad, first_off, e_word, g_word);
        }
        if (t_seen <= t_prod) ++order_errors;

        const std::uint64_t lat = t_seen > t_prod ? t_seen - t_prod : 0;
        lat_sum += lat;
        if (lat < lat_min) lat_min = lat;
        if (lat > lat_max) lat_max = lat;
        ++seen;

        hv::release(r, s, chunk);
        std::printf("TS|consumer|chunk=%" PRIu64 "|slot=%d|t_ns=%" PRIu64 "|lat_ns=%" PRIu64
                    "|bad_words=%" PRIu64 "\n",
                    chunk, s, t_seen, lat, bad);
        if (a.stall_us) {
            const std::uint64_t until = hv::now_ns() + a.stall_us * 1000ull;
            while (hv::now_ns() < until) {
            }
        }
        if (chunk % 50 == 0)
            std::fflush(stdout);
    }

    const std::uint64_t el = hv::now_ns() - t0;
    const int last = static_cast<int>((a.chunks - 1) % hv::SLOTS);
    const std::uint32_t flags = __atomic_load_n(&r.hdr(last)->flags, __ATOMIC_ACQUIRE);

    std::printf("SUMMARY|consumer|pid=%d|chunks=%" PRIu64 "|bad_chunks=%" PRIu64 "|bad_words=%" PRIu64
                "|order_errors=%" PRIu64 "|short_payloads=%" PRIu64 "|lat_min_us=%.3f|lat_mean_us=%.3f"
                "|lat_max_us=%.3f|elapsed_s=%.3f|stream_done_flag=%u\n",
                static_cast<int>(getpid()), seen, bad_chunks, bad_words, order_errors,
                short_payloads, (lat_min == ~0ull ? 0.0 : lat_min / 1e3),
                (seen ? static_cast<double>(lat_sum) / seen / 1e3 : 0.0), lat_max / 1e3,
                el / 1e9, (flags & hv::FLAG_STREAM_DONE) ? 1u : 0u);

    if (out_bad_words) *out_bad_words = bad_words;
    if (out_bad_chunks) *out_bad_chunks = bad_chunks;
    if (out_order) *out_order = order_errors;
    return (bad_words || bad_chunks || order_errors || short_payloads) ? 1 : 0;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--path") a.path = val("--path");
        else if (k == "--role") a.role = val("--role");
        else if (k == "--chunks") a.chunks = std::strtoull(val("--chunks").c_str(), nullptr, 10);
        else if (k == "--bytes") a.bytes = std::strtoull(val("--bytes").c_str(), nullptr, 10);
        else if (k == "--stall-us") a.stall_us = std::strtoull(val("--stall-us").c_str(), nullptr, 10);
        else if (k == "--msync") a.msync = val("--msync");
        else if (k == "--timeout-s") a.timeout_s = std::strtoull(val("--timeout-s").c_str(), nullptr, 10);
        else if (k == "--help" || k == "-h") {
            std::printf("usage: handoff_selftest [--path P] [--role fork|producer|consumer] "
                        "[--chunks N] [--bytes B] [--stall-us U] [--msync none|first|full]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.bytes == 0 || a.bytes > hv::SLOT_STRIDE) {
        std::fprintf(stderr, "payload %zu B does not fit SLOT_STRIDE %zu B\n", a.bytes,
                     hv::SLOT_STRIDE);
        std::exit(2);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    // Two processes share one stdout pipe in fork mode. Unbuffered/block-buffered
    // stdio flushes at arbitrary byte offsets, so one process's line can be cut
    // mid-way by the other's — which corrupts the very timestamps this test
    // exists to prove. Line buffering makes each record a single short write().
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("HV|selftest|file_bytes=%zu|slots=%d|slot_stride=%zu|chunk_bytes=%zu|file=%s\n",
                hv::FILE_BYTES, hv::SLOTS, hv::SLOT_STRIDE, a.bytes, a.path.c_str());
    std::fflush(stdout);

    if (a.role == "producer" || a.role == "consumer") {
        // Split launch: the producer owns ring creation and, if the file was
        // absent/empty, the header reset. The consumer must never reset — it
        // would wipe a publish the producer has already made.
        hv::Ring r = hv::map_ring(a.path.c_str(), a.role == "producer");
        if (!r.ok()) {
            std::fprintf(stderr, "FATAL|%s|cannot map %s for read-write\n", a.role.c_str(),
                         a.path.c_str());
            return 2;
        }
        if (a.role == "producer" && r.created) hv::reset_headers(r);
        int rc;
        if (a.role == "producer") {
            rc = run_producer(a, r);
        } else {
            rc = run_consumer(a, r, nullptr, nullptr, nullptr);
        }
        hv::unmap_ring(r);
        return rc;
    }

    // fork mode: parent owns the ring, then one producer + one consumer process.
    hv::Ring r = hv::map_ring(a.path.c_str(), true);
    if (!r.ok()) {
        std::fprintf(stderr, "FATAL|parent|cannot create/map %s\n", a.path.c_str());
        return 2;
    }
    hv::reset_headers(r);
    std::printf("HV|parent|pid=%d|mapped_bytes=%zu\n", static_cast<int>(getpid()), r.bytes);
    std::fflush(stdout);

    const pid_t prod = fork();
    if (prod == 0) {
        const int rc = run_producer(a, r);
        std::fflush(stdout);
        hv::unmap_ring(r);
        std::_Exit(rc);
    }
    if (prod < 0) {
        std::fprintf(stderr, "FATAL|parent|fork(producer) failed\n");
        return 2;
    }

    const pid_t cons = fork();
    if (cons == 0) {
        const int rc = run_consumer(a, r, nullptr, nullptr, nullptr);
        std::fflush(stdout);
        hv::unmap_ring(r);
        std::_Exit(rc);
    }
    if (cons < 0) {
        std::fprintf(stderr, "FATAL|parent|fork(consumer) failed\n");
        return 2;
    }

    int st_p = 0, st_c = 0;
    waitpid(prod, &st_p, 0);
    waitpid(cons, &st_c, 0);
    const int rp = WIFEXITED(st_p) ? WEXITSTATUS(st_p) : 128;
    const int rc = WIFEXITED(st_c) ? WEXITSTATUS(st_c) : 128;
    const bool pass = (rp == 0 && rc == 0);
    std::printf("RESULT|%s|producer_rc=%d|consumer_rc=%d|producer_pid=%d|consumer_pid=%d\n",
                pass ? "PASS" : "FAIL", rp, rc, static_cast<int>(prod), static_cast<int>(cons));
    hv::unmap_ring(r);
    return pass ? 0 : 1;
}
