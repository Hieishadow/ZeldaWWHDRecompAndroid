#include <pthread.h>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "gx2.h"
#include "gx2_cmd.h"
#include "gx2_regs.h"
#include "gx2_texture_regs.h"
#include "platform.h"
#include "runtime.h"
#include "../aspect.h"
#include <map>
#include <unistd.h>
#include <dirent.h>
#include "mem_writes.h"

using namespace Latte;
namespace gx2 {

static uint32 g_regs[kNumRegs];
static uint32* g_shadow = nullptr;
static std::unordered_map<uint32, std::vector<uint32>> g_contexts;
constexpr uint32 kRegBlock = 64;
static bool g_reg_touched[kNumRegs / kRegBlock];
static std::vector<uint16_t> g_reg_blocks;
static void touch_regs(uint32 first, uint32 n) {
    if (!n) return;
    for (uint32 b = first / kRegBlock, e = (first + n - 1) / kRegBlock; b <= e && b < kNumRegs / kRegBlock; b++)
        if (!g_reg_touched[b]) {
            g_reg_touched[b] = true;
            g_reg_blocks.push_back((uint16_t)b);
        }
}
static void touch_all_regs() { touch_regs(0, kNumRegs); }
static std::recursive_mutex g_exec_mutex;
uint32* regs() { return g_regs; }
extern "C" { uint64_t g_shader_state_gen = 1; }
extern "C" { uint64_t g_draw_state_gen = 1; }

static bool shader_irrelevant(uint32 reg) {
    if (reg >= mmSQ_ALU_CONSTANT0_0 && reg < mmSQ_ALU_CONSTANT0_0 + 0x1000) return true;
    for (uint32 base : {(uint32)mmSQ_VTX_UNIFORM_BLOCK_START, (uint32)mmSQ_PS_UNIFORM_BLOCK_START, (uint32)mmSQ_GS_UNIFORM_BLOCK_START})
        if (reg >= base && reg < base + 7 * 16) return true;
    if (reg >= mmSQ_VTX_ATTRIBUTE_BLOCK_START && reg < mmSQ_VTX_ATTRIBUTE_BLOCK_START + 7 * 16) {
        uint32 w = (reg - mmSQ_VTX_ATTRIBUTE_BLOCK_START) % 7;
        return w!= 2;
    }
    return false;
}
extern "C" { bool (*g_shader_reg_filter)(uint32 reg, uint32 oldv, uint32 newv) = nullptr; }
static const bool g_gen_stats = getenv("WWHD_GEN_STATS")!= nullptr;
static uint64_t g_gen_ctx = 0, g_gen_ctx_calls = 0, g_gen_regs = 0;
static std::unordered_map<uint32, uint64_t> g_gen_by_reg;

static void apply_regs(uint32 first, const uint32* v, uint32 n) {
    if (first + n > kNumRegs) return;
    touch_regs(first, n);
    if (memcmp(&g_regs[first], v, n * 4)!= 0) {
        static const bool coarse = getenv("WWHD_COARSE_SHADER_GEN")!= nullptr;
        bool draw = false, shader = false;
        for (uint32 i = 0; i < n &&!(draw && shader); i++) {
            if (g_regs[first + i] == v[i] || shader_irrelevant(first + i)) continue;
            draw = true;
            if (!shader && (g_shader_reg_filter &&!coarse? g_shader_reg_filter(first + i, g_regs[first + i], v[i]) : true)) {
                shader = true;
                if (g_gen_stats) { g_gen_regs++; g_gen_by_reg[first + i]++; }
            }
        }
        if (shader) g_shader_state_gen++;
        if (draw) g_draw_state_gen++;
        memcpy(&g_regs[first], v, n * 4);
    }
    if (g_shadow) memcpy(&g_shadow[first], v, n * 4);
}

struct Recording { uint32 start = 0, pos = 0, end = 0; };
static thread_local Recording t_rec;
static void execute_one(Op op, const uint32* p, uint32 n);

static const bool g_render_thread = getenv("WWHD_NO_RENDER_THREAD") == nullptr;
static std::mutex g_q_mutex;
static std::condition_variable g_q_cv, g_q_done_cv;
static std::vector<uint32> g_q_pending, g_q_work;
static bool g_q_waiting = false;
static uint64_t g_fence_issued = 0, g_fence_done = 0;

static void render_thread_main() {
    platform::set_thread_name("GX2 render");
    platform::set_thread_high_priority();
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_q_mutex);
            g_q_waiting = true;
            g_q_cv.wait(lk, [] { return!g_q_pending.empty(); });
            g_q_waiting = false;
            g_q_work.swap(g_q_pending);
        }
        {
            std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
            gfx::with_autorelease_pool([] { execute(g_q_work.data(), (uint32)g_q_work.size()); });
        }
        g_q_work.clear();
    }
}

static void enqueue(Op op, const uint32* payload, uint32 n) {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(render_thread_main).detach(); });
    std::lock_guard<std::mutex> lk(g_q_mutex);
    g_q_pending.push_back(op | (n << 8));
    g_q_pending.insert(g_q_pending.end(), payload, payload + n);
    if (g_q_waiting) g_q_cv.notify_one();
}

static void render_sync() {
    if (!g_render_thread) return;
    uint64_t id;
    { std::lock_guard<std::mutex> lk(g_q_mutex); id = ++g_fence_issued; }
    uint32 w = (uint32)id;
    enqueue(OP_FENCE, &w, 1);
    std::unique_lock<std::mutex> lk(g_q_mutex);
    g_q_done_cv.wait(lk, [&] { return g_fence_done >= id; });
}

void emit(Op op, const uint32* payload, uint32 n) {
    if (t_rec.start) {
        uint32 bytes = 4 * (n + 1);
        if (t_rec.pos + bytes > t_rec.end) { LOG("[gx2] display list overflow at %08X", t_rec.start); return; }
        uint32* w = (uint32*)mem::ptr(t_rec.pos);
        w[0] = op | (n << 8);
        memcpy(w + 1, payload, n * 4);
        t_rec.pos += bytes;
        return;
    }
    if (g_render_thread) { enqueue(op, payload, n); return; }
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    execute_one(op, payload, n);
}

static void emit_host(Op op, std::initializer_list<uint32> payload) {
    if (g_render_thread) { enqueue(op, payload.begin(), (uint32)payload.size()); return; }
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    execute_one(op, payload.begin(), (uint32)payload.size());
}

void set_regs(uint32 first, const uint32* values, uint32 count) {
    if (!count) return;
    static thread_local std::vector<uint32> buf;
    buf.resize(count + 1);
    buf[0] = first;
    memcpy(&buf[1], values, count * 4);
    emit(OP_SET_REGS, buf.data(), count + 1);
}
void set_reg(uint32 reg, uint32 value) { emit(OP_SET_REGS, {reg, value}); }

void execute(const uint32* words, uint32 count) {
    uint32 i = 0;
    while (i < count) {
        uint32 hdr = words[i];
        Op op = (Op)(hdr & 0xFF);
        uint32 n = hdr >> 8;
        if (op >= OP_COUNT || i + 1 + n > count) { LOG("[gx2] corrupt display list command %08X", hdr); return; }
        execute_one(op, &words[i + 1], n);
        i += 1 + n;
    }
}

static void set_context(uint32 ctx) {
    if (g_gen_stats) g_gen_ctx_calls++;
    if (!ctx) { g_shadow = nullptr; return; }
    auto it = g_contexts.find(ctx);
    if (it == g_contexts.end()) { g_shadow = nullptr; return; }
    g_shadow = it->second.data();
    static const bool init = (touch_regs(REGADDR::VGT_PRIMITIVE_TYPE, 1), true); (void)init;
    for (uint16_t b : g_reg_blocks) memcpy(&g_regs[b * kRegBlock], &g_shadow[b * kRegBlock], kRegBlock * 4);
    g_shader_state_gen++; g_draw_state_gen++;
    if (g_gen_stats) g_gen_ctx++;
}

constexpr uint32 kColorBufferWords = 0x9C / 4, kDepthBufferWords = 0xAC / 4, kSurfaceWords = 0x74 / 4;
static uint32 unpack_struct(const uint32* words, uint32 count, int slot) {
    static uint32 scratch = 0;
    if (!scratch) scratch = mem::host_alloc(2 * 0x100, 0x40);
    uint32 addr = scratch + slot * 0x100;
    memcpy(mem::ptr(addr), words, count * 4);
    return addr;
}

static void execute_one(Op op, const uint32* p, uint32 n) {
    switch (op) {
    case OP_NOP: break;
    case OP_SET_REGS: apply_regs(p[0], p + 1, n - 1); break;
    case OP_DRAW: gfx::draw(g_regs, p[0], p[1], 0, 0, p[2], p[3]); break;
    case OP_DRAW_INDEXED: gfx::draw(g_regs, p[0], p[1], p[2], p[3], p[4], p[5]); break;
    case OP_CLEAR_COLOR: { const uint32* q = p + kColorBufferWords; float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])}; gfx::clear_color(g_regs, unpack_struct(p, kColorBufferWords, 0), rgba); break; }
    case OP_CLEAR_DEPTH: { const uint32* q = p + kDepthBufferWords; gfx::clear_depth_stencil(g_regs, unpack_struct(p, kDepthBufferWords, 0), bitsf(q[0]), q[1], q[2]); break; }
    case OP_CLEAR_BUFFERS: { uint32 cb = unpack_struct(p, kColorBufferWords, 0), db = unpack_struct(p + kColorBufferWords, kDepthBufferWords, 1); const uint32* q = p + kColorBufferWords + kDepthBufferWords; float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])}; gfx::clear_color(g_regs, cb, rgba); gfx::clear_depth_stencil(g_regs, db, bitsf(q[4]), q[5], q[6]); break; }
    case OP_COPY_SURFACE: { uint32 src = unpack_struct(p, kSurfaceWords, 0); const uint32* q = p + kSurfaceWords; uint32 dst = unpack_struct(q + 2, kSurfaceWords, 1); const uint32* r = q + 2 + kSurfaceWords; gfx::copy_surface(src, q[0], q[1], dst, r[0], r[1]); break; }
    case OP_COPY_TO_SCAN: gfx::copy_to_scan(unpack_struct(p, kColorBufferWords, 0), p[kColorBufferWords]); break;
    case OP_CALL: execute((const uint32*)mem::ptr(p[0]), p[1] / 4); break;
    case OP_SET_CONTEXT: set_context(p[0]); break;
    case OP_INVALIDATE: gfx::invalidate(p[0], p[1], p[2]); break;
    case OP_EXPAND_COLOR: case OP_EXPAND_DEPTH: break;
    case OP_FLUSH: gfx::flush(); break;
    case OP_DRAW_DONE: gfx::draw_done(); break;
    case OP_SET_PROJ_REGS: { uint32 v[16]; memcpy(v, p + 1, sizeof v); float kx, ky; if (n == 17 && gfx::target_aspect_factors(g_regs[mmCB_COLOR0_TILE] & 0xFFFF, g_regs[mmCB_COLOR0_FRAG], kx, ky)) for (int i = 0; i < 4; i++) { v[i] = gx2::fbits(gx2::bitsf(v[i]) / kx); v[4 + i] = gx2::fbits(gx2::bitsf(v[4 + i]) / ky); } apply_regs(p[0], v, std::min<uint32>(n - 1, 16)); break; }
    case OP_PEEK_Z: gfx::peek_z(p, n); break;
    case OP_LAYOUT_ROOT: { float kx, ky; aspect::layout_root_target(p[0], gfx::target_aspect_factors(g_regs[mmCB_COLOR0_TILE] & 0xFFFF, g_regs[mmCB_COLOR0_FRAG], kx, ky)); break; }
    case OP_SWAP: if (n) gfx::set_frame_aspect(gx2::bitsf(p[0])); gfx::swap(); break;
    case OP_SETUP_CONTEXT: g_contexts[p[0]].assign(kNumRegs, 0); g_shadow = g_contexts[p[0]].data(); break;
    case OP_FENCE: { std::lock_guard<std::mutex> lk(g_q_mutex); g_fence_done = std::max<uint64_t>(g_fence_done, p[0]); g_q_done_cv.notify_all(); break; }
    default: break;
    }
}

static void set_default_state() {
    LATTE_SQ_CONFIG sq; sq.set_DX9_CONSTS(true).set_ALU_INST_PREFER_VECTOR(true).set_PS_PRIO(3).set_VS_PRIO(2).set_GS_PRIO(1).set_ES_PRIO(0);
    set_reg(REGADDR::SQ_CONFIG,
