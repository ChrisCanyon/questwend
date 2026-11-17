// OpenAI-compatible inference server.
//   POST /v1/chat/completions  (streaming SSE + non-streaming)
//   GET  /v1/models
//   GET  /health

#include "model.h"
#include "token_pieces.h"
#include "runtime.h"
#include "expert_debug_hooks.h"
#if QW_EXPERT_VISUALIZATION
#include "expert_debug_routes.h"
#endif
#include "sampler.h"
#include "questwend_bridge.h"
#include "args.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <mutex>
#include <string>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::json;
using namespace questwend;

// FIFO mutex with a waiter count: lets a generating request detect that
// someone is queued (time-slice handoff) and guarantees the waiter actually
// gets the runtime when the holder yields (std::mutex makes no such promise).
struct TicketLock {
    std::mutex m;
    std::condition_variable cv;
    uint64_t next = 0, serving = 0;
    int waiting = 0;
    std::set<uint64_t> dead;   // abandoned tickets (lock_for timeouts)
    void lock() {
        std::unique_lock<std::mutex> lk(m);
        const uint64_t t = next++;
        ++waiting;
        cv.wait(lk, [&] { return serving == t; });
        --waiting;
    }
    // Read-only telemetry may sample an idle runtime, but must never queue
    // behind an inference request or delay its next scheduled slice.
    bool try_lock() {
        std::lock_guard<std::mutex> lk(m);
        if (next != serving) return false;
        ++next;
        return true;
    }
    // Bounded wait: take a ticket, give up after `d`. On timeout the ticket is
    // abandoned (unlock() skips it) and a later retry re-queues at the back —
    // fairness is traded for letting the waiter do work (SSE heartbeats,
    // disconnect checks) between attempts.
    bool lock_for(std::chrono::milliseconds d) {
        std::unique_lock<std::mutex> lk(m);
        const uint64_t t = next++;
        ++waiting;
        const bool got = cv.wait_for(lk, d, [&] { return serving == t; });
        --waiting;
        if (!got) dead.insert(t);
        return got;
    }
    void unlock() {
        std::lock_guard<std::mutex> lk(m);
        ++serving;
        while (dead.count(serving)) { dead.erase(serving); ++serving; }
        cv.notify_all();
    }
    bool contended() {
        std::lock_guard<std::mutex> lk(m);
        return waiting > 0;
    }
    bool busy() {   // held or has waiters (a fresh request would queue)
        std::lock_guard<std::mutex> lk(m);
        return next > serving;
    }
};

// dump JSON replacing invalid UTF-8 instead of throwing: a token budget can
// cut the output mid multi-byte character (byte-level BPE), and json::dump()
// on that fragment would otherwise escape the handler as an HTTP 500.
static std::string dumpj(const nlohmann::json & j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

static std::string now_id(const char * prefix) {
    auto t = std::chrono::system_clock::now().time_since_epoch().count();
    return std::string(prefix) + std::to_string(t);
}

// local wall-clock "HH:MM:SS" for request log lines
static std::string clock_hms() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

struct ParsedToolCall { std::string name; std::string arguments; };

// Hold back a trailing incomplete UTF-8 sequence in `pend`: with byte-level
// BPE a multi-byte character can span tokens, and each SSE chunk must be
// valid UTF-8 on its own. Returns the emittable prefix of pend + piece.
static std::string utf8_clip(std::string & pend, std::string piece) {
    piece = pend + piece;
    pend.clear();
    size_t keep = piece.size();
    for (size_t i = 0; i < 4 && i < piece.size(); ++i) {
        const unsigned char c = (unsigned char) piece[piece.size() - 1 - i];
        if ((c & 0x80) == 0) break;                // ASCII: complete
        if ((c & 0xC0) == 0xC0) {                  // lead byte, i continuations after it
            const size_t need = (c & 0xF8) == 0xF0 ? 4 : (c & 0xF0) == 0xE0 ? 3 : 2;
            if (i + 1 < need) keep = piece.size() - 1 - i;   // incomplete
            break;
        }                                          // else continuation: scan back
    }
    if (keep < piece.size()) { pend = piece.substr(keep); piece.erase(keep); }
    return piece;
}

// Minimal self-contained chat UI (served at GET /). Talks to the server's own
// OpenAI-compatible /v1/chat/completions endpoint with streaming.
static const char * CHAT_HTML = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuestWend chat</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --txt:#e6e8ee; --muted:#8b93a7; --accent:#5b8cff; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; flex-direction:column; background:var(--bg); color:var(--txt);
         font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header b { font-weight:600; }
  header .model { color:var(--muted); font-size:13px; }
  header .spacer { flex:1; }
  header button, .ctl { background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:8px; padding:6px 10px; font-size:13px; cursor:pointer; }
  header label { color:var(--muted); font-size:12px; display:flex; align-items:center; gap:6px; }
  header input[type=number] { width:64px; background:var(--bg); color:var(--txt); border:1px solid var(--line);
         border-radius:6px; padding:4px 6px; }
  #chat { flex:1; overflow-y:auto; padding:20px 0; }
  .wrap { max-width:780px; margin:0 auto; padding:0 16px; }
  .msg { display:flex; gap:12px; margin:14px 0; }
  .msg .who { width:30px; height:30px; border-radius:7px; flex:none; display:flex; align-items:center;
         justify-content:center; font-size:13px; font-weight:600; }
  .msg.user .who { background:#2a3550; color:#bcd0ff; }
  .msg.assistant .who { background:#23314a; color:#9fd6c0; }
  .msg .body { white-space:pre-wrap; word-wrap:break-word; padding-top:3px; }
  .msg .body.think { color:var(--muted); font-style:italic; }
  .stats { color:var(--muted); font-size:12px; margin-top:8px; font-variant-numeric:tabular-nums; }
  footer { border-top:1px solid var(--line); padding:12px 0 18px; }
  .inrow { max-width:780px; margin:0 auto; padding:0 16px; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:10px; padding:10px 12px; font:inherit; max-height:200px; }
  .send { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:10px 18px;
         font-size:15px; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .attach { background:var(--panel); color:var(--txt); border:1px solid var(--line); border-radius:10px;
         padding:10px 12px; font-size:15px; cursor:pointer; }
  .hint { color:var(--muted); font-size:12px; text-align:center; margin-top:8px; }
  #thumbs { max-width:780px; margin:0 auto 6px; padding:0 16px; display:flex; gap:8px; }
  #thumbs img { height:56px; border-radius:8px; border:1px solid var(--line); cursor:pointer; }
  .msg .body img { max-height:160px; border-radius:8px; display:block; margin:4px 0; }
</style>
</head>
<body>
<header>
  <b>QuestWend</b><span class="model" id="model">…</span>
  <span class="spacer"></span>
  <label>temp <input type="number" id="temp" value="0.7" step="0.1" min="0" max="2"></label>
  <label><input type="checkbox" id="think" checked> think</label>
  <button id="clear">Clear</button>
</header>
<div id="chat"><div class="wrap" id="list"></div></div>
<footer>
  <div id="thumbs"></div>
  <div class="inrow">
    <button class="attach" id="attach" title="attach image">&#128206;</button>
    <input type="file" id="file" accept="image/*" multiple style="display:none">
    <textarea id="input" rows="1" placeholder="Message…  (Enter to send, Shift+Enter for newline)"></textarea>
    <button class="send" id="send">Send</button>
  </div>
  <div class="hint">streams from /v1/chat/completions</div>
</footer>
<script>
const list = document.getElementById('list');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let history = [];
let busy = false;

fetch('/v1/models').then(r=>r.json()).then(j=>{
  const m = j.data && j.data[0];
  document.getElementById('model').textContent = (m && (m.display || m.id)) || '';
}).catch(()=>{});

function addMsg(role, text, imgs){
  const m = document.createElement('div'); m.className = 'msg ' + role;
  const who = document.createElement('div'); who.className = 'who'; who.textContent = role==='user'?'You':'AI';
  const body = document.createElement('div'); body.className = 'body';
  if(imgs) for(const u of imgs){ const im = document.createElement('img'); im.src = u; body.appendChild(im); }
  if(text) body.appendChild(document.createTextNode(text));
  m.appendChild(who); m.appendChild(body); list.appendChild(m);
  document.getElementById('chat').scrollTop = 1e9;
  return body;
}

// pending image attachments (data URIs)
let pendingImgs = [];
const thumbs = document.getElementById('thumbs');
const fileInput = document.getElementById('file');
document.getElementById('attach').addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', ()=>{
  for(const f of fileInput.files){
    const r = new FileReader();
    r.onload = ()=>{ pendingImgs.push(r.result); renderThumbs(); };
    r.readAsDataURL(f);
  }
  fileInput.value = '';
});
function renderThumbs(){
  thumbs.innerHTML = '';
  pendingImgs.forEach((u, i)=>{
    const im = document.createElement('img'); im.src = u; im.title = 'click to remove';
    im.addEventListener('click', ()=>{ pendingImgs.splice(i,1); renderThumbs(); });
    thumbs.appendChild(im);
  });
}
function autosize(){ input.style.height='auto'; input.style.height = Math.min(input.scrollHeight,200)+'px'; }
input.addEventListener('input', autosize);
input.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
sendBtn.addEventListener('click', send);
document.getElementById('clear').addEventListener('click', ()=>{ history=[]; list.innerHTML=''; });

async function send(){
  const text = input.value.trim();
  if((!text && !pendingImgs.length) || busy) return;
  busy = true; sendBtn.disabled = true;
  let content = text;
  if(pendingImgs.length){
    content = pendingImgs.map(u=>({type:'image_url', image_url:{url:u}}));
    if(text) content.push({type:'text', text});
  }
  history.push({role:'user', content});
  addMsg('user', text, pendingImgs);
  pendingImgs = []; renderThumbs();
  input.value=''; autosize();
  const body = addMsg('assistant', '');
  let acc = '', accR = '';
  let timings = null;
  try {
    const resp = await fetch('/v1/chat/completions', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        messages: history, stream: true,
        temperature: parseFloat(document.getElementById('temp').value)||0,
        // no max_tokens: the server generates until the context runs out
        reasoning: document.getElementById('think').checked
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let idx;
      while((idx = buf.indexOf('\n\n')) >= 0){
        const ev = buf.slice(0, idx); buf = buf.slice(idx+2);
        const line = ev.replace(/^data:\s*/, '');
        if(line === '[DONE]') continue;
        try {
          const j = JSON.parse(line);
          if(j.timings) timings = j.timings;
          const d = j.choices && j.choices[0] && j.choices[0].delta;
          let upd = false;
          if(d && d.reasoning_content){ accR += d.reasoning_content; upd = true; }
          if(d && d.content){ acc += d.content; upd = true; }
          if(upd){ renderAssistant(body, acc, accR); document.getElementById('chat').scrollTop = 1e9; }
        } catch(e){}
      }
    }
  } catch(e){ acc += '\n[error: '+e+']'; renderAssistant(body, acc, accR); }
  if(timings) showStats(body, timings);
  const amsg = {role:'assistant', content:acc};
  if(accR) amsg.reasoning_content = accR;   // echo reasoning -> prefix cache hit
  history.push(amsg);
  busy = false; sendBtn.disabled = false; input.focus();
}
// dim the reasoning span. Reasoning arrives via reasoning_content deltas
// (thinkExt); the inline </think> parsing is kept for non-streaming replies.
function renderAssistant(el, txt, thinkExt){
  el.innerHTML = '';
  let think = thinkExt || null, answer = txt;
  const close = txt.indexOf('</think>');
  if(think === null && close >= 0){
    think  = txt.slice(0, close).replace(/^\s*<think>/, '');
    answer = txt.slice(close + 8);
  } else if(/^\s*<think>/.test(txt)){
    think = txt.replace(/^\s*<think>/, ''); answer = '';
  }
  if(think !== null && think.trim().length){
    const t = document.createElement('span'); t.className = 'body think';
    t.textContent = think.replace(/^\n+/, ''); el.appendChild(t);
  }
  if(answer.length){
    el.appendChild(document.createTextNode(answer.replace(/^\n+/, '')));
  }
}
function showStats(el, t){
  const s = document.createElement('div'); s.className = 'stats';
  const f = (x)=> (x||0).toFixed(1);
  const cached = t.cached_tokens ? `, ${t.cached_tokens} cached` : '';
  s.textContent = `TTFT ${Math.round(t.ttft_ms||0)} ms · prefill ${f(t.prefill_tps)} tok/s `
    + `(${t.prompt_tokens} in${cached}) · gen ${f(t.gen_tps)} tok/s · ${t.completion_tokens} out`;
  el.appendChild(s);
  document.getElementById('chat').scrollTop = 1e9;
}
</script>
</body>
</html>
)HTML";

int main(int argc, char ** argv) {
    CommonOptions opt;      // -m, --n-ctx, GPU/offload, ... (core/args.h)
    int  port  = 8080;
    std::string host = "127.0.0.1";
    int  cache_slots = 0;
    std::string cache_slots_dir;
    bool complete_cache_slots = false; // experimental private state format
    int  time_slice = 64;   // generation tokens per turn when requests contend
                            // (needs --cache-slots to park state; 0 = back-to-back)
    bool want_help = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--no-mmproj")        {} // retained launch compatibility; text-only server
        else if (a == "--cache-slots" && i + 1 < argc) cache_slots = std::stoi(argv[++i]);
        else if (a == "--cache-slots-dir" && i + 1 < argc) cache_slots_dir = argv[++i];
        else if (a == "--complete-cache-slots") complete_cache_slots = true;
        else if (a == "--time-slice" && i + 1 < argc) time_slice = std::stoi(argv[++i]);
        else if (a == "--heartbeat" && i + 1 < argc) set_knob("QWEN_HEARTBEAT", argv[++i]);
        else if (a == "--pf-chunk" && i + 1 < argc)  set_knob("QWEN_PF_CHUNK", argv[++i]);
        else if (a == "-h" || a == "--help") want_help = true;
        else {
            const ArgResult r = parse_common_arg(argc, argv, i, opt);
            if (r == ArgResult::Error) return 1;
            // unknown flag, or a known flag missing its value (e.g. a typo like
            // --time-clice): fail loudly instead of silently ignoring it
            if (r == ArgResult::Unknown) {
                fprintf(stderr, "error: unknown or malformed argument: %s\n"
                                "run with --help to see usage\n", a.c_str());
                return 1;
            }
        }
    }
    if (!want_help && !validate_common_options(opt)) return 1;
    if (!cache_slots_dir.empty() && cache_slots <= 0) cache_slots = 4;   // dir implies slots

    if (!want_help && complete_cache_slots && (cache_slots <= 0 || opt.use_mtp)) {
        fprintf(stderr, "error: --complete-cache-slots requires cache slots and --mtp off\n");
        return 1;
    }

    // Mirrors of the common options the request handlers below capture by value.
    const std::string & model_path = opt.model_path;
    const bool  force_cpu              = opt.force_cpu;
    const bool  use_mtp                = opt.use_mtp;
    const int   n_draft                = opt.n_draft;
    const bool  reasoning_default      = opt.reasoning;
    const float repeat_penalty_default = opt.repeat_penalty;   // when the request sends none
    int         n_ctx                  = opt.n_ctx;            // 0 = the model's trained length

    if (want_help || model_path.empty()) {
        // --help goes to stdout and succeeds; a missing -m is an error
        FILE * out = want_help ? stdout : stderr;
        fprintf(out,
            "usage: %s -m <model.gguf> [options]\n"
            "server:\n"
            "  --port <N>          listen port (default 8080)\n"
            "  --host <addr>       bind address (default 127.0.0.1)\n"
            "  --no-mmproj         accepted for launch compatibility; text input only\n"
            "  --cache-slots <N>   extra prompt-cache slots for interleaved conversations (default 0)\n"
            "  --cache-slots-dir <dir>  store the slots on disk instead of RAM (persists across restarts)\n"
            "  --complete-cache-slots  experimental QSL2 slots including PLE/QSA/ngram/logits; no MTP\n"
            "  --time-slice <N>    interleave concurrent streaming requests every N generated tokens\n"
            "                      (default 64; takes effect with --cache-slots; 0 = run back-to-back)\n"
            "  --heartbeat <on|off>  SSE keepalive chunks while queued / between prefill chunks (default on)\n"
            "  --pf-chunk <N>      server prefill slice (disconnect-abort granularity, default 4096)\n"
            "  -h, --help          show this help and exit\n", argv[0]);
        // --reasoning and --repeat-penalty are per-request overrides here
        // ("reasoning" / "repeat_penalty" in the body); the flags set the default.
        print_common_help(out);
        return want_help ? 0 : 1;
    }

    std::unique_ptr<Model> model;
    std::unique_ptr<TokenPieces> tok;
    std::unique_ptr<Runtime> rt;
    try {
        model = Model::load(model_path);
        tok   = std::make_unique<TokenPieces>(model->vocab());

        // Context length defaults to what the model was trained for. The KV
        // cache is sized from it, so a very long context costs memory that the
        // expert pool would otherwise use -- pass --n-ctx to cap it if the
        // allocation does not fit.
        if (n_ctx <= 0) {
            n_ctx = (int) model->hparams().n_ctx_train;
            fprintf(stderr, "context: %d tokens (model's trained length; --n-ctx to change)\n", n_ctx);
        }

        opt.n_ctx = n_ctx;   // resolved above from the model's trained length
        RuntimeConfig cfg;
        if (!apply_common_options(opt, cfg)) return 1;
        cfg.cache_profile_save = false;   // server only reads the profile, never overwrites it
        rt = std::make_unique<Runtime>(*model, cfg);
    } catch (const std::exception & e) {
        fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    // MTP self-speculative decode is active only if requested AND the model has a
    // nextn block; otherwise fall back to plain decoding.
    const bool mtp = use_mtp && rt->has_mtp();
    if (use_mtp && !rt->has_mtp())
        fprintf(stderr, "warning: --mtp requested but model has no nextn block; using plain decode\n");
    if (mtp) fprintf(stderr, "MTP self-speculative decode ON (draft=%d; greedy only -"
                             " request sampling parameters are ignored)\n", n_draft);

    // /v1/models id: the model name (general.name) so clients identify the
    // actual model, not the architecture; fall back to "questwend:<arch>".
    const std::string model_id = !model->hparams().general_name.empty()
        ? model->hparams().general_name
        : "questwend:" + std::string(arch_name(model->hparams().arch));

    // Build a human-readable display name: "Qwen3-30B-A3B Q8_0(qwen3)"
    auto ftype_label = [](uint32_t ft) -> const char * {
        // llama_ftype, which is what GGUF general.file_type stores -- NOT
        // ggml_ftype, whose values diverge from 10 onwards. Two of these were
        // once missing (Q2_K_S and IQ3_XS), which shifted everything above them
        // by two and made an IQ1_S model announce itself as IQ3_S. Gaps in the
        // numbering are ftypes that were removed from the format.
        switch (ft) {
            case  0: return "F32";      case  1: return "F16";
            case  2: return "Q4_0";     case  3: return "Q4_1";
            case  7: return "Q8_0";     case  8: return "Q5_0";
            case  9: return "Q5_1";     case 10: return "Q2_K";
            case 11: return "Q3_K_S";   case 12: return "Q3_K_M";
            case 13: return "Q3_K_L";   case 14: return "Q4_K_S";
            case 15: return "Q4_K_M";   case 16: return "Q5_K_S";
            case 17: return "Q5_K_M";   case 18: return "Q6_K";
            case 19: return "IQ2_XXS";  case 20: return "IQ2_XS";
            case 21: return "Q2_K_S";   case 22: return "IQ3_XS";
            case 23: return "IQ3_XXS";  case 24: return "IQ1_S";
            case 25: return "IQ4_NL";   case 26: return "IQ3_S";
            case 27: return "IQ3_M";    case 28: return "IQ2_S";
            case 29: return "IQ2_M";    case 30: return "IQ4_XS";
            case 31: return "IQ1_M";    case 32: return "BF16";
            case 36: return "TQ1_0";    case 37: return "TQ2_0";
            case 38: return "MXFP4_MOE"; case 39: return "NVFP4";
            case 40: return "Q1_0";     case 41: return "Q2_0";
            case 1024: return "";       // GUESSED: the file did not say
            default: return "";
        }
    };
    const std::string model_display = [&]() {
        const auto & hp = model->hparams();
        const char * ft = ftype_label(hp.file_type);
        std::string d;
        if (!hp.general_name.empty()) d = hp.general_name;
        if (ft && *ft) { if (!d.empty()) d += ' '; d += ft; }
        d += '('; d += arch_name(hp.arch); d += ')';
        return d;
    }();
    auto upstream_chat = std::make_shared<open_agents::ChatBridge>(model_path);
    fprintf(stderr, "chat: pinned upstream adapter, strict completion enabled\n");
    auto is_stop = [&](int32_t t){
        for (const auto & token : upstream_chat->capabilities.at("eog_tokens"))
            if (token.get<int32_t>()==t) return true;
        return false;
    };

    TicketLock tlock;          // runtime is single-threaded + stateful (FIFO queue)
    uint64_t live_owner = 0;   // request id owning the live state (0 = none); guarded by tlock
    std::atomic<uint64_t> sid_gen{1};
    if (time_slice > 0 && cache_slots <= 0)
        fprintf(stderr, "warning: --time-slice without --cache-slots: a preempted request"
                        " re-prefills its whole context on resume\n");

    // prefill progress: the offload (SSD/RAM-tier) prefill runs in chunks and
    // reports (done,total); throttle the log line to one per ~10s.
    // The runtime callback's (done,total) is per decode() call, which is too
    // local when the server itself chunks prefill (time-slicing): report
    // progress against the whole prefill instead. Each request owns its own
    // counters (Prog); `cur_prog` points at the request currently holding the
    // runtime, so interleaved requests never mix their numbers.
    using lclk = std::chrono::steady_clock;
    struct Prog {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point last  = std::chrono::steady_clock::now();
        int  base = 0;      // tokens prefilled by earlier server chunks
        int  total = 0;     // full tail tokens of this request
        bool shown = false; // any progress line shown for this request
    };
    auto cur_prog = std::make_shared<std::shared_ptr<Prog>>();
    rt->set_progress_cb([cur_prog](int done, int total) {
        const std::shared_ptr<Prog> p = *cur_prog;
        if (!p) return;
        const int gtot  = p->total > 0 ? p->total : total;
        const int gdone = p->base + done;
        const auto now = lclk::now();
        auto emit = [&] {
            const double el = std::chrono::duration<double>(now - p->start).count();
            fprintf(stderr, "  [%s] prefill %d/%d (%.0f%%) %.0f tok/s\n",
                    clock_hms().c_str(), gdone, gtot, 100.0 * gdone / gtot,
                    el > 0 ? gdone / el : 0.0);
            p->last = now;
        };
        if (gdone >= gtot) {
            // a long prefill finishing between ticks would otherwise stall the
            // log at e.g. 93%; emit a closing 100% line (only if we ever showed
            // progress, so short prefills stay quiet)
            if (p->shown) { emit(); p->shown = false; }
        } else if (now - p->last >= std::chrono::seconds(10)) {
            emit();
            p->shown = true;
        }
    });

    // ---- single-slot prompt prefix cache (guarded by infer_mtx) ----
    // The runtime keeps the KV cache / GDN state of the previous request. If
    // the new prompt's tokens start with exactly rt->kv_tokens(), skip
    // re-prefilling them and decode only the tail. Image spans inside the
    // reused prefix must hold the same image as last time (byte hash); a span
    // straddling the reuse boundary forces a full reset.
    struct ImgSpan { int first; int count; uint64_t hash; };
    auto kv_imgs = std::make_shared<std::vector<ImgSpan>>();   // image spans behind the live state

    // Longest-coverage match of `prompt` against a cached token list. Returns
    // the prompt index n (> 0) whose prefix [0, n) covers ALL cached tokens,
    // or -1. Token ids can diverge while the text is identical (the model may
    // emit a non-canonical BPE split, e.g. " "+" " where re-tokenization gives
    // "  "), so a token-id mismatch falls back to text-level alignment with
    // the seam on a prompt-token boundary.
    auto match_cov = [&tok](const std::vector<int32_t> & kv, const std::vector<ImgSpan> & kimgs,
                            const std::vector<int32_t> & prompt, const std::vector<ImgSpan> & spans,
                            const char * dbg_tag) -> int {
        if (kv.empty() || prompt.empty()) return -1;
        size_t n = 0;
        const size_t lim = std::min(kv.size(), prompt.size() - 1);   // keep >= 1 tail token
        while (n < lim && kv[n] == prompt[n]) ++n;
        bool covered = n == kv.size();
        if (n > 0 && !covered) {
            std::string skv, sp;
            for (size_t i = n; i < kv.size(); ++i) skv += tok->decode(kv[i]);
            size_t j = n;
            while (j < prompt.size() - 1 && sp.size() < skv.size()) sp += tok->decode(prompt[j++]);
            if (sp == skv) { n = j; covered = true; }   // prompt[0..j) covers all of kv
        }
        if (!covered || n == 0) {
            if (dbg_tag && getenv("QWEN_CACHE_DEBUG"))
                fprintf(stderr, "prefix cache [%s]: matched %zu of kv=%zu prompt=%zu (kv=%d '%s' prompt=%d '%s')\n",
                        dbg_tag, n, kv.size(), prompt.size(),
                        n < kv.size() ? kv[n] : -1,
                        (n < kv.size() ? tok->decode(kv[n]) : "<end>").c_str(),
                        n < prompt.size() ? prompt[n] : -1,
                        (n < prompt.size() ? tok->decode(prompt[n]) : "<end>").c_str());
            return -1;
        }
        for (const auto & s : spans)            // image span straddling the boundary
            if (s.first < (int) n && s.first + s.count > (int) n) return -1;
        std::vector<const ImgSpan *> in;        // images inside the prefix must be unchanged
        for (const auto & s : spans) if (s.first + s.count <= (int) n) in.push_back(&s);
        if (in.size() != kimgs.size()) return -1;
        for (size_t i = 0; i < in.size(); ++i)
            if (in[i]->first != kimgs[i].first || in[i]->count != kimgs[i].count ||
                in[i]->hash != kimgs[i].hash) return -1;
        return (int) n;
    };

    // ---- snapshot slots (RAM or SSD): park evicted states for later reuse ----
    struct Slot {
        std::vector<int32_t> toks;     // tokens of the saved state (match key)
        std::vector<ImgSpan> imgs;
        uint64_t stamp = 0;            // LRU
        std::vector<uint8_t> blob;     // RAM mode
        std::string path;              // SSD mode (file with meta header + state)
    };
    std::vector<Slot> slots((size_t) std::max(0, cache_slots));
    uint64_t slot_clock = 0;
    int live_origin = -1;              // slot the live state was loaded from / saved to
    const bool slots_ssd = !cache_slots_dir.empty();
    // QSL1 retains the existing default payload. QSL2 is deliberately
    // private and incompatible: callers must pin source/model/runtime/context.
    const uint32_t SLOT_MAGIC = complete_cache_slots ? 0x324C5351 : 0x314C5351;
    auto save_slot_payload = [&](const std::function<void(const void *, size_t)> & sink) {
        if (complete_cache_slots) rt->save_complete_state(sink);
        else rt->save_state(sink);
    };
    auto load_slot_payload = [&](const std::function<void(void *, size_t)> & source) {
        if (complete_cache_slots) rt->load_complete_state(source);
        else rt->load_state(source);
    };

    // A trusted, single-server cache directory is required. Startup discovers
    // only slot-N.bin, so an interrupted temporary write is never a slot.
    auto save_complete_disk_slot = [&](const std::string & path,
                                      const std::vector<int32_t> & toks,
                                      const std::vector<ImgSpan> & imgs,
                                      int slot_index) -> bool {
        const std::string temporary = path + ".tmp";
        FILE * f = fopen(temporary.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "prefix cache: complete slot %d temporary open failed\n", slot_index);
            return false;
        }
        bool ok = true;
        try {
            const uint32_t n_toks = (uint32_t) toks.size(), n_imgs = (uint32_t) imgs.size();
            auto write_checked = [&](const void * p, size_t n) {
                if (n && fwrite(p, 1, n, f) != n) throw std::runtime_error("short write");
            };
            write_checked(&SLOT_MAGIC, sizeof(SLOT_MAGIC));
            write_checked(&n_toks, sizeof(n_toks));
            write_checked(&n_imgs, sizeof(n_imgs));
            write_checked(toks.data(), toks.size() * sizeof(int32_t));
            write_checked(imgs.data(), imgs.size() * sizeof(ImgSpan));
            save_slot_payload(write_checked);
            if (fflush(f) != 0) throw std::runtime_error("flush failed");
        } catch (const std::exception & error) {
            fprintf(stderr, "prefix cache: complete slot %d save failed (%s)\n", slot_index, error.what());
            ok = false;
        }
        if (fclose(f) != 0) {
            fprintf(stderr, "prefix cache: complete slot %d close failed\n", slot_index);
            ok = false;
        }
        if (ok) {
#if defined(_WIN32)
            // Unlike remove+rename this retains the prior slot when replacement
            // fails. WRITE_THROUGH requests completion before success is returned.
            const auto from = std::filesystem::path(temporary).wstring();
            const auto to = std::filesystem::path(path).wstring();
            ok = MoveFileExW(from.c_str(), to.c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
            std::error_code error;
            std::filesystem::rename(temporary, path, error);
            ok = !error;
#endif
            if (!ok) fprintf(stderr, "prefix cache: complete slot %d final rename failed\n", slot_index);
        }
        if (!ok) remove(temporary.c_str());
        return ok;
    };

    if (slots_ssd && !slots.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_slots_dir, ec);
        // restore slot metadata from a previous run (state blobs stay on disk
        // and are validated by the runtime header when actually loaded)
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i].path = cache_slots_dir + "/slot-" + std::to_string(i) + ".bin";
            FILE * f = fopen(slots[i].path.c_str(), "rb");
            if (!f) continue;
            uint32_t magic = 0, n_toks = 0, n_imgs = 0;
            bool ok = fread(&magic, 4, 1, f) == 1 && magic == SLOT_MAGIC &&
                      fread(&n_toks, 4, 1, f) == 1 && fread(&n_imgs, 4, 1, f) == 1 &&
                      n_toks < 10u * 1024 * 1024 && n_imgs < 4096;
            if (ok) {
                slots[i].toks.resize(n_toks);
                slots[i].imgs.resize(n_imgs);
                ok = (n_toks == 0 || fread(slots[i].toks.data(), sizeof(int32_t), n_toks, f) == n_toks) &&
                     (n_imgs == 0 || fread(slots[i].imgs.data(), sizeof(ImgSpan), n_imgs, f) == n_imgs);
            }
            fclose(f);
            if (!ok) { slots[i].toks.clear(); slots[i].imgs.clear(); continue; }
            slots[i].stamp = ++slot_clock;
        }
        size_t restored = 0;
        for (const auto & s : slots) restored += !s.toks.empty();
        if (restored) fprintf(stderr, "prefix cache: restored %zu slot(s) from %s\n",
                              restored, cache_slots_dir.c_str());
    }

    // park the live state into a slot (its origin slot, or the LRU one, never
    // `avoid`). No-op if the live state is empty or unchanged since loading.
    auto save_live = [&](int avoid) {
        const auto & lt = rt->kv_tokens();
        if (lt.empty() || slots.empty()) return;
        int s = live_origin;
        if (s < 0 || s == avoid) {
            s = -1;
            uint64_t oldest = UINT64_MAX;
            for (int i = 0; i < (int) slots.size(); ++i) {
                if (i == avoid) continue;
                if (slots[i].toks.empty()) { s = i; break; }
                if (slots[i].stamp < oldest) { oldest = slots[i].stamp; s = i; }
            }
            if (s < 0) return;
        }
        if (live_origin == s && slots[s].toks.size() == lt.size()) {
            slots[s].stamp = ++slot_clock;     // unchanged since load
            return;
        }
        Slot & sl = slots[s];
        if (slots_ssd && complete_cache_slots) {
            // Prepare metadata before committing the new file. Failed
            // writes retain the old complete slot and its match key.
            auto next_toks = lt;
            auto next_imgs = *kv_imgs;
            if (!save_complete_disk_slot(sl.path, next_toks, next_imgs, s)) return;
            sl.toks = std::move(next_toks);
            sl.imgs = std::move(next_imgs);
            sl.stamp = ++slot_clock;
            return;
        }
        sl.toks = lt;
        sl.imgs = *kv_imgs;
        sl.stamp = ++slot_clock;
        if (slots_ssd) {
            FILE * f = fopen(sl.path.c_str(), "wb");
            bool ok = f != nullptr;
            if (ok) {
                const uint32_t n_toks = (uint32_t) sl.toks.size(), n_imgs = (uint32_t) sl.imgs.size();
                ok = fwrite(&SLOT_MAGIC, 4, 1, f) == 1 && fwrite(&n_toks, 4, 1, f) == 1 &&
                     fwrite(&n_imgs, 4, 1, f) == 1 &&
                     (n_toks == 0 || fwrite(sl.toks.data(), sizeof(int32_t), n_toks, f) == n_toks) &&
                     (n_imgs == 0 || fwrite(sl.imgs.data(), sizeof(ImgSpan), n_imgs, f) == n_imgs);
                if (ok) {
                    try { save_slot_payload([&](const void * p, size_t n) {
                              if (fwrite(p, 1, n, f) != n) throw std::runtime_error("short write"); });
                    } catch (const std::exception & e) {
                        fprintf(stderr, "prefix cache: slot %d save failed (%s)\n", s, e.what());
                        ok = false;
                    }
                }
                fclose(f);
            }
            if (!ok) { remove(sl.path.c_str()); sl.toks.clear(); sl.imgs.clear(); }
        } else {
            sl.blob.clear();
            sl.blob.reserve(rt->state_bytes());
            save_slot_payload([&](const void * p, size_t n) {
                const uint8_t * b = (const uint8_t *) p;
                sl.blob.insert(sl.blob.end(), b, b + n);
            });
        }
    };

    // restore a slot into the runtime; on failure the slot is dropped and the
    // runtime is reset. Returns success.
    auto load_slot = [&](int s) -> bool {
        Slot & sl = slots[s];
        try {
            if (slots_ssd) {
                FILE * f = fopen(sl.path.c_str(), "rb");
                if (!f) throw std::runtime_error("slot file missing");
                const long skip = 12 + (long) (sl.toks.size() * sizeof(int32_t))
                                     + (long) (sl.imgs.size() * sizeof(ImgSpan));
                if (fseek(f, skip, SEEK_SET) != 0) { fclose(f); throw std::runtime_error("seek failed"); }
                try {
                    // Recheck the header on every load: the file may have
                    // changed since startup metadata was inspected.
                    if (fseek(f, 0, SEEK_SET) != 0) throw std::runtime_error("seek failed");
                    uint32_t magic = 0, n_toks = 0, n_imgs = 0;
                    if (fread(&magic, 4, 1, f) != 1 || magic != SLOT_MAGIC ||
                        fread(&n_toks, 4, 1, f) != 1 || n_toks != sl.toks.size() ||
                        fread(&n_imgs, 4, 1, f) != 1 || n_imgs != sl.imgs.size())
                        throw std::runtime_error("slot format or metadata mismatch");
                    if (fseek(f, skip, SEEK_SET) != 0) throw std::runtime_error("seek failed");
                    load_slot_payload([&](void * p, size_t n) {
                        if (fread(p, 1, n, f) != n) throw std::runtime_error("short read");
                    });
                    if (complete_cache_slots && fgetc(f) != EOF)
                        throw std::runtime_error("trailing complete slot payload");
                } catch (...) { fclose(f); throw; }
                fclose(f);
            } else {
                size_t off = 0;
                load_slot_payload([&](void * p, size_t n) {
                    if (n > sl.blob.size() - off) throw std::runtime_error("short blob");
                    memcpy(p, sl.blob.data() + off, n);
                    off += n;
                });
                if (complete_cache_slots && off != sl.blob.size())
                    throw std::runtime_error("trailing complete slot payload");
            }
            if (complete_cache_slots && rt->kv_tokens() != sl.toks)
                throw std::runtime_error("complete slot token history mismatch");
        } catch (const std::exception & e) {
            fprintf(stderr, "prefix cache: slot %d load failed (%s), dropping it\n", s, e.what());
            if (slots_ssd) remove(sl.path.c_str());
            sl.toks.clear(); sl.imgs.clear(); sl.blob.clear(); sl.blob.shrink_to_fit();
            rt->reset();
            return false;
        }
        *kv_imgs = sl.imgs;
        sl.stamp = ++slot_clock;
        return true;
    };

    auto prepare_prompt = [&](std::vector<int32_t> & prompt,
                              std::vector<Runtime::EmbdOverride> & ovr,
                              std::vector<ImgSpan> spans, uint64_t owner,
                              const std::shared_ptr<Prog> & prog,
                              lclk::time_point t_recv) -> int {
        const int n_full = (int) prompt.size();
        const bool has_img = !spans.empty();
        int n = match_cov(rt->kv_tokens(), *kv_imgs, prompt, spans, "live");
        const char * src = "live";
        // mid-prompt divergence (e.g. the client rewrote the previous turn,
        // dropping its reasoning): a prompt checkpoint at or before the
        // divergence point lets the live state rewind instead of resetting.
        // Text-only lineages only — images make positions ambiguous.
        int rew = -1;
        if (n < 0 && kv_imgs->empty() && spans.empty()) {
            const auto & kv = rt->kv_tokens();
            size_t d = 0;
            const size_t lim = std::min(kv.size(), prompt.size() - 1);
            while (d < lim && kv[d] == prompt[d]) ++d;
            if (d > 0) rew = rt->best_ckpt((int) d);
        }
        if (n < 0 && !slots.empty()) {
            int best = -1, bcov = -1;
            for (int i = 0; i < (int) slots.size(); ++i) {
                if (slots[i].toks.empty()) continue;
                const int c = match_cov(slots[i].toks, slots[i].imgs, prompt, spans, nullptr);
                if (c > bcov) { bcov = c; best = i; }
            }
            if (best >= 0 && bcov > rew) {     // a fully-covering slot beats a rewind
                save_live(best);               // park the live state before overwriting it
                if (load_slot(best)) { n = bcov; live_origin = best; src = "slot"; rew = -1; }
            } else if (rew <= 0) {
                save_live(-1);                 // about to reset: park the live state
            }
        }
        if (n < 0 && rew > 0) {
            const int r = rt->rewind_to(rew);
            if (r > 0) { n = r; src = "rewind"; }
        }
        if (n < 0) { rt->reset(); live_origin = -1; n = 0; }
        if (n > 0) {
            prompt.erase(prompt.begin(), prompt.begin() + n);
            std::vector<Runtime::EmbdOverride> tail;
            for (const auto & o : ovr)
                if (o.first >= (int) n) tail.push_back({ o.first - (int) n, o.count, o.data });
            ovr = std::move(tail);
        }
        *kv_imgs = std::move(spans);
        live_owner = owner;
        // request start: time, prompt size, KV reuse, image flag, prefill size
        fprintf(stderr, "[%s] req: %d tok%s, KV reuse %d (%s), prefill %d%s\n",
                clock_hms().c_str(), n_full, has_img ? " +img" : "",
                n, n > 0 ? src : "none", (int) prompt.size(),
                rt->has_expert_cache() ? "" : " (resident)");
        // a request that sat in the queue gets an explicit 0% marker, so the
        // gap between its recv: line and the actual prefill start is visible
        if (!prompt.empty() && lclk::now() - t_recv > std::chrono::seconds(1))
            fprintf(stderr, "  [%s] prefill 0/%d (0%%)\n", clock_hms().c_str(), (int) prompt.size());
        prog->start = prog->last = lclk::now();
        prog->total = (int) prompt.size();   // full tail to prefill (overall progress)
        prog->base  = 0;
        prog->shown = false;
        *cur_prog   = prog;
        return n;
    };

    // Heartbeat on/off (--heartbeat off / QWEN_HEARTBEAT=0): A/B switch.
    // Off restores the pre-heartbeat behavior exactly — blocking lock waits
    // and no bytes written between prefill chunks (disconnects are then only
    // noticed via is_writable, and byte-silent waits may hit client timeouts).
    const bool hb_on = [] {
        const char * c = getenv("QWEN_HEARTBEAT");
        return !(c && (strcmp(c, "0") == 0 || strcmp(c, "off") == 0));
    }();
    if (!hb_on) fprintf(stderr, "heartbeat: OFF (QWEN_HEARTBEAT)\n");

    // Can the runtime be handed to a waiting request? Without cache slots the
    // evicted state cannot be parked, and the switch costs a full re-prefill
    // on BOTH sides every slice (observed as catastrophic slowdown with
    // concurrent requests, issue #2) — run back-to-back instead.
    auto yieldable = [&]() -> bool {
        if (!tlock.contended()) return false;
        if (!slots.empty()) return true;
        static std::atomic<bool> once{false};
        if (!once.exchange(true))
            fprintf(stderr, "note: request queued mid-run; without --cache-slots there is "
                            "nowhere to park the live state, so requests run back-to-back\n");
        return false;
    };

    // Optional benchmark diagnostics. These export existing access counts;
    // they never clear them, alter routing, or prefetch experts.
    auto save_routing = [&rt](int prompt_tokens, const char * phase) {
        const char * directory = getenv("QWEN_ROUTING_PROFILE_DIR");
        if (!directory || !*directory || !rt->has_expert_cache()) return;
        rt->save_expert_profile(std::string(directory) + "/prompt-" +
            std::to_string(prompt_tokens) + "-" + phase);
    };

    // completion log: prefill/decode tok/s, output tokens, expert hit-rate
    // delta since this request's prefill (c0 snapshot taken at prefill start).
    auto log_done = [&rt, &save_routing](int n_prompt, int n_cached, int gen,
                          double prefill_ms, double gen_ms, Runtime::CacheStats c0) {
        std::string hit;
        if (rt->has_expert_cache()) {
            const auto c = rt->cache_stats();
            const uint64_t h = c.hits - c0.hits, m = c.misses - c0.misses, tot = h + m;
            char b[64];
            std::snprintf(b, sizeof(b), ", expert hit %.1f%% (%llu/%llu)",
                          tot ? 100.0 * h / tot : 0.0,
                          (unsigned long long) h, (unsigned long long) tot);
            hit = b;
        }
        save_routing(n_prompt, "decode");
        const int pf = n_prompt - n_cached;
        fprintf(stderr, "[%s] done: prefill %.0f tok/s (%d tok), gen %.1f tok/s (%d tok)%s\n",
                clock_hms().c_str(),
                prefill_ms > 0 ? pf * 1000.0 / prefill_ms : 0.0, pf,
                gen_ms > 0 ? gen * 1000.0 / gen_ms : 0.0, gen, hit.c_str());
    };

    // re-attach a time-sliced request to the runtime after a pause: continue
    // if the live state is still ours, else restore our parked slot, else
    // (slot got evicted) rebuild the exact state by re-prefilling our token
    // snapshot. Returns false only when the state is unrecoverable (images:
    // the embedding overrides are gone, a re-prefill would be wrong).
    auto ensure_state = [&](uint64_t owner, const std::vector<int32_t> & my_toks,
                            const std::vector<ImgSpan> & my_imgs,
                            bool had_images, bool & need_reprefill) -> bool {
        need_reprefill = false;
        if (live_owner == owner) return true;
        int s = -1;
        for (int i = 0; i < (int) slots.size(); ++i)
            if (!slots[i].toks.empty() && slots[i].toks == my_toks) { s = i; break; }
        if (s >= 0) {
            save_live(s);
            if (load_slot(s)) {
                live_origin = s;
                live_owner  = owner;
                return true;
            }
        }
        if (had_images) return false;          // overrides gone: cannot rebuild
        save_live(-1);
        rt->reset();
        live_origin = -1;
        live_owner  = owner;
        *kv_imgs    = my_imgs;
        need_reprefill = true;                 // caller re-decodes my_toks
        return true;
    };

    auto make_sampler = [repeat_penalty_default](const json & body) {
        SamplerConfig sc;
        sc.temperature = body.value("temperature", 0.7f);
        sc.top_p       = body.value("top_p", 0.95f);
        sc.top_k       = body.value("top_k", 40);
        sc.seed        = body.value("seed", 0);
        // repetition control (plain decode; MTP is greedy and ignores sampling)
        sc.repeat_penalty    = body.value("repetition_penalty",
                               body.value("repeat_penalty", repeat_penalty_default));
        sc.presence_penalty  = body.value("presence_penalty", 0.0f);
        sc.frequency_penalty = body.value("frequency_penalty", 0.0f);
        sc.repeat_last_n     = body.value("repeat_last_n", 64);
        return sc;
    };

    httplib::Server srv;

    srv.Get("/", [reasoning_default](const httplib::Request &, httplib::Response & res) {
        std::string html = CHAT_HTML;
        // initialise the UI "think" checkbox from the server default
        if (!reasoning_default) {
            const std::string from = "id=\"think\" checked";
            auto p = html.find(from);
            if (p != std::string::npos) html.replace(p, from.size(), "id=\"think\"");
        }
        res.set_content(html, "text/html; charset=utf-8");
    });

    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        if (!upstream_chat->available()) {
            res.status=503;
            res.set_content(json{{"status","error"},{"error","Upstream chat helper unavailable; reload model runtime"}}.dump(),"application/json");
            return;
        }
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object", "list"}, {"data", json::array({
            {{"id", model_id}, {"object", "model"}, {"owned_by", "questwend"},
             {"display", model_display}}
        })}};
        res.set_content(j.dump(), "application/json");
    });

#if QW_EXPERT_VISUALIZATION
    register_expert_debug_routes(srv, rt, tlock, model);
#endif

    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content(R"({"error":"invalid json"})", "application/json"); return; }
        if (getenv("QWEN_LOG_RESPONSE"))
            fprintf(stderr, "=== request body ===\n%s\n", req.body.c_str());

        // Omitted (or <= 0) max_tokens means "generate until the context runs
        // out", matching llama.cpp's server: the caller sets a budget only when
        // it wants one. n_ctx is a safe cap because the generation loops also
        // stop on `n_past + 1 >= n_ctx`, which always trips first; both paths
        // then report finish_reason "length". OpenAI's newer spelling
        // max_completion_tokens is accepted as an alias.
        int max_tokens = body.value("max_tokens", body.value("max_completion_tokens", 0));
        if (max_tokens <= 0) max_tokens = n_ctx;
        const bool stream      = body.value("stream", false);
        const SamplerConfig sc = make_sampler(body);
        const std::string id   = now_id("chatcmpl-");
        const long long created = (long long) std::time(nullptr);

        // per-request thinking control: body "reasoning" overrides the server default
        bool reasoning = reasoning_default;
        if (body.contains("reasoning")) reasoning = body["reasoning"].get<bool>();
        else if (body.contains("enable_thinking")) reasoning = body["enable_thinking"].get<bool>();

        struct PreparedPrompt { std::vector<int32_t> ids; int gen_prompt_begin = 0; } cp;
        std::shared_ptr<open_agents::ChatTurn> chat_turn;
        std::vector<Runtime::EmbdOverride> ovr;
        std::vector<ImgSpan> spans;
        try {
            for (const auto & message : body.at("messages")) {
                const auto content = message.value("content", json());
                if (!content.is_array()) continue;
                for (const auto & part : content) {
                    if (part.value("type", std::string()) != "text")
                        throw std::runtime_error("This runtime supports text input only");
                }
            }
            json chat_request = body;
            json kwargs = body.value("chat_template_kwargs", json::object());
            if (!kwargs.is_object()) throw std::runtime_error("chat_template_kwargs must be an object");
            reasoning = kwargs.value("enable_thinking", reasoning);
            std::string effort = kwargs.value("reasoning_effort", body.value("reasoning_effort", std::string("xhigh")));
            if (effort == "high") effort = "xhigh";
            kwargs["enable_thinking"] = reasoning;
            kwargs["preserve_thinking"] = kwargs.value("preserve_thinking", true);
            kwargs["reasoning_effort"] = effort;
            chat_request["enable_thinking"] = reasoning;
            chat_request["chat_template_kwargs"] = std::move(kwargs);
            chat_turn = std::make_shared<open_agents::ChatTurn>(*upstream_chat, chat_request);
            cp.ids = chat_turn->prepared.at("tokens").get<std::vector<int32_t>>();
            cp.gen_prompt_begin = chat_turn->prepared.at("gen_prompt_begin").get<int>();
            if (getenv("QWEN_LOG_RESPONSE"))
                fprintf(stderr, "=== rendered prompt (%zu tok) ===\n%s\n", cp.ids.size(), tok->decode(cp.ids).c_str());
        } catch (const std::exception & error) {
            res.status = upstream_chat->available() ? 400 : 503;
            res.set_content(json{{"error", error.what()}}.dump(), "application/json");
            return;
        }

        std::vector<int32_t> prompt = cp.ids;
        const int n_prompt_full = (int) prompt.size();
        const bool req_mtp = mtp;

        // arrival log: the "req:" line below only appears once the runtime is
        // acquired, which can be minutes later behind a running request
        const auto t_recv = lclk::now();
        fprintf(stderr, "[%s] recv: %d tok%s%s\n", clock_hms().c_str(), n_prompt_full,
                "", tlock.busy() ? " (queued)" : "");

        // the prompt must fit the context window with room to generate
        if (n_prompt_full >= n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "prompt is " + std::to_string(n_prompt_full) +
                " tokens but the context window is " + std::to_string(n_ctx) +
                " (start the server with a larger --n-ctx)"}}.dump(), "application/json");
            return;
        }

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");

            using clk = std::chrono::steady_clock;
            struct State {
                Sampler smp;
                TicketLock * tl = nullptr;
                bool holding = false;             // we own the runtime lock
                std::vector<int32_t> prompt;
                std::vector<float> logits;
                std::shared_ptr<open_agents::ChatTurn> chat_turn;
                std::string upstream_pending;
                std::string generation_error;
                std::string text;   // accumulated output (for tool-call parsing)
                bool role_sent = false;
                std::shared_ptr<Prog> prog = std::make_shared<Prog>();   // prefill progress
                lclk::time_point t_recv;          // arrival (for the queued 0% marker)
                std::string hb;                   // heartbeat: empty-delta chunk (see below)
                std::vector<Runtime::EmbdOverride> ovr;
                std::vector<ImgSpan> spans;       // image spans for the prefix cache
                uint64_t sid = 0;                 // request id (live-state ownership)
                Runtime::CacheStats cstat0;       // expert-cache snapshot at prefill start
                bool had_images = false;
                int slice_used = 0;               // tokens emitted in the current slice
                std::vector<int32_t> my_toks;     // kv snapshot at pause (resume key)
                std::vector<ImgSpan> my_imgs;
                size_t resp_base = 0;             // kv index where the response starts (MTP)
                size_t n_sent = 0;                // response tokens delivered (MTP)
                int32_t pending = -1;             // next undecoded token after a pause (MTP)
                bool pending_valid = false;
                size_t pf_pos = 0;                // prompt tokens prefilled so far
                int gen_begin = -1;               // generation-prompt start (abs. index)
                bool pf_done = false;
                bool paused = false, done = false;
                int n_cached = 0;                 // prompt tokens reused from the cache
                int generated = 0, max_tokens = 0, prompt_tokens = 0;
                bool started = false, finished = false;
                clk::time_point t0, t_prefill, t_first;
                clk::time_point gen_log;          // last long-output progress line
                clk::time_point last_output = clk::now(); // also bound silence while buffering tool arguments
                State(const SamplerConfig & sc) : smp(sc) {}
                ~State() { if (holding && tl) tl->unlock(); }   // client disconnect mid-slice
            };
            auto st = std::make_shared<State>(sc);
            st->tl = &tlock;
            st->sid = sid_gen.fetch_add(1);
            st->prompt = prompt;
            st->max_tokens = max_tokens;
            st->prompt_tokens = (int) prompt.size();
            st->ovr   = std::move(ovr);
            st->spans = std::move(spans);
            st->chat_turn = chat_turn;
            st->gen_begin = cp.gen_prompt_begin;
            st->t_recv = t_recv;
            // Heartbeat sent while queued / between prefill chunks so client
            // body-idle timeouts don't kill a byte-silent stream. An OpenAI
            // chunk with an empty delta (not an SSE comment): every client
            // parses it and applies nothing.
            st->hb = "data: " + dumpj(json{{"id", id}, {"object", "chat.completion.chunk"},
                {"created", created}, {"model", model_id}, {"choices", json::array({
                    {{"index", 0}, {"delta", json::object()}, {"finish_reason", nullptr}}})}}) + "\n\n";
            // the generation prompt pre-opens <think> when reasoning is on, so
            // the stream starts inside the reasoning block

            st->smp.prime(prompt);   // repetition-penalty window (prompt tail)

            res.set_chunked_content_provider("text/event-stream",
                [&, st, id, created, req_mtp, n_draft](size_t, httplib::DataSink & sink) -> bool {
                    if (st->finished) return false;
                    auto release_runtime = [&]() {
                        if (!st->holding) return;
                        expert_debug_hooks::set_phase(ExpertDebug::Idle);
                        tlock.unlock();
                        st->holding = false;
                    };
                    auto finish = [&]() -> bool {
                        release_runtime();
                        const auto t_end = clk::now();
                        auto ms = [](clk::time_point a, clk::time_point b) {
                            return std::chrono::duration<double, std::milli>(b - a).count();
                        };
                        const double prefill_ms = ms(st->t0, st->t_prefill);
                        const double gen_ms     = ms(st->t_prefill, t_end);
                        const int n_prefilled = st->prompt_tokens - st->n_cached;
                        json timings = {
                            {"prompt_tokens",     st->prompt_tokens},
                            {"cached_tokens",     st->n_cached},
                            {"completion_tokens", st->generated},
                            {"ttft_ms",     ms(st->t0, st->t_first)},
                            {"prefill_tps", prefill_ms > 0 ? n_prefilled  * 1000.0 / prefill_ms : 0.0},
                            {"gen_tps",     gen_ms     > 0 ? st->generated * 1000.0 / gen_ms    : 0.0},
                        };
                        log_done(st->prompt_tokens, st->n_cached, st->generated,
                                 prefill_ms, gen_ms, st->cstat0);
                        // tool calls: parse the accumulated text; if any, emit them as a
                        // final delta and finish with "tool_calls"
                        std::vector<ParsedToolCall> calls;
                        std::string parse_error;
                        json final_delta=json::object();
                        try {
                            if (!st->generation_error.empty()) throw std::runtime_error(st->generation_error);
                            const bool length = st->generated >= st->max_tokens || st->prompt_tokens + st->generated + 1 >= n_ctx;
                            const auto parsed = st->chat_turn->finish(st->upstream_pending, length ? "length" : "stop");
                            for (const char * key : {"content","reasoning_content"}) {
                                const auto before=st->chat_turn->previous.value(key,std::string());
                                const auto after=parsed.value(key,std::string());
                                if (after.compare(0,before.size(),before)!=0)
                                    throw std::runtime_error("Final upstream parse rewrote streamed text");
                                if (after.size()>before.size()) final_delta[key]=after.substr(before.size());
                            }
                            for (const auto & call : parsed.value("tool_calls", json::array()))
                                calls.push_back({call.at("function").at("name"), call.at("function").at("arguments")});
                        } catch (const std::exception & error) { parse_error=error.what(); }
                        if (!parse_error.empty()) {
                            fprintf(stderr, "invalid tool call: %s\n", parse_error.c_str());
                            const json error = {{"error", {{"message", "Invalid model tool call: " + parse_error},
                                {"type", "invalid_tool_call"}, {"code", "invalid_tool_call"}}}};
                            const std::string event = "data: " + dumpj(error) + "\n\ndata: [DONE]\n\n";
                            sink.write(event.data(), event.size());
                            st->finished = true;
                            sink.done();
                            return false;
                        }
                        if (getenv("QWEN_LOG_RESPONSE")) {
                            fprintf(stderr, "=== raw response ===\n%s\n=== parsed %zu tool call(s) ===\n",
                                    st->text.c_str(), calls.size());
                            for (auto & c : calls)
                                fprintf(stderr, "  %s args=%s\n", c.name.c_str(), c.arguments.c_str());
                        }
                        std::string ev;
                        if (!final_delta.empty()) {
                            if (!st->role_sent) { final_delta["role"]="assistant"; st->role_sent=true; }
                            json chunk={{"id",id},{"object","chat.completion.chunk"},{"created",created},
                                {"model",model_id},{"choices",json::array({{{"index",0},{"delta",final_delta},{"finish_reason",nullptr}}})}};
                            ev+="data: "+dumpj(chunk)+"\n\n";
                        }
                        // flush withheld text: a partial marker that never completed,
                        // or withheld tool XML that turned out to be unparseable
                        // (parsed tool XML stays out of the content stream)

                        // "length" when generation was cut by the token budget or
                        // the context limit (e.g. a tool call's big content arg got
                        // truncated before its closing tags, so it could not be
                        // parsed) -> clients retry/extend instead of treating a
                        // partial reply as complete.
                        const bool truncated = calls.empty() &&
                            (st->generated >= st->max_tokens || st->prompt_tokens + st->generated + 1 >= n_ctx);
                        const char * fr = truncated ? "length" : "stop";
                        if (!calls.empty()) {
                            json tcs = json::array();
                            for (size_t k = 0; k < calls.size(); ++k)
                                tcs.push_back({{"index", (int) k}, {"id", id + "-tc" + std::to_string(k)},
                                               {"type", "function"},
                                               {"function", {{"name", calls[k].name},
                                                             {"arguments", calls[k].arguments}}}});
                            json delta = {{"tool_calls", tcs}};
                            if (!st->role_sent) { delta["role"] = "assistant"; st->role_sent = true; }
                            json tchunk = {{"id", id}, {"object", "chat.completion.chunk"},
                                {"created", created}, {"model", model_id}, {"choices", json::array({
                                    {{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}})}};
                            ev += "data: " + dumpj(tchunk) + "\n\n";
                            fr = "tool_calls";
                        }
                        json fin = {{"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id}, {"timings", timings},
                            {"usage", {{"prompt_tokens", st->prompt_tokens},
                                       {"completion_tokens", st->generated},
                                       {"total_tokens", st->prompt_tokens + st->generated}}},
                            {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", fr}}})}};
                        ev += "data: " + dumpj(fin) + "\n\ndata: [DONE]\n\n";
                        sink.write(ev.data(), ev.size());
                        st->finished = true;
                        sink.done();
                        return false;
                    };
                    auto emit_piece = [&](int32_t t) -> bool {   // SSE chunk(s) for one token
                        if (st->generated == 0) {
                            st->t_first = clk::now();
                            st->gen_log = st->t_first;
                        } else if (clk::now() - st->gen_log >= std::chrono::seconds(10)) {
                            // long-output progress, throttled like the prefill lines
                            const double el = std::chrono::duration<double>(clk::now() - st->t_first).count();
                            fprintf(stderr, "  [%s] gen %d tok, %.1f tok/s\n",
                                    clock_hms().c_str(), st->generated, el > 0 ? st->generated / el : 0.0);
                            st->gen_log = clk::now();
                        }
                        const std::string piece = tok->decode(t);
                        st->text += piece;
                        const auto complete = utf8_clip(st->upstream_pending, piece);
                        const auto delta = complete.empty() ? json::object() : st->chat_turn->feed(complete);
                        std::string ev;
                        if (!delta.empty()) {
                            json routed=delta;
                            if (!st->role_sent) { routed["role"]="assistant"; st->role_sent=true; }
                            json chunk={{"id",id},{"object","chat.completion.chunk"},{"created",created},
                                {"model",model_id},{"choices",json::array({{{"index",0},{"delta",routed},{"finish_reason",nullptr}}})}};
                            ev="data: "+dumpj(chunk)+"\n\n";
                        } else {
                            if (!hb_on || clk::now()-st->last_output<std::chrono::seconds(10)) return true;
                            ev=st->hb;
                        }
                        st->last_output=clk::now();
                        return sink.write(ev.data(),ev.size());
                    };
                    try {
                        const bool was_holding = st->holding;
                        if (!st->holding) {
                            // Waiting for the runtime can take minutes behind a long
                            // request. An SSE stream that stays byte-silent that long
                            // gets killed by client body-idle timeouts (Node/undici
                            // defaults to 300 s -> OpenCode reports "terminated"), so
                            // heartbeat (empty-delta chunk, st->hb) every 10 s; a
                            // failed write doubles as disconnect detection while
                            // still queued. QWEN_HEARTBEAT=0 restores the plain
                            // blocking wait (pre-heartbeat behavior, for A/B tests).
                            if (hb_on) {
                                while (!tlock.lock_for(std::chrono::seconds(10))) {
                                    if (!sink.is_writable() || !sink.write(st->hb.data(), st->hb.size())) {
                                        fprintf(stderr, "[%s] client gone while queued, dropping request\n", clock_hms().c_str());
                                        st->finished = true;
                                        return false;
                                    }
                                }
                            } else {
                                tlock.lock();
                            }
                            st->holding = true;
                        }
                        if (!was_holding) {
                            st->slice_used = 0;   // fresh slice on (re)acquisition
                            *cur_prog = st->prog; // progress lines report this request
                        }
                        expert_debug_hooks::set_phase(
                            st->pf_done || st->generated > 0 ? ExpertDebug::Decode : ExpertDebug::Prefill);

                        // clip the image-embedding overrides to a prompt chunk
                        // [pos, pos+n), shifting indices to be chunk-relative
                        const int ne = (int) model->hparams().n_embd;
                        auto chunk_ovr = [&](size_t pos, size_t n) {
                            std::vector<Runtime::EmbdOverride> covr;
                            for (const auto & o : st->ovr) {
                                const int lo = std::max(o.first, (int) pos);
                                const int hi = std::min(o.first + o.count, (int) (pos + n));
                                if (lo < hi) covr.push_back({ lo - (int) pos, hi - lo,
                                                              o.data + (size_t) (lo - o.first) * ne });
                            }
                            return covr;
                        };
                        // prefill slice unit: token slices would be too fine (batch
                        // efficiency), so prefill works in coarser chunks; a client
                        // disconnect / a yield to a queued request happens between
                        // chunks instead of after the whole prefill. With expert
                        // offload, per-chunk expert fetch is amortized over the chunk
                        // (layer-major prefill), so bigger chunks = less streaming:
                        // 4096 by default, QWEN_PF_CHUNK / --pf-chunk to tune.
                        // Deliberately NOT tied to --time-slice: shrinking prefill
                        // chunks to the slice length would destroy the layer-major
                        // fetch amortization on the offload tiers.
                        static const size_t pf_chunk = []{
                            const char * c = getenv("QWEN_PF_CHUNK");
                            const int v = c ? atoi(c) : 0;
                            return (size_t) (v >= 1 ? v : 4096);
                        }();

                        // ---- MTP self-speculative decode: one time slice per provider
                        // call (the whole run when --time-slice is off / uncontended) ----
                        if (req_mtp) {
                            if (!st->started) {
                                st->started = true;
                                st->t0 = clk::now();
                                st->had_images = !st->ovr.empty();
                                st->n_cached = prepare_prompt(st->prompt, st->ovr, std::move(st->spans), st->sid, st->prog, st->t_recv);
                                st->cstat0 = rt->cache_stats();
                                st->my_imgs = *kv_imgs;
                                st->resp_base = rt->kv_tokens().size() + st->prompt.size();
                                st->t_prefill = st->t_first = st->t0;   // defaults if 0 tokens
                            } else if (!was_holding) {
                                bool reprefill = false;
                                if (!ensure_state(st->sid, st->my_toks, st->my_imgs, st->had_images, reprefill)) {
                                    fprintf(stderr, "time-slice: state lost (image request), truncating stream\n");
                                    return finish();
                                }
                                if (reprefill) {
                                    expert_debug_hooks::set_phase(ExpertDebug::Prefill);
                                    rt->prefill(st->my_toks, true);   // exact rebuild incl nextn KV
                                }
                            }

                            // ---- preemptible prefill: all but the last chunk; the
                            // final chunk goes through generate_mtp below ----
                            while (st->prompt.size() - st->pf_pos > pf_chunk) {
                                // heartbeat between chunks: keeps client body-idle
                                // timeouts at bay; a failed write = client gone
                                if (!sink.is_writable() || (hb_on && !sink.write(st->hb.data(), st->hb.size()))) {
                                    fprintf(stderr, "[%s] client gone during prefill (%zu/%zu), aborting\n", clock_hms().c_str(),
                                            st->pf_pos, st->prompt.size());
                                    st->finished = true; release_runtime();
                                    return false;
                                }
                                st->prog->base = (int) st->pf_pos;   // overall-progress offset
                                rt->set_embd_overrides(chunk_ovr(st->pf_pos, pf_chunk));
                                rt->prefill(std::vector<int32_t>(st->prompt.begin() + st->pf_pos,
                                                                 st->prompt.begin() + st->pf_pos + pf_chunk), true);
                                rt->snapshot_ckpt();           // rewind point at the chunk boundary
                                st->pf_pos += pf_chunk;
                                if (yieldable()) {                       // yield mid-prefill
                                    st->my_toks = rt->kv_tokens();
                                    st->my_imgs = *kv_imgs;
                                    if (getenv("QWEN_CACHE_DEBUG"))
                                        fprintf(stderr, "[%s] time-slice: prefill yield at %zu/%zu (sid=%llu)\n", clock_hms().c_str(),
                                                st->pf_pos, st->prompt.size(), (unsigned long long) st->sid);
                                    release_runtime();
                                    return true;
                                }
                            }
                            // checkpoint at the generation-prompt boundary: a future
                            // request that rewrites this turn's output (e.g. drops
                            // the reasoning) re-tokenizes identically up to here
                            const int gen_rel = st->gen_begin < 0 ? -1 : st->gen_begin - st->n_cached;
                            if (gen_rel > (int) st->pf_pos && gen_rel < (int) st->prompt.size()) {
                                st->prog->base = (int) st->pf_pos;
                                rt->set_embd_overrides(chunk_ovr(st->pf_pos, gen_rel - st->pf_pos));
                                rt->prefill(std::vector<int32_t>(st->prompt.begin() + st->pf_pos,
                                                                 st->prompt.begin() + gen_rel), true);
                                rt->snapshot_ckpt();
                                st->pf_pos = gen_rel;
                            }
                            std::vector<int32_t> rp;             // tokens to feed generate_mtp
                            if (st->pf_pos < st->prompt.size()) {
                                st->prog->base = (int) st->pf_pos;     // overall-progress offset
                                rp.assign(st->prompt.begin() + st->pf_pos, st->prompt.end());
                                rt->set_embd_overrides(chunk_ovr(st->pf_pos, rp.size()));
                                st->pf_pos = st->prompt.size();
                            }

                            // deliver confirmed tokens a mid-cycle pause left unsent, then
                            // the pending (undecoded) one; they precede anything new
                            const std::vector<int32_t> & src = rt->kv_tokens();
                            for (size_t i = st->resp_base + st->n_sent; i < src.size(); ++i) {
                                const int32_t t = src[i];
                                if (is_stop(t)) return finish();
                                if (!emit_piece(t)) return finish();    // client gone
                                ++st->n_sent; ++st->slice_used;
                                if (++st->generated >= st->max_tokens) return finish();
                            }
                            if (st->pending_valid) {
                                const int32_t t = st->pending;
                                st->pending_valid = false;
                                if (is_stop(t) || rt->n_past() + 1 >= n_ctx) return finish();
                                if (!emit_piece(t)) return finish();
                                ++st->n_sent; ++st->slice_used;
                                if (++st->generated >= st->max_tokens) return finish();
                                rp.push_back(t);                        // decode it as the tail
                            } else if (st->started && rp.empty()) {
                                return finish();                        // nothing to resume from
                            }

                            bool first = !was_holding && st->generated == 0;
                            int32_t pending = -1;
                            st->paused = st->done = false;
                            // checkpoint once the prompt is fully in KV: the next
                            // request rewinds here if the client rewrites this turn
                            const bool ckpt = st->generated == 0;
                            rt->generate_mtp(rp, st->max_tokens - st->generated, n_draft,
                                             [&](int32_t t) -> bool {
                                if (is_stop(t) || rt->n_past() + 1 >= n_ctx) { st->done = true; return false; }
                                if (time_slice > 0 && st->slice_used >= time_slice) {
                                    if (yieldable()) { st->paused = true; return false; }
                                    st->slice_used = 0;     // nobody waiting: fresh slice
                                }
                                if (first) { st->t_prefill = st->t_first = clk::now(); first = false; }
                                if (!emit_piece(t)) { st->done = true; return false; }   // client gone
                                ++st->slice_used; ++st->n_sent;
                                st->generated++;
                                return st->generated < st->max_tokens;
                            }, &pending, ckpt);

                            if (st->paused && !st->done) {              // yield the runtime
                                st->pending = pending;
                                st->pending_valid = true;
                                st->my_toks = rt->kv_tokens();
                                st->my_imgs = *kv_imgs;
                                if (getenv("QWEN_CACHE_DEBUG"))
                                    fprintf(stderr, "[%s] time-slice: yield at %d tokens (sid=%llu)\n", clock_hms().c_str(),
                                            st->generated, (unsigned long long) st->sid);
                                release_runtime();
                                return true;                            // resumed on the next call
                            }
                            return finish();
                        }

                        // ---- plain decode: one token per provider call ----
                        if (!st->started) {
                            st->started = true;
                            st->t0 = clk::now();
                            st->had_images = !st->ovr.empty();
                            st->n_cached = prepare_prompt(st->prompt, st->ovr, std::move(st->spans), st->sid, st->prog, st->t_recv);
                            st->cstat0 = rt->cache_stats();
                            st->my_imgs = *kv_imgs;
                        } else if (!was_holding) {
                            // resumed after yielding the runtime at a slice boundary
                            bool reprefill = false;
                            if (!ensure_state(st->sid, st->my_toks, st->my_imgs, st->had_images, reprefill)) {
                                fprintf(stderr, "time-slice: state lost (image request), truncating stream\n");
                                return finish();
                            }
                            if (reprefill) {
                                expert_debug_hooks::set_phase(ExpertDebug::Prefill);
                                st->logits = rt->decode(st->my_toks);
                            }
                        }
                        // preemptible prefill: the last chunk's logits feed sampling
                        while (st->pf_pos < st->prompt.size()) {
                            // heartbeat between chunks: keeps client body-idle
                            // timeouts at bay; a failed write = client gone
                            if (!sink.is_writable() || (hb_on && !sink.write(st->hb.data(), st->hb.size()))) {
                                fprintf(stderr, "[%s] client gone during prefill (%zu/%zu), aborting\n", clock_hms().c_str(),
                                        st->pf_pos, st->prompt.size());
                                st->finished = true; release_runtime();
                                return false;
                            }
                            size_t n = std::min(pf_chunk, st->prompt.size() - st->pf_pos);
                            // split at the generation-prompt boundary so a snapshot
                            // lands exactly where a rewritten last turn diverges
                            const int gen_rel = st->gen_begin < 0 ? -1 : st->gen_begin - st->n_cached;
                            if (gen_rel > (int) st->pf_pos && gen_rel < (int) (st->pf_pos + n))
                                n = (size_t) gen_rel - st->pf_pos;
                            st->prog->base = (int) st->pf_pos;   // overall-progress offset
                            rt->set_embd_overrides(chunk_ovr(st->pf_pos, n));
                            st->logits = rt->decode(std::vector<int32_t>(st->prompt.begin() + st->pf_pos,
                                                                         st->prompt.begin() + st->pf_pos + n));
                            rt->snapshot_ckpt();   // rewind point (incl. prompt end)
                            st->pf_pos += n;
                            if (st->pf_pos < st->prompt.size() && yieldable()) {
                                st->my_toks = rt->kv_tokens();           // yield mid-prefill
                                st->my_imgs = *kv_imgs;
                                if (getenv("QWEN_CACHE_DEBUG"))
                                    fprintf(stderr, "[%s] time-slice: prefill yield at %zu/%zu (sid=%llu)\n", clock_hms().c_str(),
                                            st->pf_pos, st->prompt.size(), (unsigned long long) st->sid);
                                release_runtime();
                                return true;
                            }
                        }
                        if (!st->pf_done) {
                            st->pf_done = true;
                            save_routing(st->prompt_tokens, "prefill");
                            st->t_prefill = clk::now();
                            st->t_first   = st->t_prefill;   // overwritten on the first emitted token
                        }
                        expert_debug_hooks::set_phase(ExpertDebug::Decode);
                        int next = st->smp.sample(st->logits);
                        st->smp.accept(next);   // repetition-penalty window
                        // stop on EOS, token budget, or context limit (avoids overflow crash)
                        if (is_stop(next) || st->generated >= st->max_tokens || rt->n_past() + 1 >= n_ctx)
                            return finish();
                        if (!emit_piece(next)) { st->finished = true; release_runtime(); return false; }
                        st->generated++;
                        st->logits = rt->decode({ next });
                        if (time_slice > 0 && ++st->slice_used >= time_slice && yieldable()) {
                            st->my_toks = rt->kv_tokens();   // yield the runtime
                            st->my_imgs = *kv_imgs;
                            if (getenv("QWEN_CACHE_DEBUG"))
                                fprintf(stderr, "[%s] time-slice: yield at %d tokens (sid=%llu)\n", clock_hms().c_str(),
                                        st->generated, (unsigned long long) st->sid);
                            release_runtime();
                        }
                        return true;
                    } catch (const std::exception & e) {
                        fprintf(stderr, "stream error: %s\n", e.what());
                        st->generation_error=e.what();
                        if (st->holding) { rt->reset(); live_owner = 0; }   // state unknown
                        return finish();
                    }
                });
            return;
        }

        // non-streaming
        std::string text;
        int generated = 0, n_cached = 0;
        Runtime::CacheStats cstat0;
        using clk = std::chrono::steady_clock;
        clk::time_point t0, t_prefill = {}, t_end = {};
        try {
            struct Guard {
                TicketLock & t;
                Guard(TicketLock & t_) : t(t_) {
                    t.lock();
                    expert_debug_hooks::set_phase(ExpertDebug::Prefill);
                }
                ~Guard() {
                    expert_debug_hooks::set_phase(ExpertDebug::Idle);
                    t.unlock();
                }
            } lk(tlock);
            auto prog = std::make_shared<Prog>();
            n_cached = prepare_prompt(prompt, ovr, std::move(spans), 0, prog, t_recv);
            cstat0 = rt->cache_stats();
            t0 = clk::now();
            // checkpoint at the generation-prompt boundary (see streaming path)
            const int gen_rel = cp.gen_prompt_begin < 0 ? -1 : cp.gen_prompt_begin - n_cached;
            if (req_mtp) {
                t_prefill = t0;
                if (!ovr.empty()) rt->set_embd_overrides(ovr);   // vision
                if (gen_rel > 0 && gen_rel < (int) prompt.size() && ovr.empty()) {
                    rt->prefill(std::vector<int32_t>(prompt.begin(), prompt.begin() + gen_rel), true);
                    rt->snapshot_ckpt();
                    prompt.erase(prompt.begin(), prompt.begin() + gen_rel);
                }
                bool first = true;
                clk::time_point glog = clk::now();
                rt->generate_mtp(prompt, max_tokens, n_draft, [&](int32_t t) -> bool {
                    if (is_stop(t) || rt->n_past() + 1 >= n_ctx) return false;
                    if (first) { t_prefill = clk::now(); glog = t_prefill; first = false; }
                    text += tok->decode(t);
                    ++generated;
                    if (clk::now() - glog >= std::chrono::seconds(10)) {   // long-output progress
                        const double el = std::chrono::duration<double>(clk::now() - t_prefill).count();
                        fprintf(stderr, "  [%s] gen %d tok, %.1f tok/s\n",
                                clock_hms().c_str(), generated, el > 0 ? generated / el : 0.0);
                        glog = clk::now();
                    }
                    return generated < max_tokens;
                }, nullptr, /*ckpt_after_prefill=*/true);
            } else {
                Sampler smp(sc);
                if (!ovr.empty()) rt->set_embd_overrides(ovr);   // vision
                if (gen_rel > 0 && gen_rel < (int) prompt.size() && ovr.empty()) {
                    rt->decode(std::vector<int32_t>(prompt.begin(), prompt.begin() + gen_rel));
                    rt->snapshot_ckpt();
                    prompt.erase(prompt.begin(), prompt.begin() + gen_rel);
                }
                auto logits = rt->decode(prompt);
                rt->snapshot_ckpt();   // rewind point at prompt end
                t_prefill = clk::now();
                expert_debug_hooks::set_phase(ExpertDebug::Decode);
                smp.prime(cp.ids);     // repetition-penalty window (prompt tail)
                clk::time_point glog = clk::now();
                for (int t = 0; t < max_tokens; ++t) {
                    int next = smp.sample(logits);
                    smp.accept(next);
                    if (is_stop(next) || rt->n_past() + 1 >= n_ctx) break;
                    text += tok->decode(next);
                    ++generated;
                    if (clk::now() - glog >= std::chrono::seconds(10)) {   // long-output progress
                        const double el = std::chrono::duration<double>(clk::now() - t_prefill).count();
                        fprintf(stderr, "  [%s] gen %d tok, %.1f tok/s\n",
                                clock_hms().c_str(), generated, el > 0 ? generated / el : 0.0);
                        glog = clk::now();
                    }
                    logits = rt->decode({ next });
                }
            }
            t_end = clk::now();
        } catch (const std::exception & e) {
            fprintf(stderr, "generation error: %s\n", e.what());  // return what we have
            rt->reset();
            res.status=502;
            res.set_content(dumpj(json{{"error",{{"message",e.what()},{"type","model_generation_error"}}}}),"application/json");
            return;
            rt->reset();   // state unknown: invalidate the prefix cache
            if (t_end == clk::time_point{}) t_end = clk::now();
        }
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double prefill_ms = ms(t0, t_prefill);
        const double gen_ms     = ms(t_prefill, t_end);
        log_done(n_prompt_full, n_cached, generated, prefill_ms, gen_ms, cstat0);
        // split reasoning off, then parse tool calls out of the content
        std::string think;
        std::vector<ParsedToolCall> calls;
        std::string parse_error;
        std::string content;
        try {
            const bool length = generated >= max_tokens || n_prompt_full + generated + 1 >= n_ctx;
            const auto parsed = chat_turn->finish(text, length ? "length" : "stop");
            think=parsed.value("reasoning_content",std::string());
            content=parsed.value("content",std::string());
            for (const auto & call : parsed.value("tool_calls", json::array()))
                calls.push_back({call.at("function").at("name"),call.at("function").at("arguments")});
        } catch (const std::exception & error) { parse_error=error.what(); }
        if (!parse_error.empty()) {
            fprintf(stderr, "invalid tool call: %s\n", parse_error.c_str());
            res.status = 502;
            res.set_content(dumpj(json{{"error", {{"message", "Invalid model tool call: " + parse_error},
                {"type", "invalid_tool_call"}, {"code", "invalid_tool_call"}}}}), "application/json");
            return;
        }
        if (getenv("QWEN_LOG_RESPONSE")) {
            fprintf(stderr, "=== raw response ===\n%s\n=== parsed %zu tool call(s) ===\n",
                    text.c_str(), calls.size());
            for (auto & c : calls)
                fprintf(stderr, "  %s args=%s\n", c.name.c_str(), c.arguments.c_str());
        }
        json msg = {{"role", "assistant"}};
        msg["content"] = content;
        if (!think.empty()) msg["reasoning_content"] = think;
        if (!calls.empty()) {
            json tcs = json::array();
            for (size_t k = 0; k < calls.size(); ++k)
                tcs.push_back({{"id", id + "-tc" + std::to_string(k)}, {"type", "function"},
                               {"function", {{"name", calls[k].name},
                                             {"arguments", calls[k].arguments}}}});
            msg["tool_calls"] = tcs;
        }
        const bool truncated = calls.empty() &&
            (generated >= max_tokens || n_prompt_full + generated + 1 >= n_ctx);
        const char * fr = !calls.empty() ? "tool_calls" : (truncated ? "length" : "stop");
        json resp = {
            {"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", model_id},
            {"choices", json::array({
                {{"index", 0}, {"message", msg},
                 {"finish_reason", fr}}})},
            {"usage", {{"prompt_tokens", n_prompt_full},
                       {"completion_tokens", generated},
                       {"total_tokens", n_prompt_full + generated}}},
            {"timings", {{"prompt_tokens", n_prompt_full},
                         {"cached_tokens", n_cached},
                         {"completion_tokens", generated},
                         {"ttft_ms", prefill_ms},
                         {"prefill_tps", prefill_ms > 0 ? (n_prompt_full - n_cached) * 1000.0 / prefill_ms : 0.0},
                         {"gen_tps", gen_ms > 0 ? generated * 1000.0 / gen_ms : 0.0}}}
        };
        res.set_content(dumpj(resp), "application/json");
    });

    // generous timeouts: a single token (esp. SSD/offload prefill) can take a while
    srv.set_read_timeout(600, 0);
    srv.set_write_timeout(600, 0);
    srv.set_keep_alive_timeout(600);

    fprintf(stderr, "QuestWend server: http://%s:%d  (chat UI at /, model: %s)\n",
            host.c_str(), port, model_id.c_str());
    const bool ok = srv.listen(host.c_str(), port);
    if (!ok) fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
    return ok ? 0 : 1;
}
