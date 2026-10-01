#!/usr/bin/env bash
#
# test-api.sh — smoke & functional tester for the NVR Vision API (YOLO26).
#
# Covers openapi.yaml / SPEC.md §4:
#   GET  /health      (open, 200 {status:"ok"})
#   GET  /ready       (open, 200 {ready:true,model} | 503 {ready:false})
#   GET  /v1/models   (auth'd when API_KEY set)
#   POST /v1/detect   (multipart field "image" OR raw image/jpeg|image/png,
#                      ?min_conf=0..1 & ?classes=person,car,... filters,
#                      error shapes {error,request_id}: 400/401/405/413/503)
#
# Deps: curl (required). jq or python3 (optional, for pretty JSON + schema checks).
#
# Usage:
#   scripts/test-api.sh [options]
#
# Examples:
#   scripts/test-api.sh                                        # localhost:8080, open mode
#   scripts/test-api.sh -u http://localhost:8080 -i tests/testdata/bus.jpg
#   BASE_URL=http://192.168.1.10:8080 API_KEY=secret scripts/test-api.sh
#   scripts/test-api.sh -k secret --wait 120 --bench 10 -v
#   scripts/test-api.sh --only smoke                          # only health/ready/models
#   scripts/test-api.sh --only detect -i /tmp/snap.jpg --classes person,car
#   scripts/test-api.sh --only errors                         # negative tests only
#   scripts/test-api.sh --only bench --bench 20               # latency check only
#   scripts/test-api.sh --annotate /tmp/bus-boxes.png -i tests/testdata/bus.jpg
#   scripts/test-api.sh --annotate /tmp/out.png -i https://example.com/snap.jpg --min-conf 0.4
#
# Exit: 0 when every executed check passes, 1 otherwise.

set -uo pipefail

# ---------------------------------------------------------------- defaults ---

BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-}"
IMAGE="${IMAGE:-}"
MIN_CONF="${MIN_CONF:-0.25}"
CLASSES="${CLASSES:-person,car,bicycle}"
WAIT_SECS="${WAIT_SECS:-60}"
BENCH_N="${BENCH_N:-5}"
ONLY="all"
ANNOTATE=""
CLASSES_GIVEN=0
VERBOSE=0
QUIET=0

PASS=0
FAIL=0
SKIP=0

# ---------------------------------------------------------------- helpers ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  sed -n '2,/^# Exit/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  -u, --url URL        Base URL (default: http://localhost:8080, env BASE_URL)
  -k, --api-key KEY    API key (default: env API_KEY, empty = open/LAN mode)
  -i, --image FILE     Test image (default: tests/testdata/bus.jpg, env IMAGE)
      --min-conf F     Default min_conf for detect checks (default: 0.25)
      --classes LIST   Default classes filter (default: person,car,bicycle)
      --wait SECS      Wait up to SECS for /ready (default: 60, 0 = no wait)
      --bench N        N timed detect requests at the end (default: 5, 0 = skip)
      --only GROUP     smoke|detect|errors|auth|bench|all (default: all)
      --annotate OUT   Detect on the -i image (local file or http(s) URL),
                       print label/confidence/boxes and save an annotated
                       copy with boxes+labels to OUT (e.g. /tmp/out.png).
                       Skips the test groups. Needs python3 + pillow.
  -v, --verbose        Print response bodies
  -q, --quiet          Only print the summary
  -h, --help           Show this help
EOF
}

log()  { (( QUIET )) || printf '%s\n' "$*"; }
info() { (( QUIET )) || printf '\033[36m::\033[0m %s\n' "$*"; }
pass() { PASS=$((PASS+1)); (( QUIET )) || printf '\033[32mPASS\033[0m %s\n' "$*"; }
fail() { FAIL=$((FAIL+1)); printf '\033[31mFAIL\033[0m %s\n' "$*"; }
skip() { SKIP=$((SKIP+1)); (( QUIET )) || printf '\033[33mSKIP\033[0m %s\n' "$*"; }
vrb()  { (( VERBOSE && ! QUIET )) || return 0; printf '%s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Pretty-print JSON from stdin using jq > python3 > cat.
pjson() {
  if have jq; then jq . 2>/dev/null || cat
  elif have python3; then python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))' 2>/dev/null || cat
  else cat; fi
}

# Body text for failure messages (tolerates curl never writing the file,
# e.g. connection refused).
body_text() { cat "$1" 2>/dev/null || printf '<no response — is the server running at %s?>' "$BASE_URL"; }

# json_get <json-file> <dotted.path> — prints value or empty. jq > python3 > grep.
json_get() {
  local f="$1" path="$2"
  if have jq; then jq -r "$path // empty" "$f" 2>/dev/null
  elif have python3; then
    python3 - "$f" "$path" <<'EOF' 2>/dev/null
import json,sys
f, path = sys.argv[1], sys.argv[2].lstrip('.')
try:
    d = json.load(open(f))
    for p in path.split('.'):
        d = d[p]
    print(d if not isinstance(d,(dict,list)) else json.dumps(d))
except Exception:
    pass
EOF
  else
    grep -o "\"${path##*.}\"[[:space:]]*:[^,}]*" "$f" 2>/dev/null | head -1 | sed 's/.*:[[:space:]]*//;s/^"//;s/"$//'
  fi
}

# Global curl auth header args, rebuilt after option parsing.
AUTH_ARGS=()
rebuild_auth() {
  AUTH_ARGS=()
  if [[ -n "$API_KEY" ]]; then
    AUTH_ARGS=(-H "X-API-Key: $API_KEY")
  fi
}

# http_get <path> <out-body> -> prints "status|request-id"; body saved to file.
http_get() {
  local path="$1" out="$2"; shift 2
  local hdr tmp
  tmp="$(mktemp)"; hdr="$(mktemp)"
  local code
  code=$(curl -sS -o "$out" -D "$hdr" -w '%{http_code}' "${AUTH_ARGS[@]}" "$@" "${BASE_URL}${path}" 2>"$tmp" || true)
  if [[ ! "$code" =~ ^[0-9]+$ ]]; then
    vrb "curl stderr: $(cat "$tmp")"
    code="000"
  fi
  local rid
  rid=$(grep -i '^x-request-id:' "$hdr" | tr -d '\r' | awk '{print $2}' || true)
  rm -f "$tmp" "$hdr"
  printf '%s|%s' "$code" "$rid"
}

# http_post_file <path> <out-body> <curl-args...> -> prints "status|request-id".
http_post() {
  local path="$1" out="$2"; shift 2
  local hdr tmp
  tmp="$(mktemp)"; hdr="$(mktemp)"
  local code
  code=$(curl -sS -X POST -o "$out" -D "$hdr" -w '%{http_code}' "${AUTH_ARGS[@]}" "$@" "${BASE_URL}${path}" 2>"$tmp" || true)
  if [[ ! "$code" =~ ^[0-9]+$ ]]; then
    vrb "curl stderr: $(cat "$tmp")"
    code="000"
  fi
  local rid
  rid=$(grep -i '^x-request-id:' "$hdr" | tr -d '\r' | awk '{print $2}' || true)
  rm -f "$tmp" "$hdr"
  printf '%s|%s' "$code" "$rid"
}

expect_status() {
  # expect_status <label> <got> <want...>
  local label="$1" got="$2"; shift 2
  local w
  for w in "$@"; do
    if [[ "$got" == "$w" ]]; then pass "$label (HTTP $got)"; return 0; fi
  done
  fail "$label — want HTTP [$*], got $got$( [[ "$got" == "000" ]] && printf ' (connection failed — is the server running at %s?)' "$BASE_URL")"
  return 1
}

show_body() { (( VERBOSE && ! QUIET )) && [[ -f "$1" ]] && pjson < "$1"; }

run_group() { [[ "$ONLY" == "all" || "$ONLY" == "$1" ]]; }

now_ms() { date +%s%3N 2>/dev/null || echo $(( $(date +%s) * 1000 )); }

# run_annotate detects objects on one image (local file or http(s) URL),
# prints label/confidence/pixel boxes, and saves an annotated copy.
# Needs python3 + pillow for the drawing.
run_annotate() {
  local src="$IMAGE" out="$ANNOTATE"
  local dl="$TMPDIR_TEST/annotate-src.img"
  if [[ "$src" =~ ^https?:// ]]; then
    info "downloading $src …"
    if ! curl -sSL -m 60 -o "$dl" "$src"; then
      fail "download failed: $src"; return 1
    fi
    [[ -s "$dl" ]] || { fail "downloaded file is empty: $src"; return 1; }
  else
    [[ -f "$src" ]] || { fail "image not found: $src (pass -i FILE|URL)"; return 1; }
    dl="$src"
  fi
  if ! python3 -c 'import PIL.Image, PIL.ImageDraw, PIL.ImageFont' 2>/dev/null; then
    fail "annotate needs python3 + pillow (pip install pillow)"; return 1
  fi
  local q="min_conf=${MIN_CONF}"
  [[ -n "$CLASSES" ]] && q="$q&classes=${CLASSES}"
  info "POST /v1/detect?$q"
  local B="$TMPDIR_TEST/annotate.json" res code
  res="$(http_post "/v1/detect?$q" "$B" -F "image=@${dl}")"; code="${res%%|*}"
  if [[ "$code" != "200" ]]; then
    fail "detect failed (HTTP $code): $(body_text "$B")"; return 1
  fi
  python3 - "$dl" "$B" "$out" <<'PYEOF'
import json, sys
from PIL import Image, ImageDraw, ImageFont
src, det_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
resp = json.load(open(det_path))
dets = resp.get("detections", [])
img = Image.open(src).convert("RGB")
W, H = img.size
print("model=%s size=%dx%d inference_ms=%s objects=%d" % (
    resp.get("model"), W, H, resp.get("inference_ms"), len(dets)))
print("%3s %-12s %6s %7s %7s %7s %7s" % ("#", "label", "conf", "x_min", "y_min", "x_max", "y_max"))
for i, d in enumerate(dets, 1):
    b = d["box"]
    print("%3d %-12s %6.2f %7.0f %7.0f %7.0f %7.0f" % (
        i, d["label"], float(d["confidence"]),
        b["x_min"], b["y_min"], b["x_max"], b["y_max"]))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.load_default()
except Exception:
    font = None
palette = [(230, 57, 70), (46, 204, 113), (52, 152, 219), (241, 196, 15),
           (155, 89, 182), (26, 188, 156), (231, 76, 60), (230, 126, 34)]
lw = max(2, W // 400)
for i, d in enumerate(dets):
    c = palette[i % len(palette)]
    b = d["box"]
    x0, y0, x1, y1 = (int(round(b[k])) for k in ("x_min", "y_min", "x_max", "y_max"))
    draw.rectangle([x0, y0, x1, y1], outline=c, width=lw)
    txt = "%s %.2f" % (d["label"], float(d["confidence"]))
    tb = draw.textbbox((0, 0), txt, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty = y0 - th - 6
    if ty < 0:
        ty = y1 + 2
    draw.rectangle([x0, ty, x0 + tw + 8, ty + th + 6], fill=c)
    draw.text((x0 + 4, ty + 3), txt, fill=(255, 255, 255), font=font)
img.save(out)
print("saved: %s" % out)
PYEOF
}

# ------------------------------------------------------------------ args ---

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)      BASE_URL="$2"; shift 2 ;;
    -k|--api-key)  API_KEY="$2"; shift 2 ;;
    -i|--image)    IMAGE="$2"; shift 2 ;;
    --min-conf)    MIN_CONF="$2"; shift 2 ;;
    --classes)     CLASSES="$2"; CLASSES_GIVEN=1; shift 2 ;;
    --wait)        WAIT_SECS="$2"; shift 2 ;;
    --bench)       BENCH_N="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    --annotate)    ANNOTATE="$2"; shift 2 ;;
    -v|--verbose)  VERBOSE=1; shift ;;
    -q|--quiet)    QUIET=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
  esac
done
rebuild_auth
BASE_URL="${BASE_URL%/}"
[[ -z "$IMAGE" ]] && IMAGE="${REPO_ROOT}/tests/testdata/bus.jpg"

case "$ONLY" in all|smoke|detect|errors|auth|bench) ;; *) echo "--only must be smoke|detect|errors|auth|bench|all" >&2; exit 2 ;; esac

# ---------------------------------------------------------------- preflight ---

if ! have curl; then echo "ERROR: curl is required" >&2; exit 2; fi

log "NVR Vision API tester"
log "  base  : $BASE_URL"
log "  image : $IMAGE"
log "  auth  : $([[ -n "$API_KEY" ]] && echo "API key set (${#API_KEY} chars)" || echo "open mode (no key)")"
log "  only  : $ONLY"
log ""

if run_group detect || run_group errors || run_group bench || [[ -n "$ANNOTATE" ]]; then
  if [[ ! -f "$IMAGE" && ! "$IMAGE" =~ ^https?:// ]]; then
    fail "test image not found: $IMAGE (pass -i FILE|URL)"
    echo ""; echo "Result: $PASS passed, $FAIL failed, $SKIP skipped"
    exit 1
  fi
fi

CT=""
if [[ -f "$IMAGE" ]]; then
  case "${IMAGE,,}" in
    *.png) CT="image/png" ;;
    *.jpg|*.jpeg) CT="image/jpeg" ;;
    *) CT="image/jpeg" ;;
  esac
fi

TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

# ------------------------------------------------------ annotate mode ---
# Single-image detect + annotate. Takes precedence over the test groups;
# the classes filter defaults to all unless --classes was given.

if [[ -n "$ANNOTATE" ]]; then
  (( CLASSES_GIVEN )) || CLASSES=""
  if run_annotate; then
    echo "Result: annotated image saved to $ANNOTATE"
    exit 0
  else
    echo "Result: annotate failed"
    exit 1
  fi
fi

# ------------------------------------------------------------- 1. smoke ---

if run_group smoke; then
  info "== smoke: health / ready / models =="

  B="$TMPDIR_TEST/health.json"
  res="$(http_get "/health" "$B")"; code="${res%%|*}"
  if expect_status "GET /health" "$code" 200; then
    [[ "$(json_get "$B" '.status')" == "ok" ]] && pass "GET /health body {status:ok}" \
      || { fail "GET /health body — want {status:ok}, got: $(body_text "$B")"; }
  fi
  show_body "$B"

  # Wait for readiness (model loads in background; /ready is 503 until then).
  B="$TMPDIR_TEST/ready.json"
  if (( WAIT_SECS > 0 )); then
    info "waiting up to ${WAIT_SECS}s for /ready …"
    waited=0
    while (( waited <= WAIT_SECS )); do
      res="$(http_get "/ready" "$B")"; code="${res%%|*}"
      [[ "$code" == "200" ]] && break
      sleep 2; waited=$((waited+2))
    done
  else
    res="$(http_get "/ready" "$B")"; code="${res%%|*}"
  fi
  if [[ "$code" == "200" ]]; then
    pass "GET /ready (model: $(json_get "$B" '.model'))"
  elif [[ "$code" == "503" ]]; then
    fail "GET /ready still 503 after ${WAIT_SECS}s — model not loaded (check ORT_LIB_PATH/MODEL_PATH, docker logs)"
  else
    fail "GET /ready — want 200 (ready) or 503 (loading), got $code: $(body_text "$B")"
  fi
  show_body "$B"

  B="$TMPDIR_TEST/models.json"
  res="$(http_get "/v1/models" "$B")"; code="${res%%|*}"
  if [[ -n "$API_KEY" && "$code" == "401" ]]; then
    fail "GET /v1/models 401 with the provided API key — key rejected (this check used X-API-Key)"
  else
    if expect_status "GET /v1/models" "$code" 200; then
      [[ -n "$(json_get "$B" '.active')" ]] && pass "GET /v1/models body has active+models" \
        || fail "GET /v1/models body — want {active, models[]}, got: $(body_text "$B")"
    fi
  fi
  show_body "$B"
  echo ""
fi

# ------------------------------------------------------------- 2. detect ---

if run_group detect; then
  info "== detect: multipart + raw bytes + filters + schema =="

  # 2a. multipart (Body A)
  B="$TMPDIR_TEST/detect-multi.json"
  res="$(http_post "/v1/detect?min_conf=${MIN_CONF}&classes=${CLASSES}" "$B" -F "image=@${IMAGE}")"
  code="${res%%|*}"; rid="${res##*|}"
  if expect_status "POST /v1/detect multipart" "$code" 200; then
    show_body "$B"
    model="$(json_get "$B" '.model')"
    w="$(json_get "$B" '.width')"; h="$(json_get "$B" '.height')"
    ims="$(json_get "$B" '.inference_ms')"; dts="$(json_get "$B" '.detections')"
    if [[ -n "$model" && -n "$w" && -n "$h" && -n "$ims" && -n "$dts" ]]; then
      pass "detect schema {model,width,height,inference_ms,request_id,detections[]} — model=$model ${w}x${h} inference=${ims}ms"
    else
      fail "detect schema broken: $(body_text "$B")"
    fi
    [[ -n "$rid" ]] && pass "X-Request-ID header present ($rid)" || fail "X-Request-ID header missing"
    # per-detection shape check
    if have python3; then
      if python3 - "$B" <<'EOF' 2>/dev/null
import json,sys
r = json.load(open(sys.argv[1]))
dets = r.get("detections")
assert isinstance(dets, list), "detections not a list"
for d in dets:
    assert isinstance(d["class_id"], int), d
    assert isinstance(d["label"], str), d
    assert 0 <= float(d["confidence"]) <= 1, d
    for k in ("box", "box_norm"):
        for c in ("x_min","y_min","x_max","y_max"):
            float(d[k][c])
EOF
      then
        pass "detections[] shape {class_id,label,confidence,box,box_norm}"
        true
      else
        fail "detections[] shape broken: $(body_text "$B")"
      fi
    fi
    n="$(json_get "$B" '.detections' | grep -o 'class_id' | wc -l | tr -d ' ' || true)"
    log "  detections: ${n:-?} (min_conf=$MIN_CONF classes=$CLASSES)"
  else
    show_body "$B"
  fi

  # 2b. raw bytes (Body B) with matching Content-Type
  B="$TMPDIR_TEST/detect-raw.json"
  res="$(http_post "/v1/detect?min_conf=${MIN_CONF}" "$B" -H "Content-Type: $CT" --data-binary "@${IMAGE}")"
  code="${res%%|*}"
  if expect_status "POST /v1/detect raw ($CT)" "$code" 200; then :; else show_body "$B"; fi
  show_body "$B"

  # 2c. high threshold should filter to <= unfiltered count
  if have python3; then
    A="$TMPDIR_TEST/detect-lo.json"; HI="$TMPDIR_TEST/detect-hi.json"
    c_lo="$(http_post "/v1/detect?min_conf=0.01" "$A" -F "image=@${IMAGE}")"; c_lo="${c_lo%%|*}"
    c_hi="$(http_post "/v1/detect?min_conf=0.99" "$HI" -F "image=@${IMAGE}")"; c_hi="${c_hi%%|*}"
    if [[ "$c_lo" != "200" || "$c_hi" != "200" ]]; then
      skip "min_conf monotonic check (detect not 200: lo=$c_lo hi=$c_hi)"
    else
      lo=$(python3 -c 'import json; print(len(json.load(open("'"$A"'")).get("detections",[])))' 2>/dev/null || echo "?")
      hi=$(python3 -c 'import json; print(len(json.load(open("'"$HI"'")).get("detections",[])))' 2>/dev/null || echo "?")
      if [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ ]]; then
        if (( hi <= lo )); then pass "min_conf filter monotonic (0.01→$lo dets, 0.99→$hi dets)"
        else fail "min_conf filter inverted (0.01→$lo, 0.99→$hi)"; fi
      else
        skip "min_conf monotonic check (unparseable detect body)"
      fi
    fi
  else
    skip "min_conf monotonic check (need python3)"
  fi

  # 2d. single-class filter keeps only that label
  B="$TMPDIR_TEST/detect-person.json"
  res="$(http_post "/v1/detect?classes=person" "$B" -F "image=@${IMAGE}")"
  code="${res%%|*}"
  if [[ "$code" == "200" ]]; then
    if have python3; then
      if python3 - "$B" <<'EOF' 2>/dev/null
import json,sys
dets = json.load(open(sys.argv[1])).get("detections", [])
assert all(d["label"] == "person" for d in dets), dets
EOF
      then
        pass "classes=person keeps only person"
      else
        fail "classes=person leaked other labels: $(body_text "$B")"
      fi
    else
      pass "classes=person accepted (HTTP 200, label check needs python3)"
    fi
  else
    fail "classes=person — want 200, got $code: $(body_text "$B")"
  fi
  echo ""
fi

# ------------------------------------------------------------- 3. errors ---

if run_group errors; then
  info "== negative tests (expect 4xx/405, JSON {error,request_id}) =="
  check_err() { # check_err <label> <code> <body> <want...>
    local label="$1" code="$2" body="$3"; shift 3
    local ok=0 w
    for w in "$@"; do [[ "$code" == "$w" ]] && ok=1; done
    if (( ! ok )); then fail "$label — want [$*], got $code: $(body_text "$body")"; return; fi
    local e
    e="$(json_get "$body" '.error')"
    if [[ -n "$e" ]]; then pass "$label (HTTP $code, error: $e)"
    else fail "$label — HTTP $code but error shape broken (want {error,request_id}): $(body_text "$body")"; fi
  }

  B="$TMPDIR_TEST/e-minconf.json"
  res="$(http_post "/v1/detect?min_conf=2" "$B" -F "image=@${IMAGE}")"
  check_err "invalid min_conf=2" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-class.json"
  res="$(http_post "/v1/detect?classes=nope_not_a_class" "$B" -F "image=@${IMAGE}")"
  check_err "unknown class" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-bytes.json"
  printf 'not an image' > "$TMPDIR_TEST/payload-bad.bin"
  res="$(http_post "/v1/detect" "$B" -H "Content-Type: image/jpeg" --data-binary "@$TMPDIR_TEST/payload-bad.bin")"
  check_err "bad image bytes" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-ct.json"
  printf '{}' > "$TMPDIR_TEST/payload-ct.bin"
  res="$(http_post "/v1/detect" "$B" -H "Content-Type: application/json" --data-binary "@$TMPDIR_TEST/payload-ct.bin")"
  check_err "unsupported content-type" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-field.json"
  res="$(http_post "/v1/detect" "$B" -F "other=@${IMAGE}")"
  check_err "multipart missing field image" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-empty.json"
  : > "$TMPDIR_TEST/payload-empty.bin"
  res="$(http_post "/v1/detect" "$B" -H "Content-Type: image/jpeg" --data-binary "@$TMPDIR_TEST/payload-empty.bin")"
  check_err "empty image" "${res%%|*}" "$B" 400

  B="$TMPDIR_TEST/e-method.json"
  res="$(http_get "/v1/detect" "$B")"; check_err "GET /v1/detect (POST only)" "${res%%|*}" "$B" 405
  echo ""
fi

# ------------------------------------------------------------- 4. auth ---

if run_group auth; then
  info "== auth (/v1/* requires key only when server sets API_KEY) =="
  if [[ -z "$API_KEY" ]]; then
    B="$TMPDIR_TEST/auth-open.json"
    res="$(http_post "/v1/detect" "$B" -F "image=@${IMAGE}")"; code="${res%%|*}"
    if [[ "$code" == "200" ]]; then pass "open mode: unauthenticated detect works (server has no API_KEY)"
    elif [[ "$code" == "401" ]]; then skip "server requires a key but none given — rerun with -k KEY to test auth matrix"
    else fail "open-mode probe — got $code: $(body_text "$B")"; fi
  else
    B="$TMPDIR_TEST/auth-none.json"
    # shellcheck disable=SC2034
    SAVED_KEY="$API_KEY"; API_KEY=""; rebuild_auth
    res="$(http_post "/v1/detect" "$B" -F "image=@${IMAGE}")"; code="${res%%|*}"
    API_KEY="$SAVED_KEY"; rebuild_auth
    if [[ "$code" == "401" ]]; then pass "no key → 401"
    elif [[ "$code" == "200" ]]; then skip "server is open (API_KEY not set server-side); key not enforced"
    else fail "no key — want 401 (locked) or 200 (open), got $code"; fi

    B="$TMPDIR_TEST/auth-wrong.json"
    code="$(curl -sS -o "$B" -w '%{http_code}' -H "X-API-Key: wrong-key" -F "image=@${IMAGE}" "${BASE_URL}/v1/detect" 2>/dev/null || true)"
    if [[ "$code" == "401" ]]; then pass "wrong key → 401"
    elif [[ "$code" == "200" ]]; then skip "server is open; wrong key still accepted"
    else fail "wrong key — want 401/200, got $code"; fi

    B="$TMPDIR_TEST/auth-bearer.json"
    code="$(curl -sS -o "$B" -w '%{http_code}' -H "Authorization: Bearer $API_KEY" -F "image=@${IMAGE}" "${BASE_URL}/v1/detect" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then pass "Bearer token accepted"
    elif [[ "$code" == "401" ]]; then fail "Bearer token rejected — server may only accept X-API-Key (SPEC allows both)"
    else fail "Bearer probe — got $code: $(body_text "$B")"; fi
  fi
  echo ""
fi

# ------------------------------------------------------------- 5. bench ---

if run_group bench; then
  if (( BENCH_N > 0 )); then
    info "== bench: $BENCH_N sequential POST /v1/detect (multipart) =="
    times=()
    infer=()
    ok=0
    for i in $(seq 1 "$BENCH_N"); do
      B="$TMPDIR_TEST/bench-$i.json"
      t0="$(now_ms)"
      res="$(http_post "/v1/detect?min_conf=${MIN_CONF}" "$B" -F "image=@${IMAGE}")"
      t1="$(now_ms)"
      code="${res%%|*}"
      if [[ "$code" == "200" ]]; then
        ok=$((ok+1))
        times+=($((t1-t0)))
        im="$(json_get "$B" '.inference_ms')"
        [[ "$im" =~ ^[0-9]+(\.[0-9]+)?$ ]] && infer+=("${im%.*}")
      else
        fail "bench request $i — HTTP $code"
      fi
    done
    if (( ok > 0 )); then
      stats="$(printf '%s\n' "${times[@]}" | python3 -c '
import sys,statistics
t=sorted(float(x) for x in sys.stdin.read().split())
print(f"client_ms min={t[0]:.0f} p50={statistics.median(t):.0f} max={t[-1]:.0f} avg={sum(t)/len(t):.0f} n={len(t)}")' 2>/dev/null || echo "n=$ok")"
      pass "bench $ok/$BENCH_N ok — $stats"
      if (( ${#infer[@]} > 0 )); then
        avg_i="$(printf '%s\n' "${infer[@]}" | python3 -c 'import sys; v=[float(x) for x in sys.stdin.read().split()]; print(f"{sum(v)/len(v):.1f}")' 2>/dev/null || true)"
        log "  server inference_ms avg ≈ ${avg_i}ms (rest = HTTP+pre/post overhead)"
      fi
    else
      fail "bench — all $BENCH_N requests failed"
    fi
  else
    skip "bench (BENCH_N=0)"
  fi
  echo ""
fi

# ---------------------------------------------------------------- summary ---

echo "Result: $PASS passed, $FAIL failed, $SKIP skipped"
(( FAIL == 0 ))
