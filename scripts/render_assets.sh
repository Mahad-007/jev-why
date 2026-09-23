#!/usr/bin/env bash
# Rasterise the SVG and HTML output into PNGs for a README or a post.
#
#   ./scripts/render_assets.sh [light|dark|both]
#
# Uses headless Chrome because the charts are hand-rolled SVG with CSS custom
# properties and two themes; a converter that ignores CSS would render them
# unstyled. Nothing here is needed to use the library -- it only exists to
# produce pictures of its output.
set -uo pipefail
cd "$(dirname "$0")/.."

CHROME="${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser)}"
if [ -z "$CHROME" ]; then
  echo "no Chrome found; set CHROME=/path/to/chrome" >&2
  exit 1
fi

MODE="${1:-both}"
OUT="assets"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

shoot() {  # name svg-or-html theme width height
  local name="$1" src="$2" theme="$3" w="$4" h="$5"
  local page="$TMP/$name-$theme.html"
  local bg plane
  if [ "$theme" = "dark" ]; then plane="#0d0d0d"; else plane="#f9f9f7"; fi

  if [[ "$src" == *.svg ]]; then
    {
      printf '<!doctype html><html data-theme="%s"><head><meta charset="utf-8">' "$theme"
      printf '<style>body{margin:0;padding:20px;background:%s;' "$plane"
      printf 'font-family:system-ui,-apple-system,sans-serif}</style></head><body>'
      cat "$src"
      printf '</body></html>'
    } > "$page"
  else
    sed "s|<html lang=\"en\">|<html lang=\"en\" data-theme=\"$theme\">|" "$src" > "$page"
  fi

  "$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
    --window-size="$w,$h" --screenshot="$OUT/$name-$theme.png" "$page" 2>/dev/null \
    && echo "  wrote $OUT/$name-$theme.png" \
    || echo "  FAILED $name-$theme" >&2
}

themes=()
case "$MODE" in
  light) themes=(light) ;;
  dark)  themes=(dark) ;;
  *)     themes=(light dark) ;;
esac

for theme in "${themes[@]}"; do
  [ -f "$OUT/reliability.svg" ]  && shoot reliability  "$OUT/reliability.svg"  "$theme" 460 560
  [ -f "$OUT/faithfulness.svg" ] && shoot faithfulness "$OUT/faithfulness.svg" "$theme" 560 340
  [ -f "$OUT/report.html" ]      && shoot report       "$OUT/report.html"      "$theme" 960 940
done
echo "done"
