"""Parse Excalidraw (.excalidraw) and SVG exports for Manim animation."""

from __future__ import annotations

import base64
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

XLINK_NS = "http://www.w3.org/1999/xlink"
_DATA_URI_RE = re.compile(r"^data:image/([a-zA-Z0-9+.-]+);base64,(.+)$", re.S)
_EXT_FROM_MIME = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "webp": "webp",
    "gif": "gif",
    "svg+xml": "svg",
    "svg": "svg",
}


def _mime_to_ext(mime: str) -> str:
    mime = mime.lower().strip()
    if mime in _EXT_FROM_MIME:
        return _EXT_FROM_MIME[mime]
    if "svg" in mime:
        return "svg"
    tail = mime.split("/")[-1]
    tail = tail.replace("+xml", "").replace("+", "")
    return _EXT_FROM_MIME.get(tail, tail or "png")


def load_scene_file(path: Path) -> dict[str, Any]:
    """Load .excalidraw JSON or pass-through for SVG."""
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return {"format": "svg", "path": str(path.resolve())}
    if suffix in (".excalidraw", ".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "excalidraw":
            raise ValueError("JSON file is not Excalidraw format")
        return {"format": "excalidraw", "data": data, "path": str(path.resolve())}
    raise ValueError(f"Unsupported file type: {suffix} (use .svg or .excalidraw)")


def excalidraw_elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    elements = data.get("elements") or []
    out: list[dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        if el.get("isDeleted"):
            continue
        if el.get("type") in ("selection", "embeddable"):
            continue
        out.append(el)

    def sort_key(el: dict[str, Any]) -> tuple:
        anim = el.get("customData", {}) or {}
        order = anim.get("animOrder")
        if order is None and str(el.get("name", "")).startswith("anim:"):
            try:
                order = int(str(el["name"]).split(":")[1])
            except (ValueError, IndexError):
                order = None
        return (
            order if order is not None else 10_000,
            float(el.get("index") or 0),
        )

    out.sort(key=sort_key)
    return out


def svg_layer_paths(svg_path: Path) -> list[Path]:
    """Split SVG into one mini-SVG per animatable layer (preserves draw order)."""
    text = svg_path.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SVG: {exc}") from exc

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        raise ValueError("Root element is not <svg>")

    layers: list[str] = []

    def is_drawable(elem: ET.Element) -> bool:
        local = elem.tag.split("}")[-1]
        return local in ("path", "line", "rect", "circle", "ellipse", "polyline", "polygon", "text", "g")

    children = list(root)
    drawable_children = [c for c in children if is_drawable(c)]
    if len(drawable_children) >= 2:
        for child in drawable_children:
            layers.append(_wrap_svg_fragment(root, child))
    else:
        for elem in root.iter():
            local = elem.tag.split("}")[-1]
            if local == "path":
                layers.append(_wrap_svg_fragment(root, elem))

    if not layers:
        layers.append(text)

    cache_dir = svg_path.parent / ".excalidraw_layers" / svg_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, layer_svg in enumerate(layers):
        out = cache_dir / f"layer_{i:03d}.svg"
        out.write_text(layer_svg, encoding="utf-8")
        paths.append(out)
    return paths


def _wrap_svg_fragment(root: ET.Element, elem: ET.Element) -> str:
    view_box = root.get("viewBox") or f"0 0 {root.get('width', '800')} {root.get('height', '600')}"
    width = root.get("width", "800")
    height = root.get("height", "600")
    defs_parts: list[str] = []
    style_parts: list[str] = []
    for child in root:
        local = child.tag.split("}")[-1]
        if local == "defs":
            defs_parts.append(ET.tostring(child, encoding="unicode"))
        elif local == "style":
            style_parts.append(ET.tostring(child, encoding="unicode"))

    elem_s = ET.tostring(elem, encoding="unicode")
    bg = ""
    if root.get("style") and "background" in (root.get("style") or ""):
        bg = '<rect width="100%" height="100%" fill="#000000"/>'

    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" width="{width}" height="{height}">'
        f"{''.join(defs_parts)}{''.join(style_parts)}{bg}{elem_s}</svg>"
    )


def scene_background(scene_data: dict[str, Any]) -> str:
    if scene_data.get("format") == "excalidraw":
        app = (scene_data.get("data") or {}).get("appState") or {}
        return str(app.get("viewBackgroundColor") or "#ffffff")
    path = scene_data.get("path")
    if path:
        canvas = svg_canvas_background_color(Path(path))
        if canvas:
            return canvas
    return "#ffffff"


def svg_canvas_background_color(svg_path: Path) -> str | None:
    """Read full-canvas background fill from an Excalidraw SVG export."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return None

    _, _, vb_w, vb_h = parse_view_box(root)
    for elem in root:
        if _local_tag(elem) != "rect":
            continue
        fill = (elem.get("fill") or "").strip()
        if not fill:
            continue
        w_attr = elem.get("width", "")
        h_attr = elem.get("height", "")
        if w_attr in ("100%", str(int(vb_w)) if vb_w == int(vb_w) else str(vb_w)) and h_attr in (
            "100%",
            str(int(vb_h)) if vb_h == int(vb_h) else str(vb_h),
        ):
            return fill
        if _rect_covers_view_box(elem, vb_w, vb_h):
            return fill
    return None


def _rect_covers_view_box(elem: ET.Element, vb_w: float, vb_h: float) -> bool:
    try:
        w = float(str(elem.get("width", "0")).replace("%", ""))
        h = float(str(elem.get("height", "0")).replace("%", ""))
    except ValueError:
        return elem.get("width") == "100%" and elem.get("height") == "100%"
    return w >= vb_w * 0.95 and h >= vb_h * 0.95


def parse_view_box(root: ET.Element) -> tuple[float, float, float, float]:
    vb = root.get("viewBox") or ""
    parts = vb.split()
    if len(parts) >= 4:
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    try:
        w = float(str(root.get("width", "800")).replace("px", ""))
        h = float(str(root.get("height", "600")).replace("px", ""))
    except ValueError:
        w, h = 800.0, 600.0
    return 0.0, 0.0, w, h


def _local_tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _parse_translate(transform: str) -> tuple[float, float]:
    if not transform:
        return 0.0, 0.0
    m = re.search(r"translate\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", transform)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"translate\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", transform)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def _parent_chain(elem: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> list[ET.Element]:
    chain: list[ET.Element] = []
    cur: ET.Element | None = elem
    while cur is not None:
        chain.append(cur)
        cur = parent_map.get(cur)
    return chain


def iter_svg_text_elements(svg_path: Path) -> list[dict[str, Any]]:
    """Extract <text> nodes from Excalidraw SVG (Manim SVGMobject skips these)."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []

    parent_map: dict[ET.Element, ET.Element] = {child: parent for parent in root.iter() for child in parent}
    x0, y0, vb_w, vb_h = parse_view_box(root)
    items: list[dict[str, Any]] = []

    for idx, elem in enumerate(root.iter()):
        if _local_tag(elem) != "text":
            continue
        content = "".join(elem.itertext()).strip()
        if not content:
            continue
        tx, ty = float(elem.get("x", 0) or 0), float(elem.get("y", 0) or 0)
        for ancestor in _parent_chain(elem, parent_map)[1:]:
            ox, oy = _parse_translate(ancestor.get("transform") or "")
            tx += ox
            ty += oy
        fs_raw = str(elem.get("font-size") or "24").replace("px", "")
        try:
            font_size_px = float(fs_raw)
        except ValueError:
            font_size_px = 24.0
        fill = elem.get("fill") or "#ffffff"
        anchor = elem.get("text-anchor") or "start"
        font_family = elem.get("font-family") or ""
        font_weight = elem.get("font-weight") or "normal"
        items.append(
            {
                "order": idx,
                "text": content,
                "x": tx,
                "y": ty,
                "x0": x0,
                "y0": y0,
                "vb_w": vb_w,
                "vb_h": vb_h,
                "font_size_px": font_size_px,
                "fill": fill,
                "anchor": anchor,
                "font_family": font_family,
                "font_weight": font_weight,
            }
        )
    return items


_FONT_FACE_RE = re.compile(
    r"font-family:\s*(?:\"([^\"]+)\"|'([^']+)'|([^;\"']+))"
    r"[^;]*;\s*src:\s*url\(\s*data:font/"
    r"(woff2|woff|truetype|opentype);base64,([^)]+)\)",
    re.IGNORECASE,
)

_FONT_EXT = {
    "woff2": "woff2",
    "woff": "woff",
    "truetype": "ttf",
    "opentype": "otf",
}


def primary_font_family(font_family: str) -> str:
    """First family from a CSS font-family list."""
    if not font_family:
        return ""
    return font_family.split(",")[0].strip().strip("'\"")


def extract_svg_embedded_fonts(
    svg_path: Path,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Extract embedded @font-face blobs from Excalidraw SVG exports."""
    try:
        raw = svg_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    cache_dir = cache_dir or (svg_path.parent / ".excalidraw_render" / "fonts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fonts: dict[str, Path] = {}

    for match in _FONT_FACE_RE.finditer(raw):
        name = (match.group(1) or match.group(2) or match.group(3) or "").strip().strip("'\"")
        if not name:
            continue
        fmt = match.group(4).lower()
        ext = _FONT_EXT.get(fmt, fmt)
        try:
            payload = base64.b64decode(re.sub(r"\s+", "", match.group(5)))
        except ValueError:
            continue
        safe = re.sub(r"[^\w.-]+", "_", name) or "embedded"
        dest = cache_dir / f"{safe}.{ext}"
        if not dest.exists() or dest.read_bytes() != payload:
            dest.write_bytes(payload)
        fonts[name] = dest
    return fonts


def resolve_text_font(
    font_family: str,
    embedded: dict[str, Path],
) -> tuple[str, Path | None]:
    """Pick an embedded font file matching the SVG text's font-family."""
    primary = primary_font_family(font_family)
    if primary and primary in embedded:
        return primary, embedded[primary]
    if primary:
        for name, path in embedded.items():
            if name and name.lower() == primary.lower():
                return name, path
    # Only fall back to Excalifont when no explicit non-excal font was requested.
    if primary and primary.lower() not in ("excalifont", "virgil", "assistant"):
        return primary, None
    if "Excalifont" in embedded:
        return "Excalifont", embedded["Excalifont"]
    for name, path in embedded.items():
        if name:
            return name, path
    return primary or "sans-serif", None


def _load_ttfont(font_path: Path) -> TTFont:
    if font_path.suffix.lower() == ".ttc":
        return TTFont(str(font_path), fontNumber=0)
    return TTFont(str(font_path))


def _system_symbol_font_paths() -> list[Path]:
    """Broad-coverage fonts for symbols and emoji missing from Virgil/Excalifont."""
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Symbol.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansSymbols2-Regular.ttf"),
        Path("C:/Windows/Fonts/seguiemj.ttf"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
    ]
    return [path for path in candidates if path.is_file()]


def iter_text_font_candidates(
    font_family: str,
    embedded: dict[str, Path],
) -> list[Path]:
    """Font files to try when converting SVG text, best match first."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None or not path.is_file() or path in seen:
            return
        seen.add(path)
        ordered.append(path)

    _, primary_path = resolve_text_font(font_family, embedded)
    add(primary_path)

    for part in re.split(r",\s*", font_family or ""):
        name = part.strip().strip("'\"")
        if not name:
            continue
        if name in embedded:
            add(embedded[name])
        else:
            for key, path in embedded.items():
                if key.lower() == name.lower():
                    add(path)
                    break

    for path in _system_symbol_font_paths():
        add(path)

    if not ordered:
        for path in embedded.values():
            add(path)
    return ordered


def _glyph_render_score(text: str, font: TTFont) -> int:
    cmap = font.getBestCmap() or {}
    return sum(1 for ch in text if ord(ch) in cmap)


def _text_to_paths_with_fallback(
    text: str,
    font_family: str,
    embedded: dict[str, Path],
    font_cache: dict[Path, TTFont],
    *,
    anchor_x: float,
    baseline_y: float,
    font_size_px: float,
    fill: str,
    anchor: str,
) -> list[ET.Element]:
    """Convert text to paths, trying fallback fonts when glyphs are missing."""
    best_paths: list[ET.Element] = []
    best_score = -1
    for font_path in iter_text_font_candidates(font_family, embedded):
        if font_path not in font_cache:
            try:
                font_cache[font_path] = _load_ttfont(font_path)
            except OSError:
                continue
        font = font_cache[font_path]
        score = _glyph_render_score(text, font)
        if score <= best_score:
            continue
        paths = _text_to_path_elements(
            text,
            font,
            anchor_x=anchor_x,
            baseline_y=baseline_y,
            font_size_px=font_size_px,
            fill=fill,
            anchor=anchor,
        )
        if len(paths) > best_score:
            best_paths = paths
            best_score = len(paths)
        if score >= len(text) and len(paths) >= len(text):
            break
    return best_paths


def _parse_svg_length(value: str, *, ref: float) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.endswith("px"):
        raw = raw[:-2].strip()
    if raw.endswith("%"):
        try:
            return ref * float(raw[:-1].strip()) / 100.0
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def _rect_page_origin(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float]:
    """Top-left corner of a rect in page coordinates."""
    tx, ty = _parse_translate(elem.get("transform") or "")
    left = _float_attr(elem, "x") + tx
    top = _float_attr(elem, "y") + ty
    cur = parent_map.get(elem)
    while cur is not None:
        ptx, pty = _parse_translate(cur.get("transform") or "")
        left, top = _apply_svg_transform(left, top, cur.get("transform") or "")
        cur = parent_map.get(cur)
    return left, top


def _is_excalidraw_frame_outline_rect(
    elem: ET.Element,
    vb_w: float,
    vb_h: float,
    *,
    parent_map: dict[ET.Element, ET.Element] | None = None,
) -> bool:
    """Remove only full-bleed Excalidraw slide frames, not inset content borders."""
    if _local_tag(elem) != "rect":
        return False
    stroke = (elem.get("stroke") or "").strip().lower()
    if not stroke or stroke in ("none", "transparent"):
        return False
    fill = (elem.get("fill") or "").strip().lower()
    if fill not in ("", "none", "transparent"):
        return False
    width = _parse_svg_length(str(elem.get("width") or ""), ref=vb_w)
    height = _parse_svg_length(str(elem.get("height") or ""), ref=vb_h)
    if width is None or height is None:
        return False
    if width < vb_w * 0.85 or height < vb_h * 0.85:
        return False
    if parent_map is not None:
        left, top = _rect_page_origin(elem, parent_map)
    else:
        tx, ty = _parse_translate(elem.get("transform") or "")
        left = _float_attr(elem, "x") + tx
        top = _float_attr(elem, "y") + ty
    inset_limit = min(vb_w, vb_h) * 0.02
    if left > inset_limit or top > inset_limit:
        return False
    return True


def _svg_has_content_border(root: ET.Element, vb_w: float, vb_h: float) -> bool:
    """True when the SVG already includes a large hollow stroked slide border."""
    for elem in root.iter():
        if _local_tag(elem) != "rect":
            continue
        stroke = (elem.get("stroke") or "").strip().lower()
        if not stroke or stroke in ("none", "transparent"):
            continue
        fill = (elem.get("fill") or "").strip().lower()
        if fill not in ("", "none", "transparent"):
            continue
        width = _parse_svg_length(str(elem.get("width") or ""), ref=vb_w)
        height = _parse_svg_length(str(elem.get("height") or ""), ref=vb_h)
        if width is None or height is None:
            continue
        if width >= vb_w * 0.65 and height >= vb_h * 0.65:
            return True
    return False


def ensure_excalidraw_slide_border(svg_path: Path) -> Path:
    """Restore the rounded slide stroke missing from single-frame Excalidraw exports."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    vb_x, vb_y, vb_w, vb_h = parse_view_box(root)
    if _svg_has_content_border(root, vb_w, vb_h):
        return svg_path

    inset_x = max(8.0, min(12.0, vb_w * 0.014))
    inset_top = max(8.0, min(30.5, vb_h * 0.063))
    inset_bottom = max(8.0, min(12.0, vb_h * 0.021))
    border_w = vb_w - 2 * inset_x
    border_h = vb_h - inset_top - inset_bottom
    if border_w <= 0 or border_h <= 0:
        return svg_path

    svg_ns = "http://www.w3.org/2000/svg"
    border = ET.Element(f"{{{svg_ns}}}g")
    border.set("stroke-linecap", "round")
    border.set(
        "transform",
        f"translate({vb_x + inset_x} {vb_y + inset_top}) rotate(0 {border_w / 2} {border_h / 2})",
    )
    rect = ET.SubElement(border, f"{{{svg_ns}}}rect")
    rect.set("width", f"{border_w}px")
    rect.set("height", f"{border_h}px")
    rect.set("stroke", "#bbb")
    rect.set("fill", "none")
    rect.set("rx", "8")

    insert_at = 0
    for i, child in enumerate(list(root)):
        if _local_tag(child) == "defs":
            insert_at = i + 1
        elif _local_tag(child) == "rect" and (child.get("fill") or "").lower() not in (
            "none",
            "transparent",
            "",
        ):
            insert_at = i + 1
            break
    root.insert(insert_at, border)

    return _write_reordered_svg(svg_path, root, "border")


def strip_excalidraw_frame_outlines(svg_path: Path) -> Path:
    """Remove Excalidraw frame/slide border rects before Manim loads the SVG."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    _, _, vb_w, vb_h = parse_view_box(root)
    parent_map: dict[ET.Element, ET.Element] = {child: parent for parent in root.iter() for child in parent}
    removed = False

    for elem in list(root.iter()):
        if not _is_excalidraw_frame_outline_rect(elem, vb_w, vb_h, parent_map=parent_map):
            continue
        in_defs = False
        cur: ET.Element | None = elem
        while cur is not None:
            if _local_tag(cur) == "defs":
                in_defs = True
                break
            cur = parent_map.get(cur)
        if in_defs:
            continue
        parent = parent_map.get(elem)
        if parent is None:
            continue
        parent.remove(elem)
        removed = True

    if not removed:
        return svg_path

    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_noframe.svg"
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return out


@dataclass(frozen=True)
class AnimationUnitInfo:
    """One reorderable animation unit on an Excalidraw page."""

    index: int
    label: str
    kind: str
    unit_id: str
    hint: str = ""
    color: str | None = None
    bbox: tuple[float, float, float, float] | None = None


_NAMED_COLORS: dict[str, str] = {
    "#fff": "White",
    "#ffffff": "White",
    "#000": "Black",
    "#000000": "Black",
    "#ffda48": "Yellow",
    "#3371a2": "Blue",
    "#3572a5": "Blue",
    "#40c057": "Green",
    "#ff6b6b": "Red",
    "#fab005": "Orange",
    "#bbb": "Gray",
    "#cccccc": "Gray",
}


def _float_attr(elem: ET.Element, name: str, default: float = 0.0) -> float:
    raw = elem.get(name)
    if raw is None:
        return default
    try:
        return float(str(raw).replace("px", "").replace("%", ""))
    except ValueError:
        return default


def _normalize_hex(color: str) -> str:
    c = color.strip().lower()
    if not c.startswith("#"):
        return c
    if len(c) == 4:
        return "#" + "".join(ch * 2 for ch in c[1:])
    return c


def _color_display_name(color: str | None) -> str | None:
    if not color:
        return None
    normalized = _normalize_hex(color)
    if normalized in _NAMED_COLORS:
        return _NAMED_COLORS[normalized]
    if normalized.startswith("#") and len(normalized) == 7:
        r = int(normalized[1:3], 16)
        g = int(normalized[3:5], 16)
        b = int(normalized[5:7], 16)
        if r > 220 and g > 220 and b > 220:
            return "White"
        if r < 40 and g < 40 and b < 40:
            return "Black"
        if r > g and r > b:
            return "Red"
        if g > r and g > b:
            return "Green"
        if b > r and b > g:
            return "Blue"
        if r > 200 and g > 180 and b < 120:
            return "Yellow"
        if r > 200 and g > 120 and b < 120:
            return "Orange"
    return "Colored"


def _element_text_label_display(elem: ET.Element) -> str:
    label = (elem.get("data-excal-text") or "").strip()
    if label:
        return label
    if _local_tag(elem) == "text":
        return "".join(elem.itertext()).strip()
    for child in elem.iter():
        if child is elem:
            continue
        nested = (child.get("data-excal-text") or "").strip()
        if nested:
            return nested
        if _local_tag(child) == "text":
            text = "".join(child.itertext()).strip()
            if text:
                return text
    return ""


def _is_excalidraw_frame_title(
    unit: ET.Element,
    label: str,
    parent_map: dict[ET.Element, ET.Element],
    *,
    view_box: tuple[float, float, float, float] | None = None,
) -> bool:
    """True for Excalidraw's automatic frame name label (editor metadata, not scene content)."""
    if not _element_text_label(unit):
        return False
    center = _element_bbox_center(unit, parent_map)
    if center is None:
        return label.strip().lower() == "frame"
    cx, cy = center
    vb_x, vb_y, vb_w, vb_h = view_box or (0.0, 0.0, 800.0, 600.0)
    margin_x = min(48.0, vb_w * 0.12)
    margin_y = min(48.0, vb_h * 0.12)
    return cx <= vb_x + margin_x and cy <= vb_y + margin_y


def _is_text_animation_unit(unit: ET.Element) -> bool:
    """True for Excalidraw text units (including SVG text converted to glyph paths)."""
    return _element_unit_kind(unit) == "text"


def _element_unit_colors(elem: ET.Element) -> tuple[str | None, str | None]:
    fills: list[str] = []
    strokes: list[str] = []
    for child in elem.iter():
        fill = (child.get("fill") or "").strip()
        stroke = (child.get("stroke") or "").strip()
        if fill and fill.lower() not in ("none", "transparent"):
            fills.append(_normalize_hex(fill))
        if stroke and stroke.lower() not in ("none", "transparent"):
            strokes.append(_normalize_hex(stroke))
    primary_fill = fills[0] if fills else None
    primary_stroke = strokes[0] if strokes else None
    return primary_fill, primary_stroke


def _element_bbox_center(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float] | None:
    best: tuple[float, float, float] | None = None

    def _apply_ancestor_transforms(x: float, y: float, node: ET.Element) -> tuple[float, float]:
        chain: list[ET.Element] = []
        cur: ET.Element | None = node
        while cur is not None:
            chain.append(cur)
            cur = parent_map.get(cur)
        for ancestor in reversed(chain):
            x, y = _apply_svg_transform(x, y, ancestor.get("transform") or "")
        return x, y

    for child in elem.iter():
        tag = _local_tag(child)
        if tag == "rect":
            w = _float_attr(child, "width")
            h = _float_attr(child, "height")
            if w <= 0 or h <= 0:
                continue
            x = _float_attr(child, "x")
            y = _float_attr(child, "y")
            tx, ty = _parse_translate(child.get("transform") or "")
            cx, cy = _apply_ancestor_transforms(x + tx + w / 2, y + ty + h / 2, child)
            area = w * h
            if best is None or area > best[0]:
                best = (area, cx, cy)
        elif tag in ("use", "image"):
            x, y, w, h = _placement_from_use(child, parent_map)
            if w <= 0 or h <= 0:
                continue
            cx, cy = x + w / 2, y + h / 2
            area = w * h
            if best is None or area > best[0]:
                best = (area, cx, cy)

    if best is not None:
        return best[1], best[2]

    for node in elem.iter():
        tx, ty = _parse_translate(node.get("transform") or "")
        if tx or ty:
            return _apply_ancestor_transforms(tx, ty, node)

    return None


def _raster_unit_bbox(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float, float, float] | None:
    """BBox for embedded PNG/SVG icons (placement is already in page space)."""
    for child in elem.iter():
        tag = _local_tag(child)
        if tag not in ("use", "image"):
            continue
        x, y, w, h = _placement_from_use(child, parent_map)
        if w <= 0 or h <= 0:
            continue
        pad = max(4.0, min(w, h) * 0.08)
        return x - pad, y - pad, w + 2 * pad, h + 2 * pad
    return None


def _clamp_bbox_to_page(
    bbox: tuple[float, float, float, float],
    vb: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    vb_x, vb_y, vb_w, vb_h = vb
    x1 = max(vb_x, x)
    y1 = max(vb_y, y)
    x2 = min(vb_x + vb_w, x + w)
    y2 = min(vb_y + vb_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    clipped = x1, y1, x2 - x1, y2 - y1
    if clipped[2] * clipped[3] < max(w * h * 0.04, 16.0):
        return None
    return clipped


def _center_bbox_on_page(
    center: tuple[float, float],
    vb: tuple[float, float, float, float],
    *,
    width: float = 56.0,
    height: float = 40.0,
) -> tuple[float, float, float, float]:
    cx, cy = center
    vb_x, vb_y, vb_w, vb_h = vb
    x = cx - width / 2
    y = cy - height / 2
    clipped = _clamp_bbox_to_page((x, y, width, height), vb)
    if clipped is not None:
        return clipped
    return (
        max(vb_x, min(cx - width / 2, vb_x + vb_w - width)),
        max(vb_y, min(cy - height / 2, vb_y + vb_h - height)),
        min(width, vb_w),
        min(height, vb_h),
    )


def _element_unit_bbox(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
    vb: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounds of one animation unit in page SVG coordinates."""
    if _element_has_raster_image(elem):
        bbox = _raster_unit_bbox(elem, parent_map)
        if bbox is None:
            return None
        if vb is not None:
            clipped = _clamp_bbox_to_page(bbox, vb)
            if clipped is not None:
                return clipped
            center = _element_bbox_center(elem, parent_map)
            if center is not None:
                return _center_bbox_on_page(center, vb)
        return bbox

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    def _apply_ancestor_transforms(x: float, y: float, node: ET.Element) -> tuple[float, float]:
        chain: list[ET.Element] = []
        cur: ET.Element | None = node
        while cur is not None:
            chain.append(cur)
            cur = parent_map.get(cur)
        for ancestor in reversed(chain):
            x, y = _apply_svg_transform(x, y, ancestor.get("transform") or "")
        return x, y

    def _include_page_point(x: float, y: float) -> None:
        nonlocal min_x, min_y, max_x, max_y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    def _include_local_point(x: float, y: float, node: ET.Element) -> None:
        px, py = _apply_ancestor_transforms(x, y, node)
        _include_page_point(px, py)

    def _include_local_rect(x: float, y: float, w: float, h: float, node: ET.Element) -> None:
        _include_local_point(x, y, node)
        _include_local_point(x + w, y, node)
        _include_local_point(x, y + h, node)
        _include_local_point(x + w, y + h, node)

    for child in elem.iter():
        tag = _local_tag(child)
        if tag == "rect":
            w = _float_attr(child, "width")
            h = _float_attr(child, "height")
            if w <= 0 or h <= 0:
                continue
            x = _float_attr(child, "x")
            y = _float_attr(child, "y")
            tx, ty = _parse_translate(child.get("transform") or "")
            _include_local_rect(x + tx, y + ty, w, h, child)
        elif tag in ("use", "image"):
            x, y, w, h = _placement_from_use(child, parent_map)
            if w > 0 and h > 0:
                _include_page_point(x, y)
                _include_page_point(x + w, y + h)
        elif tag == "path":
            d = child.get("d") or ""
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)]
            if len(nums) >= 2:
                xs = nums[0::2][: len(nums) // 2]
                ys = nums[1::2][: len(nums) // 2]
                for px, py in zip(xs, ys):
                    _include_local_point(px, py, child)

    if min_x == float("inf"):
        center = _element_bbox_center(elem, parent_map)
        if center is None:
            return None
        if vb is not None:
            return _center_bbox_on_page(center, vb, width=48.0, height=32.0)
        cx, cy = center
        return cx - 24.0, cy - 16.0, 48.0, 32.0

    pad = max(4.0, (max_x - min_x) * 0.04, (max_y - min_y) * 0.04)
    bbox = min_x - pad, min_y - pad, (max_x - min_x) + 2 * pad, (max_y - min_y) + 2 * pad
    if vb is not None:
        vb_x, vb_y, vb_w, vb_h = vb
        if bbox[2] > vb_w * 0.92 or bbox[3] > vb_h * 0.92:
            center = _element_bbox_center(elem, parent_map)
            if center is not None:
                return _center_bbox_on_page(center, vb)
        clipped = _clamp_bbox_to_page(bbox, vb)
        if clipped is not None:
            return clipped
        center = _element_bbox_center(elem, parent_map)
        if center is not None:
            return _center_bbox_on_page(center, vb)
    return bbox


def _position_hint(
    center: tuple[float, float] | None,
    vb: tuple[float, float, float, float],
) -> str:
    if center is None:
        return ""
    cx, cy = center
    _, _, vb_w, vb_h = vb
    if vb_w <= 0 or vb_h <= 0:
        return ""

    x_ratio = cx / vb_w
    y_ratio = cy / vb_h
    if y_ratio < 0.33:
        v = "Top"
    elif y_ratio > 0.66:
        v = "Bottom"
    else:
        v = "Middle"
    if x_ratio < 0.33:
        h = "left"
    elif x_ratio > 0.66:
        h = "right"
    else:
        h = "center"
    if h == "center" and v == "Middle":
        return "Center of slide"
    return f"{v} {h}"


def _shape_type_name(elem: ET.Element, *, path_weight: int, area: float, vb_area: float) -> str:
    tags: dict[str, int] = {}
    for child in elem.iter():
        tag = _local_tag(child)
        tags[tag] = tags.get(tag, 0) + 1

    if tags.get("rect") == 1 and path_weight <= 2:
        fill, stroke = _element_unit_colors(elem)
        if not fill and stroke:
            if area >= vb_area * 0.7:
                return "Slide border"
            return "Outline box"
        if area >= vb_area * 0.15:
            return "Background panel"
        return "Box"

    if _element_has_raster_image(elem):
        return "Logo" if area > 0 and area < 180 * 180 else "Image"

    if path_weight > 120:
        return "Illustration"
    if path_weight > 20:
        return "Icon"
    if path_weight > 0:
        return "Drawing"
    return "Shape"


def _nearest_text_label(
    center: tuple[float, float] | None,
    text_units: list[tuple[str, tuple[float, float] | None]],
    *,
    max_dist: float,
) -> str | None:
    if center is None or not text_units:
        return None
    cx, cy = center
    best: tuple[float, str] | None = None
    for label, text_center in text_units:
        if not label or text_center is None:
            continue
        tx, ty = text_center
        dist = math.hypot(cx - tx, cy - ty)
        if dist > max_dist:
            continue
        if best is None or dist < best[0]:
            best = (dist, label)
    return best[1] if best else None


def _describe_animation_unit(
    elem: ET.Element,
    index: int,
    *,
    parent_map: dict[ET.Element, ET.Element],
    vb: tuple[float, float, float, float],
    text_units: list[tuple[str, tuple[float, float] | None]],
) -> AnimationUnitInfo:
    kind = _element_unit_kind(elem)
    display_text = _element_text_label_display(elem)
    fill, stroke = _element_unit_colors(elem)
    color = fill or stroke
    center = _element_bbox_center(elem, parent_map)
    position = _position_hint(center, vb)
    path_weight = _element_path_weight(elem)
    _, _, vb_w, vb_h = vb
    vb_area = max(vb_w * vb_h, 1.0)

    area = 0.0
    for child in elem.iter():
        if _local_tag(child) == "rect":
            area = max(area, _float_attr(child, "width") * _float_attr(child, "height"))

    if kind == "text" and display_text:
        cleaned = re.sub(r"\s+", " ", display_text).strip()
        label = cleaned if len(cleaned) <= 56 else cleaned[:53] + "..."
        hint = position or "Text"
        return AnimationUnitInfo(
            index=index,
            label=label,
            kind=kind,
            unit_id=f"unit-{index}",
            hint=hint,
            color=color,
            bbox=_element_unit_bbox(elem, parent_map, vb),
        )

    if kind == "image":
        shape = _shape_type_name(elem, path_weight=path_weight, area=area, vb_area=vb_area)
        near = _nearest_text_label(center, text_units, max_dist=max(vb_w, vb_h) * 0.55)
        if near:
            label = f"{shape} ({near})"
        elif position:
            label = f"{shape} — {position.lower()}"
        else:
            label = shape
        hint_parts = [p for p in (position, "Embedded photo/logo") if p]
        return AnimationUnitInfo(
            index=index,
            label=label,
            kind=kind,
            unit_id=f"unit-{index}",
            hint=" · ".join(hint_parts),
            color=color,
            bbox=_element_unit_bbox(elem, parent_map, vb),
        )

    shape = _shape_type_name(elem, path_weight=path_weight, area=area, vb_area=vb_area)
    color_name = _color_display_name(fill or stroke)
    near = _nearest_text_label(center, text_units, max_dist=max(vb_w, vb_h) * 0.55)
    if color_name and shape not in ("Slide border",):
        label = f"{color_name} {shape.lower()}"
    else:
        label = shape
    if near and kind == "shape" and shape not in ("Slide border", "Background panel"):
        label = f"{label} ({near})"

    hint_parts = [p for p in (position, f'Next to "{near}"' if near else "") if p]
    if not hint_parts:
        if path_weight > 0 and shape in ("Icon", "Drawing"):
            hint_parts.append(f"Hand-drawn {shape.lower()}")
        elif shape == "Slide border":
            hint_parts.append("Usually draws last")

    return AnimationUnitInfo(
        index=index,
        label=label,
        kind=kind,
        unit_id=f"unit-{index}",
        hint=" · ".join(hint_parts),
        color=color,
        bbox=_element_unit_bbox(elem, parent_map, vb),
    )


def _element_unit_kind(elem: ET.Element) -> str:
    if _element_has_raster_image(elem):
        return "image"
    if _element_text_label(elem):
        return "text"
    return "shape"


def describe_animation_units(svg_path: Path) -> list[AnimationUnitInfo]:
    """Return ordered animation units for one page SVG."""
    working = convert_svg_text_to_paths(svg_path)
    try:
        root = ET.parse(working).getroot()
    except ET.ParseError:
        return []
    units = animation_unit_elements_from_root(root)
    if not units:
        return []
    parent_map: dict[ET.Element, ET.Element] = {child: parent for parent in root.iter() for child in parent}
    vb = parse_view_box(root)

    text_units: list[tuple[str, tuple[float, float] | None]] = []
    for elem in units:
        if _element_unit_kind(elem) != "text":
            continue
        label = _element_text_label_display(elem)
        text_units.append((label, _element_bbox_center(elem, parent_map)))

    return _disambiguate_unit_labels(
        [
            _describe_animation_unit(
                elem,
                i,
                parent_map=parent_map,
                vb=vb,
                text_units=text_units,
            )
            for i, elem in enumerate(units)
        ]
    )


def _disambiguate_unit_labels(infos: list[AnimationUnitInfo]) -> list[AnimationUnitInfo]:
    """Add #2, #3 suffixes when multiple units share the same label."""
    counts: dict[str, int] = {}
    for info in infos:
        counts[info.label] = counts.get(info.label, 0) + 1

    seen: dict[str, int] = {}
    out: list[AnimationUnitInfo] = []
    for info in infos:
        if counts.get(info.label, 0) <= 1:
            out.append(info)
            continue
        seen[info.label] = seen.get(info.label, 0) + 1
        suffix = seen[info.label]
        out.append(
            AnimationUnitInfo(
                index=info.index,
                label=f"{info.label} #{suffix}",
                kind=info.kind,
                unit_id=info.unit_id,
                hint=info.hint,
                color=info.color,
                bbox=info.bbox,
            )
        )
    return out


def describe_drawing_animation_units(drawing_path: Path) -> list[dict[str, Any]]:
    """Return animation units grouped by page for an uploaded drawing."""
    suffix = drawing_path.suffix.lower()
    if suffix != ".svg":
        return []
    pages = detect_svg_pages(drawing_path)
    if not pages:
        units = describe_animation_units(drawing_path)
        try:
            root = ET.parse(drawing_path).getroot()
            _, _, vb_w, vb_h = parse_view_box(root)
        except ET.ParseError:
            vb_w, vb_h = 800.0, 450.0
        return [
            {
                "page_index": 0,
                "page_name": "Page 1",
                "page_id": "page-0",
                "page_width": vb_w,
                "page_height": vb_h,
                "units": [unit.__dict__ for unit in units],
            }
        ]
    out: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        page_svg = build_page_svg(drawing_path, page, pages)
        units = describe_animation_units(page_svg)
        out.append(
            {
                "page_index": index,
                "page_name": page.name,
                "page_id": page.page_id,
                "page_width": page.width,
                "page_height": page.height,
                "units": [unit.__dict__ for unit in units],
            }
        )
    return out


def _unit_center_x(info: AnimationUnitInfo) -> float | None:
    if not info.bbox:
        return None
    return info.bbox[0] + info.bbox[2] / 2.0


def _column_cluster_draw_order(infos: list[AnimationUnitInfo]) -> list[int]:
    """Group units into visual columns (cards), then draw left-to-right, top-to-bottom."""
    info_by_index = {info.index: info for info in infos}
    border: list[int] = []
    items: list[tuple[int, float, float]] = []
    for info in infos:
        if "slide border" in info.label.lower():
            border.append(info.index)
            continue
        cx = _unit_center_x(info)
        if cx is None:
            continue
        cy = info.bbox[1] if info.bbox else 0.0
        items.append((info.index, cx, cy))

    if len(items) < 2:
        return [info.index for info in infos]

    items.sort(key=lambda item: item[1])
    gaps = [items[i + 1][1] - items[i][1] for i in range(len(items) - 1)]
    positive_gaps = [gap for gap in gaps if gap > 1.0]
    split_gap = max(80.0, sorted(positive_gaps)[len(positive_gaps) // 2] * 0.55) if positive_gaps else 120.0

    columns: list[list[tuple[int, float, float]]] = [[items[0]]]
    for item in items[1:]:
        column_mean_x = sum(entry[1] for entry in columns[-1]) / len(columns[-1])
        if item[1] - column_mean_x <= split_gap:
            columns[-1].append(item)
        else:
            columns.append([item])

    order: list[int] = []
    for column in columns:
        order.extend(idx for idx, _cx, _cy in sorted(column, key=lambda entry: (entry[2], entry[1], entry[0])))

    seen = set(order)
    for info in infos:
        if info.index not in seen and info.index not in border:
            order.append(info.index)
    order.extend(border)
    return order


def _merged_text_anchors(infos: list[AnimationUnitInfo], *, merge_distance: float = 90.0) -> list[tuple[float, list[int]]]:
    """Cluster speech-bubble + caption text on the same character into one column."""
    anchors: list[tuple[float, list[int]]] = []
    for info in infos:
        if info.kind != "text":
            continue
        cx = _unit_center_x(info)
        if cx is None or not info.bbox:
            continue
        cy = info.bbox[1]
        if anchors:
            prev_x, prev_idxs = anchors[-1][0], anchors[-1][1]
            prev_info = next(item for item in infos if item.index == prev_idxs[-1])
            prev_cy = prev_info.bbox[1] if prev_info.bbox else cy
            if cx - prev_x < merge_distance and abs(cy - prev_cy) < 80.0:
                anchors[-1] = ((prev_x + cx) / 2.0, prev_idxs + [info.index])
                continue
        anchors.append((cx, [info.index]))
    return anchors


def infer_spatial_draw_order(svg_path: Path) -> list[int]:
    """Default draw order: left-to-right columns, connectors between characters."""
    infos = describe_animation_units(svg_path)
    if len(infos) < 2:
        return [info.index for info in infos]

    working = convert_svg_text_to_paths(svg_path)
    try:
        root = ET.parse(working).getroot()
    except ET.ParseError:
        return [info.index for info in infos]
    units = animation_unit_elements_from_root(root)
    connector_indices = {
        i for i, unit in enumerate(units) if any(unit_connector_path_flags(unit))
    }
    info_by_index = {info.index: info for info in infos}

    anchors = _merged_text_anchors(infos)
    if len(anchors) < 2:
        return _column_cluster_draw_order(infos)

    anchor_xs = [item[0] for item in anchors]
    column_count = len(anchors)
    bounds: list[float] = []
    for i in range(column_count - 1):
        if i == column_count - 2:
            bounds.append(anchor_xs[i] + 0.35 * (anchor_xs[i + 1] - anchor_xs[i]))
        else:
            bounds.append((anchor_xs[i] + anchor_xs[i + 1]) / 2.0)

    def column_for(cx: float) -> int:
        for i, bound in enumerate(bounds):
            if cx < bound:
                return i
        return column_count - 1

    buckets: list[list[int]] = [[] for _ in range(column_count)]
    gaps: list[list[int]] = [[] for _ in range(column_count - 1)]
    border: list[int] = []

    for info in infos:
        idx = info.index
        if "slide border" in info.label.lower():
            border.append(idx)
            continue
        cx = _unit_center_x(info)
        if cx is None:
            buckets[0].append(idx)
            continue
        if idx in connector_indices:
            placed = False
            for gap_i in range(column_count - 1):
                if anchor_xs[gap_i] - 40.0 < cx < anchor_xs[gap_i + 1] + 40.0:
                    gaps[gap_i].append(idx)
                    placed = True
                    break
            if not placed:
                buckets[column_for(cx)].append(idx)
            continue
        buckets[column_for(cx)].append(idx)

    def bucket_sort_key(unit_index: int) -> tuple[float, float, int]:
        info = info_by_index[unit_index]
        return (
            info.bbox[1] if info.bbox else 0.0,
            _unit_center_x(info) or 0.0,
            unit_index,
        )

    order: list[int] = []
    for column_index in range(column_count):
        order.extend(sorted(buckets[column_index], key=bucket_sort_key))
        if column_index < len(gaps):
            order.extend(sorted(gaps[column_index]))
    order.extend(sorted(border))

    seen = set(order)
    for info in infos:
        if info.index not in seen:
            order.append(info.index)
    return order


def resolve_page_unit_order(
    page_index: int,
    unit_count: int,
    *,
    page_sequences: dict[str | int, list[int]] | None = None,
    animation_sequence: list[str] | None = None,
    page_svg: Path | None = None,
) -> list[int]:
    """Resolve draw order as unit indices for one page."""
    if unit_count <= 0:
        return []
    default = list(range(unit_count))
    if page_sequences:
        raw = page_sequences.get(page_index)
        if raw is None:
            raw = page_sequences.get(str(page_index))
        if raw:
            valid = [i for i in raw if isinstance(i, int) and 0 <= i < unit_count]
            seen: set[int] = set()
            order: list[int] = []
            for idx in valid:
                if idx in seen:
                    continue
                order.append(idx)
                seen.add(idx)
            for idx in default:
                if idx not in seen:
                    order.append(idx)
            return order
    if animation_sequence and page_svg is not None and page_svg.is_file():
        ordered = reorder_svg_by_sequence(page_svg, animation_sequence)
        if ordered != page_svg:
            units = animation_unit_elements(page_svg)
            reordered = animation_unit_elements(ordered)
            if len(reordered) == len(units):
                unit_map = {id(elem): i for i, elem in enumerate(units)}
                indices = [unit_map.get(id(elem), i) for i, elem in enumerate(reordered)]
                if sorted(indices) == default:
                    return indices
    if page_svg is not None and page_svg.is_file():
        return infer_spatial_draw_order(page_svg)
    return default


def _animation_unit_drawables(root: ET.Element) -> tuple[ET.Element | None, list[ET.Element]]:
    container = animation_unit_container(root)
    if container is None:
        return None, []
    drawables = (
        _drawable_root_children(root)
        if container is root
        else _container_drawable_children(container, root)
    )
    return container, drawables


def _write_reordered_svg(svg_path: Path, root: ET.Element, stem_suffix: str) -> Path:
    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_{stem_suffix}.svg"
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return out


def reorder_svg_by_unit_order(svg_path: Path, unit_order: list[int]) -> Path:
    """Reorder SVG animation units by explicit unit indices."""
    if not unit_order:
        return svg_path
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    container, drawables = _animation_unit_drawables(root)
    if container is None or len(drawables) < 2:
        return svg_path

    index_map = {i: elem for i, elem in enumerate(drawables)}
    ordered: list[ET.Element] = []
    used: set[int] = set()
    for raw_idx in unit_order:
        idx = int(raw_idx)
        if idx in used or idx not in index_map:
            continue
        ordered.append(index_map[idx])
        used.add(idx)
    for idx, elem in enumerate(drawables):
        if idx not in used:
            ordered.append(elem)

    if ordered == drawables:
        return svg_path

    first_idx = next(i for i, child in enumerate(list(container)) if child in drawables)
    for elem in drawables:
        container.remove(elem)
    for offset, elem in enumerate(ordered):
        container.insert(first_idx + offset, elem)

    return _write_reordered_svg(svg_path, root, "unit_order")


def strip_embedded_image_uses(svg_path: Path) -> Path:
    """Remove drawable image <use> nodes; raster icons are attached separately in Manim."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    removed = False
    for elem in list(root.iter()):
        if _local_tag(elem) != "use":
            continue
        if not _elem_href(elem).startswith("#image-"):
            continue
        cur: ET.Element | None = elem
        while cur is not None:
            if _local_tag(cur) == "defs":
                break
            cur = parent_map.get(cur)
        else:
            parent = parent_map.get(elem)
            if parent is None:
                continue
            parent.remove(elem)
            removed = True

    if not removed:
        return svg_path

    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_noimages.svg"
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return out


def _path_data_valid(d: str | None) -> bool:
    return bool(d and len(d.strip()) >= 2)


def strip_invalid_svg_paths(svg_path: Path) -> Path:
    """Remove <path> nodes without a ``d`` attribute (crashes Manim's SVG parser)."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    removed = False
    for elem in list(root.iter()):
        if _local_tag(elem) != "path":
            continue
        if _path_data_valid(elem.get("d")):
            continue
        parent = parent_map.get(elem)
        if parent is None:
            continue
        parent.remove(elem)
        removed = True

    if not removed:
        return svg_path

    return _write_reordered_svg(svg_path, root, "sanitized")


_CONNECTOR_FILL_COLORS = frozenset({"#e03131", "#c92a2a", "#c2255c"})


def _mark_path_as_connector(path_elem: ET.Element) -> None:
    path_elem.set("data-excal-connector", "true")


def _path_in_text_glyph_group(
    path_elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> bool:
    cur: ET.Element | None = path_elem
    while cur is not None:
        if (cur.get("data-excal-text") or "").strip():
            return True
        cur = parent_map.get(cur)
    return False


def mark_excalidraw_connectors(svg_path: Path) -> Path:
    """Tag dashed curves and arrow bodies so Manim can keep them above later strokes."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    changed = False
    for elem in root.iter():
        if _local_tag(elem) != "path":
            continue
        if _path_in_text_glyph_group(elem, parent_map):
            continue
        if elem.get("data-excal-connector") == "true":
            continue
        if (elem.get("stroke-dasharray") or "").strip():
            _mark_path_as_connector(elem)
            changed = True
            continue
        fill = (elem.get("fill") or "").strip().lower()
        stroke = (elem.get("stroke") or "none").strip().lower()
        if fill in _CONNECTOR_FILL_COLORS and stroke in ("none", "transparent", ""):
            if len(elem.get("d") or "") >= 180:
                _mark_path_as_connector(elem)
                changed = True

    if not changed:
        return svg_path
    return _write_reordered_svg(svg_path, root, "connectors")


def unit_connector_path_flags(unit: ET.Element) -> list[bool]:
    """Return one connector flag per drawable path inside an animation unit."""
    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in unit.iter() for child in parent
    }
    flags: list[bool] = []
    for child in unit.iter():
        if _local_tag(child) != "path":
            continue
        if len(child.get("d") or "") <= 5:
            continue
        if _path_in_text_glyph_group(child, parent_map):
            flags.append(False)
            continue
        flags.append(
            child.get("data-excal-connector") == "true"
            or bool((child.get("stroke-dasharray") or "").strip())
        )
    return flags


def _viewbox_has_anchor(
    root: ET.Element,
    vb_x: float,
    vb_y: float,
    vb_w: float,
    vb_h: float,
) -> bool:
    for elem in root.iter():
        if _local_tag(elem) != "rect":
            continue
        if elem.get("data-manim-viewport") == "true":
            return True
        w = _float_attr(elem, "width")
        h = _float_attr(elem, "height")
        if w < vb_w * 0.98 or h < vb_h * 0.98:
            continue
        x = _float_attr(elem, "x", vb_x)
        y = _float_attr(elem, "y", vb_y)
        if abs(x - vb_x) <= 1.0 and abs(y - vb_y) <= 1.0:
            return True
    return False


def _fmt_svg_number(value: float) -> str:
    return str(int(value)) if abs(value - int(value)) < 1e-6 else str(value)


def normalize_svg_viewport(svg_path: Path) -> Path:
    """Sync width/height to viewBox so Manim and browsers keep the correct aspect ratio."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    vb_x, vb_y, vb_w, vb_h = parse_view_box(root)
    if vb_w <= 0 or vb_h <= 0:
        return svg_path

    changed = False
    width_text = _fmt_svg_number(vb_w)
    height_text = _fmt_svg_number(vb_h)
    if root.get("width") != width_text or root.get("height") != height_text:
        root.set("width", width_text)
        root.set("height", height_text)
        changed = True
    if root.get("preserveAspectRatio") != "xMidYMid meet":
        root.set("preserveAspectRatio", "xMidYMid meet")
        changed = True

    if not _viewbox_has_anchor(root, vb_x, vb_y, vb_w, vb_h):
        tag = root.tag
        ns = tag[: tag.index("}") + 1] if "}" in tag else ""
        anchor = ET.Element(f"{ns}rect")
        anchor.set("x", _fmt_svg_number(vb_x))
        anchor.set("y", _fmt_svg_number(vb_y))
        anchor.set("width", width_text)
        anchor.set("height", height_text)
        anchor.set("fill", "none")
        anchor.set("stroke", "none")
        anchor.set("opacity", "0")
        anchor.set("data-manim-viewport", "true")
        insert_at = 0
        for index, child in enumerate(root):
            if _local_tag(child) in {"defs", "style"}:
                insert_at = index + 1
            else:
                break
        root.insert(insert_at, anchor)
        changed = True

    if not changed:
        return svg_path

    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_viewport.svg"
    out.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return out


def prepare_svg_for_manim(
    svg_path: Path,
    animation_sequence: list[str] | None = None,
    unit_order: list[int] | None = None,
) -> Path:
    """Return an SVG ready for SVGMobject: text as paths, optional draw-order."""
    normalized = normalize_svg_viewport(svg_path)
    converted = convert_svg_text_to_paths(normalized)
    converted = strip_invalid_svg_paths(converted)
    converted = mark_excalidraw_connectors(converted)
    # Per-page unit_order is applied at animation time only; reordering the SVG DOM
    # here shifts raster placement and breaks icon alignment in cells.
    if animation_sequence and not unit_order:
        converted = reorder_svg_by_sequence(converted, animation_sequence)
    converted = ensure_excalidraw_slide_border(converted)
    converted = strip_excalidraw_frame_outlines(converted)
    return strip_invalid_svg_paths(converted)


@dataclass(frozen=True)
class SvgImagePlacement:
    """Raster image embedded in an Excalidraw SVG export."""

    file_path: Path
    x: float
    y: float
    width: float
    height: float
    order: int
    unit_index: int = 0
    rotation: float = 0.0


def _elem_href(elem: ET.Element) -> str:
    return (elem.get("href") or elem.get(f"{{{XLINK_NS}}}href") or "").strip()


def _decode_data_uri(uri: str) -> tuple[bytes, str] | None:
    match = _DATA_URI_RE.match(uri.strip())
    if not match:
        return None
    mime = match.group(1).lower()
    ext = _mime_to_ext(mime)
    try:
        payload = base64.b64decode(re.sub(r"\s+", "", match.group(2)))
    except ValueError:
        return None
    return payload, ext


def _cache_embedded_image(
    uri: str,
    cache_dir: Path,
    *,
    key: str,
) -> Path | None:
    decoded = _decode_data_uri(uri)
    if decoded is None:
        return None
    payload, ext = decoded
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", key)[:64] or "image"
    dest = cache_dir / f"{safe}.{ext}"
    if not dest.exists() or dest.read_bytes() != payload:
        dest.write_bytes(payload)
    return dest


_AFFINE_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mul_affine(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _translate_affine(tx: float, ty: float) -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def _rotate_affine(
    angle_deg: float,
    cx: float = 0.0,
    cy: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rotation = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
    if abs(cx) < 1e-12 and abs(cy) < 1e-12:
        return rotation
    return _mul_affine(
        _translate_affine(cx, cy),
        _mul_affine(rotation, _translate_affine(-cx, -cy)),
    )


def _scale_affine(
    sx: float,
    sy: float,
    cx: float = 0.0,
    cy: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    scaling = (sx, 0.0, 0.0, sy, 0.0, 0.0)
    if abs(cx) < 1e-12 and abs(cy) < 1e-12:
        return scaling
    return _mul_affine(
        _translate_affine(cx, cy),
        _mul_affine(scaling, _translate_affine(-cx, -cy)),
    )


def _affine_from_svg_transform(transform: str) -> tuple[float, float, float, float, float, float]:
    """Parse one SVG transform attribute into a 2D affine matrix."""
    if not transform.strip():
        return _AFFINE_IDENTITY

    ops: list[tuple[int, str, re.Match[str]]] = []
    patterns: list[tuple[str, str]] = [
        (r"translate\(\s*([-\d.eE+]+)(?:[\s,]+([-\d.eE+]+))?\s*\)", "translate"),
        (r"scale\(\s*([-\d.eE+]+)(?:[\s,]+([-\d.eE+]+))?\s*\)", "scale"),
        (
            r"rotate\(\s*([-\d.eE+]+)(?:[\s,]+([-\d.eE+]+))(?:[\s,]+([-\d.eE+]+))?\s*\)",
            "rotate",
        ),
        (
            r"matrix\(\s*([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)\s*\)",
            "matrix",
        ),
    ]
    for pattern, kind in patterns:
        for match in re.finditer(pattern, transform):
            ops.append((match.start(), kind, match))
    ops.sort(key=lambda item: item[0])

    matrix = _AFFINE_IDENTITY
    for _, kind, match in ops:
        if kind == "translate":
            matrix = _mul_affine(
                matrix,
                _translate_affine(float(match.group(1)), float(match.group(2) or 0)),
            )
        elif kind == "scale":
            sx = float(match.group(1))
            sy = float(match.group(2) or sx)
            matrix = _mul_affine(matrix, _scale_affine(sx, sy))
        elif kind == "rotate":
            matrix = _mul_affine(
                matrix,
                _rotate_affine(
                    float(match.group(1)),
                    float(match.group(2) or 0),
                    float(match.group(3) or 0),
                ),
            )
        else:
            matrix = _mul_affine(
                matrix,
                tuple(float(match.group(i)) for i in range(1, 7)),  # type: ignore[arg-type]
            )
    return matrix


def _apply_affine(
    x: float,
    y: float,
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _affine_angle_deg(matrix: tuple[float, float, float, float, float, float]) -> float:
    a, b, _, _, _, _ = matrix
    return math.degrees(math.atan2(b, a))


def _apply_svg_transform(x: float, y: float, transform: str) -> tuple[float, float]:
    return _apply_affine(x, y, _affine_from_svg_transform(transform))


def _elem_page_affine(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float, float, float, float, float]:
    """Accumulate ancestor transforms from one SVG element up to the root <svg>."""
    matrix = _AFFINE_IDENTITY
    cur = parent_map.get(elem)
    while cur is not None and _local_tag(cur) != "svg":
        transform = cur.get("transform")
        if transform:
            matrix = _mul_affine(_affine_from_svg_transform(transform), matrix)
        cur = parent_map.get(cur)
    return matrix


def _placement_affine_from_elem(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float, float, float, float]:
    try:
        width = float(str(elem.get("width") or "0").replace("%", ""))
        height = float(str(elem.get("height") or "0").replace("%", ""))
    except ValueError:
        width, height = 0.0, 0.0
    try:
        x = float(elem.get("x") or 0)
        y = float(elem.get("y") or 0)
    except ValueError:
        x, y = 0.0, 0.0

    matrix = _elem_page_affine(elem, parent_map)
    x, y = _apply_affine(x, y, matrix)
    return x, y, width, height, _affine_angle_deg(matrix)


def _placement_from_use(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[float, float, float, float]:
    x, y, width, height, _rotation = _placement_affine_from_elem(elem, parent_map)
    return x, y, width, height


def _symbol_image_uri(root: ET.Element, symbol_id: str) -> str | None:
    sid = symbol_id.lstrip("#")
    for elem in root.iter():
        if (elem.get("id") or "") != sid:
            continue
        if _local_tag(elem) not in ("symbol", "g"):
            continue
        for child in elem.iter():
            if _local_tag(child) == "image":
                uri = _elem_href(child)
                if uri:
                    return uri
    return None


def extract_svg_raster_placements(svg_path: Path) -> list[SvgImagePlacement]:
    """Find embedded PNG/JPEG images referenced via Excalidraw <use> symbols."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []

    parent_map: dict[ET.Element, ET.Element] = {child: parent for parent in root.iter() for child in parent}
    cache_dir = svg_path.parent / ".excalidraw_render" / "images"
    placements: list[SvgImagePlacement] = []
    order = 0
    units = animation_unit_elements_from_root(root)

    for elem in root.iter():
        tag = _local_tag(elem)
        uri = ""
        width = height = 0.0

        if tag == "use":
            href = _elem_href(elem)
            if not href.startswith("#image-"):
                continue
            uri = _symbol_image_uri(root, href) or ""
            if not uri:
                continue
            x, y, width, height, rotation = _placement_affine_from_elem(elem, parent_map)
        elif tag == "image":
            if any(_local_tag(anc) == "defs" for anc in _parent_chain(elem, parent_map)):
                continue
            uri = _elem_href(elem)
            if not uri.startswith("data:image"):
                continue
            x, y, width, height, rotation = _placement_affine_from_elem(elem, parent_map)
        else:
            continue

        if width <= 0 or height <= 0:
            continue
        key = f"{svg_path.stem}_{order}_{int(width)}x{int(height)}"
        file_path = _cache_embedded_image(uri, cache_dir, key=key)
        if file_path is None:
            continue
        units = animation_unit_elements_from_root(root)
        placements.append(
            SvgImagePlacement(
                file_path=file_path,
                x=x,
                y=y,
                width=width,
                height=height,
                order=order,
                unit_index=_element_unit_index(elem, units, parent_map),
                rotation=rotation,
            )
        )
        order += 1
    return placements


def _parent_chain(elem: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> list[ET.Element]:
    chain: list[ET.Element] = []
    cur: ET.Element | None = elem
    while cur is not None:
        chain.append(cur)
        cur = parent_map.get(cur)
    return chain


@dataclass(frozen=True)
class ExcalidrawPage:
    """One Excalidraw frame/slide treated as a full-screen animation page."""

    page_id: str
    name: str
    x: float
    y: float
    width: float
    height: float

    @property
    def sort_key(self) -> tuple[float, float]:
        return (self.y, self.x)


def _decode_excalidraw_payload(data: dict[str, Any]) -> str:
    raw = data.get("encoded") or ""
    if isinstance(raw, str):
        blob = raw.encode("latin-1")
    else:
        blob = bytes(raw)
    if data.get("compressed"):
        return zlib.decompress(blob).decode("utf-8")
    return blob.decode("utf-8")


def decode_svg_embedded_scene(svg_path: Path) -> dict[str, Any] | None:
    """Decode embedded Excalidraw JSON from SVG metadata when present."""
    try:
        text = svg_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "payload-type:" not in text or "payload-start" not in text:
        return None
    match = re.search(
        r"<!--\s*payload-start\s*-->\s*(.*?)\s*<!--\s*payload-end\s*-->",
        text,
        re.S,
    )
    if not match:
        return None
    b64 = re.sub(r"\s+", "", match.group(1))
    version_match = re.search(r"payload-version:(\d+)", text)
    is_byte_string = version_match is None or version_match.group(1) != "1"
    try:
        decoded = base64.b64decode(b64)
        payload_text = decoded.decode("latin-1") if is_byte_string else decoded.decode("utf-8")
        wrapper = json.loads(payload_text)
    except (ValueError, json.JSONDecodeError):
        return None
    if isinstance(wrapper, dict) and wrapper.get("type") == "excalidraw":
        return wrapper
    if isinstance(wrapper, dict) and "encoded" in wrapper:
        try:
            payload_text = _decode_excalidraw_payload(wrapper)
        except zlib.error:
            return None
    else:
        return None
    try:
        scene = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return scene if scene.get("type") == "excalidraw" else None


def pages_from_excalidraw_scene(scene: dict[str, Any]) -> list[ExcalidrawPage]:
    """Return frame/slide pages from embedded or .excalidraw scene data."""
    elements = scene.get("elements") or []
    frames: list[ExcalidrawPage] = []
    for el in elements:
        if not isinstance(el, dict) or el.get("isDeleted"):
            continue
        if el.get("type") not in ("frame", "magicframe"):
            continue
        frames.append(
            ExcalidrawPage(
                page_id=str(el.get("id") or ""),
                name=str(el.get("name") or "Page").strip() or "Page",
                x=float(el.get("x") or 0),
                y=float(el.get("y") or 0),
                width=float(el.get("width") or 0),
                height=float(el.get("height") or 0),
            )
        )
    frames = [p for p in frames if p.width > 10 and p.height > 10 and p.page_id]
    frames.sort(key=lambda p: p.sort_key)
    return frames


def _clip_path_frame_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"url\(#([^)]+)\)", value)
    if not match:
        return None
    frame_id = match.group(1)
    if frame_id.startswith("image-clipPath-"):
        return None
    return frame_id


def _clip_path_rect_bounds(clip_elem: ET.Element) -> tuple[float, float, float, float] | None:
    rect = next((child for child in clip_elem if _local_tag(child) == "rect"), None)
    if rect is None:
        return None
    tx, ty = _parse_translate(rect.get("transform") or "")
    try:
        w = float(str(rect.get("width") or "0"))
        h = float(str(rect.get("height") or "0"))
    except ValueError:
        return None
    if w < 10 or h < 10:
        return None
    return tx, ty, w, h


def _pages_from_svg_clip_paths(root: ET.Element) -> list[ExcalidrawPage]:
    pages: list[ExcalidrawPage] = []
    for elem in root.iter():
        if _local_tag(elem) != "clipPath":
            continue
        frame_id = elem.get("id") or ""
        if not frame_id or frame_id.startswith("image-clipPath-"):
            continue
        bounds = _clip_path_rect_bounds(elem)
        if bounds is None:
            continue
        x, y, w, h = bounds
        pages.append(
            ExcalidrawPage(
                page_id=frame_id,
                name=f"Page {len(pages) + 1}",
                x=x,
                y=y,
                width=w,
                height=h,
            )
        )
    pages.sort(key=lambda p: p.sort_key)
    return pages


def _element_page_id(elem: ET.Element, pages: list[ExcalidrawPage]) -> str | None:
    clip_id = _clip_path_frame_id(elem.get("clip-path"))
    if clip_id:
        return clip_id
    for child in elem.iter():
        if child is elem:
            continue
        clip_id = _clip_path_frame_id(child.get("clip-path"))
        if clip_id:
            return clip_id
    tx, ty = _parse_translate(elem.get("transform") or "")
    cx, cy = tx, ty
    for page in pages:
        if page.x <= cx <= page.x + page.width and page.y <= cy <= page.y + page.height:
            return page.page_id
    return None


def detect_svg_pages(svg_path: Path) -> list[ExcalidrawPage]:
    """Detect multiple Excalidraw frames/slides in an SVG export."""
    scene = decode_svg_embedded_scene(svg_path)
    if scene:
        pages = pages_from_excalidraw_scene(scene)
        if len(pages) > 1:
            return pages
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []
    pages = _pages_from_svg_clip_paths(root)
    if len(pages) > 1:
        return pages
    return []


def build_page_svg(svg_path: Path, page: ExcalidrawPage, pages: list[ExcalidrawPage]) -> Path:
    """Build a single-page SVG cropped to one frame/slide."""
    root = ET.parse(svg_path).getroot()
    svg_ns = "http://www.w3.org/2000/svg"
    drawables = _drawable_root_children(root)
    page_elems: list[ET.Element] = []
    for elem in drawables:
        elem_page = _element_page_id(elem, pages)
        if elem_page == page.page_id:
            page_elems.append(copy.deepcopy(elem))

    if not page_elems:
        raise ValueError(f"No drawable content found for page {page.name!r}")

    new_root = ET.Element(f"{{{svg_ns}}}svg")
    new_root.set("xmlns", svg_ns)
    new_root.set("viewBox", f"0 0 {page.width} {page.height}")
    new_root.set("width", str(page.width))
    new_root.set("height", str(page.height))

    for child in root:
        if _local_tag(child) == "defs":
            new_root.append(copy.deepcopy(child))

    bg = ET.Element(f"{{{svg_ns}}}rect")
    bg.set("x", "0")
    bg.set("y", "0")
    bg.set("width", str(page.width))
    bg.set("height", str(page.height))
    canvas = svg_canvas_background_color(svg_path) or "#ffffff"
    bg.set("fill", canvas)
    new_root.append(bg)

    wrapper = ET.Element(f"{{{svg_ns}}}g")
    wrapper.set("transform", f"translate({-page.x} {-page.y})")
    for elem in page_elems:
        wrapper.append(elem)
    new_root.append(wrapper)

    render_dir = svg_path.parent / ".excalidraw_render" / "pages"
    render_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w.-]+", "_", page.page_id)[:32]
    out = render_dir / f"{svg_path.stem}_page_{safe_id}.svg"
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(new_root, encoding="unicode"),
        encoding="utf-8",
    )
    return out


def pages_from_excalidraw_file(path: Path) -> list[ExcalidrawPage]:
    """Detect pages in .excalidraw JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "excalidraw":
        return []
    return pages_from_excalidraw_scene(data)


def parse_animation_sequence_phrases(note: str) -> list[str]:
    """Parse a natural-language note into ordered animation labels."""
    text = note.strip().lower()
    if not text:
        return []
    text = re.sub(
        r"^(?:ok(?:\s+so)?[, ]+|so[, ]+|please[, ]+|i want(?: it)? to(?: write| animate| show| draw)?[, ]+)+",
        "",
        text,
    )
    parts = re.split(r"\bthen\b", text)
    phrases: list[str] = []
    for part in parts:
        part = part.strip()
        part = re.sub(r"^(?:write|animate|show|draw)\s+", "", part)
        part = re.sub(r"\s+(?:first|next|last|finally)$", "", part)
        part = re.sub(r"\s+text$", "", part)
        part = part.strip(" ,.")
        if len(part) >= 2:
            phrases.append(part)
    return phrases


def _element_text_label(elem: ET.Element) -> str:
    label = (elem.get("data-excal-text") or "").strip().lower()
    if label:
        return label
    if _local_tag(elem) == "text":
        return "".join(elem.itertext()).strip().lower()
    for child in elem.iter():
        if child is elem:
            continue
        nested = (child.get("data-excal-text") or "").strip().lower()
        if nested:
            return nested
        if _local_tag(child) == "text":
            text = "".join(child.itertext()).strip().lower()
            if text:
                return text
    return ""


def _element_path_weight(elem: ET.Element) -> int:
    weight = 0
    tag = _local_tag(elem)
    if tag == "path":
        weight += max(len(elem.get("d") or ""), 1)
    for child in elem.iter():
        if child is elem:
            continue
        if _local_tag(child) == "path":
            weight += max(len(child.get("d") or ""), 1)
    return weight


def _phrase_match_score(phrase: str, label: str, *, is_path_unit: bool) -> float:
    phrase = phrase.strip().lower()
    label = label.strip().lower()
    if not phrase:
        return 0.0
    if "logo" in phrase and is_path_unit and not label:
        return 0.85
    if label:
        if phrase in label or label in phrase:
            return 1.0
        phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase))
        label_tokens = set(re.findall(r"[a-z0-9]+", label))
        if phrase_tokens and label_tokens:
            overlap = len(phrase_tokens & label_tokens) / len(phrase_tokens)
            if overlap >= 0.5:
                return 0.6 + overlap * 0.35
    if not is_path_unit and not label:
        return 0.0
    if "python" in phrase and "logo" not in phrase and "ai" in phrase and "ai" in label:
        return 0.9
    return 0.0


def _drawable_root_children(root: ET.Element) -> list[ET.Element]:
    skip_tags = {"defs", "metadata", "style", "mask"}
    _, _, vb_w, vb_h = parse_view_box(root)
    drawables: list[ET.Element] = []
    for child in list(root):
        tag = _local_tag(child)
        if tag in skip_tags:
            continue
        if tag == "rect" and _rect_covers_view_box(child, vb_w, vb_h):
            continue
        drawables.append(child)
    return drawables


def animation_unit_container(root: ET.Element) -> ET.Element | None:
    """Return the element whose children are per-item animation units."""
    drawables = _drawable_root_children(root)
    if len(drawables) == 1 and _local_tag(drawables[0]) == "g":
        inner = _container_drawable_children(drawables[0], root)
        if len(inner) >= 2:
            return drawables[0]
    if len(drawables) >= 2:
        return root
    return None


def _container_drawable_children(container: ET.Element, root: ET.Element) -> list[ET.Element]:
    skip_tags = {"defs", "metadata", "style", "mask"}
    _, _, vb_w, vb_h = parse_view_box(root)
    drawables: list[ET.Element] = []
    for child in list(container):
        tag = _local_tag(child)
        if tag in skip_tags:
            continue
        if tag == "rect" and _rect_covers_view_box(child, vb_w, vb_h):
            continue
        drawables.append(child)
    return drawables


def animation_unit_elements(svg_path: Path) -> list[ET.Element]:
    """Ordered animation units (nested slide groups or top-level drawables)."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []
    return animation_unit_elements_from_root(root)


def _is_non_animatable_unit(elem: ET.Element, vb: tuple[float, float, float, float]) -> bool:
    """Skip auto-inserted borders and empty groups from draw-order indexing."""
    if _element_has_raster_image(elem) or _element_text_label(elem):
        return False

    path_weight = _element_path_weight(elem)
    if path_weight > 0:
        return False

    _, _, vb_w, vb_h = vb
    vb_area = max(vb_w * vb_h, 1.0)
    area = 0.0
    for child in elem.iter():
        if _local_tag(child) == "rect":
            area = max(area, _float_attr(child, "width") * _float_attr(child, "height"))

    fill, stroke = _element_unit_colors(elem)
    if not fill and stroke and area >= vb_area * 0.15:
        return True

    return area <= 0.0


def _filter_animatable_units(
    units: list[ET.Element],
    vb: tuple[float, float, float, float],
) -> list[ET.Element]:
    return [unit for unit in units if not _is_non_animatable_unit(unit, vb)]


def animation_unit_elements_from_root(root: ET.Element) -> list[ET.Element]:
    """Ordered animation units from an already-parsed SVG root."""
    vb = parse_view_box(root)
    container = animation_unit_container(root)
    if container is None:
        units = _drawable_root_children(root)
    elif container is root:
        units = _drawable_root_children(root)
    else:
        units = _container_drawable_children(container, root)
    return _filter_animatable_units(units, vb)


def _element_has_raster_image(elem: ET.Element) -> bool:
    for child in elem.iter():
        if _local_tag(child) != "use":
            continue
        href = _elem_href(child)
        if href.startswith("#image-"):
            return True
    return False


def _element_unit_index(
    elem: ET.Element,
    units: list[ET.Element],
    parent_map: dict[ET.Element, ET.Element],
) -> int:
    if not units:
        return 0
    unit_ids = {id(unit) for unit in units}
    cur: ET.Element | None = elem
    while cur is not None:
        if id(cur) in unit_ids:
            return units.index(cur)
        cur = parent_map.get(cur)
    return 0


def unit_path_layer_count(unit: ET.Element) -> int:
    """Approximate Manim stroke layers contributed by one animation unit."""
    count = 0
    for child in unit.iter():
        if _local_tag(child) != "path":
            continue
        if len(child.get("d") or "") > 5:
            count += 1
    return count


def normalize_animation_sequence(phrases: list[str] | None) -> list[str]:
    """Clean phrases from chat notes or generated scene code."""
    if not phrases:
        return []
    out: list[str] = []
    for raw in phrases:
        part = raw.strip().lower()
        part = re.sub(r"^(?:write|animate|show|draw)\s+", "", part)
        part = re.sub(r"\s+(?:first|next|last|finally|text)$", "", part)
        part = part.strip(" ,.")
        if len(part) >= 2:
            out.append(part)
    return out


def reorder_svg_by_sequence(svg_path: Path, phrases: list[str]) -> Path:
    """Reorder SVG animation units to match user animation sequence notes."""
    phrases = normalize_animation_sequence(phrases)
    if not phrases:
        return svg_path
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    container = animation_unit_container(root)
    if container is None:
        return svg_path
    drawables = (
        _drawable_root_children(root)
        if container is root
        else _container_drawable_children(container, root)
    )
    if len(drawables) < 2:
        return svg_path

    ordered: list[ET.Element] = []
    used: set[int] = set()

    for phrase in phrases:
        phrase_lower = phrase.lower()
        if any(token in phrase_lower for token in ("logo", "icon", "image")):
            for elem in drawables:
                eid = id(elem)
                if eid in used:
                    continue
                if _element_has_raster_image(elem):
                    ordered.append(elem)
                    used.add(eid)
                    break
            else:
                for elem in drawables:
                    eid = id(elem)
                    if eid in used:
                        continue
                    label = _element_text_label(elem)
                    if label:
                        continue
                    if _element_path_weight(elem) < 20:
                        continue
                    ordered.append(elem)
                    used.add(eid)
                    break
            continue

        best_elem: ET.Element | None = None
        best_score = 0.35
        for elem in drawables:
            eid = id(elem)
            if eid in used:
                continue
            label = _element_text_label(elem)
            score = _phrase_match_score(phrase, label, is_path_unit=not bool(label))
            if score > best_score:
                best_score = score
                best_elem = elem
        if best_elem is not None:
            ordered.append(best_elem)
            used.add(id(best_elem))

    for elem in drawables:
        if id(elem) not in used:
            ordered.append(elem)

    if ordered == drawables:
        return svg_path

    first_idx = next(i for i, child in enumerate(list(container)) if child in drawables)
    for elem in drawables:
        container.remove(elem)
    for offset, elem in enumerate(ordered):
        container.insert(first_idx + offset, elem)

    return _write_reordered_svg(svg_path, root, "ordered")


def convert_svg_text_to_paths(svg_path: Path) -> Path:
    """Replace SVG <text> nodes with per-glyph <path> outlines using embedded fonts."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path

    parent_map: dict[ET.Element, ET.Element] = {child: parent for parent in root.iter() for child in parent}
    text_elems = [elem for elem in root.iter() if _local_tag(elem) == "text" and "".join(elem.itertext()).strip()]
    if not text_elems:
        return svg_path

    embedded = extract_svg_embedded_fonts(svg_path)
    font_cache: dict[Path, TTFont] = {}
    changed = False
    svg_ns = "http://www.w3.org/2000/svg"

    for elem in text_elems:
        content = "".join(elem.itertext()).strip()
        if not content:
            continue
        fs_raw = str(elem.get("font-size") or "24").replace("px", "")
        try:
            font_size_px = float(fs_raw)
        except ValueError:
            font_size_px = 24.0
        fill = elem.get("fill") or "#ffffff"
        anchor = elem.get("text-anchor") or "start"

        # Keep coordinates local — parent <g transform="..."> must stay intact.
        tx = float(elem.get("x", 0) or 0)
        ty = float(elem.get("y", 0) or 0)

        path_elems = _text_to_paths_with_fallback(
            content,
            elem.get("font-family") or "",
            embedded,
            font_cache,
            anchor_x=tx,
            baseline_y=ty,
            font_size_px=font_size_px,
            fill=fill,
            anchor=anchor,
        )
        if not path_elems:
            continue

        parent = parent_map.get(elem)
        if parent is None:
            continue
        idx = list(parent).index(elem)
        parent.remove(elem)
        group = ET.Element(f"{{{svg_ns}}}g")
        group.set("data-excal-text", content[:48])
        for path_elem in path_elems:
            group.append(path_elem)
        parent.insert(idx, group)
        # Bubble label to outer transform wrapper so draw-order matching sees it.
        if (
            _local_tag(parent) == "g"
            and parent.get("transform")
            and len(list(parent)) == 1
            and not parent.get("data-excal-text")
        ):
            parent.set("data-excal-text", content[:48])
        changed = True

    if not changed:
        return svg_path

    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_paths.svg"
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return out


def _hmtx_advance(hmtx: Any, glyph_name: str, scale: float) -> float:
    """Return horizontal advance for a glyph, with subset-font fallbacks."""
    for name in (glyph_name, ".notdef"):
        try:
            return float(hmtx[name][0]) * scale
        except (KeyError, IndexError, TypeError):
            continue
    try:
        metrics = getattr(hmtx, "metrics", None) or {}
        if metrics:
            return float(sum(pair[0] for pair in metrics.values()) / len(metrics)) * scale
    except (TypeError, ZeroDivisionError):
        pass
    return 250.0 * scale


def _text_to_path_elements(
    text: str,
    font: TTFont,
    *,
    anchor_x: float,
    baseline_y: float,
    font_size_px: float,
    fill: str,
    anchor: str,
) -> list[ET.Element]:
    """Layout text with Excalidraw's embedded subset font; one path per glyph."""
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap() or {}
    hmtx = font["hmtx"]
    upem = float(font["head"].unitsPerEm or 1000)
    scale = font_size_px / upem

    advances: list[tuple[str, float]] = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if not gname:
            advances.append(("", _hmtx_advance(hmtx, ".notdef", scale)))
            continue
        advances.append((gname, _hmtx_advance(hmtx, gname, scale)))

    total_width = sum(adv for _, adv in advances)
    if anchor == "middle":
        cursor_x = anchor_x - total_width / 2
    elif anchor == "end":
        cursor_x = anchor_x - total_width
    else:
        cursor_x = anchor_x

    svg_ns = "http://www.w3.org/2000/svg"
    paths: list[ET.Element] = []
    for gname, adv in advances:
        if not gname:
            cursor_x += adv
            continue
        try:
            glyph = glyph_set[gname]
        except KeyError:
            cursor_x += adv
            continue
        pen = SVGPathPen(glyph_set)
        transform = TransformPen(
            pen,
            (scale, 0, 0, -scale, cursor_x, baseline_y),
        )
        try:
            glyph.draw(transform)
        except (KeyError, TypeError, AttributeError):
            cursor_x += adv
            continue
        d = pen.getCommands()
        if not d:
            cursor_x += adv
            continue
        path_elem = ET.Element(f"{{{svg_ns}}}path")
        path_elem.set("d", d)
        path_elem.set("fill", fill)
        path_elem.set("stroke", "none")
        paths.append(path_elem)
        cursor_x += adv
    return paths
