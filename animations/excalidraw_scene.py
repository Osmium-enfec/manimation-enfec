"""Load Excalidraw SVG / .excalidraw files and animate draw-order reveal in Manim."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from manim import *
import numpy as np

from manim.utils.rate_functions import smooth

from excalidraw_parser import (
    ExcalidrawPage,
    SvgImagePlacement,
    _apply_affine,
    _elem_page_affine,
    _element_bbox_center,
    _element_has_raster_image,
    _element_path_weight,
    _element_text_label,
    _element_text_label_display,
    _element_unit_colors,
    _nearest_text_label,
    animation_unit_elements,
    build_page_svg,
    convert_svg_text_to_paths,
    describe_animation_units,
    detect_svg_pages,
    extract_svg_raster_placements,
    _is_excalidraw_frame_title,
    _is_text_animation_unit,
    pages_from_excalidraw_file,
    parse_view_box,
    prepare_svg_for_manim,
    resolve_page_unit_order,
    strip_embedded_image_uses,
    svg_canvas_background_color,
    unit_connector_path_flags,
    unit_path_layer_count,
)

# Excalidraw Y-down → Manim Y-up
_EXCAL_SCALE = 0.012
_IMAGE_REVEAL_MIN = 0.55
_IMAGE_REVEAL_MAX = 1.05


def _image_reveal_run_time(run_time: float) -> float:
    """Gentle image fade duration that scales with scene budget."""
    return max(_IMAGE_REVEAL_MIN, min(_IMAGE_REVEAL_MAX, run_time * 1.15))


def _restore_vector_opacity(mob: Mobject) -> None:
    """Restore vector opacity so fade/draw animations interpolate to a visible end state."""
    for sm in mob.get_family():
        if isinstance(sm, ImageMobject) or _is_embedded_icon(sm):
            continue
        sm.set_opacity(1)


def _image_reveal_animation(image: Mobject) -> Animation:
    """Opacity fade for raster images (FadeIn breaks when the target starts at opacity 0)."""
    image.set_opacity(0)
    return image.animate.set_opacity(1)


def _play_image_layer(
    scene: Scene,
    layer: Mobject,
    image_target: Mobject,
    *,
    run_time: float,
) -> None:
    if layer not in scene.mobjects:
        scene.add(layer)
    if isinstance(image_target, ImageMobject):
        scene.play(
            _image_reveal_animation(image_target),
            run_time=_image_reveal_run_time(run_time),
            rate_func=smooth,
        )
        return
    _restore_vector_opacity(image_target)
    scene.play(
        FadeIn(image_target, rate_func=smooth),
        run_time=_image_reveal_run_time(run_time),
    )


def _hide_draw_group(group: Mobject) -> None:
    """Hide every drawable part before the sequence reveals them one step at a time."""
    svg_root = _svg_vector_root(group)
    if svg_root is not None:
        for mob in svg_root.get_family():
            if isinstance(mob, ImageMobject):
                continue
            if _is_cell_background(mob, svg_root) or _is_background_fill(mob, svg_root):
                continue
            mob.set_opacity(0)
    for wrapper in _collect_icon_wrappers(group).values():
        _hide_layer_for_sequence(wrapper)


def _ensure_draw_group_on_scene(scene: Scene, group: Mobject) -> None:
    svg_root = _svg_vector_root(group)
    if svg_root is not None and svg_root not in scene.mobjects:
        scene.add(svg_root)


def _image_reveal_target(layer: Mobject) -> Mobject | None:
    """Return the raster image to fade in, if this layer is image-based."""
    if _is_embedded_icon(layer):
        return layer
    if _is_icon_wrapper(layer):
        return layer.submobjects[0]
    if isinstance(layer, (VGroup, Group)) and len(layer.submobjects) == 1:
        return _image_reveal_target(layer.submobjects[0])
    return None


def _is_embedded_icon(mob: Mobject) -> bool:
    return isinstance(mob, ImageMobject) or bool(getattr(mob, "excal_is_svg_icon", False))


def _is_icon_wrapper(mob: Mobject) -> bool:
    return (
        isinstance(mob, Group)
        and len(mob.submobjects) == 1
        and _is_embedded_icon(mob.submobjects[0])
    )


def _resolve_icon_path(file_path: Path) -> Path:
    """Normalize cached icon paths (legacy .svgxml → .svg)."""
    if file_path.suffix.lower() == ".svgxml" and file_path.is_file():
        svg_path = file_path.with_suffix(".svg")
        if not svg_path.exists() or svg_path.read_bytes() != file_path.read_bytes():
            svg_path.write_bytes(file_path.read_bytes())
        return svg_path
    return file_path


def _load_embedded_icon(file_path: Path) -> Mobject:
    """Load an Excalidraw embedded PNG/JPEG/SVG icon."""
    path = _resolve_icon_path(file_path)
    if path.suffix.lower() == ".svg":
        icon = SVGMobject(str(path))
        icon.excal_is_svg_icon = True
        return icon
    return ImageMobject(str(path))


def _raster_content_bbox(image_path: Path) -> tuple[int, int, int, int] | None:
    """Pixel bounds of visible raster content, excluding solid margins."""
    path = _resolve_icon_path(image_path)
    if path.suffix.lower() == ".svg":
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path)
    except OSError:
        return None

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA").getbbox()

    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if red < 250 or green < 250 or blue < 250:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < 0:
        return None
    return (min_x, min_y, max_x + 1, max_y + 1)


def _trim_icon_placement(
    placement: SvgImagePlacement,
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> SvgImagePlacement:
    """Map a pixel crop back into SVG page coordinates."""
    bx0, by0, bx1, by1 = bbox
    image_w, image_h = image_size
    if image_w <= 0 or image_h <= 0:
        return placement
    if bx0 <= 0 and by0 <= 0 and bx1 >= image_w and by1 >= image_h:
        return placement
    return SvgImagePlacement(
        file_path=placement.file_path,
        x=placement.x + placement.width * bx0 / image_w,
        y=placement.y + placement.height * by0 / image_h,
        width=placement.width * (bx1 - bx0) / image_w,
        height=placement.height * (by1 - by0) / image_h,
        order=placement.order,
        unit_index=placement.unit_index,
        rotation=placement.rotation,
    )


def _load_trimmed_raster_icon(
    placement: SvgImagePlacement,
) -> tuple[Mobject, SvgImagePlacement]:
    """Load a raster icon cropped to visible content with matching page placement."""
    path = _resolve_icon_path(placement.file_path)
    if path.suffix.lower() == ".svg":
        return _load_embedded_icon(path), placement

    try:
        from PIL import Image
    except ImportError:
        return _load_embedded_icon(path), placement

    try:
        image = Image.open(path)
    except OSError:
        return _load_embedded_icon(path), placement

    image_size = image.size
    bbox = _raster_content_bbox(path)
    if bbox is None:
        return ImageMobject(str(path)), placement

    trimmed = _trim_icon_placement(placement, bbox, image_size)
    cropped = image.crop(bbox)
    if cropped.mode not in ("RGBA", "RGB"):
        cropped = cropped.convert("RGBA" if "A" in cropped.getbands() else "RGB")
    return ImageMobject(np.array(cropped)), trimmed


def animate_excalidraw_file(
    scene: Scene,
    file_path: str | Path,
    *,
    total_run_time: float = 6.0,
    hold_time: float = 1.0,
    fade_out: bool = True,
    animation_sequence: list[str] | None = None,
    page_sequences: dict[int, list[int]] | None = None,
) -> None:
    """Main entry: load file and play draw-order animation."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Drawing not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".svg":
        pages = detect_svg_pages(path)
        bg = _svg_background(path)
        if len(pages) > 1:
            _animate_svg_pages(
                scene,
                path,
                pages,
                bg=bg,
                total_run_time=total_run_time,
                hold_time=hold_time,
                fade_out=fade_out,
                animation_sequence=animation_sequence,
                page_sequences=page_sequences,
            )
            return
        group = _load_svg_group(
            path,
            animation_sequence=animation_sequence,
            unit_order=_resolve_unit_order(0, path, animation_sequence, page_sequences),
        )
    elif suffix in (".excalidraw", ".json"):
        pages = pages_from_excalidraw_file(path)
        if len(pages) > 1:
            _animate_excalidraw_json_pages(
                scene,
                path,
                pages,
                total_run_time=total_run_time,
                hold_time=hold_time,
                fade_out=fade_out,
            )
            return
        group, bg = _load_excalidraw_group(path)
        scene.camera.background_color = bg
        _fit_to_frame(group)
        group.move_to(ORIGIN)
        _play_draw_layers(scene, group, total_run_time=total_run_time, hold_time=hold_time, fade_out=fade_out)
        return
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    scene.camera.background_color = bg
    _fit_svg_group(group)
    group.move_to(ORIGIN)
    _play_draw_layers(scene, group, total_run_time=total_run_time, hold_time=hold_time, fade_out=fade_out)


def _sort_text_glyphs(mobs: list[VMobject]) -> list[VMobject]:
    """Reading order: top-to-bottom, then left-to-right."""
    return sorted(mobs, key=lambda m: (-round(float(m.get_center()[1]), 1), float(m.get_center()[0])))


def _glyph_reveal_animation(glyph: VMobject) -> Animation:
    if float(glyph.get_fill_opacity() or 0) > 0.05:
        return Write(glyph)
    _ensure_visible_stroke(glyph)
    return Create(glyph)


def _split_unit_stroke_layers(
    unit: ET.Element,
    strokes: list[Mobject],
) -> tuple[list[Mobject], list[Mobject]]:
    """Separate connector/arrow strokes from text or illustration strokes in one unit."""
    flags = unit_connector_path_flags(unit)
    if not flags or len(flags) != len(strokes):
        return strokes, []

    text_like: list[Mobject] = []
    connectors: list[Mobject] = []
    for is_connector, stroke in zip(flags, strokes):
        if is_connector:
            connectors.append(stroke)
        else:
            text_like.append(stroke)
    return text_like, connectors


def _hide_layer_for_sequence(mob: Mobject) -> None:
    """Keep later sequence steps invisible until their turn."""
    for sm in mob.get_family():
        if isinstance(sm, ImageMobject):
            sm.set_opacity(0)
            sm.set_fill_opacity(0)
        else:
            sm.set_opacity(0)


def _finalize_layer_reveal(mob: Mobject) -> None:
    """Lock one sequence step to fully opaque before the next step starts."""
    for sm in mob.get_family():
        if isinstance(sm, ImageMobject):
            sm.set_opacity(1)
            sm.set_fill_opacity(1)
        else:
            sm.set_opacity(1)


def _connector_reveal_animation(layer: Mobject) -> Animation:
    """Reveal dashed Excalidraw connectors during their sequence step."""
    if isinstance(layer, VGroup):
        parts = [
            DashedVMobject(sm, num_dashes=14, dashed_ratio=0.42)
            if isinstance(sm, VMobject) and float(sm.get_stroke_opacity() or 0) > 0.05
            else sm
            for sm in layer.submobjects
        ]
        return Create(VGroup(*parts))

    if isinstance(layer, VMobject) and float(layer.get_stroke_opacity() or 0) > 0.05:
        _ensure_visible_stroke(layer, min_width=2.5)
        return Create(DashedVMobject(layer, num_dashes=14, dashed_ratio=0.42))

    return Create(layer)


def _play_text_typing_layer(
    scene: Scene,
    layer: Mobject,
    *,
    run_time: float,
) -> bool:
    """Reveal converted Excalidraw text glyph-by-glyph."""
    if isinstance(layer, VGroup):
        glyphs = [
            sm
            for sm in layer.submobjects
            if isinstance(sm, VMobject) and sm.get_num_points() > 0
        ]
    elif isinstance(layer, VMobject) and layer.get_num_points() > 0:
        glyphs = [layer]
    else:
        return False

    if not glyphs:
        return False

    for glyph in glyphs:
        _restore_vector_opacity(glyph)

    if len(glyphs) == 1:
        scene.play(_glyph_reveal_animation(glyphs[0]), run_time=run_time)
        _finalize_layer_reveal(layer)
        return True

    glyphs = _sort_text_glyphs(glyphs)
    lag = min(0.15, 0.85 / max(len(glyphs), 1))
    scene.play(
        LaggedStart(*[_glyph_reveal_animation(g) for g in glyphs], lag_ratio=lag),
        run_time=max(run_time, len(glyphs) * 0.035),
    )
    _finalize_layer_reveal(layer)
    return True


def _set_hidden_for_reveal(mob: Mobject) -> None:
    _hide_layer_for_sequence(mob)


def _play_animation_layer(
    scene: Scene,
    layer: Mobject,
    *,
    run_time: float,
    svg_root: Mobject | None = None,
) -> None:
    if getattr(layer, "excal_is_text_unit", False):
        if _play_text_typing_layer(scene, layer, run_time=run_time):
            return
    if getattr(layer, "excal_is_connector", False):
        _restore_vector_opacity(layer)
        scene.play(_connector_reveal_animation(layer), run_time=run_time)
        _finalize_layer_reveal(layer)
        return
    image_target = _image_reveal_target(layer)
    if image_target is not None:
        _play_image_layer(scene, layer, image_target, run_time=run_time)
        _finalize_layer_reveal(layer)
        return
    elif isinstance(layer, VMobject) and layer.get_num_points() > 0:
        has_stroke = float(layer.get_stroke_opacity() or 0) > 0.05 and float(layer.get_stroke_width() or 0) > 0
        has_fill = float(layer.get_fill_opacity() or 0) > 0.05
        _restore_vector_opacity(layer)
        if has_stroke:
            _ensure_visible_stroke(layer)
        if has_fill and not has_stroke and _is_solid_panel_fill(layer, svg_root):
            if layer not in scene.mobjects:
                scene.add(layer)
            scene.play(FadeIn(layer, rate_func=smooth), run_time=run_time * 0.75)
            _finalize_layer_reveal(layer)
            return
        if has_stroke and not has_fill:
            scene.play(FadeIn(layer, rate_func=smooth), run_time=run_time * 0.9)
        else:
            scene.play(Create(layer), run_time=run_time)
        _finalize_layer_reveal(layer)
    elif isinstance(layer, VGroup) and layer.submobjects:
        anims: list[Animation] = []
        for sm in layer.submobjects:
            nested_image = _image_reveal_target(sm)
            if nested_image is not None:
                if sm not in scene.mobjects:
                    scene.add(sm)
                if isinstance(nested_image, ImageMobject):
                    nested_image.set_opacity(0)
                    anims.append(nested_image.animate.set_opacity(1))
                else:
                    _restore_vector_opacity(nested_image)
                    anims.append(FadeIn(nested_image, rate_func=smooth))
                continue
            if not isinstance(sm, VMobject) or sm.get_num_points() <= 0:
                continue
            _restore_vector_opacity(sm)
            if float(sm.get_stroke_opacity() or 0) > 0.05:
                _ensure_visible_stroke(sm)
            has_stroke = float(sm.get_stroke_opacity() or 0) > 0.05 and float(sm.get_stroke_width() or 0) > 0
            has_fill = float(sm.get_fill_opacity() or 0) > 0.05
            if has_stroke and not has_fill:
                anims.append(FadeIn(sm, rate_func=smooth))
            else:
                anims.append(Create(sm))
        if anims:
            scene.play(
                AnimationGroup(*anims, lag_ratio=0),
                run_time=max(run_time * 0.9, _image_reveal_run_time(run_time) if len(anims) == 1 else run_time * 0.9),
                rate_func=smooth,
            )
        _finalize_layer_reveal(layer)
    elif isinstance(layer, Mobject):
        _restore_vector_opacity(layer)
        scene.play(FadeIn(layer, rate_func=smooth), run_time=run_time * 0.9)
        _finalize_layer_reveal(layer)


def _svg_vector_root(group: Mobject) -> Mobject | None:
    for sm in group.submobjects:
        if _is_embedded_icon(sm):
            continue
        return sm.submobjects[0] if len(sm.submobjects) == 1 else sm
    return None


def _play_draw_layers(
    scene: Scene,
    group: Mobject,
    *,
    total_run_time: float,
    hold_time: float,
    fade_out: bool,
) -> None:
    _add_panel_backgrounds(scene, group)
    layers = _animation_layers(group)
    svg_root = _svg_vector_root(group)
    _ensure_draw_group_on_scene(scene, group)
    _hide_draw_group(group)
    n = max(len(layers), 1)
    per = max(0.12, (total_run_time - hold_time) / n)

    for layer in layers:
        _play_animation_layer(scene, layer, run_time=per, svg_root=svg_root)

    scene.wait(hold_time)
    if fade_out:
        scene.play(FadeOut(group), run_time=0.3)


def _animate_svg_pages(
    scene: Scene,
    svg_path: Path,
    pages: list[ExcalidrawPage],
    *,
    bg: str,
    total_run_time: float,
    hold_time: float,
    fade_out: bool,
    animation_sequence: list[str] | None,
    page_sequences: dict[int, list[int]] | None = None,
) -> None:
    scene.camera.background_color = bg
    n_pages = len(pages)
    transition = min(0.35, max(0.2, total_run_time * 0.05))
    page_hold = max(0.25, hold_time / n_pages)
    draw_budget = max(0.8, (total_run_time - page_hold * n_pages - transition * (n_pages - 1)) / n_pages)

    current: Mobject | None = None
    for index, page in enumerate(pages):
        page_svg = build_page_svg(svg_path, page, pages)
        unit_order = _resolve_unit_order(index, page_svg, animation_sequence, page_sequences)
        group = _load_svg_group(
            page_svg,
            animation_sequence=animation_sequence if not unit_order else None,
            unit_order=unit_order,
        )
        _fit_svg_group(group)
        group.move_to(ORIGIN)
        if current is not None:
            scene.play(FadeOut(current), run_time=transition)

        _add_panel_backgrounds(scene, group)
        layers = _animation_layers(group)
        svg_root = _svg_vector_root(group)
        layer_count = max(len(layers), 1)
        per = max(0.12, draw_budget / layer_count)
        _ensure_draw_group_on_scene(scene, group)
        _hide_draw_group(group)
        for layer in layers:
            _play_animation_layer(scene, layer, run_time=per, svg_root=svg_root)

        scene.wait(page_hold)
        current = group

    if fade_out and current is not None:
        scene.play(FadeOut(current), run_time=0.3)


def _animate_excalidraw_json_pages(
    scene: Scene,
    path: Path,
    pages: list[ExcalidrawPage],
    *,
    total_run_time: float,
    hold_time: float,
    fade_out: bool,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    app = data.get("appState") or {}
    bg = str(app.get("viewBackgroundColor") or "#ffffff")
    scene.camera.background_color = bg

    elements = [el for el in (data.get("elements") or []) if isinstance(el, dict) and not el.get("isDeleted")]
    transition = min(0.35, max(0.2, total_run_time * 0.05))
    n_pages = len(pages)
    page_hold = max(0.25, hold_time / n_pages)
    draw_budget = max(0.8, (total_run_time - page_hold * n_pages - transition * (n_pages - 1)) / n_pages)

    current: Mobject | None = None
    for page in pages:
        page_elements = [
            el
            for el in elements
            if el.get("id") == page.page_id or el.get("frameId") == page.page_id
        ]
        if not page_elements:
            continue
        bounds = _elements_bounds(page_elements)
        parts: list[Mobject] = []
        for el in page_elements:
            mob = _element_to_mobject(el, bounds)
            if mob is not None:
                parts.append(mob)
        if not parts:
            continue
        group = VGroup(*parts)
        _fit_to_frame(group)
        group.move_to(ORIGIN)

        if current is not None:
            scene.play(FadeOut(current), run_time=transition)

        layer_count = max(len(parts), 1)
        per = max(0.12, draw_budget / layer_count)
        for mob in parts:
            if isinstance(mob, VMobject) and mob.get_num_points() > 0:
                if float(mob.get_stroke_width() or 0) > 0:
                    _ensure_visible_stroke(mob)
                scene.play(Create(mob), run_time=per)
            else:
                scene.play(FadeIn(mob, shift=0.05 * UP), run_time=per * 0.85)

        scene.wait(page_hold)
        current = group

    if fade_out and current is not None:
        scene.play(FadeOut(current), run_time=0.3)


def _fit_to_frame(group: Mobject, margin: float = 0.9) -> None:
    """Scale drawing to fill the Manim frame (scale up or down)."""
    fw = config.frame_width * margin
    fh = config.frame_height * margin
    if group.width > 0.001 and group.height > 0.001:
        group.scale(min(fw / group.width, fh / group.height))
    elif group.width > 0.001:
        group.scale(fw / group.width)
    elif group.height > 0.001:
        group.scale(fh / group.height)


def _fit_svg_group(group: Group, margin: float = 0.9) -> None:
    """Lay out every Excalidraw export in SVG viewBox space (one code path for all scenes)."""
    _layout_excalidraw_group(group, margin)


def _viewbox_layout_metrics(
    vb: tuple[float, float, float, float],
    margin: float,
) -> tuple[np.ndarray, float, float, float, float, float, float, float, float]:
    """Map the SVG viewBox onto the Manim frame."""
    vb_x, vb_y, vb_w, vb_h = vb
    page_w, page_h = _viewbox_page_size(vb_w, vb_h, margin)
    page_origin = UP * (page_h / 2) + LEFT * (page_w / 2)
    scale_x = page_w / vb_w if vb_w > 0 else 1.0
    scale_y = page_h / vb_h if vb_h > 0 else 1.0
    return page_origin, page_w, page_h, scale_x, scale_y, vb_x, vb_y, vb_w, vb_h


def _viewbox_top_left(
    x: float,
    y: float,
    page_origin: np.ndarray,
    scale_x: float,
    scale_y: float,
    vb_x: float,
    vb_y: float,
) -> np.ndarray:
    """Map one SVG page coordinate to Manim space using viewBox metrics."""
    return page_origin + RIGHT * ((x - vb_x) * scale_x) + DOWN * ((y - vb_y) * scale_y)


def _layout_excalidraw_group(group: Group, margin: float = 0.9) -> None:
    """Single layout path: vectors and raster icons share the SVG viewBox grid."""
    vb = getattr(group, "excal_view_box", None)
    if vb is None:
        _fit_to_frame(group, margin)
        return

    (
        page_origin,
        page_w,
        _page_h,
        scale_x,
        scale_y,
        vb_x,
        vb_y,
        _vb_w,
        _vb_h,
    ) = _viewbox_layout_metrics(vb, margin)

    svg = _svg_vector_root(group)
    if svg is not None and svg.width > 0.001:
        svg.scale(page_w / svg.width)
        svg.move_to(page_origin, aligned_edge=UL)

    for img in (sm for sm in group.get_family() if _is_embedded_icon(sm)):
        placement = getattr(img, "excal_placement", None)
        if placement is None:
            continue
        _stretch_mobject_to_rect(
            img,
            placement.width * scale_x,
            placement.height * scale_y,
        )
        top_left = _viewbox_top_left(
            placement.x,
            placement.y,
            page_origin,
            scale_x,
            scale_y,
            vb_x,
            vb_y,
        )
        _place_raster_icon(img, top_left, placement.rotation)

    if _group_has_manim_text(group):
        _align_raster_icons_to_manim_text(group)


def _group_has_manim_text(group: Group) -> bool:
    """True when the SVG contains vector text that icons should align to."""
    render_path = getattr(group, "excal_render_path", None)
    vb = getattr(group, "excal_view_box", None)
    svg = _svg_vector_root(group)
    if render_path is None or vb is None or svg is None:
        return False
    try:
        root = ET.parse(render_path).getroot()
    except ET.ParseError:
        return False
    path_map = _svg_path_manim_map(group, root)
    if not path_map:
        return False
    return bool(_text_label_ink_bounds(root, path_map))


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    return tag.split("}")[-1] if "}" in tag else tag


_SVG_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _svg_path_page_points(
    elem: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> list[tuple[float, float]]:
    """Sample one SVG path's coordinates in flattened page space."""
    nums = [float(n) for n in _SVG_NUM_RE.findall(elem.get("d") or "")]
    if len(nums) < 2:
        return []

    matrix = _elem_page_affine(elem, parent_map)

    points: list[tuple[float, float]] = []
    for index in range(0, len(nums) - 1, 2):
        points.append(_apply_affine(nums[index], nums[index + 1], matrix))
    return points


def _svg_path_manim_map(group: Group, root: ET.Element) -> dict[int, VMobject]:
    """Map SVG path document order to Manim path submobjects."""
    svg = _svg_vector_root(group)
    if svg is None:
        return {}

    root_mob = svg.submobjects[0] if len(svg.submobjects) == 1 else svg
    svg_paths = [elem for elem in root.iter() if _local_tag(elem) == "path"]
    manim_paths = [
        sm
        for sm in root_mob.submobjects
        if isinstance(sm, VMobject) and sm.get_num_points() > 0
    ]
    if len(svg_paths) != len(manim_paths):
        return {}

    return {index: manim_paths[index] for index in range(len(svg_paths))}


def _text_label_ink_bounds(
    root: ET.Element,
    path_map: dict[int, VMobject],
) -> list[dict[str, float | str]]:
    """Collect SVG and Manim left edges for each converted text label."""
    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    svg_paths = [elem for elem in root.iter() if _local_tag(elem) == "path"]
    entries: list[dict[str, float | str]] = []

    for group in root.iter():
        label = group.get("data-excal-text")
        if not label:
            continue
        path_children = [child for child in group if _local_tag(child) == "path"]
        if not path_children:
            continue

        ink_min_x = float("inf")
        ink_min_y = float("inf")
        ink_max_x = float("-inf")
        ink_max_y = float("-inf")
        manim_left = float("inf")
        manim_center_y_sum = 0.0
        manim_center_y_count = 0

        for path_elem in path_children:
            points = _svg_path_page_points(path_elem, parent_map)
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ink_min_x = min(ink_min_x, min(xs))
            ink_max_x = max(ink_max_x, max(xs))
            ink_min_y = min(ink_min_y, min(ys))
            ink_max_y = max(ink_max_y, max(ys))

            try:
                manim_path = path_map[svg_paths.index(path_elem)]
            except (ValueError, KeyError):
                continue
            manim_left = min(manim_left, float(manim_path.get_left()[0]))
            manim_center_y_sum += float(manim_path.get_center()[1])
            manim_center_y_count += 1

        if ink_min_x == float("inf") or manim_left == float("inf") or manim_center_y_count <= 0:
            continue

        entries.append(
            {
                "label": label,
                "ink_min_x": ink_min_x,
                "ink_min_y": ink_min_y,
                "ink_max_x": ink_max_x,
                "ink_max_y": ink_max_y,
                "center_y": (ink_min_y + ink_max_y) / 2,
                "manim_left": manim_left,
                "manim_center_y": manim_center_y_sum / manim_center_y_count,
            }
        )
    return entries


def _svg_text_label_entries(root: ET.Element) -> list[dict[str, float | str]]:
    """SVG-space label bounds for pairing icons with text units."""
    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    entries: list[dict[str, float | str]] = []
    for group in root.iter():
        label = group.get("data-excal-text")
        if not label:
            continue
        path_children = [child for child in group if _local_tag(child) == "path"]
        if not path_children:
            continue

        ink_min_x = float("inf")
        ink_min_y = float("inf")
        ink_max_x = float("-inf")
        ink_max_y = float("-inf")
        for path_elem in path_children:
            points = _svg_path_page_points(path_elem, parent_map)
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ink_min_x = min(ink_min_x, min(xs))
            ink_max_x = max(ink_max_x, max(xs))
            ink_min_y = min(ink_min_y, min(ys))
            ink_max_y = max(ink_max_y, max(ys))

        if ink_min_x == float("inf"):
            continue

        entries.append(
            {
                "label": label,
                "ink_min_x": ink_min_x,
                "ink_min_y": ink_min_y,
                "ink_max_x": ink_max_x,
                "ink_max_y": ink_max_y,
                "center_y": (ink_min_y + ink_max_y) / 2,
            }
        )
    return entries


def _text_unit_for_icon(
    placement: Any,
    render_path: Path,
    units: list[ET.Element],
) -> int | None:
    try:
        root = ET.parse(render_path).getroot()
    except ET.ParseError:
        return None

    text_entry = _text_label_for_icon(placement, _svg_text_label_entries(root))
    if text_entry is None:
        return None

    label = str(text_entry["label"])
    for unit_index, unit in enumerate(units):
        for elem in unit.iter():
            if elem.get("data-excal-text") == label:
                return unit_index
    return None


def _collect_icon_wrappers(group: Mobject) -> dict[int, Mobject]:
    wrappers: dict[int, Mobject] = {}
    for submob in group.submobjects:
        if not _is_icon_wrapper(submob):
            continue
        icon = submob.submobjects[0]
        unit_index = getattr(icon, "excal_unit_index", None)
        if unit_index is None:
            continue
        wrappers[int(unit_index)] = submob
    return wrappers


def _text_label_for_icon(
    placement: Any,
    text_entries: list[dict[str, float | str]],
) -> dict[str, float | str] | None:
    """Match one raster icon to the label sitting to its right in the same row."""
    icon_top = placement.y
    icon_bottom = placement.y + placement.height
    icon_center_y = placement.y + placement.height / 2
    icon_right = placement.x + placement.width

    best: dict[str, float | str] | None = None
    best_score = float("inf")
    for entry in text_entries:
        if float(entry["ink_max_y"]) < icon_top or float(entry["ink_min_y"]) > icon_bottom:
            continue

        gap_svg = float(entry["ink_min_x"]) - icon_right
        if gap_svg < -8 or gap_svg > 120:
            continue

        dy = abs(float(entry["center_y"]) - icon_center_y)
        score = dy * 4 + abs(gap_svg)
        if score < best_score:
            best_score = score
            best = entry
    return best


def _align_raster_icons_to_manim_text(group: Group) -> None:
    """Preserve SVG icon→text gaps using Manim-rendered text ink positions."""
    render_path = getattr(group, "excal_render_path", None)
    vb = getattr(group, "excal_view_box", None)
    svg = _svg_vector_root(group)
    if render_path is None or vb is None or svg is None:
        return

    try:
        root = ET.parse(render_path).getroot()
    except ET.ParseError:
        return

    vb_x, vb_y, vb_w, vb_h = vb
    if vb_w <= 0 or svg.width <= 0:
        return

    path_map = _svg_path_manim_map(group, root)
    if not path_map:
        return

    text_entries = _text_label_ink_bounds(root, path_map)
    if not text_entries:
        return

    for submob in group.submobjects:
        if not isinstance(submob, Group) or len(submob.submobjects) != 1:
            continue
        icon = submob.submobjects[0]
        if not _is_embedded_icon(icon):
            continue

        placement = getattr(icon, "excal_placement", None)
        if placement is None:
            continue

        text_entry = _text_label_for_icon(placement, text_entries)
        if text_entry is None:
            continue

        gap_svg = float(text_entry["ink_min_x"]) - (placement.x + placement.width)
        if gap_svg < -40:
            continue

        gap_manim = gap_svg / vb_w * svg.width
        target_right = float(text_entry["manim_left"]) - gap_manim
        shift_x = target_right - submob.get_right()[0]
        if abs(shift_x) > 1e-4:
            submob.shift(RIGHT * shift_x)

        # SVGMobject text and viewBox icon placement use opposite Y conventions;
        # snap icon center to the rendered text center instead of preserving SVG Y gap.
        target_center_y = float(text_entry["manim_center_y"])
        shift_y = target_center_y - submob.get_center()[1]
        if abs(shift_y) > 1e-4:
            submob.shift(UP * shift_y)


def _place_raster_icon(img: Mobject, top_left: np.ndarray, rotation_deg: float = 0.0) -> None:
    """Place a raster icon at a viewBox-mapped top-left, preserving SVG rotation."""
    img.move_to(top_left, aligned_edge=UL)
    if abs(rotation_deg) > 1e-3:
        img.rotate(-rotation_deg * DEGREES, about_point=top_left)


def _attach_raster_icons(
    svg: Mobject,
    render_path: Path,
    vb: tuple[float, float, float, float],
) -> list[Mobject]:
    """Load embedded raster images; layout happens in _layout_excalidraw_group."""
    placements = extract_svg_raster_placements(render_path)
    if not placements:
        return []

    _, _, vb_w, vb_h = vb
    if vb_w <= 0 or vb_h <= 0:
        return []

    attached: list[Mobject] = []
    for placement in placements:
        try:
            img, placement = _load_trimmed_raster_icon(placement)
        except OSError:
            continue
        if placement.width <= 0 or placement.height <= 0:
            continue

        img.excal_placement = placement
        img.excal_unit_index = placement.unit_index
        img.set_opacity(0)
        attached.append(Group(img))
    return attached


def _viewbox_page_size(vb_w: float, vb_h: float, margin: float) -> tuple[float, float]:
    """Map an SVG viewBox aspect ratio onto the Manim frame."""
    fw = config.frame_width * margin
    fh = config.frame_height * margin
    if vb_w <= 0 or vb_h <= 0:
        return fw, fh
    if vb_w / vb_h >= fw / fh:
        return fw, fw * vb_h / vb_w
    return fh * vb_w / vb_h, fh


def _stretch_mobject_to_rect(img: Mobject, target_w: float, target_h: float) -> None:
    """Match an icon to the exact SVG placement width/height after viewBox scaling."""
    if target_w <= 0 or target_h <= 0:
        return
    img.stretch_to_fit_width(max(target_w, 0.01))
    img.stretch_to_fit_height(max(target_h, 0.01))


def _collect_embedded_icons(group: Mobject) -> dict[int, Mobject]:
    icons: dict[int, Mobject] = {}
    for sm in group.get_family():
        if not _is_embedded_icon(sm):
            continue
        unit_index = getattr(sm, "excal_unit_index", None)
        if unit_index is None:
            continue
        icons[int(unit_index)] = sm
    return icons


def _ensure_visible_stroke(mob: VMobject, min_width: float = 2.0) -> None:
    for sm in mob.get_family():
        w = float(sm.get_stroke_width() or 0)
        if w > 0:
            sm.set_stroke(width=max(w, min_width))


def _unit_stroke_layer(strokes: list[Mobject]) -> Mobject:
    """One animation step per Excalidraw unit (not per path)."""
    if not strokes:
        return VGroup()
    if len(strokes) == 1:
        return strokes[0]
    return VGroup(*strokes)


def _resolve_draw_order(unit_order: list[int] | None, unit_count: int) -> list[int]:
    """Normalize saved draw-order indices without mutating the SVG."""
    if unit_count <= 0:
        return []
    default = list(range(unit_count))
    if not unit_order:
        return default
    seen: set[int] = set()
    ordered: list[int] = []
    for raw in unit_order:
        idx = int(raw)
        if idx in seen or idx < 0 or idx >= unit_count:
            continue
        ordered.append(idx)
        seen.add(idx)
    for idx in default:
        if idx not in seen:
            ordered.append(idx)
    return ordered


def _is_vector_logo_duplicate(unit: ET.Element) -> bool:
    """True for hand-drawn black logo traces, not cell boxes or colored decorations."""
    fill, stroke = _element_unit_colors(unit)
    fill_key = (fill or "").strip().lower()
    stroke_key = (stroke or "").strip().lower()
    path_weight = _element_path_weight(unit)

    if fill_key in ("#000000", "#000", "black") and path_weight > 500:
        return True

    if not fill_key and stroke_key in ("#000000", "#000", "black") and path_weight < 500:
        return True

    return False


def _unit_has_image_mask(unit: ET.Element) -> bool:
    return any(_local_tag(elem) == "mask" for elem in unit.iter())


def _should_shadow_logo_trace(
    unit: ET.Element,
    *,
    near_label: str | None,
    protected_labels: set[str],
    unit_label: str = "",
    has_raster_icons: bool = False,
) -> bool:
    """Hide Excalidraw image-mask slash traces when a raster icon replaces them."""
    if _element_has_raster_image(unit) or _element_text_label(unit):
        return False

    if not _is_vector_logo_duplicate(unit):
        return False

    label_lower = unit_label.lower()

    if near_label in protected_labels:
        return True

    if near_label and near_label.lower() in (
        "data",
        "science",
        "ai",
        "apps",
        "python",
        "not the only",
        "language...",
    ):
        if near_label not in protected_labels:
            return False

    if label_lower and any(
        token in label_lower
        for token in ("(data)", "(ai)", "(python)", "not the only")
    ):
        if near_label not in protected_labels:
            return False

    if _unit_has_image_mask(unit):
        if near_label is not None:
            return False
        if not has_raster_icons:
            return False
        if not label_lower:
            return True
        if ("illustration #" in label_lower or "icon #" in label_lower) and not any(
            token in label_lower for token in ("(data)", "(ai)", "(python)")
        ):
            return True
        return False

    fill, _ = _element_unit_colors(unit)
    fill_key = (fill or "").strip().lower()
    if fill_key in ("#000000", "#000", "black") and _element_path_weight(unit) > 500:
        if near_label in protected_labels:
            return True
        if near_label is None and ("illustration #" in label_lower or "icon #" in label_lower):
            return not any(token in label_lower for token in ("(data)", "(ai)", "(python)"))

    return False


def _shadowed_vector_logo_units(
    render_path: Path,
    units: list[ET.Element],
    raster_paired_text_units: set[int],
    *,
    source_svg_path: Path | None = None,
) -> set[int]:
    """Hand-drawn vector logo traces hidden when a raster icon covers the same label."""
    try:
        root = ET.parse(render_path).getroot()
    except ET.ParseError:
        return set()

    parent_map: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    vb = parse_view_box(root)
    _, _, vb_w, vb_h = vb

    protected_labels: set[str] = set()
    for idx in raster_paired_text_units:
        if idx < 0 or idx >= len(units):
            continue
        label = _element_text_label_display(units[idx]).strip()
        if label:
            protected_labels.add(label)

    text_units: list[tuple[str, tuple[float, float] | None]] = []
    for unit in units:
        if not _element_text_label(unit):
            continue
        label = _element_text_label_display(unit).strip()
        text_units.append(
            (label, _element_bbox_center(unit, parent_map))
        )

    has_raster_icons = bool(extract_svg_raster_placements(render_path))

    unit_labels: dict[int, str] = {}
    if source_svg_path is not None:
        for info in describe_animation_units(source_svg_path):
            unit_labels[info.index] = info.label

    shadowed: set[int] = set()
    for idx, unit in enumerate(units):
        label = unit_labels.get(idx, "") or _element_text_label_display(unit)
        if _is_excalidraw_frame_title(unit, label, parent_map, view_box=vb):
            shadowed.add(idx)
            continue
        if _element_has_raster_image(unit) or _element_text_label(unit):
            continue
        if _element_path_weight(unit) <= 0:
            continue
        center = _element_bbox_center(unit, parent_map)
        near = _nearest_text_label(
            center,
            text_units,
            max_dist=max(vb_w, vb_h) * 0.35,
        )
        if _should_shadow_logo_trace(
            unit,
            near_label=near,
            protected_labels=protected_labels,
            unit_label=unit_labels.get(idx, ""),
            has_raster_icons=has_raster_icons,
        ):
            shadowed.add(idx)
    return shadowed


def _animation_layers(group: Mobject) -> list[Mobject]:
    """Collect drawable strokes and raster images in draw-order."""
    render_path = getattr(group, "excal_render_path", None)
    unit_order = getattr(group, "excal_unit_order", None)
    vector_roots = [sm for sm in group.submobjects if not _is_icon_wrapper(sm)]
    icon_wrappers = _collect_icon_wrappers(group)

    all_strokes: list[Mobject] = []
    for svg_root in vector_roots:
        root = svg_root.submobjects[0] if len(svg_root.submobjects) == 1 else svg_root
        found = _collect_stroke_layers(root, svg_root=root)
        if found:
            all_strokes.extend(found)
        elif len(svg_root.submobjects) > 1:
            all_strokes.extend(
                [sm for sm in svg_root.submobjects if not _is_background_fill(sm, root)]
            )

    if render_path is not None and (icon_wrappers or animation_unit_elements(render_path)):
        units = animation_unit_elements(render_path)
        if units:
            layers: list[Mobject] = []
            offset = 0
            stroke_ranges: list[tuple[int, int]] = []
            for unit in units:
                count = unit_path_layer_count(unit)
                start = offset
                if count > 0 and offset < len(all_strokes):
                    offset = min(offset + count, len(all_strokes))
                stroke_ranges.append((start, offset))

            icon_before_text: dict[int, Mobject] = {}
            orphan_icons: dict[int, Mobject] = {}
            for icon_unit, wrapper in icon_wrappers.items():
                icon = wrapper.submobjects[0]
                placement = getattr(icon, "excal_placement", None)
                text_unit = (
                    _text_unit_for_icon(placement, render_path, units)
                    if placement is not None
                    else None
                )
                if text_unit is not None:
                    icon_before_text[text_unit] = wrapper
                else:
                    orphan_icons[icon_unit] = wrapper

            shadowed_vectors = _shadowed_vector_logo_units(
                render_path,
                units,
                set(icon_before_text.keys()),
                source_svg_path=getattr(group, "excal_source_path", None),
            )

            draw_order = _resolve_draw_order(unit_order, len(units))
            for unit_index in draw_order:
                if unit_index in shadowed_vectors:
                    continue
                unit_parts: list[Mobject] = []
                if unit_index in icon_before_text:
                    unit_parts.append(icon_before_text[unit_index])
                if unit_index in orphan_icons:
                    unit_parts.append(orphan_icons[unit_index])
                start, end = stroke_ranges[unit_index]
                if start < end:
                    unit_strokes = all_strokes[start:end]
                    text_strokes, connector_strokes = _split_unit_stroke_layers(
                        units[unit_index],
                        unit_strokes,
                    )
                    if text_strokes:
                        layer = _unit_stroke_layer(text_strokes)
                        if _is_text_animation_unit(units[unit_index]):
                            layer.excal_is_text_unit = True
                        unit_parts.append(layer)
                    if connector_strokes:
                        connector_layer = _unit_stroke_layer(connector_strokes)
                        connector_layer.excal_is_connector = True
                        unit_parts.append(connector_layer)
                if not unit_parts:
                    continue
                if len(unit_parts) == 1:
                    layers.append(unit_parts[0])
                    continue
                combined = VGroup(*unit_parts)
                if any(getattr(part, "excal_is_text_unit", False) for part in unit_parts):
                    combined.excal_is_text_unit = True
                if any(getattr(part, "excal_is_connector", False) for part in unit_parts):
                    combined.excal_is_connector = True
                layers.append(combined)

            if offset < len(all_strokes):
                tail = all_strokes[offset:]
                shadowed_stroke_indices: set[int] = set()
                cursor = 0
                for unit_index, unit in enumerate(units):
                    count = unit_path_layer_count(unit)
                    if unit_index in shadowed_vectors:
                        shadowed_stroke_indices.update(range(cursor, cursor + count))
                    cursor += count
                for stroke_index, stroke in enumerate(tail, start=offset):
                    if stroke_index in shadowed_stroke_indices:
                        continue
                    if stroke not in layers:
                        layers.append(stroke)
            for wrapper in icon_wrappers.values():
                if wrapper not in layers:
                    layers.append(wrapper)
            if layers:
                return layers

    if not all_strokes and vector_roots:
        all_strokes = list(vector_roots)
    layers = list(all_strokes)
    for wrapper in sorted(icon_wrappers.values(), key=lambda mob: getattr(mob.submobjects[0], "excal_unit_index", 0)):
        if wrapper not in layers:
            layers.append(wrapper)
    if layers:
        return layers
    return [group]


def _collect_stroke_layers(
    mob: Mobject,
    acc: list[Mobject] | None = None,
    *,
    svg_root: Mobject | None = None,
) -> list[Mobject]:
    if acc is None:
        acc = []
    if svg_root is None:
        svg_root = mob
    if isinstance(mob, VMobject) and mob.get_num_points() > 1:
        child_strokes = [
            c
            for c in mob.submobjects
            if isinstance(c, VMobject)
            and c.get_num_points() > 1
            and not _is_background_fill(c, svg_root)
            and not _is_frame_outline(c, svg_root)
            and not _is_cell_background(c, svg_root)
        ]
        if not child_strokes:
            if (
                not _is_background_fill(mob, svg_root)
                and not _is_frame_outline(mob, svg_root)
                and not _is_cell_background(mob, svg_root)
            ):
                acc.append(mob)
            return acc
    for child in mob.submobjects:
        _collect_stroke_layers(child, acc, svg_root=svg_root)
    return acc


def _is_background_fill(mob: Mobject, root: Mobject | None = None) -> bool:
    if not isinstance(mob, VMobject):
        return False
    fill = float(mob.get_fill_opacity() or 0)
    if fill < 0.75:
        return False
    if root is None:
        return False
    if root.width < 0.01 or root.height < 0.01:
        return False
    if mob.width >= root.width * 0.92 and mob.height >= root.height * 0.92:
        return True
    return False


def _is_frame_outline(mob: Mobject, root: Mobject | None = None) -> bool:
    """Skip hollow full-bleed slide frames, not inset content borders."""
    if not isinstance(mob, VMobject):
        return False
    if float(mob.get_fill_opacity() or 0) > 0.08:
        return False
    if float(mob.get_stroke_opacity() or 0) < 0.05:
        return False
    if root is None or root.width < 0.01 or root.height < 0.01:
        return False
    if mob.width < root.width * 0.88 or mob.height < root.height * 0.88:
        return False
    inset_x = abs(float(mob.get_left()[0] - root.get_left()[0]))
    inset_y = abs(float(mob.get_top()[1] - root.get_top()[1]))
    if inset_x > root.width * 0.02 or inset_y > root.height * 0.02:
        return False
    return True


def _fill_rgb(mob: VMobject) -> tuple[float, float, float] | None:
    try:
        rgb = mob.get_fill_color()
        return float(rgb[0]), float(rgb[1]), float(rgb[2])
    except (TypeError, IndexError, AttributeError):
        return None


def _is_solid_panel_fill(mob: VMobject, root: Mobject | None = None) -> bool:
    """Large saturated fill-only shapes (note cards), not text glyph paths."""
    if float(mob.get_fill_opacity() or 0) < 0.35:
        return False
    if float(mob.get_stroke_opacity() or 0) > 0.05 and float(mob.get_stroke_width() or 0) > 0:
        return False
    if root is not None and root.width > 0.01 and root.height > 0.01:
        if mob.width < root.width * 0.08 or mob.height < root.height * 0.08:
            return False
    rgb = _fill_rgb(mob)
    if rgb is None:
        return False
    if max(rgb) > 0.97 and min(rgb) > 0.94:
        return False
    return (max(rgb) - min(rgb)) > 0.12


def _is_neutral_light_panel_fill(mob: VMobject) -> bool:
    """Pale gray grid/card panels (not white, not vivid note-card colors)."""
    rgb = _fill_rgb(mob)
    if rgb is None:
        return False
    mx, mn = max(rgb), min(rgb)
    if mx > 0.97 and mn > 0.94:
        return False
    return (mx - mn) < 0.08 and mx > 0.85


def _is_cell_background(mob: Mobject, root: Mobject | None = None) -> bool:
    """Light neutral panel fills — show immediately behind icons/text."""
    if not isinstance(mob, VMobject):
        return False
    if float(mob.get_fill_opacity() or 0) < 0.35:
        return False
    if not _is_neutral_light_panel_fill(mob):
        return False
    if root is None or root.width < 0.01 or root.height < 0.01:
        return False
    if mob.width >= root.width * 0.88 and mob.height >= root.height * 0.88:
        return False
    return mob.width >= root.width * 0.12 and mob.height >= root.height * 0.08


def _add_panel_backgrounds(scene: Scene, group: Mobject) -> None:
    """Add light grid panels to the scene before stroke animation begins."""
    for svg_root in group.submobjects:
        if _is_embedded_icon(svg_root):
            continue
        root = svg_root.submobjects[0] if len(svg_root.submobjects) == 1 else svg_root
        for mob in root.get_family():
            if _is_cell_background(mob, root):
                mob.set_opacity(1)
                mob.set_z_index(0)
                if mob not in scene.mobjects:
                    scene.add(mob)


def _resolve_unit_order(
    page_index: int,
    page_svg: Path,
    animation_sequence: list[str] | None,
    page_sequences: dict[int, list[int]] | None,
) -> list[int] | None:
    converted = convert_svg_text_to_paths(page_svg)
    # Match the prepared render catalog so saved unit indices map to raster icons.
    render_catalog = prepare_svg_for_manim(converted)
    unit_count = len(animation_unit_elements(render_catalog))
    if unit_count < 2:
        return None
    has_saved = bool(
        page_sequences
        and (page_index in page_sequences or str(page_index) in page_sequences)
    )
    if not has_saved and not animation_sequence:
        return None
    order = resolve_page_unit_order(
        page_index,
        unit_count,
        page_sequences=page_sequences,
        animation_sequence=animation_sequence,
        page_svg=converted,
    )
    if has_saved:
        return order
    default = list(range(unit_count))
    if order != default:
        return order
    return None


def _load_svg_group(
    path: Path,
    animation_sequence: list[str] | None = None,
    unit_order: list[int] | None = None,
) -> Group:
    """Load SVG vectors plus embedded raster images from Excalidraw exports."""
    cleaned = _strip_svg_background_rect(path)
    render_path = prepare_svg_for_manim(
        cleaned,
        animation_sequence=animation_sequence,
        unit_order=unit_order,
    )
    mobject_path = strip_embedded_image_uses(render_path)
    svg = SVGMobject(str(mobject_path))
    _hide_background_submobjects(svg)

    try:
        root = ET.parse(render_path).getroot()
    except ET.ParseError:
        root = None
    vb = parse_view_box(root) if root is not None else (0.0, 0.0, max(svg.width, 1.0), max(svg.height, 1.0))

    icons = _attach_raster_icons(svg, render_path, vb)
    group = Group(svg, *icons)
    group.excal_render_path = render_path
    group.excal_source_path = path
    group.excal_view_box = vb
    group.excal_unit_order = unit_order
    return group


def _strip_svg_background_rect(svg_path: Path) -> Path:
    """Remove full-canvas white/colored background rects from Excalidraw SVG exports."""
    text = svg_path.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return svg_path

    view_box = root.get("viewBox") or ""
    vb_parts = view_box.split()
    vb_w = float(vb_parts[2]) if len(vb_parts) >= 4 else 800.0
    vb_h = float(vb_parts[3]) if len(vb_parts) >= 4 else 600.0

    removed = False
    for elem in list(root):
        local = elem.tag.split("}")[-1]
        if local != "rect":
            continue
        w_attr = elem.get("width", "")
        h_attr = elem.get("height", "")
        fill = (elem.get("fill") or "").lower()
        if w_attr in ("100%", str(vb_w)) and h_attr in ("100%", str(vb_h)):
            root.remove(elem)
            removed = True
        elif fill in ("#ffffff", "white", "#fff") and _rect_covers_canvas(elem, vb_w, vb_h):
            root.remove(elem)
            removed = True

    if not removed:
        return svg_path

    render_dir = svg_path.parent / ".excalidraw_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{svg_path.stem}_nobg.svg"
    out.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return out


def _rect_covers_canvas(elem: ET.Element, vb_w: float, vb_h: float) -> bool:
    try:
        w = float(str(elem.get("width", "0")).replace("%", ""))
        h = float(str(elem.get("height", "0")).replace("%", ""))
    except ValueError:
        return elem.get("width") == "100%" and elem.get("height") == "100%"
    return w >= vb_w * 0.95 and h >= vb_h * 0.95


def _hide_background_submobjects(mob: Mobject) -> None:
    for sm in mob.get_family():
        if _is_background_fill(sm, mob):
            sm.set_opacity(0)
            sm.set_fill(opacity=0)


def _svg_background(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "viewBackgroundColor" in text:
        m = re.search(r'"viewBackgroundColor"\s*:\s*"([^"]+)"', text)
        if m:
            return m.group(1)
    canvas = svg_canvas_background_color(path)
    if canvas:
        return canvas
    # Excalidraw dark mode export
    if re.search(r'viewBackgroundColor["\']?\s*[:=]\s*["\']#1', text, re.I):
        return "#1e1e1e"
    return "#ffffff"


def _load_excalidraw_group(path: Path) -> tuple[VGroup, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    elements = _excalidraw_elements(data)
    app = data.get("appState") or {}
    bg = str(app.get("viewBackgroundColor") or "#ffffff")

    if not elements:
        raise ValueError("Excalidraw file has no drawable elements")

    bounds = _elements_bounds(elements)
    parts: list[Mobject] = []
    for el in elements:
        mob = _element_to_mobject(el, bounds)
        if mob is not None:
            parts.append(mob)
    return VGroup(*parts), bg


def _excalidraw_elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    elements = data.get("elements") or []
    out = [el for el in elements if isinstance(el, dict) and not el.get("isDeleted")]
    out.sort(key=lambda e: (float(e.get("index") or 0),))
    return out


def _elements_bounds(elements: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for el in elements:
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        w = float(el.get("width") or 0)
        h = float(el.get("height") or 0)
        xs.extend([x, x + w])
        ys.extend([y, y + h])
        for pt in el.get("points") or []:
            if len(pt) >= 2:
                xs.append(x + float(pt[0]))
                ys.append(y + float(pt[1]))
    if not xs:
        return 0, 0, 800, 600
    return min(xs), min(ys), max(xs), max(ys)


def _map_point(x: float, y: float, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    mx = (x - cx) * _EXCAL_SCALE
    my = -(y - cy) * _EXCAL_SCALE
    return np.array([mx, my, 0.0])


def _stroke(el: dict[str, Any]) -> dict[str, Any]:
    return {
        "color": el.get("strokeColor") or WHITE,
        "width": max(1.5, float(el.get("strokeWidth") or 2)),
        "fill": el.get("backgroundColor") or el.get("fillStyle"),
        "opacity": float(el.get("opacity") or 100) / 100.0,
    }


def _element_to_mobject(el: dict[str, Any], bounds: tuple[float, float, float, float]) -> Mobject | None:
    etype = el.get("type")
    st = _stroke(el)
    x, y = float(el.get("x", 0)), float(el.get("y", 0))

    if etype == "rectangle":
        w, h = float(el.get("width") or 1), float(el.get("height") or 1)
        center = _map_point(x + w / 2, y + h / 2, bounds)
        rect = RoundedRectangle(
            width=max(w * _EXCAL_SCALE, 0.05),
            height=max(h * _EXCAL_SCALE, 0.05),
            corner_radius=float(el.get("roundness") or 0) * _EXCAL_SCALE * 8,
            color=st["color"],
            stroke_width=st["width"],
        )
        if st["fill"] and st["fill"] != "transparent":
            rect.set_fill(st["fill"], opacity=0.25)
        rect.move_to(center)
        return rect

    if etype == "ellipse":
        w, h = float(el.get("width") or 1), float(el.get("height") or 1)
        center = _map_point(x + w / 2, y + h / 2, bounds)
        ell = Ellipse(
            width=max(w * _EXCAL_SCALE, 0.05),
            height=max(h * _EXCAL_SCALE, 0.05),
            color=st["color"],
            stroke_width=st["width"],
        )
        ell.move_to(center)
        return ell

    if etype == "diamond":
        w, h = float(el.get("width") or 1), float(el.get("height") or 1)
        center = _map_point(x + w / 2, y + h / 2, bounds)
        d = Polygon(
            UP * h * _EXCAL_SCALE / 2,
            RIGHT * w * _EXCAL_SCALE / 2,
            DOWN * h * _EXCAL_SCALE / 2,
            LEFT * w * _EXCAL_SCALE / 2,
            color=st["color"],
            stroke_width=st["width"],
        )
        d.move_to(center)
        return d

    if etype in ("arrow", "line"):
        pts = el.get("points") or [[0, 0], [100, 0]]
        if len(pts) < 2:
            return None
        start = _map_point(x + float(pts[0][0]), y + float(pts[0][1]), bounds)
        end = _map_point(x + float(pts[-1][0]), y + float(pts[-1][1]), bounds)
        if etype == "arrow":
            return Arrow(
                start,
                end,
                color=st["color"],
                stroke_width=st["width"],
                buff=0,
                max_tip_length_to_length_ratio=0.15,
            )
        return Line(start, end, color=st["color"], stroke_width=st["width"])

    if etype == "freedraw":
        pts = el.get("points") or []
        if len(pts) < 2:
            return None
        coords = [_map_point(x + float(p[0]), y + float(p[1]), bounds) for p in pts if len(p) >= 2]
        vm = VMobject(color=st["color"], stroke_width=st["width"])
        vm.set_points_as_corners(coords)
        return vm

    if etype == "text":
        raw = str(el.get("text") or el.get("originalText") or "")
        if not raw.strip():
            return None
        fs = max(14, int(float(el.get("fontSize") or 20) * _EXCAL_SCALE * 28))
        pos = _map_point(x, y, bounds)
        t = Text(raw, font_size=fs, color=st["color"])
        t.move_to(pos)
        return t

    if etype == "image":
        return None

    return None
