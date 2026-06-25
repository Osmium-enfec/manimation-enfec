import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  GripVertical,
  Image as ImageIcon,
  Loader2,
  RotateCcw,
  Shapes,
  Type,
} from "lucide-react";
import type { ExcalidrawAnimationUnit, ExcalidrawDrawOrderPreviewBridge, ExcalidrawPageUnits, Project } from "../api";
import { fetchExcalidrawUnits, saveExcalidrawSequence } from "../api";

interface ExcalidrawSequenceEditorProps {
  projectId: string;
  drawingRef?: string;
  onSaved: (pageSequences: Record<string, number[]>, project?: Project, code?: string) => void;
  onPreviewBridgeChange?: (bridge: ExcalidrawDrawOrderPreviewBridge | null) => void;
  onFocusPagePreview?: () => void;
  disabled?: boolean;
}

function kindIcon(kind: string) {
  if (kind === "image") return <ImageIcon size={12} />;
  if (kind === "text") return <Type size={12} />;
  return <Shapes size={12} />;
}

function kindLabel(kind: string) {
  if (kind === "image") return "Image";
  if (kind === "text") return "Text";
  return "Shape";
}

function orderFromPage(page: ExcalidrawPageUnits): number[] {
  const known = new Set(page.units.map((unit) => unit.index));
  if (page.saved_order?.length) {
    const saved = page.saved_order.filter((idx) => known.has(idx));
    for (const unit of page.units) {
      if (!saved.includes(unit.index)) saved.push(unit.index);
    }
    return saved;
  }
  return page.units.map((unit) => unit.index);
}

function buildFullOrder(page: ExcalidrawPageUnits, currentOrder: number[]): number[] {
  const known = new Set(page.units.map((unit) => unit.index));
  const result: number[] = [];
  for (const idx of currentOrder) {
    if (known.has(idx)) result.push(idx);
  }
  for (const unit of page.units) {
    if (!result.includes(unit.index)) result.push(unit.index);
  }
  return result;
}

function orderKey(order: number[]): string {
  return order.join(",");
}

function ordersEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

export function ExcalidrawSequenceEditor({
  projectId,
  drawingRef,
  onSaved,
  onPreviewBridgeChange,
  onFocusPagePreview,
  disabled,
}: ExcalidrawSequenceEditorProps) {
  const [pages, setPages] = useState<ExcalidrawPageUnits[]>([]);
  const [pageIndex, setPageIndex] = useState(0);
  const [order, setOrder] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const [selectedUnitIndex, setSelectedUnitIndex] = useState<number | null>(null);
  const [hoveredUnitIndex, setHoveredUnitIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const listItemRefs = useRef<Map<number, HTMLLIElement>>(new Map());
  const skipAutoSaveRef = useRef(true);
  const lastSavedKeyRef = useRef("");
  const persistInFlightRef = useRef(false);
  const pagesRef = useRef(pages);
  pagesRef.current = pages;

  const activePage = pages[pageIndex];

  const orderedUnits = useMemo(() => {
    if (!activePage) return [] as ExcalidrawAnimationUnit[];
    const byIndex = new Map(activePage.units.map((unit) => [unit.index, unit]));
    const fullOrder = buildFullOrder(activePage, order);
    return fullOrder
      .map((idx) => byIndex.get(idx))
      .filter((unit): unit is ExcalidrawAnimationUnit => unit != null);
  }, [activePage, order]);

  const syncOrderState = useCallback((nextOrder: number[], options?: { skipAutoSave?: boolean }) => {
    lastSavedKeyRef.current = orderKey(nextOrder);
    setOrder((current) => (ordersEqual(current, nextOrder) ? current : nextOrder));
    if (options?.skipAutoSave !== false) skipAutoSaveRef.current = true;
  }, []);

  const persistOrder = useCallback(
    async (manual: boolean) => {
      if (!activePage || persistInFlightRef.current) return;
      const fullOrder = buildFullOrder(activePage, order);
      const key = orderKey(fullOrder);
      if (key === lastSavedKeyRef.current) return;

      persistInFlightRef.current = true;
      if (manual) setSaving(true);
      setError(null);
      try {
        const res = await saveExcalidrawSequence(projectId, activePage.page_index, fullOrder);
        lastSavedKeyRef.current = key;
        onSaved(res.page_sequences, res.project, res.code);
        setPages((current) =>
          current.map((page) =>
            page.page_index === activePage.page_index ? { ...page, saved_order: fullOrder } : page
          )
        );
      } catch (e) {
        setError(String(e));
      } finally {
        persistInFlightRef.current = false;
        if (manual) setSaving(false);
      }
    },
    [activePage, onSaved, order, projectId]
  );

  const loadUnits = useCallback(async () => {
    if (!drawingRef) {
      setPages([]);
      setOrder([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetchExcalidrawUnits(projectId);
      setPages(res.pages);
      const first = res.pages[0];
      setPageIndex(0);
      if (first) {
        syncOrderState(orderFromPage(first));
      } else {
        syncOrderState([]);
      }
      setSelectedUnitIndex(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [drawingRef, projectId, syncOrderState]);

  useEffect(() => {
    void loadUnits();
  }, [loadUnits]);

  useEffect(() => {
    const page = pagesRef.current[pageIndex];
    if (!page) {
      syncOrderState([]);
      setSelectedUnitIndex(null);
      return;
    }
    syncOrderState(orderFromPage(page));
    setSelectedUnitIndex(null);
  }, [pageIndex, syncOrderState]);

  useEffect(() => {
    if (!activePage || skipAutoSaveRef.current) {
      skipAutoSaveRef.current = false;
      return;
    }
    const fullOrder = buildFullOrder(activePage, order);
    if (orderKey(fullOrder) === lastSavedKeyRef.current) return;

    const timer = window.setTimeout(() => {
      void persistOrder(false);
    }, 450);
    return () => window.clearTimeout(timer);
  }, [order, activePage, persistOrder]);

  useEffect(() => {
    if (selectedUnitIndex == null) return;
    listItemRefs.current.get(selectedUnitIndex)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedUnitIndex]);

  const selectUnit = useCallback(
    (unitIndex: number) => {
      setSelectedUnitIndex(unitIndex);
      onFocusPagePreview?.();
    },
    [onFocusPagePreview]
  );

  const hoverUnit = useCallback((unitIndex: number | null) => {
    setHoveredUnitIndex(unitIndex);
  }, []);

  useEffect(() => {
    if (!onPreviewBridgeChange) return;
    if (loading || !activePage || activePage.units.length === 0) {
      onPreviewBridgeChange(null);
      return;
    }
    onPreviewBridgeChange({
      page: activePage,
      orderedUnits,
      selectedUnitIndex,
      hoveredUnitIndex,
      selectUnit,
      hoverUnit,
    });
  }, [
    activePage,
    hoverUnit,
    hoveredUnitIndex,
    loading,
    onPreviewBridgeChange,
    orderedUnits,
    selectUnit,
    selectedUnitIndex,
  ]);

  useEffect(() => {
    return () => onPreviewBridgeChange?.(null);
  }, [onPreviewBridgeChange]);

  const moveItem = (from: number, to: number) => {
    if (from === to || from < 0 || to < 0) return;
    setOrder((prev) => {
      if (!activePage) return prev;
      const known = new Set(activePage.units.map((unit) => unit.index));
      const current: number[] = [];
      for (const idx of prev) {
        if (known.has(idx)) current.push(idx);
      }
      for (const unit of activePage.units) {
        if (!current.includes(unit.index)) current.push(unit.index);
      }
      if (from >= current.length || to >= current.length) return current;
      const next = [...current];
      const [item] = next.splice(from, 1);
      next.splice(to, 0, item);
      return next;
    });
  };

  const handleReset = () => {
    if (!activePage) return;
    const documentOrder = activePage.units.map((unit) => unit.index);
    skipAutoSaveRef.current = false;
    setOrder(documentOrder);
  };

  const handleApply = () => {
    void persistOrder(true);
  };

  const isDirty =
    !!activePage && orderKey(buildFullOrder(activePage, order)) !== lastSavedKeyRef.current;

  if (!drawingRef) return null;

  return (
    <div className="excal-sequence-panel">
      <div className="excal-sequence-header">
        <div className="excal-sequence-title-block">
          <strong>Draw order</strong>
          <span className="excal-sequence-subtitle">
            Reorder draws for the active page. Changes save automatically; top draws first.
          </span>
        </div>
        {pages.length > 1 && (
          <select
            className="excal-sequence-page-select"
            value={pageIndex}
            disabled={disabled || loading}
            onChange={(e) => setPageIndex(Number(e.target.value))}
          >
            {pages.map((page) => (
              <option key={page.page_id} value={page.page_index}>
                {page.page_name}
              </option>
            ))}
          </select>
        )}
        {pages.length === 1 && activePage && (
          <span className="voice-motion-chip-meta">{activePage.page_name}</span>
        )}
      </div>

      {loading ? (
        <div className="excal-sequence-empty">
          <Loader2 size={14} className="spin" /> Loading elements…
        </div>
      ) : !activePage || activePage.units.length === 0 ? (
        <div className="excal-sequence-empty">No animatable elements on this page.</div>
      ) : (
        <div className="excal-sequence-body excal-sequence-body-list-only">
          <div className="excal-sequence-list-wrap">
            <div className="excal-sequence-list-label">Draw sequence</div>
            <ol className="excal-sequence-list">
              {orderedUnits.map((unit, index) => {
                const isSelected = selectedUnitIndex === unit.index;
                const isHovered = hoveredUnitIndex === unit.index;
                return (
                  <li
                    key={unit.unit_id}
                    ref={(node) => {
                      if (node) listItemRefs.current.set(unit.index, node);
                      else listItemRefs.current.delete(unit.index);
                    }}
                    className={[
                      "excal-sequence-item",
                      dragIndex === index ? "dragging" : "",
                      dropIndex === index && dragIndex !== null && dragIndex !== index ? "drop-target" : "",
                      isSelected ? "selected" : "",
                      isHovered ? "hovered" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    draggable={!disabled}
                    onClick={() => selectUnit(unit.index)}
                    onMouseEnter={() => hoverUnit(unit.index)}
                    onMouseLeave={() =>
                      setHoveredUnitIndex((current) => (current === unit.index ? null : current))
                    }
                    onDragStart={() => setDragIndex(index)}
                    onDragEnd={() => {
                      setDragIndex(null);
                      setDropIndex(null);
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDropIndex(index);
                    }}
                    onDragLeave={() => setDropIndex((current) => (current === index ? null : current))}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (dragIndex !== null) moveItem(dragIndex, index);
                      setDragIndex(null);
                      setDropIndex(null);
                    }}
                  >
                    <span className="excal-sequence-step" title="Draw step">
                      {index + 1}
                    </span>
                    <button
                      type="button"
                      className="excal-sequence-grip"
                      title="Drag to reorder"
                      disabled={disabled}
                      onMouseDown={(e) => e.stopPropagation()}
                    >
                      <GripVertical size={14} />
                    </button>
                    {unit.color ? (
                      <span
                        className="excal-sequence-swatch"
                        style={{ background: unit.color }}
                        title={`Color ${unit.color}`}
                      />
                    ) : (
                      <span className={`excal-sequence-kind kind-${unit.kind}`}>{kindIcon(unit.kind)}</span>
                    )}
                    <div className="excal-sequence-copy">
                      <span className="excal-sequence-label">{unit.label}</span>
                      <span className="excal-sequence-meta">
                        {kindLabel(unit.kind)}
                        {unit.hint ? ` · ${unit.hint}` : ""}
                      </span>
                    </div>
                    <span className="excal-sequence-actions">
                      <button
                        type="button"
                        className="btn-ghost xs"
                        disabled={disabled || index === 0}
                        onClick={(e) => {
                          e.stopPropagation();
                          moveItem(index, index - 1);
                        }}
                        title="Move up (draw earlier)"
                      >
                        <ArrowUp size={12} />
                      </button>
                      <button
                        type="button"
                        className="btn-ghost xs"
                        disabled={disabled || index === orderedUnits.length - 1}
                        onClick={(e) => {
                          e.stopPropagation();
                          moveItem(index, index + 1);
                        }}
                        title="Move down (draw later)"
                      >
                        <ArrowDown size={12} />
                      </button>
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      )}

      <div className="excal-sequence-toolbar">
        <button
          type="button"
          className="btn-ghost sm excal-sequence-toolbar-btn"
          disabled={disabled || !activePage || !isDirty}
          onClick={handleReset}
          title="Restore SVG document order"
        >
          <RotateCcw size={13} />
          Reset
        </button>
        <button
          type="button"
          className="btn-primary sm excal-sequence-toolbar-btn excal-sequence-save-btn"
          disabled={disabled || saving || !activePage || orderedUnits.length < 2 || !isDirty}
          onClick={handleApply}
        >
          {saving ? <Loader2 size={13} className="spin" /> : null}
          <span>{saving ? "Saving…" : "Save now"}</span>
        </button>
      </div>

      {error && <div className="error-bar compact">{error}</div>}
    </div>
  );
}
