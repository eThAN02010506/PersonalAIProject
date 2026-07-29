import { type FocusEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type HoverHelpProps = {
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
};

type TooltipPosition = {
  left: number;
  top: number;
  width: number;
};

const HOVER_DELAY_MS = 650;
const TOOLTIP_MAX_WIDTH = 340;
const VIEWPORT_MARGIN = 12;

export function HoverHelp({
  title,
  description,
  children,
  className = "",
}: HoverHelpProps) {
  const timer = useRef<number | null>(null);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  const hide = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    setPosition(null);
  };

  const schedule = (target: HTMLElement) => {
    hide();
    timer.current = window.setTimeout(() => {
      const bounds = target.getBoundingClientRect();
      const width = Math.min(TOOLTIP_MAX_WIDTH, window.innerWidth - VIEWPORT_MARGIN * 2);
      const unclampedLeft = bounds.left + bounds.width / 2;
      const left = Math.min(
        Math.max(unclampedLeft, width / 2 + VIEWPORT_MARGIN),
        window.innerWidth - width / 2 - VIEWPORT_MARGIN,
      );
      setPosition({ left, top: bounds.bottom + 10, width });
      timer.current = null;
    }, HOVER_DELAY_MS);
  };

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const handleBlur = (event: FocusEvent<HTMLSpanElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) hide();
  };

  return (
    <span
      className={`hover-help-target ${className}`.trim()}
      onMouseEnter={(event) => schedule(event.currentTarget)}
      onMouseLeave={hide}
      onFocusCapture={(event) => schedule(event.currentTarget)}
      onBlurCapture={handleBlur}
    >
      {children}
      {position &&
        // 原因：顶部工具栏可以横向滚动，放在控件内部的气泡会被 overflow 裁切。
        // 作用：Portal 将说明渲染到页面顶层，同时仍以当前控件的位置为锚点。
        createPortal(
          <span
            className="hover-help-popup"
            role="tooltip"
            style={{
              left: position.left,
              top: position.top,
              width: position.width,
            }}
          >
            <strong>{title}</strong>
            <span>{description}</span>
          </span>,
          document.body,
        )}
    </span>
  );
}
