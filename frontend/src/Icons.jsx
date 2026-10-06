import React from "react";

const paths = {
  plus: <path d="M12 5v14M5 12h14" />,
  chat: <path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-2 2V11.5A8.5 8.5 0 0 1 10.5 3h2A8.5 8.5 0 0 1 21 11.5Z" />,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></>,
  graph: <><rect x="3" y="9" width="5" height="6" rx="1"/><rect x="16" y="3" width="5" height="6" rx="1"/><rect x="16" y="15" width="5" height="6" rx="1"/><path d="M8 12h4V6h4M12 12v6h4"/></>,
  settings: <><path d="m9 3-.7 2.3-2 .9L4 5.7 2 9l1.6 1.8v2.4L2 15l2 3.3 2.3-.5 2 .9L9 21h4l.7-2.3 2-.9 2.3.5 2-3.3-1.6-1.8v-2.4L20 9l-2-3.3-2.3.5-2-.9L13 3Z"/><circle cx="11" cy="12" r="3"/></>,
  panel: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></>,
  arrow: <path d="M12 19V5m-6 6 6-6 6 6"/>,
  down: <path d="m6 9 6 6 6-6"/>,
  right: <path d="m9 6 6 6-6 6"/>,
  close: <path d="m6 6 12 12M6 18 18 6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/></>,
  trash: <><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></>,
  edit: <><path d="m14 5 5 5M4 20l5-1L21 7l-5-5L4 14Z"/></>,
  export: <><path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/></>,
  spark: <><path d="m12 3 2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3Z"/></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></>,
  bolt: <path d="m13 2-9 12h7l-1 8 10-12h-7Z"/>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/></>,
  stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
};
export function Icon({ name, size = 18, ...props }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name] || paths.chat}</svg>;
}
export function Mark({ size = 30 }) {
  return <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M8 29V11l12 7 12-7v18l-12-7-12 7Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"/><path d="M20 18v14M8 11l12-7 12 7" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"/><circle cx="20" cy="20" r="3" fill="currentColor"/></svg>;
}
