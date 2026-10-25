// Small shared pieces for states every page has: waiting, nothing to show, and failure.

/** A placeholder with the rough shape of the page that is loading. Announced once, not row by row. */
export function Skeleton({ variant = "page", label = "Loading" }) {
  const rows = { list: 8, page: 5, article: 6 }[variant] ?? 5;
  return (
    <div className={`skeleton skeleton-${variant}`} role="status" aria-busy="true" aria-label={label}>
      <span className="skeleton-title" aria-hidden="true" />
      {Array.from({ length: rows }, (_, i) => <span key={i} className="skeleton-line" aria-hidden="true" style={{ "--w": `${92 - ((i * 17) % 38)}%` }} />)}
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="empty">
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

/** Says what failed and offers to try again. `error` is an Error. */
export function ErrorNotice({ what, error, onRetry, children }) {
  return (
    <div role="alert" className="error-notice">
      <p><strong>Cannot load {what}.</strong> {error?.message}</p>
      {children}
      {onRetry && <button type="button" onClick={onRetry}>Try again</button>}
    </div>
  );
}

/** A labelled progress bar. */
export function Bar({ value, max, label }) {
  const pct = max ? Math.round((100 * value) / max) : 0;
  return (
    <div className="bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={value}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}
