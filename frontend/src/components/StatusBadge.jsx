const STATUS_STYLES = {
  success: "bg-emerald-100 text-emerald-700 ring-emerald-200",
  processing: "bg-amber-100 text-amber-700 ring-amber-200",
  failed: "bg-rose-100 text-rose-700 ring-rose-200",
};

function normalizeStatus(status) {
  if (!status) {
    return "processing";
  }
  return String(status).toLowerCase();
}

function StatusBadge({ status }) {
  const normalizedStatus = normalizeStatus(status);
  const badgeStyle = STATUS_STYLES[normalizedStatus] ?? STATUS_STYLES.processing;
  const label = normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1);

  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${badgeStyle}`}>
      {label}
    </span>
  );
}

export default StatusBadge;
