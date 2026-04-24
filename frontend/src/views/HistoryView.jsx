import StatusBadge from "../components/StatusBadge";

const mockHistory = [
  { id: "1", fileName: "security_feed_0426.mp4", status: "success", timestamp: "2026-04-24T09:10:00Z" },
  { id: "2", fileName: "warehouse_snapshot.png", status: "processing", timestamp: "2026-04-24T08:45:00Z" },
  { id: "3", fileName: "incident_audio_11.wav", status: "failed", timestamp: "2026-04-23T18:20:00Z" },
];

function HistoryView() {
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6">
      <section className="rounded-2xl bg-white p-6 shadow-sm">
        <h1 className="text-3xl font-bold text-slate-900">Upload History</h1>
        <p className="mt-2 text-sm text-slate-600">Track previous uploads and current processing status.</p>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
        <ul className="divide-y divide-slate-200">
          {mockHistory.map((item) => (
            <li key={item.id} className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="font-medium text-slate-900">{item.fileName}</p>
                <p className="text-sm text-slate-500">{new Date(item.timestamp).toLocaleString()}</p>
              </div>
              <StatusBadge status={item.status} />
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default HistoryView;
