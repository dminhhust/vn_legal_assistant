import React, { useCallback, useEffect, useState } from "react";
import { getChecklist, generateChecklist, updateChecklistItemStatus } from "../api.js";

const STATUS_ICONS = { pending: "🔲", done: "✅", dismissed: "🚫" };
const STATUSES = ["pending", "done", "dismissed"];

export default function Dashboard({ userId, goTo }) {
  const [items, setItems] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [generateMessage, setGenerateMessage] = useState(null);
  const [updatingId, setUpdatingId] = useState(null);
  const [updateError, setUpdateError] = useState(null);

  const fetchChecklist = useCallback(async () => {
    try {
      setItems(await getChecklist(userId));
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, [userId]);

  useEffect(() => {
    fetchChecklist();
  }, [fetchChecklist]);

  if (!userId) {
    return (
      <div>
        <div className="alert warning">No profile yet.</div>
        <div className="link-list">
          <button className="link" onClick={() => goTo("onboarding")}>
            ➡️ 📝 Go to Onboarding first
          </button>
        </div>
      </div>
    );
  }

  const handleGenerate = async () => {
    setGenerating(true);
    setGenerateMessage(null);
    try {
      const created = await generateChecklist(userId);
      setGenerateMessage(`Generated ${created.length} checklist item(s).`);
      await fetchChecklist();
    } catch (err) {
      setGenerateMessage(`Generation failed: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const handleStatusChange = async (itemId, newStatus) => {
    setUpdatingId(itemId);
    setUpdateError(null);
    try {
      await updateChecklistItemStatus(userId, itemId, newStatus);
      await fetchChecklist();
    } catch (err) {
      setUpdateError(`Update failed: ${err.message}`);
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <div>
      <h2>✅ Your Obligation Checklist</h2>
      <p>
        This checklist is <strong>not generated automatically</strong>. Click the button below
        to run it live — the app will search the ingested legal corpus for obligations that
        apply to your profile, extract structured deadlines with an LLM, and compute due
        dates, right now.
      </p>

      <button className="btn primary" onClick={handleGenerate} disabled={generating}>
        {generating ? "Running retrieval + extraction pipeline..." : "🚀 Generate My Checklist"}
      </button>

      {generateMessage && <div className="alert info">{generateMessage}</div>}
      {loadError && <div className="alert error">Could not load checklist: {loadError}</div>}
      {updateError && <div className="alert error">{updateError}</div>}

      <hr />

      {!loadError && items === null && <div className="caption">Loading checklist...</div>}

      {!loadError && items && items.length === 0 && (
        <div className="alert info">No checklist items yet — click Generate My Checklist above.</div>
      )}

      {!loadError && items && items.length > 0 && (
        <div className="checklist">
          {items.map((item) => (
            <div className="card" key={item.id}>
              <div className="card-row">
                <div className="card-main">
                  <div className="card-title">
                    {STATUS_ICONS[item.status] || "🔲"} {item.title}
                  </div>
                  <div className="caption">
                    Category: {item.category}
                    {item.due_date ? ` · Due: ${item.due_date}` : ""}
                  </div>
                  <div className="card-desc">{item.description}</div>
                  <div className="caption">⚠️ {item.penalty_summary}</div>
                  <div className="caption">📖 Source: {item.source_citation}</div>
                </div>
                <div className="card-status">
                  <label className="caption">Status</label>
                  <select
                    value={item.status}
                    disabled={updatingId === item.id}
                    onChange={(e) => handleStatusChange(item.id, e.target.value)}
                  >
                    {STATUSES.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}