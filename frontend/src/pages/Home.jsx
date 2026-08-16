import React, { useEffect, useState } from "react";
import { getHealth } from "../api.js";

export default function Home({ userId, goTo }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((data) => !cancelled && setStatus(data))
      .catch((err) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <h2>Backend status</h2>
      {error && <div className="alert error">Could not reach backend: {error}</div>}
      {status && (
        <div className="alert success">
          Backend reachable: {status.status}
          {status.llm_providers_available && status.llm_providers_available.length > 0 ? (
            <div className="caption">
              LLM providers available: {status.llm_providers_available.join(", ")}
            </div>
          ) : (
            <div className="caption">
              No LLM provider keys detected — the checklist generator and chat will not be
              able to make real model calls until at least one of ANTHROPIC_API_KEY /
              OPENAI_API_KEY / GOOGLE_API_KEY is set.
            </div>
          )}
        </div>
      )}
      {!status && !error && <div className="caption">Checking backend...</div>}

      <hr />

      {userId ? (
        <div className="alert info">Signed in as user {userId}.</div>
      ) : (
        <div className="alert info">No profile yet — start with Onboarding.</div>
      )}

      <div className="link-list">
        <button className="link" onClick={() => goTo("onboarding")}>
          ➡️ 📝 Onboarding — personal info collection
        </button>
        <button className="link" onClick={() => goTo("dashboard")}>
          ➡️ ✅ Dashboard — generate & view your obligation checklist
        </button>
        <button className="link" onClick={() => goTo("chat")}>
          ➡️ 💬 Chat — ask the RAG legal assistant
        </button>
      </div>
    </div>
  );
}
