import React, { useState } from "react";
import { sendChat } from "../api.js";

export default function Chat({ userId, goTo, chatHistory, setChatHistory }) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);

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

  const handleSend = async (event) => {
    event.preventDefault();
    const message = input.trim();
    if (!message || sending) return;
    setError(null);
    setInput("");
    // Sends the full visible conversation (minus the new turn) with
    // every request — the backend is deliberately stateless.
    const history = chatHistory.map((t) => ({ role: t.role, content: t.content }));
    const userTurn = { role: "user", content: message };
    setChatHistory([...chatHistory, userTurn]);
    setSending(true);
    try {
      const data = await sendChat(userId, message, history);
      const assistantTurn = { role: "assistant", content: data.text, tools: data.tool_calls_made };
      setChatHistory((prev) => [...prev, assistantTurn]);
    } catch (err) {
      setError(`Request failed: ${err.message}`);
      // keep the user turn visible, but surface the failure
      setChatHistory((prev) => [...prev, { role: "assistant", content: `(error) ${err.message}`, error: true }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div>
      <h2>💬 Ask the Legal Assistant</h2>
      <div className="caption">
        This assistant can search the ingested legal corpus and check/update your checklist.
        It's not a substitute for professional legal advice.
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="chat-history">
        {chatHistory.length === 0 && <div className="caption">No messages yet.</div>}
        {chatHistory.map((turn, i) => (
          <div key={i} className={`chat-turn ${turn.role}`}>
            <div className={`chat-bubble ${turn.role}${turn.error ? " error" : ""}`}>
              {turn.content}
            </div>
            {turn.tools && turn.tools.length > 0 && (
              <div className="caption">Tools used: {turn.tools.join(", ")}</div>
            )}
          </div>
        ))}
        {sending && <div className="caption">Thinking...</div>}
      </div>

      <form className="chat-input" onSubmit={handleSend}>
        <input
          type="text"
          placeholder="Ask a question, e.g. 'What tax deadlines apply to me?'"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={sending}
        />
        <button type="submit" className="btn primary" disabled={sending || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}