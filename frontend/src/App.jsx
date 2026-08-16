import React, { useEffect, useState } from "react";
import Home from "./pages/Home.jsx";
import Onboarding from "./pages/Onboarding.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Chat from "./pages/Chat.jsx";

const USER_ID_KEY = "vn_legal_user_id";
const CHAT_HISTORY_KEY = "vn_legal_chat_history";

export const TABS = {
  home: { label: "🏠 Home" },
  onboarding: { label: "📝 Onboarding" },
  dashboard: { label: "✅ Dashboard" },
  chat: { label: "💬 Chat" },
};

export default function App() {
  const [tab, setTab] = useState("home");
  const [userId, setUserIdState] = useState(() => localStorage.getItem(USER_ID_KEY));
  const [chatHistory, setChatHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  });

  const setUserId = (id) => {
    if (id) localStorage.setItem(USER_ID_KEY, id);
    else localStorage.removeItem(USER_ID_KEY);
    setUserIdState(id);
  };

  useEffect(() => {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory));
  }, [chatHistory]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🇻🇳 Personal Legal Assistant — MVP</h1>
        <p className="caption">
          A redesigned, focused MVP: personal info collection, an auto-generated obligation
          checklist you activate manually, a RAG legal chatbot, and a real legal-source
          crawler behind the scenes.
        </p>
      </header>

      <nav className="app-nav">
        {Object.entries(TABS).map(([key, { label }]) => (
          <button
            key={key}
            className={`nav-tab${tab === key ? " active" : ""}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {tab === "home" && <Home userId={userId} goTo={setTab} />}
        {tab === "onboarding" && <Onboarding userId={userId} setUserId={setUserId} goTo={setTab} />}
        {tab === "dashboard" && <Dashboard userId={userId} goTo={setTab} />}
        {tab === "chat" && (
          <Chat userId={userId} goTo={setTab} chatHistory={chatHistory} setChatHistory={setChatHistory} />
        )}
      </main>

      <footer className="app-footer">
        <span>
          {userId ? `Signed in as user ${userId}` : "No profile yet — start with Onboarding."}
        </span>
      </footer>
    </div>
  );
}
