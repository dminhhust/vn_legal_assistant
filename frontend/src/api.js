// API layer — single place that talks to the FastAPI backend.
//
// The browser always calls `/api/*` (never the backend origin
// directly): in dev, Vite proxies it (vite.config.js); in production,
// nginx proxies it (frontend/nginx.conf.template). That keeps the app
// origin-agnostic and CORS-free in every deployment shape.
const BASE = "/api";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// Equivalent of the old Streamlit `friendly_error_message`: FastAPI
// error bodies carry a `detail` field that is the actually-useful
// message (e.g. the 503 "needs at least one working LLM provider"
// text); surface it instead of a bare status line.
async function request(path, options = {}) {
  let resp;
  try {
    resp = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new ApiError(`Network error: ${err.message}`);
  }
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body && body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // non-JSON error body — keep the status line
    }
    throw new ApiError(detail, resp.status);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

export const getHealth = () => request("/health");

export const createProfile = (payload) => request("/profile", {
  method: "POST",
  body: JSON.stringify(payload),
});

export const getProfile = (userId) => request(`/profile/${encodeURIComponent(userId)}`);

export const getChecklist = (userId) => request(`/checklist/${encodeURIComponent(userId)}`);

export const generateChecklist = (userId) => request(`/checklist/${encodeURIComponent(userId)}/generate`, {
  method: "POST",
});

export const updateChecklistItemStatus = (userId, itemId, status) =>
  request(`/checklist/${encodeURIComponent(userId)}/items/${encodeURIComponent(itemId)}`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });

export const sendChat = (userId, message, history) =>
  request("/chat", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, message, history }),
  });
