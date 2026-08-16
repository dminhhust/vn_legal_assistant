import React, { useState } from "react";
import { ApiError, createProfile } from "../api.js";

const MARITAL_STATUSES = ["", "single", "married", "divorced", "widowed"];
const GENDERS = ["", "male", "female", "other", "prefer_not_to_say"];
const OCCUPATIONS = ["", "employee", "freelancer", "business_owner", "student", "retired", "unemployed"];
const INCOME_SOURCES = ["salary", "freelance", "business", "investment", "rental", "other"];

export default function Onboarding({ userId, setUserId, goTo }) {
  const [form, setForm] = useState({
    username: "",
    age: 30,
    marital_status: "",
    gender: "",
    dependents: 0,
    province: "",
    occupation_type: "",
    income_sources: [],
    has_business: false,
    business_sector: "",
    owns_property: false,
    owns_vehicle: false,
    reminder_lead_days: 3,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [created, setCreated] = useState(null);

  const set = (key) => (event) => {
    const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    setForm((f) => ({ ...f, [key]: value }));
  };

  const toggleIncomeSource = (source) => {
    setForm((f) => ({
      ...f,
      income_sources: f.income_sources.includes(source)
        ? f.income_sources.filter((s) => s !== source)
        : [...f.income_sources, source],
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setCreated(null);
    if (!form.username.trim()) {
      setError("Username is required.");
      return;
    }
    setSubmitting(true);
    try {
      const payload = {
        username: form.username.trim(),
        age: form.age || null,
        gender: form.gender || null,
        marital_status: form.marital_status || null,
        province: form.province.trim() || null,
        dependents: Number(form.dependents) || 0,
        occupation_type: form.occupation_type || null,
        income_sources: form.income_sources,
        has_business: form.has_business,
        business_sector: form.business_sector.trim() || null,
        owns_property: form.owns_property,
        owns_vehicle: form.owns_vehicle,
        reminder_lead_days: Number(form.reminder_lead_days) || 3,
      };
      const data = await createProfile(payload);
      setUserId(data.user_id);
      setCreated(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setError(`Username '${form.username}' is already taken — pick another.`);
      } else {
        setError(`Request failed: ${err.message}`);
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (userId && !created) {
    return (
      <div>
        <div className="alert success">
          You're already onboarded as user <code>{userId}</code>.
        </div>
        <div className="caption">
          Editing an existing profile isn't wired into this page yet — use the API directly
          if needed.
        </div>
        <div className="link-list">
          <button className="link" onClick={() => goTo("dashboard")}>
            ➡️ ✅ Go generate your checklist
          </button>
          <button className="link" onClick={() => { setUserId(null); }}>
            ↪️ Switch to a different user
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2>📝 Onboarding</h2>
      <div className="caption">
        A few quick questions — used only to figure out which legal obligations apply to you.
      </div>

      {created && (
        <div className="alert success">
          Profile created!
          {created.traits && created.traits.length > 0 && (
            <div className="caption">Derived traits: {created.traits.join(", ")}</div>
          )}
          <div className="link-list">
            <button className="link" onClick={() => goTo("dashboard")}>
              ➡️ ✅ Go generate your obligation checklist
            </button>
          </div>
        </div>
      )}

      {error && <div className="alert error">{error}</div>}

      {!created && (
        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <label>Username</label>
            <input
              type="text"
              placeholder="e.g. minh_nguyen"
              value={form.username}
              onChange={set("username")}
            />
          </div>

          <h3>Identity</h3>
          <div className="grid-2">
            <div className="field">
              <label>Age</label>
              <input type="number" min="0" max="120" value={form.age} onChange={set("age")} />
            </div>
            <div className="field">
              <label>Marital status</label>
              <select value={form.marital_status} onChange={set("marital_status")}>
                {MARITAL_STATUSES.map((s) => (
                  <option key={s} value={s}>{s || "(none)"}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Gender</label>
              <select value={form.gender} onChange={set("gender")}>
                {GENDERS.map((g) => (
                  <option key={g} value={g}>{g || "(none)"}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Number of dependents</label>
              <input type="number" min="0" max="20" value={form.dependents} onChange={set("dependents")} />
            </div>
          </div>
          <div className="field">
            <label>Province / city of residence</label>
            <input type="text" placeholder="e.g. Hanoi" value={form.province} onChange={set("province")} />
          </div>

          <h3>Work</h3>
          <div className="field">
            <label>Occupation type</label>
            <select value={form.occupation_type} onChange={set("occupation_type")}>
              {OCCUPATIONS.map((o) => (
                <option key={o} value={o}>{o || "(none)"}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Income sources</label>
            <div className="checkbox-row">
              {INCOME_SOURCES.map((src) => (
                <label key={src} className="checkbox">
                  <input
                    type="checkbox"
                    checked={form.income_sources.includes(src)}
                    onChange={() => toggleIncomeSource(src)}
                  />
                  {src}
                </label>
              ))}
            </div>
          </div>
          <div className="field">
            <label className="checkbox">
              <input type="checkbox" checked={form.has_business} onChange={set("has_business")} />
              I own a business
            </label>
          </div>
          <div className="field">
            <label>Business sector (if applicable)</label>
            <input
              type="text"
              placeholder="e.g. retail"
              value={form.business_sector}
              onChange={set("business_sector")}
            />
          </div>

          <h3>Assets</h3>
          <div className="grid-2">
            <div className="field">
              <label className="checkbox">
                <input type="checkbox" checked={form.owns_property} onChange={set("owns_property")} />
                I own property (land / housing)
              </label>
            </div>
            <div className="field">
              <label className="checkbox">
                <input type="checkbox" checked={form.owns_vehicle} onChange={set("owns_vehicle")} />
                I own a vehicle
              </label>
            </div>
          </div>

          <h3>Preferences</h3>
          <div className="field">
            <label>Remind me this many days before a deadline: {form.reminder_lead_days}</label>
            <input
              type="range"
              min="1"
              max="30"
              value={form.reminder_lead_days}
              onChange={set("reminder_lead_days")}
            />
          </div>

          <button type="submit" className="btn primary" disabled={submitting}>
            {submitting ? "Creating profile..." : "Create profile"}
          </button>
        </form>
      )}
    </div>
  );
}