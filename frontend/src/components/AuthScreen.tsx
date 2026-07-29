import { Bot, LoaderCircle, LockKeyhole } from "lucide-react";
import { type FormEvent, useState } from "react";

import { api } from "../lib/api";
import type { UserAccount } from "../lib/types";

type AuthScreenProps = {
  bootstrapRequired: boolean;
  onAuthenticated: (user: UserAccount) => void;
};

export function AuthScreen({
  bootstrapRequired,
  onAuthenticated,
}: AuthScreenProps) {
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const status = bootstrapRequired
        ? await api.bootstrap({ username, displayName, password })
        : await api.login(username, password);
      if (!status.user) throw new Error("The account session was not created.");
      onAuthenticated(status.user);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Authentication failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-screen">
      <section className="auth-panel" aria-labelledby="auth-title">
        <header>
          <div className="brand-mark"><Bot size={19} /></div>
          <div>
            <h1 id="auth-title">Qwopus Agent</h1>
            <p>{bootstrapRequired ? "Create the first administrator" : "Sign in"}</p>
          </div>
        </header>
        <form onSubmit={(event) => void submit(event)}>
          {bootstrapRequired && (
            <label>
              <span>Display name</span>
              <input
                autoComplete="name"
                autoFocus
                maxLength={80}
                onChange={(event) => setDisplayName(event.target.value)}
                required
                value={displayName}
              />
            </label>
          )}
          <label>
            <span>Username</span>
            <input
              autoComplete="username"
              autoFocus={!bootstrapRequired}
              maxLength={32}
              minLength={3}
              onChange={(event) => setUsername(event.target.value)}
              required
              value={username}
            />
          </label>
          <label>
            <span>Password</span>
            <input
              autoComplete={bootstrapRequired ? "new-password" : "current-password"}
              maxLength={256}
              minLength={bootstrapRequired ? 8 : 1}
              onChange={(event) => setPassword(event.target.value)}
              required
              type="password"
              value={password}
            />
          </label>
          {error && <div className="error-banner">{error}</div>}
          <button className="primary-button" disabled={submitting} type="submit">
            {submitting
              ? <LoaderCircle className="spin" size={16} />
              : <LockKeyhole size={16} />}
            {bootstrapRequired ? "Initialize" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}
