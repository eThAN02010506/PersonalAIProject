import { LoaderCircle, Plus, Shield, UserRound, X } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { UserAccount } from "../lib/types";

type AccountDialogProps = {
  user: UserAccount;
  onClose: () => void;
  onUserChanged: (user: UserAccount) => void;
};

export function AccountDialog({
  user,
  onClose,
  onUserChanged,
}: AccountDialogProps) {
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"admin" | "member">("member");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadUsers = useCallback(async () => {
    if (user.role !== "admin") return;
    setUsers(await api.listUsers());
  }, [user.role]);

  useEffect(() => {
    void loadUsers().catch((reason) => {
      setError(reason instanceof Error ? reason.message : "Could not load accounts");
    });
  }, [loadUsers]);

  const changePassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy("password");
    setError(null);
    try {
      const status = await api.changePassword(currentPassword, newPassword);
      if (status.user) onUserChanged(status.user);
      setCurrentPassword("");
      setNewPassword("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Password change failed");
    } finally {
      setBusy(null);
    }
  };

  const createAccount = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy("create");
    setError(null);
    try {
      await api.createUser({ username, displayName, password, role });
      setUsername("");
      setDisplayName("");
      setPassword("");
      setRole("member");
      await loadUsers();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Account creation failed");
    } finally {
      setBusy(null);
    }
  };

  const toggleActive = async (account: UserAccount) => {
    setBusy(account.id);
    setError(null);
    try {
      await api.setUserActive(account.id, !account.active);
      await loadUsers();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Account update failed");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <section
        className="model-dialog account-dialog"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="account-dialog-title"
      >
        <header className="dialog-header">
          <div>
            <h2 id="account-dialog-title">{user.display_name}</h2>
            <span>@{user.username} · {user.role}</span>
          </div>
          <button className="icon-button" onClick={onClose} title="Close" type="button">
            <X size={18} />
          </button>
        </header>
        <div className="account-dialog-body">
          <form className="account-form" onSubmit={(event) => void changePassword(event)}>
            <h3><UserRound size={15} /> Password</h3>
            <div className="account-field-grid">
              <label>
                <span>Current password</span>
                <input
                  autoComplete="current-password"
                  onChange={(event) => setCurrentPassword(event.target.value)}
                  required
                  type="password"
                  value={currentPassword}
                />
              </label>
              <label>
                <span>New password</span>
                <input
                  autoComplete="new-password"
                  minLength={8}
                  onChange={(event) => setNewPassword(event.target.value)}
                  required
                  type="password"
                  value={newPassword}
                />
              </label>
            </div>
            <button className="secondary-button" disabled={busy !== null} type="submit">
              {busy === "password" && <LoaderCircle className="spin" size={14} />}
              Change password
            </button>
          </form>

          {user.role === "admin" && (
            <>
              <form className="account-form" onSubmit={(event) => void createAccount(event)}>
                <h3><Plus size={15} /> Add account</h3>
                <div className="account-field-grid account-create-grid">
                  <label>
                    <span>Display name</span>
                    <input
                      maxLength={80}
                      onChange={(event) => setDisplayName(event.target.value)}
                      required
                      value={displayName}
                    />
                  </label>
                  <label>
                    <span>Username</span>
                    <input
                      maxLength={32}
                      minLength={3}
                      onChange={(event) => setUsername(event.target.value)}
                      required
                      value={username}
                    />
                  </label>
                  <label>
                    <span>Initial password</span>
                    <input
                      minLength={8}
                      onChange={(event) => setPassword(event.target.value)}
                      required
                      type="password"
                      value={password}
                    />
                  </label>
                  <label>
                    <span>Role</span>
                    <select
                      onChange={(event) => setRole(event.target.value as "admin" | "member")}
                      value={role}
                    >
                      <option value="member">Member</option>
                      <option value="admin">Administrator</option>
                    </select>
                  </label>
                </div>
                <button className="primary-button" disabled={busy !== null} type="submit">
                  {busy === "create" && <LoaderCircle className="spin" size={14} />}
                  Create account
                </button>
              </form>

              <section className="account-list">
                <h3><Shield size={15} /> Accounts</h3>
                {users.map((account) => (
                  <div className="account-row" key={account.id}>
                    <span className={`account-state ${account.active ? "active" : ""}`} />
                    <div>
                      <strong>{account.display_name}</strong>
                      <small>@{account.username} · {account.role}</small>
                    </div>
                    <button
                      className="secondary-button"
                      disabled={busy !== null || account.id === user.id}
                      onClick={() => void toggleActive(account)}
                      type="button"
                    >
                      {busy === account.id && <LoaderCircle className="spin" size={13} />}
                      {account.active ? "Disable" : "Enable"}
                    </button>
                  </div>
                ))}
              </section>
            </>
          )}
          {error && <div className="error-banner">{error}</div>}
        </div>
      </section>
    </div>
  );
}
