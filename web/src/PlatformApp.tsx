"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, post, resourceUrl } from "./platform-api";
type Row = Record<string, any>;
const terminal = ["succeeded", "failed", "cancelled", "partially_succeeded"];
const label: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  retry_wait: "等待重试",
  cancel_requested: "取消中",
  cancelled: "已取消",
  succeeded: "已完成",
  failed: "失败",
  partially_succeeded: "部分完成",
  pending: "待处理",
};
const mergeMessages = (old: Row[], fresh: Row[]) =>
  Array.from(new Map([...old, ...fresh].map((m) => [m.id, m])).values()).sort(
    (a, b) => a.seq - b.seq,
  );
const markdown = (x: any): string =>
  typeof x === "string"
    ? x
    : x?.text || x?.markdown || x?.answer || JSON.stringify(x, null, 2);
export default function PlatformApp() {
  const [user, setUser] = useState<Row | null>(null),
    [wid, setWid] = useState(""),
    [projects, setProjects] = useState<Row[]>([]),
    [pid, setPid] = useState("");
  const [convs, setConvs] = useState<Row[]>([]),
    [cid, setCid] = useState(""),
    [messages, setMessages] = useState<Row[]>([]),
    [docs, setDocs] = useState<Row[]>([]),
    [tasks, setTasks] = useState<Row[]>([]),
    [reports, setReports] = useState<Row[]>([]);
  const [email, setEmail] = useState(""),
    [password, setPassword] = useState(""),
    [register, setRegister] = useState(false),
    [text, setText] = useState(""),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false),
    [tab, setTab] = useState("chat"),
    [selected, setSelected] = useState<string[]>([]),
    [progress, setProgress] = useState("");
  const [papers, setPapers] = useState<Row[]>([]);
  const [tokens, setTokens] = useState<Row[]>([]),
    [sessions, setSessions] = useState<Row[]>([]),
    [adminTasks, setAdminTasks] = useState<Row[]>([]),
    [newToken, setNewToken] = useState("");
  const key = useRef<string | null>(null),
    epoch = useRef(0),
    fileInput = useRef<HTMLInputElement>(null);
  const wp = `/workspaces/${wid}`,
    pp = `${wp}/projects/${pid}`;
  const fail = (e: unknown) =>
    setError(e instanceof Error ? e.message : "操作失败");
  const identify = useCallback(async () => {
    try {
      const u = await api("/me"),
        ws = await api<Row[]>("/workspaces");
      setUser(u);
      setWid(ws[0]?.id || "");
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void identify();
  }, [identify]);
  useEffect(() => {
    if (!wid) return;
    let live = true;
    api<Row[]>(`${wp}/projects`)
      .then((p) => {
        if (live) {
          setProjects(p);
          setPid(p[0]?.id || "");
        }
      })
      .catch(fail);
    return () => {
      live = false;
    };
  }, [wid]);
  const refresh = useCallback(async () => {
    if (!wid || !pid) return;
    const current = epoch.current;
    const [d, c, t, r, p] = await Promise.all([
      api<Row[]>(`${pp}/documents`),
      api<Row[]>(`${pp}/conversations`),
      api<Row[]>(`${wp}/tasks?project_id=${pid}`),
      api<Row[]>(`${pp}/reports`),
      api<Row[]>(`${pp}/papers`),
    ]);
    if (current !== epoch.current) return;
    setDocs(d);
    setConvs(c);
    setTasks(t);
    setReports(r);
    setPapers(p);
  }, [wid, pid]);
  useEffect(() => {
    epoch.current++;
    setCid("");
    setMessages([]);
    setSelected([]);
    setTasks([]);
    if (pid) void refresh().catch(fail);
  }, [pid, refresh]);
  useEffect(() => {
    if (!cid) {
      setMessages([]);
      return;
    }
    let live = true;
    api<Row[]>(`${wp}/conversations/${cid}/messages?latest=true`)
      .then((m) => {
        if (live) setMessages(m);
      })
      .catch(fail);
    return () => {
      live = false;
    };
  }, [cid, wid]);
  useEffect(() => {
    if (!pid) return;
    let live = true;
    const timer = setInterval(() => {
      void refresh().catch(fail);
      if (cid)
        void api<Row[]>(`${wp}/conversations/${cid}/messages?latest=true`)
          .then((m) => {
            if (live) setMessages((old) => mergeMessages(old, m));
          })
          .catch(fail);
    }, 3000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [pid, cid, refresh]);
  const active = tasks.find(
    (t) => t.conversation_id === cid && !terminal.includes(t.status),
  );
  useEffect(() => {
    if (!active) return;
    const s = new EventSource(resourceUrl(`${wp}/tasks/${active.id}/events`));
    s.onmessage = (e) => {
      try {
        setProgress(JSON.parse(e.data).event_type);
      } catch {}
    };
    s.onerror = () => {
      void refresh().catch(fail);
    };
    return () => s.close();
  }, [active?.id, wid, refresh]);
  useEffect(() => {
    if (tab !== "account" || !user) return;
    let live = true;
    void Promise.all([api<Row[]>("/credentials"), api<Row[]>("/sessions")])
      .then(([a, b]) => {
        if (!live) return;
        setTokens(a);
        setSessions(b);
      })
      .catch((e) => {
        if (live) fail(e);
      });
    void api<Row[]>("/admin/tasks")
      .then((rows) => {
        if (live) setAdminTasks(rows);
      })
      .catch(() => {
        if (live) setAdminTasks([]);
      });
    return () => {
      live = false;
    };
  }, [tab, user]);
  async function auth(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (register) {
        const r = await api("/auth/register", post({ email, password }));
        if (!r.user.email_verified_at) {
          setNotice("注册成功，请查看验证邮件。");
          setRegister(false);
          return;
        }
      }
      await api("/auth/login", post({ email, password }));
      await identify();
      setPassword("");
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function createProject() {
    const name = prompt("研究项目名称");
    if (!name) return;
    try {
      const p = await api(`${wp}/projects`, post({ name }));
      setProjects((old) => [...old, p]);
      setPid(p.id);
    } catch (e) {
      fail(e);
    }
  }
  async function editProject(remove = false) {
    const project = projects.find((p) => p.id === pid);
    if (!project) return;
    try {
      if (remove) {
        if (!confirm("删除项目会取消未完成任务，工作区文件仍保留。继续？"))
          return;
        await api(`${pp}?version=${project.version}`, { method: "DELETE" });
        const next = projects.filter((p) => p.id !== pid);
        setProjects(next);
        setPid(next[0]?.id || "");
      } else {
        const name = prompt("项目名称", project.name);
        if (!name) return;
        const updated = await api(pp, {
          method: "PATCH",
          body: JSON.stringify({ version: project.version, name }),
        });
        setProjects((old) => old.map((p) => (p.id === pid ? updated : p)));
      }
    } catch (e) {
      fail(e);
    }
  }
  async function newChat() {
    const c = await api(`${pp}/conversations`, post());
    setConvs((old) => [c, ...old]);
    setCid(c.id);
    return c.id;
  }
  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (busy || text.trim().length < 2) return;
    setBusy(true);
    setError("");
    try {
      const id = cid || (await newChat());
      key.current ??= crypto.randomUUID();
      await api(`${wp}/conversations/${id}/messages`, {
        ...post({ message: text, document_ids: selected }),
        headers: { "Idempotency-Key": key.current },
      });
      key.current = null;
      setText("");
      setMessages(await api(`${wp}/conversations/${id}/messages?latest=true`));
      await refresh();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function upload(f: File) {
    setBusy(true);
    try {
      const digest = Array.from(
        new Uint8Array(
          await crypto.subtle.digest("SHA-256", await f.arrayBuffer()),
        ),
      )
        .map((v) => v.toString(16).padStart(2, "0"))
        .join("");
      const grant = await api(
        `${pp}/uploads/init`,
        post({ filename: f.name, size: f.size, sha256: digest }),
      );
      const form = new FormData();
      Object.entries(grant.fields).forEach(([k, v]) => form.set(k, String(v)));
      form.set("file", f);
      const uploaded = await fetch(grant.url, { method: "POST", body: form });
      if (!uploaded.ok) throw new Error("文件上传失败，请重试");
      const d = await api(`${wp}/uploads/${grant.id}/complete`, post());
      await api(`${pp}/imports`, {
        ...post({ document_ids: [d.document_id] }),
        headers: { "Idempotency-Key": crypto.randomUUID() },
      });
      setNotice(
        d.duplicate
          ? "已复用论文，正在确认索引。"
          : "上传完成，后台正在解析与索引。",
      );
      await refresh();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function action(id: string, operation: string) {
    try {
      await api(`${wp}/tasks/${id}/${operation}`, post());
      await refresh();
    } catch (e) {
      fail(e);
    }
  }
  async function logout() {
    await api("/auth/logout", post());
    epoch.current++;
    setNewToken("");
    setUser(null);
    setWid("");
    setPid("");
    setProjects([]);
    setMessages([]);
    setTasks([]);
    // A full document navigation discards all account data and pending React callbacks.
    // A same-component login must never inherit another user's one-time credentials.
    window.location.replace("/");
  }
  if (loading)
    return (
      <main className="grid h-screen place-items-center">正在恢复会话…</main>
    );
  if (!user)
    return (
      <main className="min-h-screen bg-paper-50 flex items-center justify-center p-6">
        <form
          onSubmit={auth}
          className="w-full max-w-md rounded-2xl border border-paper-200 bg-white p-8 space-y-5"
        >
          <h1 className="text-3xl font-display">Research Copilot</h1>
          <p className="text-sm text-ink-400">你的论文、研究过程与知识工作区</p>
          <h2>{register ? "创建账号" : "登录工作区"}</h2>
          <label className="block">
            邮箱
            <input
              aria-label="邮箱"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
          <label className="block">
            密码
            <input
              aria-label="密码"
              type="password"
              minLength={8}
              maxLength={256}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
          {error && (
            <p role="alert" className="text-red-700">
              {error}
            </p>
          )}
          {notice && <p role="status">{notice}</p>}
          <button
            disabled={busy}
            className="w-full bg-ink-800 text-white p-3 rounded"
          >
            {busy ? "请稍候…" : register ? "注册" : "登录"}
          </button>
          <button
            type="button"
            onClick={() => {
              setRegister(!register);
              setError("");
            }}
            className="underline"
          >
            {register ? "已有账号？登录" : "创建新账号"}
          </button>
          <div className="flex gap-3 text-xs">
            <button
              type="button"
              onClick={() =>
                void api("/auth/reset/request", post({ email }))
                  .then(() => setNotice("如账号存在，重置邮件已发送。"))
                  .catch(fail)
              }
            >
              忘记密码
            </button>
            <button
              type="button"
              onClick={() =>
                void api("/auth/verify/request", post({ email }))
                  .then(() => setNotice("如账号存在，验证邮件已发送。"))
                  .catch(fail)
              }
            >
              重发验证邮件
            </button>
            <button
              type="button"
              onClick={() => {
                const p = new URLSearchParams(location.search),
                  a = p.get("action"),
                  t = p.get("token");
                if (!a || !t) {
                  setError("请通过邮件链接打开此页面");
                  return;
                }
                void api(
                  `/auth/${a}/confirm`,
                  post({ token: t, ...(a === "reset" ? { password } : {}) }),
                )
                  .then(() => {
                    setNotice("操作成功，请登录");
                    history.replaceState({}, "", location.pathname);
                  })
                  .catch(fail);
              }}
            >
              完成邮件操作
            </button>
          </div>
        </form>
      </main>
    );
  return (
    <div className="h-screen flex bg-paper-50 text-ink-800">
      <aside className="w-64 shrink-0 border-r p-4 flex flex-col gap-4 overflow-y-auto">
        <h1 className="font-display text-xl">Research Copilot</h1>
        <span className="text-xs break-all">{user.email}</span>
        <select
          aria-label="研究项目"
          className="border rounded p-2"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
        >
          <option value="">选择研究项目</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        <button
          onClick={() => void createProject()}
          className="border rounded p-2"
        >
          ＋ 新建项目
        </button>
        {pid && (
          <>
            <div className="flex gap-3 text-xs">
              <button onClick={() => void editProject()}>重命名项目</button>
              <button onClick={() => void editProject(true)}>删除项目</button>
            </div>
            <button
              onClick={() => void newChat().catch(fail)}
              className="bg-ink-800 text-white rounded p-2"
            >
              ＋ 新建对话
            </button>
            {convs.map((c) => (
              <button
                key={c.id}
                onClick={() => {
                  setCid(c.id);
                  setTab("chat");
                }}
                className={`text-left p-2 rounded ${cid === c.id ? "bg-paper-200" : ""}`}
              >
                {c.title || "研究对话"}
              </button>
            ))}
          </>
        )}
        <button onClick={() => void logout().catch(fail)} className="mt-auto">
          退出登录
        </button>
      </aside>
      <main className="flex-1 min-w-0 flex flex-col">
        <header className="border-b p-4 flex gap-5">
          {[
            ["chat", "研究对话"],
            ["papers", "论文库"],
            ["tasks", "任务进度"],
            ["reports", "研究报告"],
            ["account", "账号与管理"],
          ].map(([id, title]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={tab === id ? "font-bold border-b-2" : ""}
            >
              {title}
            </button>
          ))}
        </header>
        {error && (
          <div role="alert" className="bg-red-50 text-red-700 p-3">
            {error}
            <button className="float-right" onClick={() => setError("")}>
              关闭
            </button>
          </div>
        )}
        {notice && (
          <div role="status" className="bg-green-50 p-3">
            {notice}
            <button className="float-right" onClick={() => setNotice("")}>
              关闭
            </button>
          </div>
        )}
        {!pid && tab !== "account" ? (
          <section className="m-auto text-center">
            <h2 className="text-2xl">开始一个研究项目</h2>
            <p className="my-4">导入论文，分析证据，撰写综述。</p>
            <button onClick={() => void createProject()}>创建项目</button>
          </section>
        ) : (
          <>
            {tab === "chat" && (
              <>
                <section className="flex-1 overflow-y-auto p-6">
                  <div className="max-w-3xl mx-auto space-y-5">
                    {messages.length > 0 && messages[0].seq > 1 && (
                      <button
                        className="underline"
                        onClick={() =>
                          void api<Row[]>(
                            `${wp}/conversations/${cid}/messages?latest=true&before=${messages[0].seq}`,
                          )
                            .then((m) =>
                              setMessages((old) => mergeMessages(old, m)),
                            )
                            .catch(fail)
                        }
                      >
                        加载更早的消息
                      </button>
                    )}
                    {!messages.length && (
                      <p className="text-ink-400 mt-12">
                        选择论文，提出你的研究问题。离开页面后任务仍会继续。
                      </p>
                    )}
                    {messages.map((m) => (
                      <article
                        key={m.id}
                        className={`p-4 rounded-xl ${m.role === "user" ? "bg-paper-200" : "bg-white border"}`}
                      >
                        <div className="text-xs mb-2">
                          {m.role === "user" ? "你" : "Research Copilot"}
                        </div>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {markdown(m.content)}
                        </ReactMarkdown>
                      </article>
                    ))}
                    {active && (
                      <div role="status">
                        {label[active.status]} · {progress || "准备研究"}
                        <button
                          onClick={() => void action(active.id, "cancel")}
                          className="ml-4 underline"
                        >
                          取消
                        </button>
                      </div>
                    )}
                  </div>
                </section>
                <form onSubmit={send} className="border-t p-4">
                  <div className="max-w-3xl mx-auto flex gap-3">
                    <textarea
                      aria-label="研究问题"
                      value={text}
                      onChange={(e) => {
                        setText(e.target.value);
                        key.current = null;
                      }}
                      maxLength={3000}
                      placeholder="比较这些论文的方法、实验与局限…"
                      className="flex-1 border rounded p-3"
                    />
                    <button
                      disabled={busy || text.trim().length < 2 || !!active}
                      className="bg-ink-800 text-white px-5 rounded disabled:opacity-40"
                    >
                      发送
                    </button>
                  </div>
                  <p className="max-w-3xl mx-auto text-xs mt-2">
                    {selected.length
                      ? `已选 ${selected.length} 篇论文`
                      : "未指定时使用项目中可用论文"}{" "}
                    · 回答请结合原文核对
                  </p>
                </form>
              </>
            )}
            {tab === "papers" && (
              <section className="overflow-auto p-6">
                <div className="flex justify-between mb-5">
                  <h2>项目论文库</h2>
                  <button
                    disabled={busy}
                    onClick={() => fileInput.current?.click()}
                    className="bg-ink-800 text-white p-2 rounded"
                  >
                    上传 PDF
                  </button>
                  <input
                    ref={fileInput}
                    type="file"
                    accept="application/pdf"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void upload(f);
                      e.target.value = "";
                    }}
                  />
                </div>
                {docs.map((d) => (
                  <article
                    key={d.id}
                    className="border bg-white rounded p-4 mb-3 flex gap-4"
                  >
                    <input
                      type="checkbox"
                      aria-label={`选择 ${d.title}`}
                      checked={selected.includes(d.id)}
                      onChange={(e) =>
                        setSelected((old) =>
                          e.target.checked
                            ? [...old, d.id]
                            : old.filter((id) => id !== d.id),
                        )
                      }
                    />
                    <div className="flex-1">
                      <a
                        target="_blank"
                        rel="noreferrer"
                        href={resourceUrl(`${wp}/files/${d.file_id}/content`)}
                        className="underline"
                      >
                        {d.title}
                      </a>
                      <p className="text-xs mt-2">
                        解析：{label[d.parse_status] || d.parse_status} · 索引：
                        {label[d.index_status] || d.index_status}
                      </p>
                    </div>
                    <button
                      className="mr-3"
                      onClick={() =>
                        void api(`${pp}/papers/${d.paper_id}`, {
                          method: "DELETE",
                        })
                          .then(() => refresh())
                          .catch(fail)
                      }
                    >
                      从项目移除
                    </button>
                    <button
                      onClick={() => {
                        if (
                          confirm(
                            "从工作区删除文档会影响关联的所有项目。继续？",
                          )
                        )
                          void api(`${wp}/documents/${d.id}`, {
                            method: "DELETE",
                          })
                            .then(() => {
                              setSelected((old) =>
                                old.filter((id) => id !== d.id),
                              );
                              return refresh();
                            })
                            .catch(fail);
                      }}
                    >
                      删除
                    </button>
                  </article>
                ))}
                {papers
                  .filter((p) => !docs.some((d) => d.paper_id === p.id))
                  .map((p) => (
                    <article
                      key={p.id}
                      className="border bg-white rounded p-4 mb-3"
                    >
                      <strong>{p.title}</strong>
                      <p className="text-xs">学术元数据 · 尚未关联可用文档</p>
                      {p.metadata_json?.arxiv_id && (
                        <button
                          className="underline mr-3"
                          onClick={() =>
                            void api(`${pp}/imports`, {
                              ...post({ paper_ids: [p.id] }),
                              headers: {
                                "Idempotency-Key": crypto.randomUUID(),
                              },
                            })
                              .then(() => refresh())
                              .catch(fail)
                          }
                        >
                          导入公开 PDF
                        </button>
                      )}
                      <button
                        onClick={() =>
                          void api(`${pp}/papers/${p.id}`, { method: "DELETE" })
                            .then(() => refresh())
                            .catch(fail)
                        }
                      >
                        从项目移除
                      </button>
                    </article>
                  ))}
                {!docs.length && <p>还没有论文，上传 PDF 后会自动解析。</p>}
              </section>
            )}
            {tab === "tasks" && (
              <section className="overflow-auto p-6 space-y-3">
                {tasks.map((t) => (
                  <article key={t.id} className="border rounded bg-white p-4">
                    <div className="flex justify-between">
                      <strong>
                        {t.kind === "import" ? "论文导入" : "研究任务"}
                      </strong>
                      <span>{label[t.status] || t.status}</span>
                    </div>
                    <p className="text-xs my-2">{t.id}</p>
                    {t.error_code && (
                      <p className="text-red-700">{t.error_code}</p>
                    )}
                    {(["failed", "cancelled"].includes(t.status) ||
                      (t.kind === "import" &&
                        t.status === "partially_succeeded")) && (
                      <button onClick={() => void action(t.id, "retry")}>
                        重试
                      </button>
                    )}
                    {!terminal.includes(t.status) && (
                      <button onClick={() => void action(t.id, "cancel")}>
                        取消
                      </button>
                    )}
                  </article>
                ))}
              </section>
            )}
            {tab === "account" && (
              <section className="overflow-auto p-6 space-y-5">
                <h2 className="text-xl">个人凭证</h2>
                <p>
                  凭证仅在创建时显示，请妥善保存。CLI
                  通过凭证访问已授权的工作区。
                </p>
                <button
                  className="underline"
                  onClick={() => {
                    const name = prompt("凭证名称");
                    if (!name) return;
                    void api(
                      "/credentials",
                      post({ name, scopes: ["read", "write"] }),
                    )
                      .then(async (t) => {
                        setNewToken(t.token);
                        setTokens(await api("/credentials"));
                      })
                      .catch(fail);
                  }}
                >
                  创建读写凭证（30 天）
                </button>
                {newToken && (
                  <div>
                    <code className="break-all">{newToken}</code>
                    <button onClick={() => setNewToken("")}>隐藏</button>
                  </div>
                )}
                {tokens.map((t) => (
                  <div key={t.id}>
                    {t.name} · {t.revoked_at ? "已撤销" : "有效"}{" "}
                    {!t.revoked_at && (
                      <button
                        className="underline"
                        onClick={() =>
                          void api(`/credentials/${t.id}`, { method: "DELETE" })
                            .then(async () =>
                              setTokens(await api("/credentials")),
                            )
                            .catch(fail)
                        }
                      >
                        撤销
                      </button>
                    )}
                  </div>
                ))}
                <h2 className="text-xl">登录会话</h2>
                {sessions.map((s) => (
                  <div key={s.id}>
                    {new Date(s.created_at).toLocaleString()} ·{" "}
                    {s.revoked_at ? "已撤销" : "有效"}{" "}
                    {!s.revoked_at && (
                      <button
                        className="underline"
                        onClick={() =>
                          void api(`/sessions/${s.id}`, { method: "DELETE" })
                            .then(async () => {
                              setSessions(await api("/sessions"));
                              await identify();
                            })
                            .catch(fail)
                        }
                      >
                        撤销会话
                      </button>
                    )}
                  </div>
                ))}
                {adminTasks.length > 0 && (
                  <>
                    <h2 className="text-xl">管理任务</h2>
                    {adminTasks.map((t) => (
                      <div className="border p-3" key={t.id}>
                        {t.id} · {label[t.status]} · {t.error_code}
                        {[
                          ["cancel", "取消"],
                          ["retry", "受控重试"],
                        ].map(([action, title]) => (
                          <button
                            key={action}
                            className="underline ml-3"
                            onClick={() =>
                              void api(`/admin/tasks/${t.id}/${action}`, post())
                                .then(async () =>
                                  setAdminTasks(await api("/admin/tasks")),
                                )
                                .catch(fail)
                            }
                          >
                            {title}
                          </button>
                        ))}
                      </div>
                    ))}
                  </>
                )}
              </section>
            )}
            {tab === "reports" && (
              <section className="overflow-auto p-6 space-y-4">
                {!reports.length && <p>研究报告会保存在这里。</p>}
                {reports.map((r) => (
                  <article
                    key={`${r.id}-${r.report_version}`}
                    className="bg-white border rounded p-5"
                  >
                    <h2 className="text-xl mb-3">
                      {r.title || "研究报告"} · v{r.report_version}
                    </h2>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {markdown(r.content)}
                    </ReactMarkdown>
                    <button
                      className="underline mt-4"
                      onClick={() => {
                        const a = document.createElement("a"),
                          url = URL.createObjectURL(
                            new Blob([markdown(r.content)], {
                              type: "text/markdown",
                            }),
                          );
                        a.href = url;
                        a.download = "research-report.md";
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      导出 Markdown
                    </button>
                  </article>
                ))}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
