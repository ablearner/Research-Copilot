# VS Code Remote / Codex 登录排查记录

## 现象

Codex 插件在 VS Code Remote 里反复登录失败，日志里出现两类错误：

- `fetch failed`
- `403 Forbidden`，`unsupported_country_region_territory`

表面上看像是代理可用，但 Codex 进程实际并没有继承到正确的代理环境。

## 根因

1. WSL 交互 shell 里原来会自动导出代理，但 VS Code Remote / Codex 进程并不一定吃到同一份环境。
2. VS Code Server 的机器级设置里曾经有 `http.proxy` 配置，容易和 shell 环境互相干扰。
3. 最终确认，Codex app-server 进程自身环境里一度没有代理变量，所以它直连到了受限出口，导致登录被服务端拒绝。

## 处理过程

1. 取消了 `.bashrc` 里自动导出的代理设置，避免把代理注入到所有 shell 和 VS Code Remote 进程。
2. 清空了 VS Code Server 的机器级代理配置，避免 VS Code 自己再覆盖一层。
3. 重新重载 VS Code Remote，让远端扩展主机重新启动。
4. 重启后验证新的 shell 出口已经变成新加坡，说明代理链路本身是通的。
5. 再检查 Codex 进程环境，确认它最终能继承到同样的代理环境后，登录恢复正常。

## 关键验证

- 当前 shell 的 `proxy_test` 输出为新加坡出口。
- Codex 登录日志里不再持续出现 `unsupported_country_region_territory`。
- VS Code Remote 重载后生成了新的 `exthost` 日志，说明远端扩展主机已经重新启动。

## 如果下次又失败，优先检查

1. 当前 VS Code Remote 里 `proxy_test` 是否还是新加坡出口。
2. Codex app-server 进程环境里是否还继承到 `http_proxy`、`https_proxy`、`all_proxy`。
3. `~/.vscode-server/data/Machine/settings.json` 是否又被写回了代理配置。
4. `~/.bashrc` 是否又出现了自动导出代理的内容。

## 结论

这次问题的本质不是 Codex 本体损坏，而是 VS Code Remote / WSL 的启动环境和实际代理出口没有对齐。只要让 Codex 进程真正继承到可用的代理，并且出口在支持地区，登录就会恢复。