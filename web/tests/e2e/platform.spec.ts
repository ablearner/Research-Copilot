import { test, expect } from "@playwright/test";
import fs from "node:fs";

test("register, private paper import, queued Agent, refresh and cross-user isolation", async ({
  page,
  browser,
}) => {
  const email = `browser-${Date.now()}@example.org`;
  await page.goto("/");
  await page.getByRole("button", { name: "创建新账号", exact: true }).click();
  await page.getByLabel("邮箱", { exact: true }).fill(email);
  await page.getByLabel("密码", { exact: true }).fill("test-password-123");
  await page.getByRole("button", { name: "注册", exact: true }).click();
  await expect(page.getByRole("button", { name: "退出登录" })).toBeVisible();
  await page.getByRole("button", { name: "账号与管理", exact: true }).click();
  await expect(page.getByRole("heading", { name: "个人凭证" })).toBeVisible();
  page.once("dialog", (d) => d.accept("Browser Research"));
  await page.getByRole("button", { name: "＋ 新建项目", exact: true }).click();
  await expect(page.getByLabel("研究项目")).toContainText("Browser Research");
  await page.getByRole("button", { name: "论文库", exact: true }).click();
  await page.locator("input[type=file]").setInputFiles({
    name: "evidence.pdf",
    mimeType: "application/pdf",
    buffer: fs.readFileSync("tests/fixtures/evidence.pdf"),
  });
  await expect(page.getByText("evidence.pdf", { exact: true })).toBeVisible();
  await expect(page.getByText("解析：已完成 · 索引：已完成")).toBeVisible({
    timeout: 60000,
  });
  const url = await page
    .getByText("evidence.pdf", { exact: true })
    .getAttribute("href");
  const stranger = await browser.newContext();
  const other = await stranger.newPage();
  await other.goto("/");
  await other.getByRole("button", { name: "创建新账号", exact: true }).click();
  await other
    .getByLabel("邮箱", { exact: true })
    .fill(`other-${Date.now()}@example.org`);
  await other.getByLabel("密码", { exact: true }).fill("test-password-123");
  await other.getByRole("button", { name: "注册", exact: true }).click();
  await expect(other.getByRole("button", { name: "退出登录" })).toBeVisible();
  expect(
    (await other.request.get(`${new URL(page.url()).origin}${url}`)).status(),
  ).toBe(404);
  await stranger.close();
  await page.getByRole("button", { name: "研究对话", exact: true }).click();
  await page
    .getByLabel("研究问题")
    .fill("What accuracy does method Alpha achieve?");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator("article")).toHaveCount(2, { timeout: 60000 });
  await page.reload();
  await page.getByRole("button", { name: "New research", exact: true }).click();
  await expect(page.locator("article")).toHaveCount(2);
  await page.screenshot({
    path: "test-results/research-workspace.png",
    fullPage: true,
  });
  expect((await page.request.get("/api/files/anything.pdf")).status()).toBe(
    404,
  );
  page.once("dialog", (d) => d.accept("Renamed Research"));
  await page.getByRole("button", { name: "重命名项目", exact: true }).click();
  await expect(page.getByLabel("研究项目")).toContainText("Renamed Research");
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "删除项目", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "开始一个研究项目" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "退出登录" }).click();
  await expect(
    page.getByRole("button", { name: "登录", exact: true }),
  ).toBeVisible();
});

test("switching accounts clears one-time private credentials", async ({
  page,
}) => {
  async function register(email: string) {
    if (
      await page
        .getByRole("button", { name: "创建新账号", exact: true })
        .isVisible()
    )
      await page
        .getByRole("button", { name: "创建新账号", exact: true })
        .click();
    await page.getByLabel("邮箱", { exact: true }).fill(email);
    await page.getByLabel("密码", { exact: true }).fill("test-password-123");
    await page.getByRole("button", { name: "注册", exact: true }).click();
    await expect(page.getByRole("button", { name: "退出登录" })).toBeVisible();
    await page.getByRole("button", { name: "账号与管理", exact: true }).click();
  }
  await page.goto("/");
  await register(`credential-a-${Date.now()}@example.org`);
  page.once("dialog", (d) => d.accept("Private A credential"));
  await page.getByRole("button", { name: "创建读写凭证（30 天）" }).click();
  await expect(page.locator("code")).toContainText("rc_");
  const secret = await page.locator("code").innerText();
  await page.getByRole("button", { name: "退出登录" }).click();
  await expect(page.getByLabel("邮箱", { exact: true })).toBeVisible();
  await register(`credential-b-${Date.now()}@example.org`);
  await expect(page.getByText(secret, { exact: true })).toHaveCount(0);
  await expect(
    page.getByText("Private A credential", { exact: false }),
  ).toHaveCount(0);
});
