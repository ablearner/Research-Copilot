import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Research-Copilot",
  description:
    "Multi-agent research assistant with literature discovery, grounded QA, and document-chart tooling",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN" className="antialiased">
      <body className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
        {children}
      </body>
    </html>
  );
}
