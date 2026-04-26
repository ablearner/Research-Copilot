"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles } from "lucide-react";

type NavItem = {
  href: string;
  label: string;
  matches: string[];
};

const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Research-Copilot", matches: ["/", "/research"] },
];

function isActivePath(pathname: string, item: NavItem): boolean {
  return item.matches.some(
    (match) =>
      pathname === match || (match !== "/" && pathname.startsWith(`${match}/`))
  );
}

export function AppNavigation() {
  const pathname = usePathname();

  return (
    <nav className="inline-flex items-center gap-1 rounded-lg border border-gray-200 bg-white p-1">
      {NAV_ITEMS.map((item) => {
        const active = isActivePath(pathname, item);
        const itemClass = `group relative flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[12px] font-medium transition-colors ${
          active
            ? "bg-gray-900 text-white"
            : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
        }`;

        if (active) {
          return (
            <span
              key={item.href}
              aria-current="page"
              className={itemClass}
            >
              <Sparkles className="h-3.5 w-3.5 text-gray-400" />
              {item.label}
            </span>
          );
        }

        return (
          <Link
            key={item.href}
            href={item.href}
            scroll={false}
            className={itemClass}
          >
            <Sparkles className="h-3.5 w-3.5 text-gray-400" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
