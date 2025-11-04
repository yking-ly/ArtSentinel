"use client";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const [isDark, setIsDark] = useState<boolean>(false);

  useEffect(() => {
    setMounted(true);
    // read initial state
    const has = document.documentElement.classList.contains("dark");
    setIsDark(has);
  }, []);

  const toggle = () => {
    const el = document.documentElement.classList;
    const next = !isDark;
    setIsDark(next);
    el.toggle("dark", next);
    try {
      localStorage.setItem("as-theme", next ? "dark" : "light");
    } catch {}
  };

  if (!mounted) return null;

  return (
    <button
      aria-label="Toggle dark mode"
      onClick={toggle}
      className="theme-toggle fixed right-4 top-4 z-50"
    >
      {isDark ? (
        // Sun icon
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M12 4V2M12 22v-2M4.93 4.93L3.51 3.51M20.49 20.49l-1.42-1.42M22 12h-2M4 12H2M19.07 4.93l1.42-1.42M4.93 19.07l-1.42 1.42"
            stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.5"/>
        </svg>
      ) : (
        // Moon icon
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"
            stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )}
    </button>
  );
}
