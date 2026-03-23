/** @type {import('tailwindcss').Config} */
const { fontFamily } = require("tailwindcss/defaultTheme");

module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Dark mode always on via class on <html>
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mont: ["var(--font-mont)", ...fontFamily.sans],
      },
      colors: {
        dark: "#0a0e17",
        darkAlt: "#111827",
        darkCard: "#151d2e",
        light: "#e2e8f0",
        lightMuted: "#94a3b8",
        primary: "#3b82f6",       // blue-500
        primaryDark: "#60a5fa",   // blue-400
        accent: "#f59e0b",        // amber-500
        accentLight: "#fbbf24",   // amber-400
        emerald: "#10b981",
        rose: "#f43f5e",
      },
      animation: {
        "spin-slow": "spin 8s linear infinite",
        "pulse-slow": "pulse 4s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 5px rgba(59,130,246,0.3), 0 0 20px rgba(59,130,246,0.1)" },
          "100%": { boxShadow: "0 0 10px rgba(59,130,246,0.5), 0 0 40px rgba(59,130,246,0.2)" },
        },
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px)",
        "radial-glow": "radial-gradient(ellipse at center, rgba(59,130,246,0.08) 0%, transparent 70%)",
      },
      backgroundSize: {
        "grid": "60px 60px",
      },
      boxShadow: {
        "glow": "0 0 15px rgba(59,130,246,0.3)",
        "glow-lg": "0 0 30px rgba(59,130,246,0.2), 0 0 60px rgba(59,130,246,0.1)",
        "card": "0 4px 20px rgba(0,0,0,0.3)",
      },
    },
    screens: {
      "2xl": { max: "1535px" },
      xl: { max: "1279px" },
      lg: { max: "1023px" },
      md: { max: "767px" },
      sm: { max: "639px" },
      xs: { max: "479px" },
    },
  },
  plugins: [
    function ({ addVariant }) {
      addVariant("child", "& > *");
      addVariant("child-hover", "& > *:hover");
    },
  ],
};
