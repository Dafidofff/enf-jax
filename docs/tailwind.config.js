/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js,md}", "./*.{html,js,md}", "./_layouts/*.{html,md,js}"],
  theme: {
    extend: {
      fontFamily: {
        fragment: ["Fragment", "mono"],
        dm: ["DM", "sans-serif"],
        computer_modern_bright: ["Computer Modern Bright", "sans-serif"],
        computer_modern_concrete: ["Computer Modern Concrete", "sans-serif"],
        computer_modern_sans: ["Computer Modern Sans", "sans-serif"],
        computer_modern_serif: ["Computer Modern Serif", "serif"],
        computer_modern_typewriter: ["Computer Modern Typewriter", "mono"],
        open_sans: ["Open Sans", "sans-serif"],
      },
    },
    patterns: {
      opacity: {
          100: "1",
          80: ".80",
          60: ".60",
          40: ".40",
          20: ".20",
          10: ".10",
          5: ".05",
      },
      size: {
          1: "0.25rem",
          2: "0.5rem",
          4: "1rem",
          6: "1.5rem",
          8: "2rem",
          16: "4rem",
          20: "5rem",
          24: "6rem",
          32: "8rem",
          64: "16rem",
          128: "32rem"
      }
    }
  },
  plugins: [
    require('tailwindcss-bg-patterns'),
  ],
}
