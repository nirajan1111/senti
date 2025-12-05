/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        positive: '#22c55e',
        negative: '#ef4444',
        neutral: '#6b7280',
      },
    },
  },
  plugins: [],
}
