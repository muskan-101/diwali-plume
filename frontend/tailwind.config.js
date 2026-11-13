/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          main: '#0B1220',
          sidebar: '#080E19',
          card: '#121D2D',
          elevated: '#172538',
        },
        border: {
          subtle: '#243247',
          active: '#36C5F0',
        },
        accent: {
          cyan: '#36C5F0',
          teal: '#27C7A5',
          amber: '#F4B942',
          coral: '#F06464',
        },
        txt: {
          main: '#EDF3FA',
          muted: '#91A1B5',
          dim: '#627285',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
