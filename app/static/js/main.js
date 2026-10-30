/**
 * TruthLens Main Frontend Interactions & Scripting
 * - Dark/Light Theme Switcher (LocalStorage persisted)
 * - Password Show/Hide & Real-time Entropy Meter
 * - Dynamic Input Mode Tab Switching
 * - Chart.js Initializers
 * - Print / Share Trigger
 */

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initPasswordToggles();
  initPasswordStrengthMeter();
  initInputModeSwitcher();
});

/* ================= Theme Toggle ================= */
function initThemeToggle() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  const htmlRoot = document.documentElement;

  // Retrieve saved theme or system preference
  const savedTheme = localStorage.getItem("truthlens-theme");
  if (savedTheme) {
    htmlRoot.setAttribute("data-theme", savedTheme);
    updateThemeBtnIcon(toggleBtn, savedTheme);
  }

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = htmlRoot.getAttribute("data-theme") || "light";
      const nextTheme = current === "dark" ? "light" : "dark";
      htmlRoot.setAttribute("data-theme", nextTheme);
      localStorage.setItem("truthlens-theme", nextTheme);
      updateThemeBtnIcon(toggleBtn, nextTheme);

      // Trigger redraw of any active charts with updated theme colors
      if (window.truthlensCharts) {
        window.truthlensCharts.forEach(c => c.update());
      }
    });
  }
}

function updateThemeBtnIcon(btn, theme) {
  if (!btn) return;
  btn.textContent = theme === "dark" ? "☀ LIGHT" : "☽ DARK";
}

/* ================= Password Utilities ================= */
function initPasswordToggles() {
  const toggleBtns = document.querySelectorAll(".toggle-password-btn");
  toggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (input) {
        if (input.type === "password") {
          input.type = "text";
          btn.textContent = "HIDE";
        } else {
          input.type = "password";
          btn.textContent = "SHOW";
        }
      }
    });
  });
}

function initPasswordStrengthMeter() {
  const pwdInput = document.getElementById("reg_password");
  const meter = document.getElementById("password-strength-bar");
  const label = document.getElementById("password-strength-text");

  if (!pwdInput || !meter || !label) return;

  pwdInput.addEventListener("input", () => {
    const val = pwdInput.value;
    let score = 0;
    if (val.length >= 8) score += 25;
    if (/[A-Z]/.test(val)) score += 25;
    if (/[0-9]/.test(val)) score += 25;
    if (/[^A-Za-z0-9]/.test(val)) score += 25;

    meter.style.width = score + "%";
    if (score < 50) {
      meter.style.backgroundColor = "var(--coral)";
      label.textContent = "Weak (requires mixed case, digits & symbols)";
      label.style.color = "var(--coral)";
    } else if (score < 100) {
      meter.style.backgroundColor = "var(--yellow)";
      label.textContent = "Moderate (add symbols for maximum security)";
      label.style.color = "var(--yellow)";
    } else {
      meter.style.backgroundColor = "var(--teal)";
      label.textContent = "Strong password";
      label.style.color = "var(--teal)";
    }
  });
}

/* ================= Input Mode Switcher ================= */
function initInputModeSwitcher() {
  const modeRadios = document.querySelectorAll("input[name='input_mode']");
  const textGroup = document.getElementById("input-group-text");
  const urlGroup = document.getElementById("input-group-url");
  const fileGroup = document.getElementById("input-group-file");

  if (!modeRadios.length) return;

  function switchMode(selected) {
    if (textGroup) textGroup.style.display = selected === "text" ? "block" : "none";
    if (urlGroup) urlGroup.style.display = selected === "url" ? "block" : "none";
    if (fileGroup) fileGroup.style.display = selected === "file" ? "block" : "none";
  }

  modeRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      switchMode(e.target.value);
    });
    if (radio.checked) {
      switchMode(radio.value);
    }
  });
}

/* ================= Printable Report ================= */
function printReport() {
  window.print();
}

