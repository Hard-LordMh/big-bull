/* BIG BULL landing page — app URL wiring.
 *
 * Replace APP_URL below with your live Streamlit Cloud address, e.g.
 *   "https://big-bull.streamlit.app"
 * Then redeploy (or just push) and every CTA on the page updates at once.
 */
const APP_URL = "https://big-bull.streamlit.app";

document.querySelectorAll("#nav-cta, #hero-cta, #band-cta, #foot-cta").forEach((el) => {
  el.setAttribute("href", APP_URL);
  el.setAttribute("target", "_blank");
  el.setAttribute("rel", "noopener noreferrer");
});

// Mark external CTAs so it is obvious they leave the page.
const ctas = document.querySelectorAll("#nav-cta, #hero-cta, #band-cta, #foot-cta");
ctas.forEach((el) => {
  const arrow = document.createElement("span");
  arrow.textContent = "→";
  arrow.style.marginLeft = "0.5rem";
  arrow.setAttribute("aria-hidden", "true");
  el.appendChild(arrow);
});

// Fade sections in on scroll (respects reduced-motion).
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!reduced && "IntersectionObserver" in window) {
  document.querySelectorAll(".card, .feat, .step, .hstat").forEach((el, i) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(10px)";
    el.style.transition = "opacity .45s ease, transform .45s ease";
    new IntersectionObserver((entries, obs) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const delay = Number(el.dataset.delay || 0);
        setTimeout(() => {
          el.style.opacity = "1";
          el.style.transform = "translateY(0)";
        }, delay);
        obs.unobserve(el);
      });
    }, { threshold: 0.12 }).observe(el);
  });

  // Stagger siblings for a softer cascade.
  document.querySelectorAll(".grid-2, .grid-3, .steps, .hero-stats").forEach((group) => {
    [...group.children].forEach((child, i) => {
      child.dataset.delay = String(i * 60);
    });
  });
}