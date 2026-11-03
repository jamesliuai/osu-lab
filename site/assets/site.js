// Small enhancements only; the page reads completely without script (the header then stays hidden, and the
// opening carries the same document links).
(() => {
  // Reveal the header once the opening's links have scrolled above the top of the view.
  const header = document.querySelector(".site-header");
  const opening = document.querySelector(".hero .actions");
  if (header) {
    if ("IntersectionObserver" in window && opening) {
      new IntersectionObserver(([entry]) => {
        header.classList.toggle("is-revealed", !entry.isIntersecting && entry.boundingClientRect.top < 0);
      }).observe(opening);
    } else {
      header.classList.add("is-revealed");
    }
  }

  // Mark the section in view in the contents list and the header's section links.
  const links = new Map();
  document.querySelectorAll('.toc a[href^="#"], .section-nav a[href^="#"]').forEach((a) => {
    const id = a.hash.slice(1);
    if (!links.has(id)) links.set(id, []);
    links.get(id).push(a);
  });
  if ("IntersectionObserver" in window && links.size) {
    const seen = new Map();
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((e) => seen.set(e.target.id, e.isIntersecting));
      const current = [...links.keys()].find((id) => seen.get(id));
      links.forEach((group, id) => group.forEach((a) => {
        if (id === current) a.setAttribute("aria-current", "true");
        else a.removeAttribute("aria-current");
      }));
    }, { rootMargin: "-25% 0px -65% 0px" });
    links.forEach((_, id) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }

  // Sideways-scrolling areas take keyboard focus only while they actually overflow.
  const scrollers = document.querySelectorAll(".scroller, .table-scroll");
  const updateScrollers = () => scrollers.forEach((el) => {
    if (el.scrollWidth > el.clientWidth + 1) el.setAttribute("tabindex", "0");
    else el.removeAttribute("tabindex");
  });
  updateScrollers();
  window.addEventListener("resize", updateScrollers);

  // Copy buttons (shown only when script runs).
  document.querySelectorAll("button.copy[data-copy]").forEach((btn) => {
    const source = document.getElementById(btn.dataset.copy);
    if (!source) return;
    btn.hidden = false;
    const label = btn.textContent;
    btn.addEventListener("click", async () => {
      let done = false;
      try {
        await navigator.clipboard.writeText(source.textContent);
        done = true;
      } catch {
        const range = document.createRange();
        range.selectNodeContents(source);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
      btn.textContent = done ? "Copied" : "Selected; press Ctrl+C";
      setTimeout(() => { btn.textContent = label; }, 1800);
    });
  });
})();
