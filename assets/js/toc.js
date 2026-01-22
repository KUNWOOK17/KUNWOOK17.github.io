document.addEventListener("DOMContentLoaded", () => {
  const toc = document.getElementById("toc");
  if (!toc) return;

  const headings = document.querySelectorAll(".post-content h2, .post-content h3");

  headings.forEach((heading) => {
    if (!heading.id) {
      heading.id = heading.textContent
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9가-힣]+/g, "-");
    }

    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    link.className = heading.tagName === "H3" ? "toc-h3" : "toc-h2";

    toc.appendChild(link);
  });
});
