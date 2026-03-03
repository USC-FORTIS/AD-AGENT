const nodes = document.querySelectorAll(".reveal");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.15 }
);

nodes.forEach((node, i) => {
  node.style.transitionDelay = `${i * 70}ms`;
  observer.observe(node);
});
