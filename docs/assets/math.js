document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".arithmatex").forEach(function (element) {
    renderMathInElement(element, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      trust: false
    });
  });
});
